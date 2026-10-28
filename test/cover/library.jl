# A library computes a function Tempest can compute, its template: Tempest
# finds the groups that compute it, by tracing it at their extents and element
# types and comparing the two at the index level, and hands the library each
# array as a strided view. A product binds whatever its factors' layout and
# batch; a product with work before or after it binds to nothing, and so does
# a template whose literals differ from the group's at the group's extents.
# A library's own launches compute what it binds.
include("../fixtures/programs.jl")

product(A, B) = C.contract(A, B, (1, 3, 4), (3, 2, 4), (1, 2, 4))
const PRODUCT = C.Template(product; inputs = (A = (:m, :k, :b), B = (:k, :n, :b)), outputs = (:C,))

bound(t, f, args...) = [C.match_template(t, 1, T.CPUTarget(), g) for g in C.group_views(C.plan(C.rewrite(C.lower(trace(f, args...)))))]
layout(s) = (s.size, s.strides, s.offset)

@testset "a template binds the groups that compute it" begin
    a, b = data(6, 4), data(4, 5)
    m = only(bound(PRODUCT, (a, b) -> a * b, a, b))
    # An axis no axis of the group stands for has extent one.
    @test m.extents == (m = 6, k = 4, b = 1, n = 5)
    @test map(layout, m.arrays) == (A = ((6, 4, 1), (1, 6, 0), 0), B = ((4, 5, 1), (1, 4, 0), 0), C = ((6, 5, 1), (1, 6, 0), 0))
    @test only(bound(PRODUCT, (a, b) -> permutedims(a) * b, data(4, 6), b)).arrays.A.strides == (4, 1, 0)
    h = only(bound(PRODUCT, (a, b) -> a * b, BFloat16.(single(a)), BFloat16.(single(b))))
    @test h.arrays.A.eltype === h.arrays.C.eltype === BFloat16
    # Attention's scores, keys by queries summing features: heads and batch
    # are one axis of the batch, whose strides chain.
    q, k = data(16, 32, 2, 3), data(16, 32, 2, 3)
    s = only(bound(PRODUCT, (q, k) -> C.contract(k, q, (1, 2, 3, 4), (1, 5, 3, 4), (2, 5, 3, 4)), q, k))
    @test s.extents == (m = 32, k = 16, b = 6, n = 32)
    @test s.arrays.A.strides == (16, 1, 512) && s.arrays.C.strides == (32, 1, 1024)
    # Stored batch first, heads after: no one stride steps the batch of all three.
    @test all(isnothing, bound(PRODUCT, (q, k) -> C.contract(k, q, (1, 2, 3, 4), (1, 5, 3, 4), (2, 5, 4, 3)), q, k))
    @test all(isnothing, bound(PRODUCT, (a, b) -> a * b .+ 1, a, b))
    @test all(isnothing, bound(PRODUCT, (a, b) -> tanh.(a) * b, a, b))
    @test all(isnothing, bound(PRODUCT, a -> sum(a; dims = 1), a))
    # Groups that read tuples, as a linear recurrence's scan does, are told
    # apart before the template is traced at their types.
    @test all(isnothing, bound(PRODUCT, (a, b) -> sum(abs2, accumulate(RECURRENCE, tuple.(a, b))), 0.9 .+ 0.1 .* data(30; scale = 0.3), data(30)))
    # A mean divides by its extent: bound at the group's extents, its literal
    # is the group's, and a sum divided by another number is not a mean.
    mean = C.Template(x -> sum(x; dims = 1) ./ size(x, 1); inputs = (X = (:r, :c),), outputs = (:Y,))
    @test only(bound(mean, x -> sum(x; dims = 1) ./ 6, a)).extents == (r = 6, c = 4)
    @test all(isnothing, bound(mean, x -> sum(x; dims = 1) ./ 5, a))
    # Over eight rows the divisor is a product by an eighth, as Tempest writes
    # a quotient whose reciprocal is exact.
    @test only(bound(mean, x -> sum(x; dims = 1) ./ 8, data(8, 4))).extents == (r = 8, c = 4)
end

# A library on the CPU: products computed from their bindings alone, element
# by element, and free by its estimate, so the cover takes every product it
# binds, in a step's forward and backward alike, and the products of chunks
# of one value, which the planner groups, in one launch.
struct HostLibrary <: C.Library end
C.templates(::HostLibrary, ::T.CPUTarget) = (PRODUCT,)
C.estimate(::HostLibrary, b, target, g) = [C.Work(0.0)]
element(s, i...) = 1 + s.offset + sum(map(*, i, s.strides))
function host_product(b)
    A, B, Y = b.arrays.A, b.arrays.B, b.arrays.C
    e = b.extents
    for z in 0:(e.b - 1), j in 0:(e.n - 1), i in 0:(e.m - 1)
        Y.buffer[element(Y, i, j, z)] = sum(A.buffer[element(A, i, l, z)] * B.buffer[element(B, l, j, z)] for l in 0:(e.k - 1))
    end
end
C.build(::HostLibrary, target, b::C.Binding) = (() -> host_product(b), "host product $(b.extents.m)×$(b.extents.n)×$(b.extents.k)")
# Instances alike, as products of chunks of one value are: one launch.
C.supports(::HostLibrary, ::T.CPUTarget, bs::Vector{C.Binding}) = true
C.build(::HostLibrary, target, bs::Vector{C.Binding}) = (() -> foreach(host_product, bs), "host products ×$(length(bs))")

@testset "a library computes what it binds" begin
    C.register!(@__MODULE__, HostLibrary())
    try
        chunks(a, s...) = ntuple(k -> C.contract(a[:, (8k - 7):8k], s[k], (1, 4), (4, 2, 3), (1, 2, 3)), 4)
        whole, states = data(6, 32; scale = 0.1), [data(8, 5, 3; scale = 0.1k) for k in 1:4]
        ex = prepare(trace(chunks, whole, states...), whole, states...)
        @test occursin("host products ×4", sprint(explain, ex))
        @test agree(C.leaves(outputs(execute!(ex))), C.leaves(interpret(trace(chunks, whole, states...), whole, states...)))
        for (f, args) in (((a, b) -> sum(abs2, tanh.(permutedims(a) * b)), (data(4, 6), data(4, 5))),
                ((w, x) -> sum(abs2, tanh.(w * x)), (data(5, 6; scale = 0.3), data(6, 4))),
                ((q, k) -> sum(abs2, C.contract(k, q, (1, 2, 3, 4), (1, 5, 3, 4), (2, 5, 3, 4))), (data(8, 6, 2, 3), data(8, 6, 2, 3; scale = 0.7))))
            for p in (trace(f, args...), differentiate(trace(f, args...)))
                ex = prepare(p, args...)
                @test occursin("host product", sprint(explain, ex))
                @test agree(C.leaves(outputs(execute!(ex))), filter(!isnothing, C.leaves(interpret(p, args...))))
            end
        end
    finally
        C.register!(@__MODULE__)
    end
end
