# The index level computes what the program level computes: every program and
# its differentiated form, lowered and interpreted value by value, agrees with
# the program interpreter, and layout leaves no trace among the values.
include("../fixtures/programs.jl")

flat(x) = filter(!isnothing, C.leaves(x))

@testset "index level" begin
    @testset "$name" for (name, f, args) in [PROGRAMS; REPEATED_WRITES]
        for p in (trace(f, args...), differentiate(trace(f, args...)))
            graph = C.lower(p)
            @test agree(C.interpret(graph, args...), flat(interpret(p, args...)))
            # A scatter by index arrays is a sum by key, and a write an
            # overwrite by key into the contents written over; by integers and
            # ranges, either is a select of a read through a formula.
            @test all(graph.values) do v
                d = v.definition
                !(d isa C.Keyed) || any(e -> !isempty(IR.indirect_values(e)), d.key) &&
                    (d.fold isa IR.SumFold && d.into === nothing || d.fold isa IR.Overwrite && d.into !== nothing)
            end
            # Filling out to a larger shape is a formula, not a value. (A
            # call's argument is materialized for an implementation that
            # claims the call; nothing else reads it.)
            users = C.readers(graph)
            @test !any(k -> graph.values[k].definition isa C.Pointwise && graph.values[k].definition.f === identity &&
                            !isempty(users[k]), eachindex(graph.values))
            # A product that sums no axis, an outer product, is the pointwise
            # product, not a fold.
            @test !any(v -> v.definition isa C.Reduction && isempty(v.definition.reduced), graph.values)
        end
    end
end

# Slices concatenated in order are the array they were taken from, read
# through a formula, along the axis they were taken along or another: the
# select between them, and its adjoint, the sum of what each slice receives,
# compute no value. Slices out of order, or of different arrays, select.
@testset "pieces that cover a value" begin
    x, w = data(4, 3, 5), data(20, 3; scale = 0.3)
    slices(x, order) = [x[:, :, t] for t in order]
    selects(graph) = any(((v, l),) -> l && v.definition isa C.Pointwise && v.definition.f === ifelse,
        zip(graph.values, C.live_values(graph)))
    for (f, covers) in (((x, w) -> sum(abs2, vcat(slices(x, 1:5)...) .* w), true),
            ((x, w) -> sum(abs2, hcat(slices(x, 1:5)...) .* reshape(w, 4, 15)), true),
            ((x, w) -> sum(abs2, vcat(slices(x, [2, 1, 3, 4, 5])...) .* w), false),
            ((x, w) -> sum(abs2, vcat(slices(x, 1:4)..., w[1:4, :]) .* w), false))
        for p in (trace(f, x, w), differentiate(trace(f, x, w)))
            graph = C.lower(p)
            @test agree(C.interpret(graph, x, w), flat(interpret(p, x, w)))
            @test selects(graph) != covers
        end
    end
end

# An index out of range is an error in the interpreters.
@testset "gathers out of range" begin
    x = data(8, 6)
    p = trace((x, i) -> x[:, i], x, [1, 2])
    @test_throws BoundsError interpret(p, x, [1, 7])
    @test_throws BoundsError C.interpret(C.lower(p), x, [1, 7])
end

# An index a program computes from others by sums, differences and products
# by constants is a formula of theirs, read in the kernel that indexes by it,
# where a kernel of its own stored it: an offset, a stride, the linear index
# of Cartesian indices over two and three axes, read and written, and a
# gather by indices each step of a map computes. A minimum, a clamp, a
# remainder, a division or a choice is none, and is stored.
@testset "indices computed by sums are formulas" begin
    x, i = data(8, 3), Int32[3, 1, 4, 2]
    K, k = data(4, 2, 16, 3), data(4, 2, 4, 3; scale = 0.3)
    p = Int32.(mod1.([3, 6, 9, 12] .+ 2 .* (1:3)', 16))
    K3, a, b, c = data(4, 6, 5, 3), [6, 1, 3], [5, 2, 5], [1, 3, 2]
    J = [mod1(3b + q, 8) for q in 1:2, b in 1:3]
    columns = reshape(1:3, 1, :)
    formulas = [
        ((x, i) -> x[i .+ 1, :], (x, i)),
        ((x, i) -> x[2 .* i .- 1, :] .+ x[i .- 1 .+ 1, :], (x, i)),
        ((x, i) -> (y = copy(x); y[i .+ 1, :] .= 0; y), (x, i)),
        ((K, p) -> K[:, :, CartesianIndex.(p, columns)], (K, p)),
        ((K, k, p) -> (K = copy(K); K[:, :, CartesianIndex.(p, columns)] .= k; K), (K, k, p)),
        ((K3, a, b, c) -> K3[:, CartesianIndex.(a, b, c)], (K3, a, b, c)),
        ((x, J) -> stack(map((v, j) -> v[j], eachcol(x), eachcol(J))), (x, J)),
    ]
    stored = [
        ((x, i) -> x[clamp.(i .+ 4, 1, 8), :], (x, i)),
        ((x, i) -> x[mod1.(i .+ 6, 8), :], (x, i)),
        ((x, i) -> x[(i .- 1) .÷ 2 .+ 1, :], (x, i)),
        ((x, i) -> x[ifelse.(i .> 2, i, Int32(1)), :], (x, i)),
    ]
    for (expected, programs) in ((1, formulas), (2, stored)), (f, args) in programs
        c = T.compile(f, map(copy, args)...; target = T.CPUTarget(), measure = :none, capture = false)
        @test length(c.executable.kernels) == expected
        @test c(map(copy, args)...) ≈ f(map(copy, args)...)
    end
    # The gradient of a gather by an offset scatters at the same formula,
    # prepared as the program interprets it.
    g(x, i) = T.gradient(x -> sum(abs2, x[i .+ 1, :] .* x[2 .* i, :]), x)[1]
    @test T.compile(g, x, i; target = T.CPUTarget(), measure = :none, capture = false)(x, i) ≈ interpret(trace(g, x, i), x, i)
end
