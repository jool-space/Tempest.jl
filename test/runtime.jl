# Prepared programs compute what programs compute, through generated kernels,
# on the CPU (and on CUDA when it is loaded and functional).
using Test
include("cases.jl")

scalar(x) = ndims(x) == 0 ? x[] : x
agree(a, b; rtol = 1.0e-10) = all(map((x, y) -> isapprox(scalar(Array(x)), y; rtol, atol = rtol * 1.0e-2), a, b))
host(x) = x isa AbstractArray ? x : x

single(x) = x isa AbstractArray{<:AbstractFloat} ? Float32.(x) : x isa AbstractFloat ? Float32(x) : x

function check_prepared(device, expected_target; rtol = 1.0e-10, precision = identity)
    @testset "$name" for (name, f, args) in PROGRAMS
        args = map(precision, args)
        for p in (trace(f, args...), differentiate(trace(f, args...)))
            reference = filter(!isnothing, T.leaves(interpret(p, args...)))
            ex = prepare(p, map(device, args)...)
            @test ex.target == expected_target
            got = filter(!isnothing, T.leaves(execute!(ex)))
            @test agree(got, reference; rtol)
        end
    end
end

@testset "prepared on the CPU" begin
    check_prepared(identity, T.CPUTarget())
end

# An implementation that proposes each product alone, grown from its fold, in
# place of the generator's product with the work after it; it claims a lone
# product as free, so the cover takes it. A candidate over several domains
# (the product and a sum over its result) does not form a group.
struct HostProducts <: T.Implementation end
lone_product(p, g) = g.kind === :reduction && all(in(g.reductions), T.members(g)) && T.product_axes(p.graph, g) !== nothing
T.candidates(::HostProducts, ::T.CPUTarget, p) = [T.grow(p.graph, r, set -> length(set) <= 1)
    for g in p.groups if g.kind === :reduction && T.product_axes(p.graph, g) !== nothing for r in g.reductions]
T.claims(::HostProducts, ::T.CPUTarget, p, g) = lone_product(p, g) ? [:host] : []
T.estimate(::HostProducts, claim, target, p, g) = [T.Work(0.0)]
T.build_claim(::HostProducts, claim, target, p, g, buffers) = T.build_floor(target, p, g, buffers)

@testset "candidates from implementations" begin
    a, b = data(6, 5; scale = 0.3), data(5, 4)
    p = trace((a, b) -> tanh.(a * b), a, b)
    @test length(T.plan(T.lower(p)).groups) == 1 # the generator fuses the tanh
    T.register!(@__MODULE__, HostProducts())
    try
        ex = prepare(p, a, b)
        @test length(ex.accepted) == 1 && length(first(first(ex.accepted))) == 1
        @test length(ex.plan.groups) == 2
        @test isapprox(only(T.leaves(execute!(ex))), interpret(p, a, b); rtol = 1.0e-12)
        @test occursin("from HostProducts replaces", sprint(explain, ex))
        # The product, the tanh and a sum over the result span two domains.
        graph = T.lower(trace((a, b) -> sum(tanh.(a * b); dims = 2), a, b))
        r = only(g.reductions[1] for g in T.plan(graph).groups if T.product_axes(graph, g) !== nothing)
        @test_throws T.NotAGroup T.plan(graph; fixed = [T.grow(graph, r, set -> length(set) <= 3)])
    finally
        T.register!(@__MODULE__)
    end
end

# The floor evaluates recomputed folds too: an implementation that asks for
# them and leaves the kernels to the floor.
struct RecomputeOnTheFloor <: T.Implementation end
T.recomputes_folds(::RecomputeOnTheFloor, target) = true
T.claims(::RecomputeOnTheFloor, target, p, g) = isempty(T.recomputed_in(g)) ? () : (:floor,)
T.build_claim(::RecomputeOnTheFloor, claim, target, p, g, buffers) = error("left to the floor")

function with_recomputing_floor(f)
    T.register!(@__MODULE__, RecomputeOnTheFloor())
    try
        f()
    finally
        T.register!(@__MODULE__)
    end
end

attention_cases = [
    ("causal attention, grouped heads", (q, k, v) -> T.attention(q, k, v; causal = true),
     (data(4, 4, 9, 2; scale = 0.3), data(4, 2, 9, 2; scale = 0.2), data(4, 2, 9, 2; scale = 0.1))),
    ("attention", (q, k, v) -> T.attention(q, k, v), (data(8, 2, 7, 3; scale = 0.3), data(8, 2, 7, 3; scale = 0.2), data(8, 2, 7, 3; scale = 0.1))),
]

@testset "recomputed folds on the CPU floor" begin
    with_recomputing_floor() do
        @testset "$name" for (name, f, args) in attention_cases
            p = trace(f, args...)
            ex = prepare(p, args...)
            @test !isempty(T.recomputed_in(only(ex.plan.groups)))
            @test agree(T.leaves(execute!(ex)), [interpret(p, args...)])
        end
    end
end

using CUDACore
if CUDACore.functional()
    @testset "recomputed folds on the CUDA floor" begin
        with_recomputing_floor() do
            @testset "$name" for (name, f, args) in attention_cases
                p = trace(f, args...)
                ex = prepare(p, map(CuArray, args)...; target = T.CUDATarget(@__MODULE__))
                @test all(d -> d isa T.FloorKernels, ex.kernels)
                @test agree(T.leaves(execute!(ex)), [interpret(p, args...)]; rtol = 1.0e-8)
            end
        end
    end
    @testset "prepared on CUDA" begin
        check_prepared(x -> x isa AbstractArray ? CuArray(x) : x, T.CUDATarget())
    end
    using cuTile
    @testset "prepared on CUDA with cuTile, Float32" begin
        check_prepared(x -> x isa AbstractArray ? CuArray(x) : x, T.CUDATarget(); rtol = 2.0e-4, precision = single)
    end
    @testset "products on cuTile" begin
        w, x, b = Float32.(data(96, 80; scale = 0.1)), Float32.(data(80, 112)), Float32.(data(96))
        for target in (T.CUDATarget(cuTile), T.CUDATarget(cuTile; precision = :tf32))
            p = trace((w, x, b) -> gelu.(w * x .+ b), w, x, b)
            ex = prepare(p, CuArray(w), CuArray(x), CuArray(b); target)
            @test any(d -> !(d isa T.FloorKernels) && d.schedule.sketch === :product, ex.kernels)
            @test isapprox(Array(only(T.leaves(execute!(ex)))), interpret(p, w, x, b); rtol = target.precision === :tf32 ? 1.0e-2 : 1.0e-4)
        end
        # A conversion to BFloat16 is part of the epilogue: one kernel, and
        # only the BFloat16 result is stored.
        p = trace((w, x, b) -> Core.BFloat16.(gelu.(w * x .+ b)), w, x, b)
        ex = prepare(p, CuArray(w), CuArray(x), CuArray(b); target = T.CUDATarget(cuTile; precision = :tf32))
        @test length(ex.kernels) == 1 && only(ex.kernels).schedule.sketch === :product
        y = only(T.leaves(execute!(ex)))
        @test eltype(y) === Core.BFloat16
        @test isapprox(Float32.(Array(y)), Float32.(interpret(p, w, x, b)); rtol = 1.0e-2)
        # Half operands accumulate in Float32 and round once.
        h, y = Float16.(data(64, 48; scale = 0.1)), Float16.(data(48, 32))
        p = trace((h, y) -> h * y, h, y)
        ex = prepare(p, CuArray(h), CuArray(y); target = T.CUDATarget(cuTile))
        @test any(d -> !(d isa T.FloorKernels) && d.schedule.sketch === :product, ex.kernels)
        @test isapprox(Array(only(T.leaves(execute!(ex)))), interpret(p, h, y); rtol = eps(Float16))
        # A product spelled as a broadcast and a sum is a product too.
        a, c = Float32.(data(64, 48; scale = 0.1)), Float32.(data(48, 80))
        ex = prepare(trace(spelled_product, a, c), CuArray(a), CuArray(c); target = T.CUDATarget(cuTile))
        @test any(d -> !(d isa T.FloorKernels) && d.schedule.sketch === :product, ex.kernels)
        @test isapprox(Array(only(T.leaves(execute!(ex)))), a * c; rtol = 1.0e-4)
        # A long sum over a small output splits among blocks (split-K) and merges.
        u, z = Float32.(data(64, 3000; scale = 0.01)), Float32.(data(3000, 48))
        for target in (T.CUDATarget(cuTile), T.CUDATarget(cuTile; precision = :tf32))
            p = trace((u, z) -> u * z, u, z)
            ex = prepare(p, CuArray(u), CuArray(z); target)
            @test any(d -> !(d isa T.FloorKernels) && d.schedule.sketch === :product && d.schedule.splits > 1, ex.kernels)
            @test isapprox(Array(only(T.leaves(execute!(ex)))), interpret(p, u, z); rtol = target.precision === :tf32 ? 1.0e-2 : 1.0e-4)
        end
    end
    using cuBLAS
    @testset "products on cuBLAS" begin
        @test Base.get_extension(Tempest, :cuBLASExt) !== nothing
        built(ex) = [d for d in ex.kernels if d isa Base.get_extension(Tempest, :cuBLASExt).BuiltProduct]
        a, b = Float32.(data(96, 80; scale = 0.1)), Float32.(data(80, 112))
        at, bt = permutedims(a), permutedims(b)
        A, B = Float32.(data(32, 24, 5; scale = 0.1)), Float32.(data(24, 40, 5; scale = 0.2))
        # Factors in either orientation, a result stored transposed, a batch,
        # and half factors accumulating in Float32.
        layouts = [((a, b) -> a * b, (a, b)), ((at, b) -> transpose(at) * b, (at, b)),
            ((a, bt) -> a * transpose(bt), (a, bt)), ((at, bt) -> transpose(at) * transpose(bt), (at, bt)),
            ((a, b) -> T.contract(a, b, (1, 2), (2, 3), (3, 1)), (a, b)), ((A, B) -> T.batched_mul(A, B), (A, B)),
            ((a, b) -> a * b, (Float16.(a), Float16.(b))), ((a, b) -> a * b, (Core.BFloat16.(a), Core.BFloat16.(b)))]
        for (f, args) in layouts, target in (T.CUDATarget(cuBLAS), T.CUDATarget(cuBLAS; precision = :tf32))
            p = trace(f, args...)
            ex = prepare(p, map(CuArray, args)...; target)
            @test length(built(ex)) == 1
            rtol = eltype(args[1]) === Float32 ? (target.precision === :tf32 ? 1.0e-2 : 1.0e-5) : 2 * eps(eltype(args[1]))
            @test isapprox(Float32.(Array(only(T.leaves(execute!(ex))))), Float32.(interpret(p, args...)); rtol)
        end
        # With work after the product, cuBLAS's candidate is the product
        # alone, and the work after it a kernel of its own.
        w = Float32.(data(96; scale = 0.3))
        p = trace((a, b, w) -> tanh.(a * b .+ w), a, b, w)
        ex = prepare(p, CuArray(a), CuArray(b), CuArray(w); target = T.CUDATarget(cuBLAS))
        @test length(built(ex)) == 1 && length(ex.accepted) == 1 && length(ex.kernels) == 2
        @test isapprox(Array(only(T.leaves(execute!(ex)))), interpret(p, a, b, w); rtol = 1.0e-5)
        ex = prepare(trace((a, b) -> a * b, a, b), CuArray(a), CuArray(b); target = T.CUDATarget(cuTile))
        @test isempty(built(ex))
    end
    @testset "choosing between claims" begin
        # Half-precision products cuBLAS and cuTile both claim, close by
        # estimate: measured at preparation and kept in the tuning record.
        a, b = Float16.(data(1024, 256; scale = 0.1)), Float16.(data(256, 1024))
        p = trace((a, b) -> a * b, a, b)
        target = T.CUDATarget()
        prepared() = prepare(p, CuArray(a), CuArray(b); target)
        ex = prepared()
        x = only(ex.decisions)
        @test x.how === :measured && length(x.options) == 2
        @test occursin("chosen by measurement", sprint(explain, ex))
        path = T.record_path(target)
        lines = readlines(path)
        @test length(lines) >= 2
        # Given the record, preparing again measures nothing and chooses alike.
        again = prepared()
        @test readlines(path) == lines && only(again.decisions).choice.key == x.choice.key
        # A measurement overrides the estimate only beyond the noise margin.
        rewrite(f) = write(path, join((string(k, '\t', f(k, parse(Float64, t))) for (k, t) in split.(lines, '\t')), '\n') * '\n')
        other = only(o for o in x.options if o.key != x.choice.key)
        rewrite((k, t) -> k == other.key ? 1.0e-7 : 1.0e-5)
        @test only(prepared().decisions).choice.key == other.key
        rewrite((k, t) -> 1.0e-5)
        @test only(prepared().decisions).choice.key == first(x.options).key
        @test isapprox(Array(only(T.leaves(execute!(prepared())))), interpret(p, a, b); rtol = 2 * eps(Float16))
    end
    @testset "implementation sets" begin
        w, x, b = Float32.(data(96, 80; scale = 0.1)), Float32.(data(80, 112)), Float32.(data(96))
        p = trace((w, x, b) -> gelu.(w * x .+ b), w, x, b)
        # Loading a package makes its implementations available; a target
        # names the ones it uses, and the floor is always there.
        @test T.used_packages(T.CUDATarget()) == [cuBLAS, cuTile]
        ex = prepare(p, CuArray(w), CuArray(x), CuArray(b); target = T.CUDATarget(cuTile))
        @test ex.packages == [cuTile]
        @test occursin("implementations from cuTile, and the floor", sprint(explain, ex))
        @test_throws ArgumentError prepare(p, CuArray(w), CuArray(x), CuArray(b); target = T.CUDATarget(Test))
    end
end

# Sizes that take several blocks: full reductions merge partial states, and
# folds over other than the swept axes (the loss, a weight gradient) merge
# theirs, under every sketch.
rmsnorm32(x, w) = x ./ sqrt.(sum(abs2, x; dims = 1) ./ size(x, 1) .+ 1.0f-5) .* w
function layernorm32(x, w)
    m = sum(x; dims = 1) ./ size(x, 1)
    (x .- m) ./ sqrt.(sum(abs2, x .- m; dims = 1) ./ size(x, 1) .+ 1.0f-5) .* w
end

const LARGE = [
    ("vector sum", x -> sum(abs2, x), (data(1 << 16),)),
    ("dot product", (a, b) -> sum(a .* b), (data(1 << 16), data(1 << 16; scale = 0.3))),
    ("matrix sum and shift", x -> sum(exp.(x .- maximum(x))), (data(512, 300),)),
    ("RMSNorm training, resident", (x, w) -> sum(abs2, rmsnorm32(x, w)), (data(256, 700), data(256; scale = 0.1))),
    ("softmax training, resident", (x, t) -> sum(softmax_columns(x) .* t), (data(256, 700), data(256, 700; scale = 0.7))),
    ("softmax training, streamed", (x, t) -> sum(softmax_columns(x) .* t), (data(9000, 40), data(9000, 40; scale = 0.7))),
    ("row softmax training, split", (x, t) -> sum(softmax_rows(x) .* t), (data(100, 3000), data(100, 3000; scale = 0.7))),
    ("layer norm training", (x, w) -> sum(abs2, layernorm32(x, w)), (data(256, 700), data(256; scale = 0.1))),
]

if CUDACore.functional()
    @testset "several blocks with cuTile, Float32" begin
        @testset "$name" for (name, f, args) in LARGE
            args = map(single, args)
            for p in (trace(f, args...), differentiate(trace(f, args...)))
                reference = T.leaves(interpret(p, args...))
                ex = prepare(p, map(x -> x isa AbstractArray ? CuArray(x) : x, args)...)
                @test all(d -> !(d isa T.FloorKernels), ex.kernels)
                @test agree(T.leaves(execute!(ex)), reference; rtol = 2.0e-3)
            end
        end
    end
end

# A fold of the user's with a tuple state, Welford's count, mean and sum of
# squared deviations, generated like the built-in folds under every sketch.
function welford(a, b)
    n = a[1] + b[1]
    w = ifelse(n == 0, zero(n), b[1] / n)
    d = b[2] - a[2]
    (n, a[2] + d * w, a[3] + b[3] + d * d * a[1] * w)
end
const VARIANCE = T.GenericFold(; init = T -> (zero(T), zero(T), zero(T)), lift = x -> (one(x), x, zero(x)),
    combine = welford, observe = s -> s[3] / s[1])

if CUDACore.functional()
    @testset "a fold with a tuple state on cuTile" begin
        for (sketch, x, dims) in ((:resident, data(300, 200), 1), (:streamed, data(9000, 40), 1), (:split, data(100, 3000), 2),
            (:full, data(512, 300), (1, 2)))
            x = single(x)
            p = trace(x -> T.fold(VARIANCE, x; dims), x)
            ex = prepare(p, CuArray(x); target = T.CUDATarget(cuTile))
            @test only(ex.kernels).schedule.sketch === sketch
            @test isapprox(Array(only(T.leaves(execute!(ex)))), interpret(p, x); rtol = 1.0e-4)
        end
    end
end

# Positions on tiles, and formulas of the axes the grid binds computed per
# block: a causal mask after a product, positions in pointwise work and in a
# fold, and key heads shared among query heads, spelled as a broadcast that
# reads key head h ÷ G.
causal_scores(a, b) = (s = a' * b; ifelse.(T.axis_positions(s, 1) .<= T.axis_positions(s, 2), s, -Inf32))
function grouped_scores(q, k)
    D, H, S, B = size(q)
    G = H ÷ size(k, 2)
    shared = reshape(reshape(k, D, 1, size(k, 2), S, B) .* ones(Float32, 1, G, 1, 1, 1), D, H, S, B)
    T.contract(shared, q, (1, 2, 3, 4), (1, 2, 5, 4), (3, 5, 2, 4))
end

const POSITIONAL = [
    ("causal mask after a product", (a, b) -> exp.(causal_scores(a, b) .- T.logsumexp(causal_scores(a, b); dims = 1)),
     (data(24, 70; scale = 0.1), data(24, 70; scale = 0.2))),
    ("positions in pointwise work", x -> x .* T.axis_positions(x, 2) .+ T.positions(x), (data(50, 30, 3),)),
    ("positions in a fold", x -> sum(ifelse.(T.axis_positions(x, 1) .<= T.axis_positions(x, 2), x, 0.0f0); dims = 1), (data(300, 200),)),
    ("grouped heads by division", grouped_scores, (data(16, 6, 40, 2; scale = 0.1), data(16, 2, 40, 2; scale = 0.2))),
    ("shared columns by division in a fold",
     x -> sum(reshape(reshape(x, 64, 1, 3, 10) .* ones(Float32, 1, 2, 1, 1), 64, 6, 10); dims = 1), (data(64, 3, 10),)),
]

if CUDACore.functional()
    @testset "positions and per-block formulas on cuTile" begin
        @testset "$name" for (name, f, args) in POSITIONAL
            args = map(single, args)
            p = trace(f, args...)
            ex = prepare(p, map(CuArray, args)...; target = T.CUDATarget(cuTile))
            @test all(d -> !(d isa T.FloorKernels), ex.kernels)
            @test agree(T.leaves(execute!(ex)), T.leaves(interpret(p, args...)); rtol = 2.0e-4)
        end
    end
end

# Attention generated from its reference: one kernel, a fold over keys whose
# scores and logsumexp are recomputed in it, online or in two passes.
const GENERATED_ATTENTION = [
    ("causal, grouped heads", (16, 4, 2, 200, 2), true),
    ("grouped heads", (16, 4, 2, 200, 2), false),
    ("causal, odd sizes", (32, 3, 1, 70, 3), true),
    ("one head", (8, 1, 1, 133, 1), false),
]

if CUDACore.functional()
    @testset "attention generated on cuTile" begin
        @testset "$name" for (name, (D, H, KV, S, B), causal) in GENERATED_ATTENTION
            q, k, v = single(data(D, H, S, B; scale = 0.1)), single(data(D, KV, S, B; scale = 0.2)), single(data(D, KV, S, B; scale = 0.3))
            p = trace((q, k, v) -> T.attention(q, k, v; causal), q, k, v)
            for online in (true, false)
                ex = prepare(p, map(CuArray, (q, k, v))...; target = T.CUDATarget(cuTile; online))
                kernel = only(ex.kernels)
                @test !(kernel isa T.FloorKernels)
                @test occursin("online softmax", string(kernel)) == online
                @test agree(T.leaves(execute!(ex)), [interpret(p, q, k, v)]; rtol = 1.0e-4)
            end
        end
        # The reference's gradient: every product generated, also those that
        # sum over key heads shared by one query head each.
        q, k, v = single(data(16, 2, 70, 2; scale = 0.1)), single(data(16, 2, 70, 2; scale = 0.2)), single(data(16, 2, 70, 2; scale = 0.3))
        d = differentiate(trace((q, k, v) -> sum(abs2, T.attention(q, k, v; causal = true)), q, k, v))
        ex = prepare(d, map(CuArray, (q, k, v))...; target = T.CUDATarget(cuTile))
        @test all(d -> !(d isa T.FloorKernels), ex.kernels)
        @test agree(filter(!isnothing, T.leaves(execute!(ex))), filter(!isnothing, T.leaves(interpret(d, q, k, v))); rtol = 1.0e-4)
    end
end

include("cache.jl")
