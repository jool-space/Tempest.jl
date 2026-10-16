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

using CUDACore
if CUDACore.functional()
    @testset "prepared on CUDA" begin
        check_prepared(x -> x isa AbstractArray ? CuArray(x) : x, T.CUDATarget())
    end
    using cuTile
    @testset "prepared on CUDA with cuTile, Float32" begin
        check_prepared(x -> x isa AbstractArray ? CuArray(x) : x, T.CUDATarget(); rtol = 2.0e-4, precision = single)
    end
    @testset "products on cuTile" begin
        w, x, b = Float32.(data(96, 80; scale = 0.1)), Float32.(data(80, 112)), Float32.(data(96))
        for target in (T.CUDATarget(), T.CUDATarget(precision = :tf32))
            p = trace((w, x, b) -> gelu.(w * x .+ b), w, x, b)
            ex = prepare(p, CuArray(w), CuArray(x), CuArray(b); target)
            @test any(d -> !(d isa T.FloorKernels) && d.schedule.sketch === :product, ex.kernels)
            @test isapprox(Array(only(T.leaves(execute!(ex)))), interpret(p, w, x, b); rtol = target.precision === :tf32 ? 1.0e-2 : 1.0e-4)
        end
        # A conversion to BFloat16 is part of the epilogue: one kernel, and
        # only the BFloat16 result is stored.
        p = trace((w, x, b) -> Core.BFloat16.(gelu.(w * x .+ b)), w, x, b)
        ex = prepare(p, CuArray(w), CuArray(x), CuArray(b); target = T.CUDATarget(precision = :tf32))
        @test length(ex.kernels) == 1 && only(ex.kernels).schedule.sketch === :product
        y = only(T.leaves(execute!(ex)))
        @test eltype(y) === Core.BFloat16
        @test isapprox(Float32.(Array(y)), Float32.(interpret(p, w, x, b)); rtol = 1.0e-2)
        # Half operands accumulate in Float32 and round once.
        h, y = Float16.(data(64, 48; scale = 0.1)), Float16.(data(48, 32))
        p = trace((h, y) -> h * y, h, y)
        ex = prepare(p, CuArray(h), CuArray(y))
        @test any(d -> !(d isa T.FloorKernels) && d.schedule.sketch === :product, ex.kernels)
        @test isapprox(Array(only(T.leaves(execute!(ex)))), interpret(p, h, y); rtol = eps(Float16))
        # A product spelled as a broadcast and a sum is a product too.
        a, c = Float32.(data(64, 48; scale = 0.1)), Float32.(data(48, 80))
        ex = prepare(trace(spelled_product, a, c), CuArray(a), CuArray(c))
        @test any(d -> !(d isa T.FloorKernels) && d.schedule.sketch === :product, ex.kernels)
        @test isapprox(Array(only(T.leaves(execute!(ex)))), a * c; rtol = 1.0e-4)
        # A long sum over a small output splits among blocks (split-K) and merges.
        u, z = Float32.(data(64, 3000; scale = 0.01)), Float32.(data(3000, 48))
        for target in (T.CUDATarget(), T.CUDATarget(precision = :tf32))
            p = trace((u, z) -> u * z, u, z)
            ex = prepare(p, CuArray(u), CuArray(z); target)
            @test any(d -> !(d isa T.FloorKernels) && d.schedule.sketch === :product && d.schedule.splits > 1, ex.kernels)
            @test isapprox(Array(only(T.leaves(execute!(ex)))), interpret(p, u, z); rtol = target.precision === :tf32 ? 1.0e-2 : 1.0e-4)
        end
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
            ex = prepare(p, map(CuArray, args)...)
            @test all(d -> !(d isa T.FloorKernels), ex.kernels)
            @test agree(T.leaves(execute!(ex)), T.leaves(interpret(p, args...)); rtol = 2.0e-4)
        end
    end
end
