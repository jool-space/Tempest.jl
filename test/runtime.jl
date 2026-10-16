# Prepared programs compute what programs compute, through generated kernels,
# on the CPU (and on CUDA when it is loaded and functional).
using Test
include("cases.jl")

scalar(x) = ndims(x) == 0 ? x[] : x
agree(a, b; rtol = 1.0e-10) = all(map((x, y) -> isapprox(scalar(Array(x)), y; rtol, atol = rtol * 1.0e-2), a, b))
host(x) = x isa AbstractArray ? x : x

single(x) = x isa AbstractArray ? Float32.(x) : x isa Number ? Float32(x) : x

function check_prepared(device, expected_target; rtol = 1.0e-10, precision = identity)
    @testset "$name" for (name, f, args) in PROGRAMS
        args = map(precision, args)
        for p in (trace(f, args...), differentiate(trace(f, args...)))
            reference = T.leaves(interpret(p, args...))
            ex = prepare(p, map(device, args)...)
            @test ex.target == expected_target
            got = T.leaves(execute!(ex))
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
