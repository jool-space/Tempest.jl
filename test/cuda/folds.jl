# Folds on cuTile: sizes that take several blocks, whose partial states merge,
# a fold of the user's with a tuple state under every sketch, and extremes as
# Base finds them, on tiles and on the floor.
include("../fixtures/programs.jl")

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

@testset "several blocks with cuTile, Float32" begin
    @testset "$name" for (name, f, args) in LARGE
        args = map(single, args)
        for p in (trace(f, args...), differentiate(trace(f, args...)))
            reference = C.leaves(interpret(p, args...))
            ex = prepare(p, map(x -> x isa AbstractArray ? CuArray(x) : x, args)...)
            @test all(d -> !(d isa C.FloorKernels), ex.kernels)
            @test agree(C.leaves(outputs(execute!(ex))), reference; rtol = 2.0e-3)
        end
    end
end

# A fold of the user's with a tuple state (`VARIANCE`, in
# fixtures/programs.jl), generated like the built-in folds under every sketch,
# named and as Base spells it, its observe composed from the broadcast after.
@testset "a fold with a tuple state on cuTile" begin
    for (sketch, x, dims) in ((:resident, data(300, 200), 1), (:streamed, data(9000, 40), 1), (:split, data(100, 3000), 2),
        (:full, data(512, 300), (1, 2))), f in (x -> reduce(VARIANCE, x; dims), x -> base_variance(x; dims))
        x = single(x)
        p = trace(f, x)
        ex = prepare(p, CuArray(x); target = T.CUDATarget(cuTile))
        @test only(ex.kernels).schedule.sketch === sketch
        @test isapprox(Array(only(C.leaves(outputs(execute!(ex))))), interpret(p, x); rtol = 1.0e-4)
    end
end

# Extremes on cuTile find what Base finds: the first of equals, a NaN, 0.0
# above -0.0, within a block's tile and among the blocks a long column splits
# into, in Float32, BFloat16s.BFloat16 and Int32. Where they are along a
# vector or one axis, taken as integers, is a fold on tiles; a
# `CartesianIndex` is no tile's element, and is made on the floor.
@testset "extremes on cuTile" begin
    x = Float32.(sin.(reshape(1:(50000 * 3), 50000, 3)) .* 100)
    x[777, 1] = x[40000, 1] = 500
    x[123, 2] = x[9000, 2] = NaN
    x[:, 3] .= .-abs.(x[:, 3]) .- 1
    x[5, 3], x[6, 3], x[9, 3] = 0.0f0, -0.0f0, 0.0f0
    on_tiles = (x -> getindex.(argmax(x; dims = 1), 1), x -> argmax(x[:, 1]))
    found_in_floats(r) = map(v -> v isa Union{BFloat16s.BFloat16, AbstractArray{BFloat16s.BFloat16}} ? Float32.(v) : v, found(r))
    @testset "$(eltype(x))" for x in (x, BFloat16s.BFloat16.(x), Int32.(replace(round.(x), NaN => 0)))
        reference = x isa AbstractArray{BFloat16s.BFloat16} ? Float32.(x) : x
        for f in (on_tiles..., x -> findmax(x; dims = 1), x -> findmin(x[1:1000, :]; dims = 2), argmin, x -> extrema(x; dims = 1))
            ex = prepare(trace(f, x), CuArray(x); target = T.CUDATarget(cuTile))
            f in on_tiles && @test !any(k -> k isa C.FloorKernels, ex.kernels)
            @test isequal(found_in_floats(outputs(execute!(ex))), found(f(reference)))
        end
    end
end
