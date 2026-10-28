# Scans on cuTile, a block per tile of the kept axes sweeping the scanned ones:
# along the first axis, a middle one, the last, and two at once; builtin folds
# and folds of the user's, a recurrence among them, whose combine does not
# commute; with maps around them and a fold beside them; and reversed, in a
# cumulative sum's derivative. Base's `accumulate` by a function of the
# user's runs in sequence, and an exclusive scan, which derivatives make, has
# no tile realization yet: both stay on the floor.
include("../fixtures/programs.jl")

@testset "scans on cuTile" begin
    x = single(data(1000, 64))
    a, b = single(0.9 .+ 0.1 .* data(700, 32; scale = 0.3)), single(data(700, 32; scale = 0.7))
    scans = [
        (x -> cumsum(x; dims = 1), (x,)), (x -> cumsum(x; dims = 2), (x,)), (cumsum, (single(data(5000)),)),
        (x -> cumsum(x; dims = 2), (single(data(16, 300, 8)),)), (x -> accumulate(+, x), (single(data(30, 40)),)),
        (x -> cumprod(1 .+ 0.001f0 .* x; dims = 1), (x,)), (x -> accumulate(max, x; dims = 1), (x,)),
        ((a, b) -> accumulate(RECURRENCE, tuple.(a, b); dims = 1), (a, b)),
        (x -> accumulate(VARIANCE, x; dims = 1), (single(data(300, 20)),)),
        (x -> exp.(cumsum(sin.(x); dims = 1) .* 0.01f0), (x,)), (x -> (cumsum(x; dims = 1), sum(x; dims = 1)), (x,)),
        (x -> cumsum(x; dims = 1), (BFloat16s.BFloat16.(data(500, 16)),)),
        (x -> T.gradient(x -> sum(cumsum(x; dims = 2) .^ 2), x), (single(data(16, 300, 8)),)),
    ]
    sketches(ex) = [k.schedule.sketch for k in ex.kernels if hasproperty(k, :schedule)]
    for (f, args) in scans
        p = trace(f, args...)
        ex = prepare(p, map(CuArray, args)...; target = T.CUDATarget(T.cuTileGenerator()))
        @test :scan in sketches(ex)
        @test agree(C.leaves(outputs(execute!(ex))), C.leaves(interpret(p, args...)); rtol = 1.0e-4)
    end
    for (f, args) in [((a, b) -> recurrence(a, b), (single(data(700)), single(data(700)))),
        (x -> T.gradient(x -> sum(cumprod(1 .+ 0.001f0 .* x; dims = 1)), x), (x,))]
        p = trace(f, args...)
        ex = prepare(p, map(CuArray, args)...; target = T.CUDATarget(T.cuTileGenerator()))
        @test !(:scan in sketches(ex))
        @test agree(C.leaves(outputs(execute!(ex))), C.leaves(interpret(p, args...)); rtol = 1.0e-4)
    end
end
