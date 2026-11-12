# A map over slices, as `mapslices`, `map` or a broadcast over `eachslice`,
# `eachcol` or `eachrow`, and `stack` of a function over them state it, traces
# its function once, as a loop with no state: one computation over the axes
# the slices are taken along, whatever their count, its results Base's. A
# slice's function may hold a reduction, a product, a scan or a gather by
# indices the steps share; its derivative is the loop's.
@testset "maps over slices" begin
    unit(v) = v ./ sum(v)
    loops(p) = unique(n.op.body for n in p.nodes if n.op isa C.Loop)
    for B in (2, 5)
        x, y, x3 = 3 .+ data(4, B), data(4, B; scale = 0.3), data(3, 2, B; scale = 0.7)
        W, idx = data(3, 4; scale = 0.2), [4, 1, 3]
        cases = [
            (x -> Int32.(vec(mapslices(argmax, x; dims = 1))), (x,)),
            (x -> mapslices(sum, x; dims = 1), (x,)),
            (x -> mapslices(maximum, x; dims = 2), (x,)),
            (x -> mapslices(norm, x; dims = 1), (x,)),
            (x -> mapslices(v -> sum(abs2, v) + 1, x; dims = 1), (x,)),
            (x -> mapslices(cumsum, x; dims = 1), (x,)),
            (x -> mapslices(unit, x; dims = 1), (x,)),
            ((W, x) -> mapslices(v -> W * v, x; dims = 1), (W, x)),
            (x -> mapslices(v -> v[idx], x; dims = 1), (x,)),
            (x3 -> mapslices(m -> m' * m, x3; dims = (1, 2)), (x3,)),
            (x3 -> mapslices(unit, x3; dims = 3), (x3,)),
            (x -> map(sum, eachcol(x)), (x,)),
            (x -> map(argmax, eachslice(x; dims = 2)), (x,)),
            (x -> map(sum, eachrow(x)), (x,)),
            (x3 -> map(norm, eachslice(x3; dims = (2, 3))), (x3,)),
            ((x, y) -> map(dot, eachcol(x), eachcol(y)), (x, y)),
            (x -> sum.(eachcol(x)), (x,)),
            (x -> stack(map(unit, eachcol(x))), (x,)),
            (x -> stack(unit, eachcol(x)), (x,)),
            (x3 -> stack(m -> m * m', eachslice(x3; dims = 3)), (x3,)),
        ]
        for (f, args) in cases
            p = trace(f, args...)
            @test length(loops(p)) == 1
            @test agree((interpret(p, args...),), (f(args...),); rtol = 1.0e-12)
        end
        # Two axes sliced are one of their positions.
        @test only(loops(trace(x3 -> map(norm, eachslice(x3; dims = (2, 3))), x3))).steps == 2B
        @test T.gradient(x -> sum(abs2, mapslices(unit, x; dims = 1)), x)[1] ≈ T.gradient(x -> sum(abs2, x ./ sum(x; dims = 1)), x)[1]
        @test agree(T.gradient((W, x) -> sum(abs2, stack(v -> tanh.(W * v), eachcol(x))), W, x),
            T.gradient((W, x) -> sum(abs2, tanh.(W * x)), W, x))
        compiled = T.compile(x -> Int32.(vec(mapslices(argmax, x; dims = 1))), single(x); target = T.CPUTarget(), capture = false)
        @test compiled(single(x)) == Int32.(vec(mapslices(argmax, single(x); dims = 1)))
    end

    # The interpreter runs a loop a step at a time; lowering batches its step
    # over the steps, which a gather by indices that differ from step to
    # step does not yet. Closed by batching such a gather as one by the
    # linear indices of each step's elements.
    @testset "a gather by indices a step slices or computes" begin
        x, J = data(4, 3), [1 4 2; 3 3 1]
        gathered(x, J) = stack(map((v, j) -> v[j], eachcol(x), eachcol(J)))
        @test agree((interpret(trace(gathered, x, J), x, J),), (gathered(x, J),))
        @test_broken message(() -> C.lower(trace(gathered, x, J))) == ""
    end
end
