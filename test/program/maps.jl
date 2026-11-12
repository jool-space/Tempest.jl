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

    # A gather or a write by index arrays each step slices or computes is,
    # over the steps, one by the linear index each step's elements name in
    # its slice and the steps together (`batched_indexing!`): one
    # computation at any number of slices, its gradient a scatter alike.
    @testset "a gather or a write by indices a step slices or computes" begin
        for B in (2, 3, 8)
            x, x3 = data(4, B), data(2, 4, B; scale = 0.5)
            J = [mod1(3b + k, 4) for k in 1:2, b in 1:B]
            programs = [
                ((x, J) -> stack(map((v, j) -> v[j], eachcol(x), eachcol(J))), (x, J)),
                ((x, J) -> stack(map((v, j) -> v[j] .* sum(v), eachcol(x), eachcol(J))), (x, J)),
                ((x3, J) -> stack(map((m, j) -> m[:, j], eachslice(x3; dims = 3), eachcol(J))), (x3, J)),
                ((x3, J) -> stack(map((m, j) -> m[j, :]', eachslice(permutedims(x3, (2, 1, 3)); dims = 3), eachcol(J))), (x3, J)),
                ((x, J) -> stack(map((v, j) -> (y = copy(v); y[j] .= 0; y), eachcol(x), eachcol(J))), (x, J)),
            ]
            for (f, args) in programs
                p = trace(f, args...)
                @test count(n -> n.op isa C.Loop, p.nodes) >= 1
                ex = prepare(p, args...; target = T.CPUTarget())
                @test agree((outputs(execute!(ex)),), (f(args...),); rtol = 1.0e-12)
            end
            gathered(x, J) = sum(abs2, stack(map((v, j) -> v[j] .* sum(v), eachcol(x), eachcol(J))))
            unrolled(x, J) = sum(abs2, stack([x[J[:, b], b] .* sum(x[:, b]) for b in axes(x, 2)]))
            @test agree(T.gradient(gathered, x, J)[1:1], T.gradient(unrolled, x, J)[1:1])
        end
    end
end

# The linear index into axes merged into one, from the index along each,
# which states indices running along several axes together: Cartesian
# indices read and written over two and three axes, as Base reads and
# writes them.
@testset "linear indices over merged axes" begin
    x = data(3, 4, 5)
    i, j, k = [3, 1, 2], [4, 4, 1], [2, 5, 5]
    # On its own, over traced indices that broadcast against each other.
    merged(x) = (a = C.handle(0 .* x[1, 1:3, 1] .+ i); b = C.handle(0 .* x[1, 1:3, 1] .+ j); c = C.handle(0 .* x[1, 1:3, 1]' .+ k');
        (C.TracedArray(a.tape, C.linear_index((a, b), (3, 4)).id), C.TracedArray(a.tape, C.linear_index((a, b, c), (3, 4, 5)).id)))
    @test interpret(trace(merged, x), x) == (LinearIndices((3, 4))[CartesianIndex.(i, j)], LinearIndices(x)[CartesianIndex.(i, j, k')])
    reads = [
        (x -> x[CartesianIndex.(i, j), :], (x,)),
        (x -> x[:, CartesianIndex.(j, k)], (x,)),
        (x -> x[CartesianIndex.(i, j, k)], (x,)),
    ]
    writes = [
        (x -> (y = copy(x); y[CartesianIndex.(i, j), :] .= 0; y), (x,)),
        (x -> (y = copy(x); y[:, CartesianIndex.(j, k)] = x[:, 1:3, 1]; y), (x,)),
        (x -> (y = copy(x); y[CartesianIndex.(i, j, k)] .= -1; y), (x,)),
    ]
    for (f, args) in [reads; writes]
        @test interpret(trace(f, args...), args...) == f(args...)
        @test outputs(execute!(prepare(trace(f, args...), args...; target = T.CPUTarget()))) == f(args...)
    end
end
