# Base's `accumulate` over slices of traced arrays traces its step once, as a
# loop whose body is the step's program: a matrix state over chunks, as
# chunked DeltaNet carries one, with a result of each step beside it; a
# recurrent cell whose weight the step captures, its transposes computed
# once outside; their derivatives, loops reversed; and the writes a step may
# not make.
@testset "recurrences" begin
    step((S, _), (w, u, kd, g)) = (g .* S .+ kd' * (u .- w * S), w * S)
    chunks(W, U, Kd, G, S0) = Tuple(x for t in accumulate(step, zip(eachslice(W; dims = 3), eachslice(U; dims = 3),
        eachslice(Kd; dims = 3), eachslice(G; dims = 3)); init = (S0, nothing)) for x in t)
    W, U, Kd = data(4, 3, 5; scale = 0.3), data(4, 2, 5; scale = 0.2), data(4, 3, 5; scale = 0.1)
    G, S0 = 0.5 .+ 0.1 .* data(1, 1, 5), data(3, 2; scale = 0.4)
    p = trace(chunks, W, U, Kd, G, S0)
    loops = [n for n in p.nodes if n.op isa C.Loop]
    @test length(loops) == 2 && only(unique(n.op.body for n in loops)).steps == 5
    @test agree(interpret(p, W, U, Kd, G, S0), chunks(W, U, Kd, G, S0); rtol = 1.0e-12)

    cell(Wh, xs, h0) = Tuple(accumulate((h, x) -> tanh.(transpose(transpose(Wh)) * h .+ x), eachslice(xs; dims = 2); init = h0))
    Wh, xs, h0 = data(3, 3; scale = 0.5), data(3, 6; scale = 0.2), data(3; scale = 0.1)
    p = trace(cell, Wh, xs, h0)
    body = only(n.op.body for n in p.nodes if n.op isa C.Loop).program
    @test count(n -> n.op isa C.Argument, body.nodes) == 3 && !any(n -> n.op isa C.Permute, body.nodes)
    @test agree(interpret(p, Wh, xs, h0), cell(Wh, xs, h0); rtol = 1.0e-12)

    # Its derivative is a loop too, the steps reversed: a step replays the
    # forward one from the state it started at, which the forward loop gives
    # beside its results, and pulls back through it the state's cotangent and
    # what the step's results receive; what the captures receive is summed
    # over the steps. It agrees with the gradient of the loop written out.
    squares(f) = (xs...) -> sum(x -> sum(abs2, x), f(xs...))
    slice(x, t) = x[ntuple(_ -> Colon(), ndims(x) - 1)..., t]
    unrolled(op, init, xs...) = (s = init; Tuple(s = op(s, map(x -> slice(x, t), xs)) for t in axes(first(xs), ndims(first(xs)))))
    run(xs, s0) = Tuple(accumulate((s, x) -> s * 0.5 + sum(abs2, x), eachslice(xs; dims = 2); init = s0))
    for (f, g, args) in ((chunks, (xs...) -> Tuple(x for t in unrolled(step, (last(xs), nothing), xs[1:4]...) for x in t), (W, U, Kd, G, S0)),
            (cell, (Wh, xs, h0) -> unrolled((h, (x,)) -> tanh.(transpose(transpose(Wh)) * h .+ x), h0, xs), (Wh, xs, h0)),
            (run, (xs, s0) -> unrolled((s, (x,)) -> s * 0.5 + sum(abs2, x), s0, xs), (data(3, 4), 0.25)))
        @test all(map((a, b) -> isapprox(a, b; rtol = 1.0e-12), T.gradient(squares(f), args...), T.gradient(squares(g), args...)))
        bodies = unique(n.op.body for n in differentiate(trace(squares(f), args...)).nodes if n.op isa C.Loop)
        @test length(bodies) == 2 && all(b -> b.steps == size(args[end - 1], ndims(args[end - 1])), bodies)
    end

    outside(x) = (a = copy(x[:, 1]); accumulate((s, y) -> (a[1] = s[1]; s .+ y), eachslice(x; dims = 2); init = a))
    @test occursin("writes an array made before it", message(() -> trace(outside, data(3, 4))))
    inside(x) = Tuple(accumulate((s, y) -> (t = copy(s); t[1] = y[2]; t .+ y), eachslice(x; dims = 2); init = x[:, 1]))
    @test agree(interpret(trace(inside, data(3, 4)), data(3, 4)), inside(data(3, 4)); rtol = 1.0e-12)
    retyped(x) = accumulate((s, y) -> Float32.(s .+ y), eachslice(x; dims = 2); init = x[:, 1])
    @test occursin("changes from", message(() -> trace(retyped, data(3, 4))))
end
