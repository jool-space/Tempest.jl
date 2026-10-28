# Base's `accumulate` over slices of traced arrays traces its step once, as a
# loop whose body is the step's program: a matrix state over chunks, as
# chunked DeltaNet carries one, with a result of each step beside it; a
# recurrent cell whose weight the step captures, its transposes computed
# once outside; and the writes a step may not make.
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

    outside(x) = (a = copy(x[:, 1]); accumulate((s, y) -> (a[1] = s[1]; s .+ y), eachslice(x; dims = 2); init = a))
    @test occursin("writes an array made before it", message(() -> trace(outside, data(3, 4))))
    inside(x) = Tuple(accumulate((s, y) -> (t = copy(s); t[1] = y[2]; t .+ y), eachslice(x; dims = 2); init = x[:, 1]))
    @test agree(interpret(trace(inside, data(3, 4)), data(3, 4)), inside(data(3, 4)); rtol = 1.0e-12)
    retyped(x) = accumulate((s, y) -> Float32.(s .+ y), eachslice(x; dims = 2); init = x[:, 1])
    @test occursin("changes from", message(() -> trace(retyped, data(3, 4))))
end
