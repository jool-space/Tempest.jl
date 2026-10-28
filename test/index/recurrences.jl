# A recurrence's step, spliced into the program over a last axis of its
# steps: its products are values over the steps like any others, and its
# state one value, carried one step back from what the step computes. The
# index level's interpreter, a pass per step, agrees with the program's.
@testset "recurrences lowered" begin
    step((S, _), (w, u, kd, g)) = (g .* S .+ kd' * (u .- w * S), w * S)
    chunks(W, U, Kd, G, S0) = Tuple(x for t in accumulate(step, zip(eachslice(W; dims = 3), eachslice(U; dims = 3),
        eachslice(Kd; dims = 3), eachslice(G; dims = 3)); init = (S0, nothing)) for x in t)
    W, U, Kd = data(4, 3, 5; scale = 0.3), data(4, 2, 5; scale = 0.2), data(4, 3, 5; scale = 0.1)
    G, S0 = 0.5 .+ 0.1 .* data(1, 1, 5), data(3, 2; scale = 0.4)
    p = trace(chunks, W, U, Kd, G, S0)
    g = C.lower(p)
    state = only(v for v in g.values if v.definition isa C.Carried)
    @test state.dims == (3, 2, 5) && state.definition.axis == 3
    @test count(v -> v.definition isa C.Reduction && last(v.dims) == 5, g.values) == 2
    @test agree(Tuple(C.interpret(g, W, U, Kd, G, S0)), C.leaves(interpret(p, W, U, Kd, G, S0)); rtol = 1.0e-12)

    cell(Wh, xs, h0) = Tuple(accumulate((h, x) -> tanh.(transpose(Wh) * h .+ x), eachslice(xs; dims = 2); init = h0))
    Wh, xs, h0 = data(3, 3; scale = 0.5), data(3, 6; scale = 0.2), data(3; scale = 0.1)
    p = trace(cell, Wh, xs, h0)
    @test agree(Tuple(C.interpret(C.lower(p), Wh, xs, h0)), C.leaves(interpret(p, Wh, xs, h0)); rtol = 1.0e-12)
    # A number for a state, a sum in the step; and one step, which carries nothing.
    run(xs, s0) = Tuple(accumulate((s, x) -> s * 0.5 + sum(abs2, x), eachslice(xs; dims = 2); init = s0))
    for xs in (data(3, 4), data(3, 1))
        p = trace(run, xs, 0.25)
        @test agree(Tuple(C.interpret(C.lower(p), xs, 0.25)), C.leaves(interpret(p, xs, 0.25)); rtol = 1.0e-12)
    end
end
