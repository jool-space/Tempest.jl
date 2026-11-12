# A handle names a node of its tape while its function is traced: a roll a
# gradient makes of a loop leaves the nodes it rolls where they are, so that
# a value the function stores where it outlives it, in a `Ref`, a variable it
# captures, a global or a container, is still what it was after the gradient
# returns. Once the trace has ended a handle names nothing, and reading one is
# refused where it is read, never read as another program's value.
@testset "handles that outlive what they name" begin
    x = data(3, 6; scale = 0.3) ./ 4
    # A loop that rolls, a fold in each step, or one that does not; it
    # stores its value at its second step.
    function steps(x, put, rolls)
        h = x[:, 1]
        for t in 2:5
            h = rolls ? tanh.(h .* sum(x[:, t])) : tanh.(h .* 0.5)
            t == 3 && put(h)
        end
        h
    end
    function after_gradient(x, rolls)
        r = Ref{Any}(nothing)
        v, g = T.withgradient(x -> sum(steps(x, h -> r[] = h, rolls)), x)
        (v, g[1], r[] .+ 0)
    end
    plain(x) = (r = Ref{Any}(nothing); steps(x, h -> r[] = h, true); r[])

    # Read after the loop, in the same function, the value is what it was.
    p = trace(x -> (r = Ref{Any}(nothing); h = steps(x, h -> r[] = h, true); (sum(h), r[] .+ 0)), x)
    @test count(n -> n.op isa C.Loop, p.nodes) == 1
    @test interpret(p, x)[2] ≈ plain(x)
    # Read after a gradient that rolled the loop, it is what it was, where
    # the program read zeros; the gradient is the rolled loop's.
    p = trace(x -> after_gradient(x, true), x)
    @test count(n -> n.op isa C.Loop, p.nodes) >= 2
    results = interpret(p, x)
    @test results[3] ≈ plain(x)
    @test results[1] ≈ sum(steps(x, identity, true))
    @test results[2] ≈ interpret(trace(x -> T.gradient(x -> sum(steps(x, identity, true)), x)[1], x), x)
    # As one the gradient did not roll.
    p = trace(x -> after_gradient(x, false), x)
    @test count(n -> n.op isa C.Loop, p.nodes) == 0
    kept = Ref{Any}(nothing); steps(x, h -> kept[] = h, false)
    @test interpret(p, x)[3] ≈ kept[]

    # Untraced, a gradient traces its function: what it stored is read after
    # the trace ended, where it was a handle Julia's code went on with.
    @test occursin("was read after the trace that made it ended", message(() -> after_gradient(x, false)))
    left = Ref{Any}(nothing)
    T.gradient(x -> sum(steps(x, h -> left[] = h, true)), x)
    @test left[] isa C.TracedArray && C.stale(left[])
    @test endswith(sprint(show, left[]), "stale)")
    @test occursin("after the trace that made it ended", message(() -> left[] .+ 1))
    # In a later trace, it is not a value of that trace's program, and a
    # gradient whose function closes over it without reading it is taken.
    @test occursin("after the trace that made it ended", message(() -> trace(y -> y .+ left[], x[:, 1])))
    @test T.gradient(y -> (left; sum(abs2, y)), x)[1] ≈ 2 .* x
    # Its program's results are its function's, as before.
    @test T.withgradient(x -> sum(steps(x, identity, true)), x).val ≈ sum(steps(x, identity, true))
end

# A roll keeps the program's meaning: traced with nothing rolled, every
# iteration's nodes in place, a program interprets as it does rolled, its
# gradient too.
@testset "rolled and unrolled programs interpret the same" begin
    W, xs, h0 = data(4, 4; scale = 0.3) ./ 4, data(4, 7; scale = 0.7), data(4; scale = 1.1)
    cell(W, xs, h) = (for t in axes(xs, 2); h = tanh.(W * h .+ xs[:, t]) .+ 0.01 .* sum(h); end; sum(abs2, h))
    unrolled(f, args...) = Base.ScopedValues.with(() -> trace(f, args...), C.ROLLING => false)
    for f in (cell, (W, xs, h) -> T.withgradient((W, h) -> cell(W, xs, h), W, h))
        p, q = trace(f, W, xs, h0), unrolled(f, W, xs, h0)
        @test count(n -> n.op isa C.Loop, p.nodes) >= 1
        @test !any(n -> n.op isa C.Loop, q.nodes)
        @test all(map(≈, C.leaves(interpret(p, W, xs, h0)), C.leaves(interpret(q, W, xs, h0))))
    end
end
