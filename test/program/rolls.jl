# A loop rolled from the tape keeps the program's meaning, and is the loop its
# states make it: a state carried and read is a recurrence; a state each step
# only writes, at positions of its own, is no recurrence, but a map of what
# the steps write, written at once; and no state at all is a map.

loops(p) = [n.op.body for n in p.nodes if n.op isa C.Loop]
unrolled(f, args...) = Base.ScopedValues.with(() -> trace(f, args...), C.ROLLING => false)
kernels(f, args...) = length(T.compile(f, args...; target = T.CPUTarget(), measure = :none, capture = false).executable.kernels)

# Traced with nothing rolled, every iteration's nodes in place, a program
# interprets as it does rolled, its gradient too.
@testset "rolled and unrolled programs interpret the same" begin
    W, xs, h0 = data(4, 4; scale = 0.3) ./ 4, data(4, 7; scale = 0.7), data(4; scale = 1.1)
    cell(W, xs, h) = (for t in axes(xs, 2); h = tanh.(W * h .+ xs[:, t]) .+ 0.01 .* sum(h); end; sum(abs2, h))
    for f in (cell, (W, xs, h) -> T.withgradient((W, h) -> cell(W, xs, h), W, h))
        p, q = trace(f, W, xs, h0), unrolled(f, W, xs, h0)
        @test count(n -> n.op isa C.Loop, p.nodes) >= 1
        @test !any(n -> n.op isa C.Loop, q.nodes)
        @test all(map(≈, C.leaves(interpret(p, W, xs, h0)), C.leaves(interpret(q, W, xs, h0))))
    end
end

@testset "a state each step writes apart is a map and one write" begin
    normalized(v) = v ./ sum(v)
    for B in (3, 5)
        x, W = 3 .+ data(5, B; scale = 0.4), data(6, 5; scale = 0.3)
        # Each slice written as a step computes it, into an array the
        # program makes, an argument, from a product, and a number each.
        written = [
            (x -> (z = zero(x); for b in axes(x, 2); z[:, b] .= normalized(x[:, b]); end; z), (x,)),
            ((z, x) -> (for b in axes(x, 2); z[:, b] .= normalized(x[:, b]); end; z), (zero(x), x)),
            ((z, W, x) -> (for b in axes(x, 2); z[:, b] .= W * x[:, b]; end; z), (zeros(6, B), W, x)),
            ((s, x) -> (for b in axes(x, 2); s[b] = sum(x[:, b]); end; s), (zeros(B), x)),
            # A slice every other position, and the steps the other way.
            ((z, x) -> (for b in axes(x, 2); z[:, 2b - 1] .= normalized(x[:, b]); end; z), (zeros(5, 2B), x)),
            ((z, x) -> (for b in reverse(axes(x, 2)); z[:, b] .= normalized(x[:, b]); end; z), (zero(x), x)),
        ]
        for (k, (f, args)) in enumerate(written)
            p = trace(f, map(copy, args)...)
            @test only(loops(p)).carried == []
            @test interpret(p, map(copy, args)...) ≈ f(map(copy, args)...)
            k <= 4 && @test kernels(f, map(copy, args)...) == 1
        end
        # A recurrence that writes each state into an array of them: the
        # state carried, the array written at once; its gradient as the
        # unrolled program's.
        function states(W, x)
            H = zero(x)
            h = x[:, 1]
            for t in axes(x, 2)
                h = tanh.(W[1:5, :] * h .+ x[:, t])
                H[:, t] = h
            end
            H
        end
        p = trace(states, W, x)
        @test only(loops(p)).carried == [1]
        @test interpret(p, W, x) ≈ states(W, x)
        g = (W, x) -> T.gradient((W, x) -> sum(abs2, states(W, x)), W, x)
        @test all(map(≈, interpret(trace(g, W, x), W, x), interpret(unrolled(g, W, x), W, x)))
    end
    # Where something reads a state the steps write before the last has
    # written it, or a step reads what it wrote, the state is carried, and
    # the program the same.
    x = 3 .+ data(5, 4; scale = 0.4)
    function read_between(x)
        z = zero(x)
        s = sum(z)
        for b in axes(x, 2)
            z[:, b] .= x[:, b] .* s .+ 1
            s = sum(z)
        end
        z
    end
    @test interpret(trace(read_between, x), x) ≈ read_between(x)
end
