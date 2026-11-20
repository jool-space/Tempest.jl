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
    for B in (2, 3, 5)
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
        B == 2 && continue
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

# A loop with no state carried is a map over its steps and rolls from two; a
# recurrence, whose steps rolled run one at a time where unrolled they fuse,
# rolls from three whose steps fold, shift or write apart.
@testset "a map rolls from two steps, a recurrence from three" begin
    x = data(5, 3; scale = 0.4)
    pairsum(x) = sum(abs2, x[:, 1]) * sum(abs2, x[:, 2])
    p = trace(pairsum, x)
    @test !isempty(loops(p)) && all(l -> isempty(l.carried), loops(p))
    @test interpret(p, x) ≈ pairsum(x)
    folded(x, n) = (h = x[:, 1]; for t in 1:n; h = tanh.(h .* sum(x[:, t])); end; sum(h))
    @test isempty(loops(trace(x -> folded(x, 2), x)))
    @test only(loops(trace(x -> folded(x, 3), x))).carried == [1]
    # Selections alike a step apart that do not slide evenly, a view's
    # contents recorded anew at each write into its parent next to the
    # steps' slices, are no loop's: the first column twice, then the second.
    function viewed(x)
        a = copy(x)
        v = view(a, :, 1)
        h = a[:, 1]
        for t in 2:5
            h = tanh.(h .* sum(a[:, t]))
            a[:, t] = h
        end
        sum(v) + sum(a) + sum(h)
    end
    y = data(5, 6; scale = 0.3)
    @test interpret(trace(viewed, y), copy(y)) ≈ viewed(copy(y))
end

# The derivative of `f` at `x` along its `i`th element, by central differences.
slope(f, x, i; e = 1.0e-6) = (f(setindex!(copy(x), x[i] + e, i)) - f(setindex!(copy(x), x[i] - e, i))) / 2e
compiled_on_host(f, x) = T.compile(f, x; target = T.CPUTarget(), measure = :none, capture = false)(x)

# Traced, compiled and with its gradient, `f` computes what it does untraced.
function as_untraced(f, x)
    @test interpret(trace(f, x), x) ≈ f(x)
    @test compiled_on_host(f, x) ≈ f(x)
    @test isapprox(only(compiled_on_host(y -> T.gradient(f, y), x)), map(i -> slope(f, x, i), CartesianIndices(x)); rtol = 1.0e-6)
end

# Two states, each stepped from the other's value before the step: over
# numbers, over columns, and written out with no loop, traced, compiled and
# with their gradients they compute what the code does untraced.
@testset "two states each stepped from the other's earlier value" begin
    function numbers(x)
        a = x[1]
        b = a * a
        for t in 2:3
            a, b = tanh(b - x[t]), a * x[t]
        end
        a + b^2
    end
    function columns(x)
        a = x[:, 1]
        b = a .* a
        for t in 2:3
            a, b = tanh.(b .- x[:, t]), a .* x[:, t]
        end
        sum(a) + sum(abs2, b)
    end
    function written(x)
        a0 = x[1]; b0 = a0 * a0
        a1 = tanh(b0 - x[2]); b1 = a0 * x[2]
        a2 = tanh(b1 - x[3]); b2 = a1 * x[3]
        a2 + b2^2
    end
    for (f, x) in ((numbers, data(5; scale = 0.5)), (columns, data(3, 4; scale = 0.5)), (written, data(5; scale = 0.5)))
        as_untraced(f, x)
    end
end

# A value of the steps two steps read at one place is read at two ages, the
# first step's own and the second's of the step before, which no loop's step
# can: `a * a`, then `a * x[2]`, of `a = x[1]`. Such steps are no loop's and
# stay as traced, where states that cycle, swap, are read at two ages or
# squared are a loop's, which rolls where its steps fold.
@testset "a value two steps read at one place is no loop's" begin
    x, m = data(6; scale = 0.5), data(3, 5; scale = 0.5)
    unrolled_programs = [
        (x -> (a = x[1]; a * a + a * x[2]), x),
        (x -> (a = x[:, 1]; sum(a .* a) * sum(a .* x[:, 2]) * sum(a .* x[:, 3])), m),
        (x -> (a = x[1]; b = a * a; for t in 2:5; a, b = tanh(b - x[t]), a * x[t]; end; a + b^2), x),
        (x -> (a = x[:, 1]; b = a .* a; for t in 2:5; a, b = tanh.(b .- x[:, t]), a .* x[:, t]; end; sum(a) + sum(abs2, b)), m),
    ]
    for (f, x) in unrolled_programs
        @test isempty(loops(trace(f, x)))
        as_untraced(f, x)
        if CUDA_FUNCTIONAL
            @test agree(on_host(T.compile(f, CuArray(x))(CuArray(x))), f(x))
            @test agree(on_host(only(T.compile(y -> T.gradient(f, y), CuArray(x))(CuArray(x)))), map(i -> slope(f, x, i), CartesianIndices(x)); rtol = 1.0e-6)
        end
    end
    rolled_programs = [
        (x -> (a = x[:, 1]; b = a .* a; for t in 2:5; a, b = tanh.(b .- sum(x[:, t])), a .* x[:, t]; end; sum(a) + sum(abs2, b)), m),
        (x -> (a = x[:, 1]; b = 2 .* a; c = a .* a; for t in 2:5; a, b, c = tanh.(c .+ sum(x[:, t])), a .* x[:, t], b .- x[:, t]; end; sum(a) + sum(b) + sum(c)), m),
        (x -> (a = x[:, 1]; b = x[:, 2]; for t in 2:5; a, b = b .* sum(x[:, t]), a .+ x[:, t]; end; sum(a .* b)), m),
        (x -> begin p, q = x[:, 1], x[:, 2]; for t in 2:5; p, q = q, tanh.(q .+ p .* sum(x[:, t])); end; sum(p) + sum(q) end, m),
        (x -> (a = x[:, 1]; s = a .* a; for t in 2:5; a, s = tanh.(s .- sum(x[:, t])), a .* a; end; sum(a) + sum(s)), m),
    ]
    for (f, x) in rolled_programs
        @test !isempty(loops(trace(f, x)))
        as_untraced(f, x)
    end
    # Numbers cycled, swapped and read at two ages.
    for f in (x -> (a = x[1]; b = 2a; c = a * a; for t in 2:5; a, b, c = tanh(c + x[t]), a * x[t], b - x[t]; end; a + b + c),
            x -> (a = x[1]; b = x[2]; for t in 2:5; a, b = b * x[t], a + x[t]; end; a * b),
            x -> begin p, q = x[1], x[2]; for t in 2:5; p, q = q, tanh(q + p * x[t]); end; p + q end)
        as_untraced(f, x)
    end
end

# A loop rolled from a vector's elements, `x[t]`, takes each as the number its
# step's nodes read: a map over them, and a state stepped from a number each
# step computes once, its interpretation, its gradient's too, as the code.
@testset "a loop rolled over a vector's elements takes numbers" begin
    x = data(6; scale = 0.5)
    for f in (x -> tanh(x[1]) * tanh(x[2]),
            x -> (a = x[1]; s = a * a; for t in 2:5; a, s = tanh(s - x[t]), a * a; end; a + s))
        @test !isempty(loops(trace(f, x)))
        as_untraced(f, x)
        @test only(interpret(trace(y -> T.gradient(f, y), x), x)) ≈ only(compiled_on_host(y -> T.gradient(f, y), x))
    end
end

# Selections that slide, as a scan's, but over a state the loop carries, or
# over one array at the first step and another after it: no scan of an array
# the loop captures, and so no loop's.
@testset "a selection that slides over a state, or over two arrays, is no scan" begin
    x, m = data(9; scale = 0.5), data(3, 9; scale = 0.5)
    for (f, x) in ((x -> (h = x; for t in 1:3; h = h .* tanh(h[t]); end; sum(h)), x),
            (m -> (h = m; for t in 1:3; h = h .* sum(h[:, t]); end; sum(h)), m),
            (m -> (a = m .+ 1; b = m .* 2; sum(a[:, 1]) * sum(b[:, 2]) * sum(b[:, 3])), m),
            (m -> (a = m .+ 1; b = m .* 2; sum(a[:, 1]) * sum(b[:, 2])), m),
            (m -> (a = m .+ 1; z = sum(m); b = m .* 2; sum(a[:, 1]) * sum(b[:, 2]) + z), m))
        as_untraced(f, x)
    end
end

# A loop drawn from `rng`, with its arguments: over a vector's elements or a
# matrix's columns, forward or reversed, of two to four steps; a map writing
# each step's value apart, or one to three states, each stepped from two
# operands drawn from the states as the step started, those stepped already
# in the step, the step's slice of `x`, an element of a state at the step's
# index, or the step's slice of `y` at the first step and of `x` after; the
# second operand summed or not, which makes a recurrence's steps fold.
function drawn_loop(rng)
    columns = rand(rng, Bool)
    steps = rand(rng, 2:5)
    # The steps' order, made where the loop runs: compile takes a range the
    # function holds, an array, for an argument.
    reversed = rand(rng, Bool)
    order() = reversed ? (steps:-1:1) : (1:steps)
    H = steps + 1
    x, y = columns ? (data(H, steps; scale = 0.4), data(H, steps; scale = 0.7)) : (data(steps; scale = 0.4), data(steps; scale = 0.7))
    slice(a, t) = columns ? a[:, t] : a[t]
    # The step's slice of `x`, or of `y` at the first step.
    entered(x, y, t) = t == first(order()) ? slice(y, t) : slice(x, t)
    if rand(rng) < 0.25
        from_y = rand(rng, Bool)
        mapped(x, y) = (z = zero(x); for t in order(); columns ? (z[:, t] .= tanh.(from_y ? entered(x, y, t) : slice(x, t))) : (z[t] = tanh(from_y ? entered(x, y, t) : slice(x, t))); end; sum(abs2, z))
        return mapped, (x, y)
    end
    nstates = rand(rng, 1:3)
    kinds = columns ? (:old, :new, :slice, :element, :entered) : (:old, :new, :slice, :entered)
    function draw(i)
        k = rand(rng, kinds)
        k === :new && i == 1 && (k = :old)
        (k, rand(rng, 1:(k === :new ? i - 1 : nstates)))
    end
    rules = [(draw(i), draw(i), rand(rng, Bool), 0.3 + 0.2i) for i in 1:nstates]
    function operand((kind, j), olds, news, x, y, t)
        kind === :old && return olds[j]
        kind === :new && return news[j]
        kind === :slice && return slice(x, t)
        kind === :element && return olds[j][t]
        entered(x, y, t)
    end
    function looped(x, y)
        states = ntuple(i -> columns ? x[:, 1] .* (0.5 + 0.1i) : x[1] * (0.5 + 0.1i), nstates)
        for t in order()
            news = ()
            for (a, b, summed, c) in rules
                u, v = operand(a, states, news, x, y, t), operand(b, states, news, x, y, t)
                # A state keeps its shape: a vector where the loop is over columns.
                s = tanh.(c .* u .+ (summed ? sum(v) : v) .* (1 - c) .+ (columns ? zero(x[:, 1]) : 0))
                news = (news..., s)
            end
            states = news
        end
        sum(s -> sum(abs2, s), states)
    end
    looped, (x, y)
end

# Gradients equal where both are of an argument, or neither is.
same_gradient(a, b; rtol = 1.0e-10) = a === nothing ? b === nothing : isapprox(a, b; rtol)

# A program the planner refuses to compile, naming what it does not support
# yet, as against one it compiles wrong.
refused(err) = err isa ArgumentError && occursin("is not supported yet", err.msg)

# Every loop of the family rolls, where it rolls, into the program its trace
# as it ran is (see `unrolled`): interpreted, compiled, and with its gradient
# interpreted and compiled. A gradient the planner refuses is a gap, the one
# `runtime/recurrences.jl` records of a state carried from a value of
# another shape.
@testset "a loop drawn from the family rolls into the program it traced as" begin
    rng = Random.Xoshiro(20261120)
    rolled = 0
    compiled(f, args) = T.compile(f, args...; target = T.CPUTarget(), measure = :none, capture = false)(args...)
    for _ in 1:64
        f, args = drawn_loop(rng)
        expected = interpret(unrolled(f, args...), args...)
        p = trace(f, args...)
        rolled += !isempty(loops(p))
        @test interpret(p, args...) ≈ expected
        @test compiled(f, args) ≈ expected
        g = (xs...) -> T.gradient(f, xs...)
        gradient = C.leaves(interpret(unrolled(g, args...), args...))
        @test all(map(same_gradient, C.leaves(interpret(trace(g, args...), args...)), gradient))
        planned = try
            C.leaves(compiled(g, args))
        catch err
            refused(err) || rethrow()
            nothing
        end
        if planned === nothing
            @test_broken planned !== nothing
        else
            @test all(map((a, b) -> same_gradient(a, b; rtol = 1.0e-8), planned, gradient))
        end
    end
    # The family reaches the roller.
    @test rolled >= 12
end
