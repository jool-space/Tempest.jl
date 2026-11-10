# A recurrence prepared: one group computes its states, the values of its
# cycle read outside it and what each state is carried from, a step at a
# time, an item per position its step does not read them across; what each
# step computes besides is planned as any value over the steps, reading what
# the group stored, a batch over them; its gradient a recurrence over the
# steps reversed. On the CPU, and on CUDA, by the floor and beside the
# generators, which step through it holding its state on chip; its kernels,
# and its gradient's, do not grow with its steps.
@testset "recurrences prepared" begin
    step((S, _), (w, u, kd, g)) = (g .* S .+ kd' * (u .- w * S), w * S)
    chunks(W, U, Kd, G, S0) = (r = accumulate(step, zip(eachslice(W; dims = 3), eachslice(U; dims = 3),
        eachslice(Kd; dims = 3), eachslice(G; dims = 3)); init = (S0, nothing)); (stack(first.(r)), stack(last.(r))))
    inputs(n) = (data(4, 3, n; scale = 0.3), data(4, 2, n; scale = 0.2), data(4, 3, n; scale = 0.1), 0.5 .+ 0.1 .* data(1, 1, n), data(3, 2; scale = 0.4))
    cell(Wh, xs, h0) = Tuple(accumulate((h, x) -> tanh.(transpose(Wh) * h .+ x), eachslice(xs; dims = 2); init = h0))
    run(xs, s0) = Tuple(accumulate((s, x) -> s * 0.5 + sum(abs2, x), eachslice(xs; dims = 2); init = s0))
    cases = [(chunks, inputs(5)), (chunks, inputs(1)), (cell, (data(3, 3; scale = 0.5), data(3, 6; scale = 0.2), data(3; scale = 0.1))),
        (run, (data(3, 4), 0.25))]
    devices = CUDA_FUNCTIONAL ? [(identity, 1.0e-12), (x -> x isa AbstractArray ? CuArray(x) : x, 1.0e-12),
        (x -> x isa AbstractArray ? CuArray(single(x)) : single(x), 1.0e-5)] : [(identity, 1.0e-12)]
    for (f, args) in cases, (device, rtol) in devices
        p = trace(f, map(x -> device(x) isa CuArray{Float32} ? single(x) : x, args)...)
        ex = prepare(p, map(device, args)...)
        @test agree(C.leaves(outputs(execute!(ex))), C.leaves(interpret(trace(f, args...), args...)); rtol)
    end
    # Its gradient prepared: the reversed loop is a recurrence too.
    grad(f) = (xs...) -> T.withgradient((ys...) -> sum(x -> sum(abs2, x), f(ys...)), xs...)
    for (f, args) in cases, (device, rtol) in devices
        p = trace(grad(f), map(x -> device(x) isa CuArray{Float32} ? single(x) : x, args)...)
        ex = prepare(p, map(device, args)...)
        @test agree(C.leaves(outputs(execute!(ex))), C.leaves(interpret(trace(grad(f), args...), args...)); rtol)
    end
    # An LSTM carries two states, its hidden state and its cell, each step
    # reading both: one recurrence, whose step computes the cell before the
    # hidden state that reads it. Both start from one array, so their terms
    # where they start are one. Its gradient's loop is another, of the two
    # states' adjoints.
    σ(x) = 1 / (1 + exp(-x))
    function lstm_cell(p, (h, c), x)
        i, f = σ.(p.Wi * x .+ p.Ui * h .+ p.bi), σ.(p.Wf * x .+ p.Uf * h .+ p.bf)
        g, o = tanh.(p.Wg * x .+ p.Ug * h .+ p.bg), σ.(p.Wo * x .+ p.Uo * h .+ p.bo)
        c = f .* c .+ i .* g
        (o .* tanh.(c), c)
    end
    lstm(p, xs, h0) = Tuple(map(first, accumulate((s, x) -> lstm_cell(p, s, x), eachslice(xs; dims = 2); init = (h0, h0))))
    weights = (; (Symbol(:W, k) => data(3, 2; scale = 0.3) for k in "ifgo")..., (Symbol(:U, k) => data(3, 3; scale = 0.3) for k in "ifgo")...,
        (Symbol(:b, k) => data(3; scale = 0.1) for k in "ifgo")...)
    lstm_args = (weights, data(2, 5; scale = 0.5), data(3; scale = 0.1))
    lstm_grad = (p, xs, h0) -> T.withgradient((q, x) -> sum(h -> sum(abs2, h), lstm(q, x, h0)), p, xs)
    states(p) = [length(C.recurrence_states(p.graph, g.outputs)) for g in p.groups if g.kind isa C.RecurrenceKind]
    for (f, loops) in ((lstm, [2]), (lstm_grad, [2, 2]))
        p = trace(f, lstm_args...)
        eager = C.leaves(f(lstm_args...))
        planned = C.plan(C.lower(p))
        @test states(planned) == loops
        @test agree(C.execute(planned, lstm_args...), eager; rtol = 1.0e-10)
        @test agree(C.leaves(outputs(execute!(prepare(p, lstm_args...)))), eager; rtol = 1.0e-10)
        if CUDA_FUNCTIONAL
            device(x) = x isa AbstractArray ? CuArray(single(x)) : x isa NamedTuple ? map(device, x) : x
            host(x) = x isa AbstractArray ? single(x) : x isa NamedTuple ? map(host, x) : x
            q = trace(f, map(host, lstm_args)...)
            for target in (T.CUDATarget(; precision = :full), T.CUDATarget(T.cuBLASProducts(); precision = :full))
                ex = prepare(q, map(device, lstm_args)...; target)
                @test count(k -> k isa C.SteppedKernels || occursin("cuTile recurrence", string(k)), ex.kernels) == length(loops)
                @test agree(map(x -> x isa AbstractArray ? Array(x) : x, C.leaves(outputs(execute!(ex)))), eager; rtol = 1.0e-4)
            end
        end
    end
    # A first state learned as a vector and repeated over the batch: each
    # step's slice where the steps start is it spread over the batch, and
    # the steps run as plans of their own, forward and back. Read so, the
    # start had left the recurrence to the floor's kernel stepping through
    # it, an LSTM's training step 1.76 s against 5.6 ms.
    batched(p, xs, h0) = (h = repeat(h0, 1, size(xs, 2));
        Tuple(map(first, accumulate((s, x) -> lstm_cell(p, s, x), eachslice(xs; dims = 3); init = (h, h)))))
    batched_grad = (p, xs, h0) -> T.withgradient((q, x, h) -> sum(h -> sum(abs2, h), batched(q, x, h)), p, xs, h0)
    batch_args = (weights, data(2, 4, 5; scale = 0.5), data(3; scale = 0.1))
    for (f, loops) in ((batched, 1), (batched_grad, 2))
        eager = C.leaves(f(batch_args...))
        ex = prepare(trace(f, batch_args...), batch_args...)
        @test count(k -> k isa C.SteppedKernels, ex.kernels) == loops
        @test agree(C.leaves(outputs(execute!(ex))), eager; rtol = 1.0e-10)
        if CUDA_FUNCTIONAL
            device(x) = x isa AbstractArray ? CuArray(single(x)) : x isa NamedTuple ? map(device, x) : x
            host(x) = x isa AbstractArray ? single(x) : x isa NamedTuple ? map(host, x) : x
            ex = prepare(trace(f, map(host, batch_args)...), map(device, batch_args)...; target = T.CUDATarget(T.cuBLASProducts(); precision = :full))
            @test count(k -> k isa C.SteppedKernels, ex.kernels) == loops
            @test agree(map(x -> x isa AbstractArray ? Array(x) : x, C.leaves(outputs(execute!(ex)))), eager; rtol = 1.0e-4)
        end
    end
    # The weights' gradient sums each step's part: pinned to ride the
    # backward's steps, a fold of its group, and the plan agrees.
    args = (data(3, 3; scale = 0.5), data(3, 6; scale = 0.2), data(3; scale = 0.1))
    planned = prepare(trace(grad(cell), args...), args...; build = false).plan
    sums = C.riding_sums(planned)
    @test !isempty(sums)
    ridden = C.plan(planned.graph; ride = sums)
    @test any(g -> g.kind isa C.RecurrenceKind && sort(g.reductions) == sort(sums), ridden.groups)
    @test agree(C.execute(ridden, args...), C.leaves(interpret(trace(grad(cell), args...), args...)); rtol = 1.0e-12)
    # The product each step returns is of its cycle: the group stores it at
    # each step, along the state's columns and steps, its rows an axis of
    # their own. The state one step on it stores where it is returned.
    products(xs...) = last(chunks(xs...))
    for (f, stores) in ((chunks, true), (products, false))
        p = C.plan(C.lower(trace(f, inputs(5)...)))
        g = only(g for g in p.groups if g.kind isa C.RecurrenceKind)
        @test C.recurrence_axes(p.graph, g) == (axis = 3, parallel = [2], inner = [1, 4])
        @test g.domain == (3, 2, 5, 4) && length(g.outputs) == 3 && p.stored[last(g.outputs)] == stores
    end
    # The reversed loop's group keeps the forward's axes: a map of what it
    # stores, read outside it, it leaves to that reader to recompute.
    p = C.plan(C.lower(trace(grad(chunks), inputs(5)...)))
    @test [g.domain for g in p.groups if g.kind isa C.RecurrenceKind] == [(3, 2, 5, 4), (3, 2, 5, 4)]
    kernels(f, n) = length(prepare(trace(f, inputs(n)...), inputs(n)...).kernels)
    @test kernels(chunks, 5) == kernels(chunks, 9) && kernels(grad(chunks), 5) == kernels(grad(chunks), 9)
    if CUDA_FUNCTIONAL
        # In Float32 the generator steps through it, the state's tile on chip,
        # in one kernel; at a state's size, reading it in pieces of that tile.
        # Its gradient's loop in another, reading the steps from the last.
        state(c, dk, dv, n) = (data(c, dk, n; scale = 0.3), data(c, dv, n; scale = 0.2), data(c, dk, n; scale = 0.01),
            0.5 .+ 0.1 .* data(1, 1, n), data(dk, dv; scale = 0.4))
        for args in (inputs(5), state(64, 128, 128, 3)), (f, loops) in ((chunks, 1), (grad(chunks), 2))
            ex = prepare(trace(f, map(single, args)...), map(x -> CuArray(single(x)), args)...)
            @test count(k -> occursin("cuTile recurrence", string(k)), ex.kernels) == loops
            @test agree(C.leaves(outputs(execute!(ex))), C.leaves(interpret(trace(f, args...), args...)); rtol = 1.0e-4)
        end
        # The cell's gradient, its sums riding the steps, by cuTile's kernels:
        # a step's slice of three numbers starts 12 bytes after the last's,
        # and the reversed steps start at the last, aligned to 16. Each
        # kernel is built on its least aligned slice, which it is compiled
        # for; built on the first, it read the next ones wrong.
        args = (data(3, 3; scale = 0.5), data(3, 5; scale = 0.2), data(3; scale = 0.1))
        q = trace(grad(cell), map(single, args)...)
        sums = C.riding_sums(prepare(q, map(single, args)...; build = false).plan)
        ex = prepare(q, map(x -> CuArray(single(x)), args)...; ride = sums, measure = :none, target = T.CUDATarget(T.cuTileGenerator()))
        @test count(k -> k isa C.SteppedKernels, ex.kernels) == 1
        @test agree(C.leaves(outputs(execute!(ex))), C.leaves(interpret(trace(grad(cell), args...), args...)); rtol = 1.0e-4)
        # A product in a recurrence's step whose factor the step computes,
        # held in the kernel and not loaded: at three hidden units in tiles of
        # four, its padded lanes are the step's 0/0, which the product had
        # multiplied by the weight's zeros there, NaN.
        rnn(W, a) = (s = a; for t in 1:7; z = transpose(W) * s; s = tanh.(z ./ (1 .+ abs.(z .- sum(z; dims = 1)))); end; sum(abs2, s))
        args = (data(3, 3; scale = 0.5), data(3, 2; scale = 0.5))
        q = trace(grad(rnn), map(single, args)...)
        ex = prepare(q, map(x -> CuArray(single(x)), args)...; measure = :none, target = T.CUDATarget(T.cuTileGenerator()))
        @test count(k -> occursin("cuTile recurrence", string(k)), ex.kernels) == 2
        @test agree(C.leaves(outputs(execute!(ex))), C.leaves(interpret(trace(grad(rnn), args...), args...)); rtol = 1.0e-4)
    end
end

# A state no cycle of states runs through is no recurrence: a state whose
# next value reads no state, or reads only states that do not read it, is
# its next value one step back, or where it starts, an array over the steps
# like any other. A state lagging another, `s1, s2 = W * x, s1`, and one
# lagging a recurrence, at any step order.
@testset "states no cycle runs through" begin
    W = data(3, 3; scale = 0.3)
    lagged(xs, a, b) = (r = last(accumulate(((s, l), x) -> (W * x, 2 .* s), eachslice(xs; dims = 2); init = (a, b))); sum(r[1]) + sum(abs2, r[2]))
    looped(order) = (xs, a, b) -> (s = a; l = b; for t in order(size(xs, 2)); s, l = W * xs[:, t], 2 .* s; end; sum(s) + sum(abs2, l))
    trailing(xs, a, b) = (s = a; l = b; for t in axes(xs, 2); s, l = tanh.(W * s .+ xs[:, t]), sum(s) .* s; end; sum(s) + sum(abs2, l))
    args = (data(3, 5; scale = 0.4), data(3; scale = 0.2), data(3; scale = 0.3))
    for f in (lagged, looped(n -> 1:n), looped(n -> n:-1:1), trailing)
        p = trace(f, args...)
        @test count(n -> n.op isa C.Loop, p.nodes) > 0
        @test only(C.leaves(outputs(execute!(prepare(p, args...)))))[] ≈ f(args...)
        grad = (ys...) -> T.withgradient(f, ys...)
        @test agree(C.leaves(outputs(execute!(prepare(trace(grad, args...), args...)))), C.leaves(grad(args...)))
    end
end

# States assigned at once, each carried from a value reading the other, are
# one recurrence, though neither's value is carried from one reading it: two
# states crossing, two swapping, and a second-order recurrence, the last
# state carried from the state as it was, whose loops roll through
# `accumulate`. Two states given one value, written twice and computed once,
# are each carried from a value of its own. A state whose next value only
# another state's next value reads is kept with what it is carried from: the
# recurrence had been left out, its states without a kernel.
@testset "states assigned at once" begin
    W, U = data(3, 3; scale = 0.3), data(3, 2; scale = 0.3)
    crossing(xs, a, b) = (s1 = a; s2 = b; for t in axes(xs, 2); s1, s2 = tanh.(W * s2 .+ U * xs[:, t]), 0.5 .* s1 .+ 0.5 .* tanh.(s2); end; sum(s1) + sum(abs2, s2))
    swapping(xs, a, b) = (r = last(accumulate(((s1, s2), x) -> (tanh.(s2 .+ U * x), s1), eachslice(xs; dims = 2); init = (a, b))); sum(r[1]) + sum(abs2, r[2]))
    args = (data(2, 5; scale = 0.4), data(3; scale = 0.2), data(3; scale = 0.3))
    second(xs, a, b) = (r = last(accumulate(((y, last), x) -> (tanh.(1.5 .* y .- 0.5 .* last .+ U * x), y), eachslice(xs; dims = 2); init = (a, b))); sum(abs2, r[1]) + sum(r[2]))
    twins(xs, a, b) = (s1 = a; s2 = b; r = sum(a) * 0.1;
        for t in axes(xs, 2); z = s2 ./ (1 .+ abs.(U * xs[:, t])) .+ s1; s2 = tanh.(z); s1 = tanh.(z); r = 0.5 * r + 0.01 * sum(abs2, z); end;
        sum(abs2, s1) + sum(s2) + r)
    σ(x) = 1 / (1 + exp(-x))
    function crossed(xs, a, b)
        s1 = a; s2 = b; s3 = a .* 0.5
        for t in axes(xs, 2)
            z3 = (U * xs[:, t]) ./ (1 .+ abs.(s3))
            z4 = s2 .* (sum(U * xs[:, t]) * 0.1)
            s1, s2, s3 = σ.(z3) .* s3 .+ (1 .- σ.(z3)) .* tanh.(z3), tanh.(z4), σ.(z4) .* s1 .+ (1 .- σ.(z4)) .* tanh.(z4)
        end
        sum(s2) + sum(abs2, s3)
    end
    for f in (crossing, swapping, second, twins, crossed)
        p = trace(f, args...)
        @test count(n -> n.op isa C.Loop, p.nodes) > 0
        @test only(C.leaves(outputs(execute!(prepare(p, args...)))))[] ≈ f(args...)
        grad = (ys...) -> T.withgradient(f, ys...)
        @test agree(C.leaves(outputs(execute!(prepare(trace(grad, args...), args...)))), C.leaves(grad(args...)))
    end
end

# A recurrence's state reads, after its first step, what its value is
# carried from reads: a kernel reading the state runs after the
# recurrence's, which reads that. No kernel joins a value the recurrence
# reads with a sum over its states, which the cover had proposed: a map the
# step reads and a number carried beside the states, from the sum.
@testset "a kernel reading a recurrence's state runs after what it reads" begin
    W, U, b = data(4, 4; scale = 0.3), data(4, 3; scale = 0.3), data(4; scale = 0.2)
    σ(x) = 1 / (1 + exp(-x))
    function f(xs, a)
        s1 = a; s2 = a; r = sum(a) * 0.1
        for t in axes(xs, 3)
            z2 = tanh.(transpose(W) * (U * xs[:, :, t]))
            z3 = 0.5 .* z2 .+ b
            z4 = z3 .+ s2
            s1, s2 = 0.5 .* s2 .+ 0.5 .* tanh.(z3), σ.(z4) .* s1 .+ (1 .- σ.(z4)) .* tanh.(z2)
            r = 0.5 * r + 0.01 * sum(abs2, z4)
        end
        sum(abs2, s1) + sum(s2) * 0.3 + r
    end
    args = (data(3, 3, 3; scale = 0.4), data(4, 3; scale = 0.5))
    @test only(C.leaves(outputs(execute!(prepare(trace(f, args...), args...)))))[] ≈ f(args...)
end

# A sum rides a recurrence's steps where it adds them up: not the state
# after them over an axis as long as they are, nor a state of a second
# recurrence, whose step reads the first's, at each step, by the first's
# steps, which would compute the second's state too; by the second's, each
# offered once. Nor by the first's, in a gradient, a sum reading the
# second's cycle, which its states reach through what they are carried from.
@testset "sums riding a recurrence add its steps up" begin
    W = data(2, 2; scale = 0.5)
    after(a, xs) = (h = a; for t in axes(xs, 3); h = tanh.(W * h .+ xs[:, :, t]); end; sum(h))
    function second(a, b, xs)
        s, u = a, b
        acc = []
        for t in axes(xs, 3)
            z = tanh.(W * s .+ xs[:, :, t])
            s = 0.5 .* s .+ 0.5 .* z
            u = 0.5 .* u .+ 0.5 .* tanh.(z)
            push!(acc, u)
        end
        sum(abs2, stack(acc))
    end
    for (f, args, offered) in ((after, (data(2, 3), data(2, 3, 3)), 0), (second, (data(2, 3), data(2, 3), data(2, 3, 3)), 1))
        p = trace(f, args...)
        @test count(n -> n.op isa C.Loop, p.nodes) == 1
        planned = prepare(p, args...; build = false).plan
        sums = C.riding_sums(planned)
        @test length(sums) == offered
        for r in sums
            @test C.execute(C.plan(planned.graph; ride = [r]), args...)[1] ≈ f(args...)
        end
    end
    σ(x) = 1 / (1 + exp(-x))
    function crossing(xs, a1, a2, U)
        s1 = a1; s2 = a2; s3 = s1
        for t in axes(xs, 2)
            z3 = s1 .+ U * xs[:, t] .- s2 .* (sum(s1) * 0.1)
            z4 = s2 .* (sum(U * xs[:, t]) * 0.1)
            s1, s2, s3 = σ.(z3) .* s1 .+ (1 .- σ.(z3)) .* tanh.(z3), 0.5 .* s2 .+ 0.5 .* tanh.(z4), 0.5 .* s3 .+ 0.5 .* tanh.(z4)
        end
        sum(abs2, s1) + sum(s2) * 0.3 + sum(s3) * 0.3
    end
    grad = (ys...) -> T.withgradient(crossing, ys...)
    args = (data(2, 5), data(2), data(2), data(2, 2; scale = 0.5))
    planned = prepare(trace(grad, args...), args...; build = false).plan
    for r in C.riding_sums(planned)
        @test agree(C.execute(C.plan(planned.graph; ride = [r]), args...), C.leaves(grad(args...)))
    end
end

# A step run alone reads an input's slice a step on as it reads the step's:
# a loop over `xs[:, :, t]` from its second slice has a gradient whose sum
# for the weights reads `xs` one step past the step's position, and every
# sum offered rides.
@testset "a sum riding a recurrence reads an input steps on" begin
    function shifted(xs, a1, a2, U)
        s1, s2 = a1, a2
        for t in 2:4
            z = U * xs[:, :, t] .+ s1 .* s2
            e = exp.(z .- maximum(z; dims = 1))
            s2 = 0.5 .* s2 .+ 0.5 .* tanh.(e)
        end
        sum(s2) * 0.3
    end
    grad = (ys...) -> T.withgradient(shifted, ys...)
    args = (data(2, 1, 5), data(1, 1; scale = 0.5), data(1, 1; scale = 0.7), data(1, 2; scale = 0.1))
    q = trace(grad, args...)
    sums = C.riding_sums(prepare(q, args...; build = false).plan)
    @test length(sums) == 2
    for r in sums
        @test agree(C.leaves(outputs(execute!(prepare(q, args...; ride = [r])))), C.leaves(grad(args...)))
    end
end

# A recurrence a sum rides runs a step at a time, from its start in storage:
# a start computed and read once, `tanh.(a)`, is stored for it rather than
# recomputed in the recurrence's kernel, where the steps could not read it.
@testset "a sum riding a recurrence whose start is computed" begin
    function started(xs, a, U)
        s = tanh.(a)
        acc = []
        for t in 2:4
            s = 0.5 .* s .+ 0.5 .* tanh.(U * xs[:, :, t] .+ s)
            push!(acc, s)
        end
        sum(abs2, stack(acc)) * 0.1
    end
    grad = (ys...) -> T.withgradient(started, ys...)
    args = (data(2, 1, 5), data(1, 1; scale = 0.5), data(1, 2; scale = 0.3))
    q = trace(grad, args...)
    sums = C.riding_sums(prepare(q, args...; build = false).plan)
    @test length(sums) == 2
    for r in sums
        @test agree(C.leaves(outputs(execute!(prepare(q, args...; ride = [r])))), C.leaves(grad(args...)))
    end
end

# A sum riding a recurrence is computed in its steps with what they compute
# for it, which nothing else reads: an LSTM's weight gradient sums each
# step's product of the input and the gate's gradient there. A kernel of
# one of those products beside the ride, as an implementation's candidate
# of the products made, is one whose work nothing reads, and its plan fails
# the verifier; the LSTM's cuTile claims for it stored nothing and failed to
# build. The ride is about those values too, so the cover takes no proposal
# of a kernel of one with it.
@testset "a sum riding a recurrence is about what its steps compute for it" begin
    σ(x) = 1 / (1 + exp(-x))
    function cell(p, (h, c), x)
        i, f = σ.(p.Wi * x .+ p.Ui * h), σ.(p.Wf * x .+ p.Uf * h)
        g, o = tanh.(p.Wg * x .+ p.Ug * h), σ.(p.Wo * x .+ p.Uo * h)
        c = f .* c .+ i .* g
        (o .* tanh.(c), c)
    end
    function loss(p, xs)
        s = (p.h0, p.h0)
        for t in axes(xs, 3)
            s = cell(p, s, xs[:, :, t])
        end
        sum(abs2, first(s))
    end
    w(o, i) = data(o, i; scale = 0.3)
    p = (; Wi = w(3, 2), Wf = w(3, 2), Wg = w(3, 2), Wo = w(3, 2), Ui = w(3, 3), Uf = w(3, 3), Ug = w(3, 3), Uo = w(3, 3), h0 = data(3, 2))
    args = (p, data(2, 2, 4))
    d = C.Decider(T.CPUTarget(), C.lower(trace((q, xs) -> T.withgradient(loss, q, xs), args...)))
    rides = C.rides(C.generator_plan(d))
    @test !isempty(rides)
    kernels = 0
    for (r, within, how) in rides, v in within
        own = try
            C.plan(d.graph; fixed = [[v]], ride = [r])
        catch e
            e isa C.NotAGroup || rethrow()
            continue
        end
        kernels += 1
        @test_throws C.InvalidPlan C.verify(own)
        @test C.plan_taking(d, [C.Candidate([[v]]), C.Ride(r, within, how)]) === nothing
        @test C.verify(C.plan_taking(d, [C.Ride(r, within, how)])) isa C.Plan
    end
    @test kernels > 0
end

# A result that is a map of a recurrence's cycle is stored, though what it
# reads is: the gradient of a delta rule's values is the step's map of the
# state's adjoint, which its group had left for the result to recompute, and
# no kernel wrote it.
@testset "a result of a recurrence's cycle is stored" begin
    function delta(S, q, k, v, β)
        outputs = []
        for t in axes(q, 2)
            δ = β[t] .* (v[:, t] .- S' * k[:, t])
            S = S .+ k[:, t] * δ'
            push!(outputs, S' * q[:, t])
        end
        stack(outputs), S
    end
    args = (data(4, 3; scale = 0.3), data(4, 6; scale = 0.4), data(4, 6; scale = 0.2), data(3, 6; scale = 0.5), 0.5 .+ 0.1 .* data(6))
    loss(xs...) = (r = delta(xs...); sum(abs2, r[1]) + sum(r[2]))
    grad = (xs...) -> T.withgradient(loss, xs...)
    @test agree(C.leaves(outputs(execute!(prepare(trace(grad, args...), args...)))), C.leaves(grad(args...)); rtol = 1.0e-10)
end

# A value computed from one step of a recurrence, which nothing else reads of
# it, rides its steps as a sum over them does: an output at each step is
# written at each, the state the steps leave at the last, and the states of
# every step are stored nowhere when nothing else reads them, as the delta
# rule's outputs in Qwen3.5-4B, which had stored 2 MiB of state a step to
# read each step's output from. Its gradient reads every step's state, which
# the steps then store.
@testset "values riding a recurrence's steps" begin
    function delta(S, q, k, v, β)
        outputs = []
        for t in axes(q, 2)
            δ = β[t] .* (v[:, t] .- S' * k[:, t])
            S = S .+ k[:, t] * δ'
            push!(outputs, S' * q[:, t])
        end
        stack(outputs), S
    end
    args = (data(4, 3; scale = 0.3), data(4, 6; scale = 0.4), data(4, 6; scale = 0.2), data(3, 6; scale = 0.5), 0.5 .+ 0.1 .* data(6))
    eager = C.leaves(delta(args...))
    p = trace(delta, args...)
    planned = prepare(p, args...; build = false).plan
    recurrence(q) = only(g for g in q.groups if g.kind isa C.RecurrenceKind)
    state = first(recurrence(planned).outputs)
    # Nothing outside reads the state at any step: it is stored at none.
    @test !planned.stored[state]
    riders = C.rides(planned)
    @test sort(last.(riders)) == [:each, :last] && isempty(C.riding_sums(planned))
    ridden = C.plan(planned.graph; ride = C.riding_values(planned))
    @test length(ridden.groups) == 1 && sort(C.structure(ridden).stored) == sort(first.(riders))
    @test agree(C.execute(ridden, args...), eager; rtol = 1.0e-12)
    explained = sprint(C.explain, ridden)
    @test occursin("at each step, riding the steps", explained) && occursin("at the last step, riding the steps", explained)
    for ride in ([first(r) for r in riders if last(r) === :each], C.riding_values(planned))
        ex = prepare(p, args...; ride)
        @test agree(C.leaves(outputs(execute!(ex))), eager; rtol = 1.0e-12)
    end
    # The outputs alone ride: the steps run a step at a time, as plans of their own.
    @test count(k -> k isa C.SteppedKernels, prepare(p, args...; ride = [first(r) for r in riders if last(r) === :each]).kernels) == 1
    loss(xs...) = (r = delta(xs...); sum(abs2, r[1]) + sum(r[2]))
    grad = (xs...) -> T.withgradient(loss, xs...)
    q = trace(grad, args...)
    @test agree(C.leaves(outputs(execute!(prepare(q, args...)))), C.leaves(grad(args...)); rtol = 1.0e-10)
    backward = prepare(q, args...; build = false).plan
    @test any(g -> g.kind isa C.RecurrenceKind && backward.stored[first(g.outputs)], backward.groups)
    if CUDA_FUNCTIONAL
        # The cover takes both: one kernel steps through the delta rule, the
        # state on chip, writing each step's output and the last state.
        q32 = trace(delta, map(single, args)...)
        ex = prepare(q32, map(x -> CuArray(single(x)), args)...; measure = :none)
        @test length(ex.kernels) == 1 && occursin("cuTile recurrence", string(only(ex.kernels)))
        @test agree(map(Array, C.leaves(outputs(execute!(ex)))), eager; rtol = 1.0e-4)
        ex = prepare(trace(grad, map(single, args)...), map(x -> CuArray(single(x)), args)...; measure = :none)
        @test agree(map(x -> x isa AbstractArray ? Array(x) : x, C.leaves(outputs(execute!(ex)))), C.leaves(grad(args...)); rtol = 1.0e-4)
    end
end
