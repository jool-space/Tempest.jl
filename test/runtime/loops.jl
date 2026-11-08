# Loops rolled from the tape: a `for` loop traces as its iterations, and the
# window of them that repeats is the loop `accumulate` traces, so a program
# does not grow with its steps; a recurrence no one kernel holds runs a step
# at a time, its step's kernels planned once and launched once per step, and
# a sum over its steps may be added as they run.
@testset "loops rolled and stepped" begin
    dense(l, u) = l.weight * u .+ l.bias
    field(p, u) = dense(p.dec, tanh.(dense(p.enc, u)))
    function rk4(f, u, steps)
        dt = 1.0f0 / Float32(steps)
        for _ in 1:steps
            k1 = f(u)
            k2 = f(u .+ dt / 2 .* k1)
            k3 = f(u .+ dt / 2 .* k2)
            k4 = f(u .+ dt .* k3)
            u = u .+ dt / 6 .* (k1 .+ 2.0f0 .* k2 .+ 2.0f0 .* k3 .+ k4)
        end
        u
    end
    params = (enc = (weight = single(data(8, 4; scale = 0.3)), bias = single(data(8; scale = 0.1))),
        dec = (weight = single(data(4, 8; scale = 0.3)), bias = single(data(4; scale = 0.1))))
    u0, y = single(data(4, 6)), single(data(4, 6; scale = 0.5))
    loss(p, u0, y, steps) = sum(abs2.(rk4(u -> field(p, u), u0, steps) .- y)) / length(y)
    train(steps) = (p, u0, y) -> T.withgradient(q -> loss(q, u0, y, steps), p)
    loops(p) = count(n -> n.op isa C.Loop, p.nodes)
    stepped(ex) = count(k -> k isa C.SteppedKernels, ex.kernels)
    # The program and its kernels are as many at 3 steps as at 12, and agree
    # with the loop run untraced.
    programs = Dict(steps => trace(train(steps), params, u0, y) for steps in (3, 12))
    @test length(programs[3].nodes) == length(programs[12].nodes) && loops(programs[3]) > 0
    for (steps, p) in programs
        eager = train(steps)(params, u0, y)
        @test agree(C.leaves(interpret(p, params, u0, y)), C.leaves(eager); rtol = 1.0e-5)
        ex = prepare(p, params, u0, y)
        @test stepped(ex) == 2
        @test agree(C.leaves(outputs(execute!(ex))), C.leaves(eager); rtol = 1.0e-5)
    end
    @test length(prepare(programs[3], params, u0, y).kernels) == length(prepare(programs[12], params, u0, y).kernels)
    # Outside a gradient, the loop is rolled as the trace ends.
    forward = trace((p, u0, y) -> loss(p, u0, y, 12), params, u0, y)
    @test loops(forward) == 1
    @test interpret(forward, params, u0, y) ≈ loss(params, u0, y, 12)
    # Each step's state read after the loop: a slice of the loop's, where the
    # steps it reads leave room for the loop before them; else unrolled.
    function trajectory(u, steps)
        us = typeof(u)[]
        for _ in 1:steps
            u = tanh.(params.enc.weight' * (params.enc.weight * u)) .+ u
            push!(us, u)
        end
        sum(sum.(abs2, us))
    end
    p = trace(u -> trajectory(u, 8), u0)
    @test interpret(p, u0) ≈ trajectory(u0, 8)
    @test only(C.leaves(outputs(execute!(prepare(p, u0)))))[] ≈ trajectory(u0, 8)
    # Two states starting from one value, as an LSTM's hidden state and
    # cell may, are two arguments of the step.
    function two(W, x, h)
        c = h
        for _ in 1:6
            z = W * h .+ x
            c = tanh.(z) .* c .+ 0.1
            h = tanh.(z) .* tanh.(c)
        end
        sum(abs2, h)
    end
    args = (data(4, 4; scale = 0.3), data(4), data(4; scale = 0.5))
    p = trace(two, args...)
    @test loops(p) == 1
    @test interpret(p, args...) ≈ two(args...)
    # Two repeats are cheaper unrolled.
    twice = trace(u -> sum(tanh.(tanh.(u))), u0)
    @test loops(twice) == 0
    if CUDA_FUNCTIONAL
        device(x) = x isa AbstractArray ? CuArray(x) : map(device, x)
        p = programs[12]
        ex = prepare(p, device(params), device(u0), device(y))
        @test agree(map(x -> x isa AbstractArray ? Array(x) : x, C.leaves(outputs(execute!(ex)))), C.leaves(train(12)(params, u0, y)); rtol = 1.0e-3)
    end
    # Each parameter's gradient sums every evaluation's part over the steps:
    # added after them, from the parts the backward's steps store, or as they
    # run, where the parts' operands are fresh. The cover prices both; pinned
    # to ride, each sum is a fold of the backward's group, which stores
    # nothing only the sums read, and its steps add each part.
    p = programs[12]
    eager = train(12)(params, u0, y)
    planned = prepare(p, params, u0, y; build = false).plan
    sums = C.riding_sums(planned)
    @test length(sums) == 4
    backward(q) = only(g for g in q.groups if g.kind === :recurrence && !isempty(g.reductions))
    ridden = C.plan(planned.graph; ride = sums)
    @test sort(backward(ridden).reductions) == sort(sums)
    # What the backward's steps compute stays its group's: no join takes a
    # value of it into a kernel that reads it, which would write it again,
    # and a set fixed apart that holds one forms no plan.
    recurrent(v) = planned.owner[v] > 0 && planned.groups[planned.owner[v]].kind === :recurrence
    @test !any(q -> recurrent(q.value), C.joins(planned))
    reader = findfirst(g -> g.kind === :reduction && any(recurrent, C.group_reads(planned.graph, g)), planned.groups)
    member = first(filter(recurrent, collect(C.group_reads(planned.graph, planned.groups[reader]))))
    @test_throws C.NotAGroup C.plan(planned.graph; fixed = [[C.members(planned.groups[reader]); member]])
    @test count(m -> ridden.stored[m], backward(ridden).outputs) < maximum(g -> g.kind === :recurrence ? length(g.outputs) : 0, planned.groups)
    ex = prepare(p, params, u0, y; ride = sums)
    @test sort(only(k.sums for k in ex.kernels if k isa C.SteppedKernels && !isempty(k.sums))) == sort(sums)
    @test agree(C.leaves(outputs(execute!(ex))), C.leaves(eager); rtol = 1.0e-5)
    if CUDA_FUNCTIONAL
        ex = prepare(p, device(params), device(u0), device(y); ride = sums)
        @test count(k -> k isa C.SteppedKernels && !isempty(k.sums), ex.kernels) == 1
        @test agree(map(x -> x isa AbstractArray ? Array(x) : x, C.leaves(outputs(execute!(ex)))), C.leaves(eager); rtol = 1.0e-3)
    end
end

# A loop that selects each step's slice of an array, `xs[:, :, t]`, rolls as
# `accumulate` over the slices does: the loop scans the array, or the range
# of it the steps select, reversed where the index falls; an array the loop
# captures, copied at each use, is computed once; two states may start from
# one value. A selection stepping by more than one stays unrolled.
@testset "loops over slices rolled" begin
    W, U = data(4, 4; scale = 0.3), data(4, 3; scale = 0.2)
    cell(h, x) = tanh.(W * h .+ U * x)
    run(order) = (xs, h) -> (for t in order(size(xs, 3)); h = cell(h, xs[:, :, t]); end; sum(abs2, h))
    σ(x) = 1 / (1 + exp(-x))
    function lstm(xs, h)
        c = h
        for t in axes(xs, 2)
            z = W * h .+ U * xs[:, t]
            c = σ.(z) .* c .+ tanh.(z)
            h = σ.(z) .* tanh.(c)
        end
        sum(abs2, h)
    end
    forcing(ts, u) = (for t in eachindex(ts); u = tanh.(W * u) .* ts[t] .+ u; end; sum(abs2, u))
    xs, h = data(3, 2, 6; scale = 0.4), data(4, 2; scale = 0.2)
    cases = [(run(n -> 1:n), (xs, h), 1), (run(n -> 2:(n - 1)), (xs, h), 1), (run(n -> n:-1:1), (xs, h), 1),
        (lstm, (data(3, 6; scale = 0.4), data(4; scale = 0.1)), 1), (forcing, (data(6; scale = 0.2), h), 1),
        (run(n -> 1:2:n), (xs, h), 0)]
    for (f, args, loops) in cases
        p = trace(f, args...)
        @test count(n -> n.op isa C.Loop, p.nodes) == loops
        @test interpret(p, args...) ≈ f(args...)
        @test only(C.leaves(outputs(execute!(prepare(p, args...)))))[] ≈ f(args...)
        grad = (ys...) -> T.withgradient(f, ys...)
        @test agree(C.leaves(outputs(execute!(prepare(trace(grad, args...), args...)))), C.leaves(grad(args...)))
    end
    if CUDA_FUNCTIONAL
        args = (single(data(3, 6; scale = 0.4)), single(data(4; scale = 0.1)))
        grad = (ys...) -> T.withgradient(lstm, ys...)
        ex = prepare(trace(grad, args...), map(CuArray, args)...)
        @test agree(C.leaves(outputs(execute!(ex))), C.leaves(grad(args...)); rtol = 1.0e-3)
    end
end

# A stencil's steps: each position reads its neighbours at the step before,
# the grid shifted by one, its index wrapped where the shift runs off it, so
# no position's steps are apart from the others' and the floor's kernel
# steps through every position alone. The step is a program of its own, its
# reads quotients and remainders of its positions, and runs a step at a
# time, forward and back.
@testset "stencils stepped" begin
    function interior(u)
        i = reshape(Int32(1):Int32(size(u, 1)), :, 1)
        j = reshape(Int32(1):Int32(size(u, 2)), 1, :)
        (i .> 1) .& (i .< size(u, 1)) .& (j .> 1) .& (j .< size(u, 2))
    end
    function diffuse(u, inside)
        up, down = vcat(u[1:1, :], u[1:(end - 1), :]), vcat(u[2:end, :], u[end:end, :])
        left, right = hcat(u[:, 1:1], u[:, 1:(end - 1)]), hcat(u[:, 2:end], u[:, end:end])
        ifelse.(inside, u .+ eltype(u)(0.2) .* (up .+ down .+ left .+ right .- 4 .* u), u)
    end
    evolve(u, q) = (inside = interior(u); last(accumulate((v, s) -> diffuse(v, inside) .+ s, eachslice(q; dims = 3); init = u)))
    heat(u, q) = T.withgradient(v -> sum(abs2, evolve(v, q)), u)
    args = (data(8, 8), data(8, 8, 5; scale = 0.01))
    stepped(ex) = [k isa C.SteppedKernels for (k, g) in zip(ex.kernels, ex.plan.groups) if g.kind === :recurrence]
    ex = prepare(trace(heat, args...), args...)
    @test stepped(ex) == [true, true]
    @test agree(C.leaves(outputs(execute!(ex))), C.leaves(heat(args...)); rtol = 1.0e-10)
    # Written as a plain loop over the source's slices, its steps hold no
    # fold, but each selects its neighbours from the step before, which an
    # unrolled step does not fuse into the next: rolled, as the accumulate.
    function plain(u, q)
        inside = interior(u)
        for t in axes(q, 3)
            u = diffuse(u, inside) .+ q[:, :, t]
        end
        u
    end
    looped(u, q) = T.withgradient(v -> sum(abs2, plain(v, q)), u)
    p = trace(looped, args...)
    @test count(n -> n.op isa C.Loop, p.nodes) > 0
    ex = prepare(p, args...)
    @test stepped(ex) == [true, true]
    @test agree(C.leaves(outputs(execute!(ex))), C.leaves(looped(args...)); rtol = 1.0e-10)
    # A loop of maps alone, each position reading its own, stays unrolled.
    @test !any(n -> n.op isa C.Loop, trace(u -> (for _ in 1:6; u = tanh.(u) .* 0.5; end; sum(u)), args[1]).nodes)
    if CUDA_FUNCTIONAL
        ex = prepare(trace(heat, map(single, args)...), map(x -> CuArray(single(x)), args)...)
        @test stepped(ex) == [true, true]
        @test agree(map(x -> x isa AbstractArray ? Array(x) : x, C.leaves(outputs(execute!(ex)))), C.leaves(heat(args...)); rtol = 1.0e-4)
    end
end
