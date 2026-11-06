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
