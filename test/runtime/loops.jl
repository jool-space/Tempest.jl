# Loops rolled from the tape: a `for` loop traces as its iterations, and the
# window of them that repeats is the loop `accumulate` traces, so a program
# does not grow with its steps; a recurrence no one kernel holds runs a step
# at a time, its step's kernels planned once and launched once per step.
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
    # Two repeats are cheaper unrolled.
    twice = trace(u -> sum(tanh.(tanh.(u))), u0)
    @test loops(twice) == 0
    if CUDA_FUNCTIONAL
        device(x) = x isa AbstractArray ? CuArray(x) : map(device, x)
        p = programs[12]
        ex = prepare(p, device(params), device(u0), device(y))
        @test agree(map(x -> x isa AbstractArray ? Array(x) : x, C.leaves(outputs(execute!(ex)))), C.leaves(train(12)(params, u0, y)); rtol = 1.0e-3)
    end
end
