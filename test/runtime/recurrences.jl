# A recurrence prepared: one group computes its state, the values of its
# cycle read outside it and what the state is carried from, a step at a
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
    grad(f) = (xs...) -> T.gradient((ys...) -> sum(x -> sum(abs2, x), f(ys...)), xs...)
    for (f, args) in cases, (device, rtol) in devices
        p = trace(grad(f), map(x -> device(x) isa CuArray{Float32} ? single(x) : x, args)...)
        ex = prepare(p, map(device, args)...)
        @test agree(C.leaves(outputs(execute!(ex))), C.leaves(interpret(trace(grad(f), args...), args...)); rtol)
    end
    # The product each step returns is of its cycle: the group stores it at
    # each step, along the state's columns and steps, its rows an axis of
    # their own. The state one step on it stores where it is returned.
    products(xs...) = last(chunks(xs...))
    for (f, stores) in ((chunks, true), (products, false))
        p = C.plan(C.lower(trace(f, inputs(5)...)))
        g = only(g for g in p.groups if g.kind === :recurrence)
        @test C.recurrence_axes(p.graph, g) == (axis = 3, parallel = [2], inner = [1, 4])
        @test g.domain == (3, 2, 5, 4) && length(g.outputs) == 3 && p.stored[last(g.outputs)] == stores
    end
    kernels(f, n) = length(prepare(trace(f, inputs(n)...), inputs(n)...).kernels)
    @test kernels(chunks, 5) == kernels(chunks, 9) && kernels(grad(chunks), 5) == kernels(grad(chunks), 9)
    if CUDA_FUNCTIONAL
        # In Float32 the generator steps through it, the state's tile on chip,
        # in one kernel; at a state's size, reading it in pieces of that tile.
        state(c, dk, dv, n) = (data(c, dk, n; scale = 0.3), data(c, dv, n; scale = 0.2), data(c, dk, n; scale = 0.01),
            0.5 .+ 0.1 .* data(1, 1, n), data(dk, dv; scale = 0.4))
        for args in (inputs(5), state(64, 128, 128, 3))
            ex = prepare(trace(chunks, map(single, args)...), map(x -> CuArray(single(x)), args)...)
            @test count(k -> occursin("cuTile recurrence", string(k)), ex.kernels) == 1
            @test agree(C.leaves(outputs(execute!(ex))), C.leaves(interpret(trace(chunks, args...), args...)); rtol = 1.0e-4)
        end
    end
end
