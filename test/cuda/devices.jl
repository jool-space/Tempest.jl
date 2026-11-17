# A program is prepared for, and runs on, the device its arrays are on,
# whatever device the calling task is on, and leaves the task's device and
# stream as they were. A program whose arrays are on several devices runs
# over them, each value on one and a crossing where it is read on another.
# What needs two devices is skipped on a machine with one.
include("../fixtures/devices.jl")

# The device the task is on, and its stream there.
task_device() = (CUDACore.deviceid(CUDACore.device()), CUDACore.stream())

cu32(x) = CuArray(Float32.(x))

@testset "the device of the arrays" begin
    x = cu32(data(8, 8))
    @test C.device_of(x) == 0
    @test C.device_of(view(x, :, 1:2)) == 0
    @test C.device_of(data(4)) === nothing
    @test C.target_for(x).device == 0
    @test T.CUDATarget(; device = 1).device == 1
    @test contains(repr(T.CUDATarget(T.cuTileGenerator(); device = 1)), "device = 1")
    @test !contains(repr(T.CUDATarget(T.cuTileGenerator())), "device")
    @test T.CUDATarget().device === nothing
    @test compile(x -> x .+ 1, x; target = T.CUDATarget()).executable.target.device == 0


    @testset "host arrays go to the target's device as before" begin
        h = data(16)
        c = compile(h -> 2 .* h, h; target = T.CUDATarget())
        @test c.executable.target.device == 0
        @test Array(c(h)) ≈ 2 .* h
    end

    @testset "a call leaves the task's stream as it was" begin
        c = compile(x -> sum(x * x; dims = 1), x)
        s = CUDACore.CuStream()
        CUDACore.stream!(s) do
            @test Array(c(x)) ≈ sum(Array(x) * Array(x); dims = 1)
            @test task_device() == (0, s)
        end
    end
end

@testset "two devices" begin
    if CUDACore.ndevices() < 2
        @test_skip "needs two devices"
    else
        # The task stays on device 0; the arrays are on device 1.
        CUDACore.device!(0)
        before = task_device()
        a = CUDACore.device!(() -> cu32(data(64, 64)), 1)
        b = CUDACore.device!(() -> cu32(data(64)), 1)
        f(a, b) = sum(a * a; dims = 1) .+ b'

        @testset "arrays on one device run there, whatever device the task is on" begin
            c = compile(f, a, b)
            @test c.executable.target.device == 1
            @test task_device() == before
            y = c(a, b)
            @test C.device_of(y) == 1
            @test task_device() == before
            @test Array(y) ≈ f(Array(a), Array(b))
            # Again, on other arrays of the device.
            a2 = CUDACore.device!(() -> cu32(data(64, 64; scale = 0.5)), 1)
            @test Array(c(a2, b)) ≈ f(Array(a2), Array(b))
            @test task_device() == before
        end

        @testset "uncaptured, and through the executable" begin
            c = compile(f, a, b; capture = false)
            @test Array(c(a, b)) ≈ f(Array(a), Array(b))
            ex = prepare(trace(f, a, b), a, b)
            @test ex.target.device == 1
            @test Array(only(C.leaves(outputs(execute!(ex))))) ≈ f(Array(a), Array(b))
            capture!(ex)
            @test Array(only(C.leaves(outputs(replay!(ex))))) ≈ f(Array(a), Array(b))
            @test task_device() == before
        end

        @testset "a program that writes its arguments writes them on their device" begin
            g!(w, d) = (w .-= 0.1f0 .* d; nothing)
            w = CUDACore.device!(() -> cu32(data(32)), 1)
            d = CUDACore.device!(() -> cu32(data(32; scale = 2)), 1)
            expected = Array(w) .- 0.1f0 .* Array(d)
            c = compile(g!, w, d)
            c(w, d)
            @test Array(w) ≈ expected
            @test task_device() == before
        end

        @testset "a call that throws leaves the task where it was" begin
            c = compile(f, a, b)
            @test_throws ArgumentError c(a, CUDACore.device!(() -> cu32(data(65)), 1))
            @test task_device() == before
        end

        @testset "arrays on two devices are planned over both, home where the first is" begin
            b0 = cu32(data(64))
            c = compile(f, a, b0)
            @test c.executable.devices == [1, 0]
            @test Array(c(a, b0)) ≈ f(Array(a), Array(b0))
            ex = prepare(trace(f, a, b0), a, b0; verify = true)
            @test Array(only(C.leaves(outputs(execute!(ex))))) ≈ f(Array(a), Array(b0))
            @test task_device() == before
        end

        @testset "a target naming no device is at home where the arrays are" begin
            c = compile(f, a, b; target = T.CUDATarget(T.cuTileGenerator()))
            @test c.executable.target.device == 1
            @test c.executable.devices == [1]
            @test Array(c(a, b)) ≈ f(Array(a), Array(b))
            # With every array on the host, on the calling task's device.
            h = data(8, 8)
            c = CUDACore.device!(() -> compile(x -> x .+ 1, h; target = T.CUDATarget()), 1)
            @test c.executable.devices == [1]
            @test task_device() == before
        end

        @testset "a target naming another device than the arrays' is home there" begin
            x0 = cu32(data(8, 8))
            c = compile(x -> x .+ 1, x0; target = T.CUDATarget(; device = 1))
            @test c.executable.devices == [1, 0]
            @test Array(c(x0)) ≈ Array(x0) .+ 1
            @test task_device() == before
        end

        @testset "a call bringing arrays of another device than planned is refused by name" begin
            c = compile(f, a, b)
            a0, b0 = CuArray(Array(a)), CuArray(Array(b))
            @test_throws "argument 1 is on device 0, where the program was prepared with it on device 1" c(a0, b0)
            @test_throws "argument 2 is on device 0, where the program was prepared with it on device 1" c(a, b0)
            ex = prepare(trace(f, a, b), a, b)
            @test_throws "input 1 is on device 0, where the program was prepared with it on device 1" C.point!(ex, a0, b0)
            @test_throws "input 1 is on device 0, where the program was prepared with it on device 1" bind!(ex, a0, b0)
            @test_throws "input 2 is on device 0, where the program was prepared with it on device 1" bind!(ex, 2 => b0)
            # The program still runs on its own device's arrays.
            @test Array(c(a, b)) ≈ f(Array(a), Array(b))
            @test task_device() == before
        end
    end
end

# The placed model's arrays on the GPU: block `k`'s weights on device
# `device(k)`, the input on the first's and the target on the last's.
function placed_on(ws, x, t, device)
    on(d, a) = CUDACore.device!(() -> CuArray(a), d)
    (Tuple(map(a -> on(device(k), a), w) for (k, w) in enumerate(ws)), on(device(1), x), on(device(length(ws)), t))
end

# The model over `n` devices, a run of blocks alike on each, and all on device 0.
placed_over(ws, x, t, n) = placed_on(ws, x, t, k -> (k - 1) * n ÷ length(ws))
placed_whole(ws, x, t) = placed_on(ws, x, t, _ -> 0)

# The leaves of a run's results on the host, a number for a number.
host(a::Number) = a
host(a::AbstractArray{<:Any, 0}) = Array(a)[]
host(a) = Array(a)
host_leaves(r) = Any[host(a) for a in C.leaves(r)]

# Whether each gradient program `program` returns of an argument among
# `inputs` is on that argument's device.
gradients_home(ex, program, inputs) =
    all(((r, a),) -> C.device_of(ex.results[r]) == C.device_of(C.argument_leaves(program, inputs)[a]), gradient_pairs(program))

@testset "the placed model over two devices" begin
    if CUDACore.ndevices() < 2
        @test_skip "needs two devices"
    else
        CUDACore.device!(0)
        before = task_device()
        ws, x, t = placed_model(Float32)
        placed, whole = placed_over(ws, x, t, 2), placed_whole(ws, x, t)
        ws2, x2, t2 = again(ws, x, t)
        placed2, whole2 = placed_over(ws2, x2, t2, 2), placed_whole(ws2, x2, t2)
        for (f, arity) in ((placed_half, 2), (placed_step, 3))
            @testset "$f" begin
                program = trace(f, (ws, x, t)[1:arity]...)
                one = execute!(prepare(program, whole[1:arity]...))
                ex = prepare(program, placed[1:arity]...; verify = true)
                @test ex.devices == [0, 1]
                @test crossed_bytes(ex.plan) == fill(sizeof(x), arity - 1)
                f === placed_step && @test gradients_off(ex.plan, program) == 0
                execute!(ex)
                @test all(splat(≈), zip(host_leaves(outputs(ex)), host_leaves(outputs(one))))
                f === placed_step && @test gradients_home(ex, program, placed)
                # Each run of steps on one device is a graph of its own, and
                # each crossing runs between them as it is replayed.
                capture!(ex)
                segments = ex.captured.segments
                @test segment_devices(ex.captured) == (f === placed_half ? [0, 1, 1] : [0, 1, 1, 0, 0])
                @test [s.graph isa Vector{C.Scheduled} for s in segments] == (f === placed_half ? [false, true, false] : [false, true, false, true, false])
                foreach(a -> fill!(a, 0), C.leaves(outputs(ex)))
                replay!(ex)
                @test all(splat(≈), zip(host_leaves(outputs(ex)), host_leaves(outputs(one))))
                # Compiled, each call's arrays bound again on their devices.
                c, w = compile(f, placed[1:arity]...), compile(f, whole[1:arity]...)
                @test all(splat(≈), zip(host_leaves(c(placed[1:arity]...)), host_leaves(w(whole[1:arity]...))))
                @test all(splat(≈), zip(host_leaves(c(placed2[1:arity]...)), host_leaves(w(whole2[1:arity]...))))
                @test all(splat(≈), zip(host_leaves(c(placed[1:arity]...)), host_leaves(w(whole[1:arity]...))))
                @test task_device() == before
            end
        end

        @testset "a result read on another task's stream waits for the call" begin
            # Large enough that a read not ordered after the call would find
            # the result unwritten, and often enough to have caught a read
            # that raced in two calls of five.
            large = placed_model(Float32; blocks = 4, width = 256, tokens = 4096)
            first_arrays = placed_over(large..., 2)
            second_arrays = placed_over(map(w -> map(a -> 0.5f0 .* a, w), large[1]), 2 .* large[2], large[3] .- 1, 2)
            program = trace(placed_step, large...)
            ex = prepare(program, first_arrays...)
            expected = [host_leaves(outputs(execute!(prepare(program, arrays...)))) for arrays in (first_arrays, second_arrays)]
            read_elsewhere(r) = fetch(Threads.@spawn host_leaves(r))
            # The calls of `run` on the first arrays then the second whose
            # results, read on another task, are not the call's.
            wrong(run; trials = 25) = count(1:trials) do _
                any((first_arrays, second_arrays) .=> expected) do (arrays, e)
                    !all(splat(==), zip(read_elsewhere(run(arrays...)), e))
                end
            end
            @test wrong((xs...) -> outputs(execute!(bind!(ex, xs...)))) == 0
            capture!(ex)
            @test wrong((xs...) -> outputs(replay!(bind!(ex, xs...)))) == 0
            @test wrong(compile(placed_step, first_arrays...)) == 0
        end

        @testset "a program that writes an argument on the second device" begin
            step!(x, w) = (w .= w .- 0.5f0 .* x; sum(w))
            h, g = data(64), data(64; scale = 2)
            expected = g .- 0.5f0 .* h
            ch, cg = cu32(h), CUDACore.device!(() -> cu32(g), 1)
            c = compile(step!, ch, cg)
            @test c.executable.argument_devices == [0, 1]
            @test c(ch, cg) ≈ sum(expected)
            @test C.device_of(cg) == 1 && Array(cg) ≈ expected
            @test Array(ch) ≈ h
            @test task_device() == before
        end

        @testset "an argument on another device than planned is refused by name" begin
            ex = prepare(trace(placed_half, ws, x), placed[1:2]...)
            moved = CUDACore.device!(() -> CuArray(x), 1)
            @test_throws "input 2 is on device 1, where the program was prepared with it on device 0" bind!(ex, placed[1], moved)
            @test_throws "input 2 is on device 1, where the program was prepared with it on device 0" bind!(ex, 2 => moved)
            c = compile(placed_half, placed[1:2]...)
            @test_throws "argument 2 is on device 1, where the program was prepared with it on device 0" c(placed[1], moved)
            # On the host, an array is copied to whichever device its argument is on.
            @test host_leaves(c(placed[1], x)) ≈ host_leaves(c(placed[1:2]...))
            @test task_device() == before
        end

        @testset "the placed model over three devices" begin
            if CUDACore.ndevices() < 3
                @test_skip "needs three devices"
            else
                ws, x, t = placed_model(Float32; blocks = 6)
                placed = placed_over(ws, x, t, 3)
                program = trace(placed_step, ws, x, t)
                one = execute!(prepare(program, placed_whole(ws, x, t)...))
                ex = prepare(program, placed...; verify = true)
                @test ex.devices == [0, 1, 2]
                execute!(ex)
                @test all(splat(≈), zip(host_leaves(outputs(ex)), host_leaves(outputs(one))))
                @test gradients_home(ex, program, placed)
                replay!(capture!(ex))
                @test all(splat(≈), zip(host_leaves(outputs(ex)), host_leaves(outputs(one))))
                @test Set(segment_devices(ex.captured)) == Set([0, 1, 2])
            end
        end
    end
end

# A crossing runs on a lane of its own, beside both devices' kernels: what
# overwrites its source, in the call or in the next, waits for it, and the
# next call's steps for the last call's. Each copy is large, 128 MiB, so that
# it takes longer than the kernels around it, and calls are made back to
# back, unsynchronized, their results compared to the bit with the same
# program on one device.
@testset "crossings beside the kernels" begin
    if CUDACore.ndevices() < 2
        @test_skip "needs two devices"
    else
        CUDACore.device!(0)
        before = task_device()
        on(d, a) = CUDACore.device!(() -> CuArray(a), d)
        n = 32 << 20
        # The state on device 0, updated in place by the first kernel of each
        # call, crosses to device 1, where it is added into the other state.
        step!(w, v) = (w .= w .+ 1.0f0; v .= v .+ w; 2.0f0 .* v)
        w, v = rand(Xoshiro(1), Float32, n), rand(Xoshiro(2), Float32, n)

        @testset "the next call overwrites what a crossing of the last reads" begin
            placed, whole = compile(step!, on(0, w), on(1, v)), compile(step!, on(0, w), on(0, v))
            @test any(C.crosses, placed.executable.steps)
            pw, pv, ww, wv = on(0, w), on(1, v), on(0, w), on(0, v)
            got = [placed(pw, pv) for _ in 1:20]
            expected = [whole(ww, wv) for _ in 1:20]
            @test all(k -> Array(got[k]) == Array(expected[k]), 1:20)
            @test Array(pw) == Array(ww) && Array(pv) == Array(wv)
            @test task_device() == before
        end

        @testset "a recording replayed back to back is the execution" begin
            program = trace(step!, w, v)
            live, recorded = prepare(program, on(0, w), on(1, v); point = true), prepare(program, on(0, w), on(1, v); point = true)
            capture!(recorded)
            foreach(_ -> execute!(live), 1:20)
            foreach(_ -> replay!(recorded), 1:20)
            @test host_leaves(outputs(recorded)) == host_leaves(outputs(live))
            # Pointed at other arrays, the recording is changed in place.
            graphs = [s.graph for s in recorded.captured.segments]
            C.point!(recorded, on(0, w), on(1, v))
            C.point!(live, on(0, w), on(1, v))
            @test !any(recorded.settles)
            @test [s.graph for s in recorded.captured.segments] == graphs
            replay!(recorded)
            execute!(live)
            @test host_leaves(outputs(recorded)) == host_leaves(outputs(live))
        end

        @testset "a copy back into the storage a crossing read" begin
            # A value crossed to device 1 and back, the copy back in the slot
            # of the value first crossed, once that crossing has read it.
            there_and_back(w0, x, w1, w2) = (h = tanh.(w0 * x); y = tanh.(w1 * h); tanh.(w2 * y) .+ x)
            # The two copies, 112 MiB each, cost less than the 256 MiB of `w1` that
            # computing on device 0 would copy instead.
            m, k = 8192, 3584
            rng = Xoshiro(3)
            xs = (randn(rng, Float32, m, m) ./ m, randn(rng, Float32, m, k), randn(rng, Float32, m, m) ./ m, randn(rng, Float32, m, m) ./ m)
            placed = compile(there_and_back, on(0, xs[1]), on(0, xs[2]), on(1, xs[3]), on(0, xs[4]))
            ex = placed.executable
            crossing = findall(C.crosses, ex.steps)
            @test length(crossing) == 2
            @test !isempty(intersect(ex.touches[crossing[1]].reads, ex.touches[crossing[2]].writes))
            whole = compile(there_and_back, map(a -> on(0, a), xs)...)
            pa, wa = map(a -> on(0, a), xs), map(a -> on(0, a), xs)
            pa = (pa[1], pa[2], on(1, xs[3]), pa[4])
            got = [placed(pa...) for _ in 1:5]
            expected = [whole(wa...) for _ in 1:5]
            @test all(k -> Array(got[k]) == Array(expected[k]), 1:5)
        end

        @testset "a crossing of a span copies those bytes" begin
            # Device 1 reads a middle range of six columns of `a`, three chunks
            # of the staged copy, from a column three in: less than `b`, which
            # computing on device 0 would copy instead.
            n = 1 << 20
            through(a, b) = a[:, 4:9] .* b
            rng = Xoshiro(4)
            a, b = randn(rng, Float32, n, 12), randn(rng, Float32, n, 6, 3)
            placed = compile(through, on(0, a), on(1, b))
            @test crossed_bytes(placed.executable.plan) == [sizeof(Float32) * n * 6]
            @test occursin("from element $(3n) to %", sprint(C.explain, placed.executable))
            whole = compile(through, on(0, a), on(0, b))
            @test Array(placed(on(0, a), on(1, b))) == Array(whole(on(0, a), on(0, b)))
        end
        @test task_device() == before
    end
end

# Data parallelism in its own spelling over the devices there are, two and
# three: each replica updated from its gradients reduced on its own device,
# from the others' gradients crossed to it, stays its fellows' to the bit.
@testset "replicas over $n devices stay alike to the bit" for n in (2, 3)
    if CUDACore.ndevices() < n
        @test_skip "needs $n devices"
    else
        on(d, a) = CUDACore.device!(() -> CuArray(a), d)
        ws, x, t = placed_model(Float32; tokens = 96)
        rs, xs, ts = replicas_of(ws, x, t, n)
        args = (Tuple(map(w -> map(a -> on(k - 1, a), w), r) for (k, r) in enumerate(rs)),
            Tuple(on(k - 1, a) for (k, a) in enumerate(xs)), Tuple(on(k - 1, a) for (k, a) in enumerate(ts)))
        c = compile(replicated_sgd!, args...)
        @test c.executable.devices == collect(0:(n - 1))
        foreach(_ -> c(args...), 1:3)
        replicas = [map(Array, collect(Iterators.flatten(r))) for r in args[1]]
        @test all(r -> r == replicas[1], replicas)
        copies = [s for s in c.executable.steps if C.crosses(s)]
        @test Set((s.from, s.device) for s in copies) == Set((a, b) for a in 0:(n - 1), b in 0:(n - 1) if a != b)
    end
end

# The recording of data parallelism over the devices there are, three at
# most: a crossing waits for one event, the signal of the step that wrote its
# source, or for none where its lane is ordered after that step already (see
# `stream_schedule`); and each signal a crossing waits for is recorded in its
# device's graph directly after its step, on that step's nodes alone and
# before any later step of the segment, so that the crossing starts once its
# source is written, not once the segment ends.
@testset "a crossing waits for the signal recorded directly after its source's writer" begin
    n = min(CUDACore.ndevices(), 3)
    if n < 2
        @test_skip "needs two devices"
    else
        CX = Base.get_extension(Tempest, :CUDAExt)
        on(d, a) = CUDACore.device!(() -> CuArray(a), d)
        rs, xs, ts = replicas_of(placed_model(Float32; tokens = 96)..., n)
        args = (Tuple(map(w -> map(a -> on(k - 1, a), w), r) for (k, r) in enumerate(rs)),
            Tuple(on(k - 1, a) for (k, a) in enumerate(xs)), Tuple(on(k - 1, a) for (k, a) in enumerate(ts)))
        ex = capture!(prepare(trace(replicated_sgd!, rs, xs, ts), args...))
        r, steps, touches = ex.captured, ex.steps, ex.touches
        # The step that wrote what crossing `k` copies.
        writer(k) = only(unique(findlast(j -> key in touches[j].writes, 1:(k - 1)) for key in touches[k].reads))
        # The nodes `node` depends on, at any remove.
        function before(node)
            seen, todo = Set{CUDACore.CUgraphNode}(), CX.dependencies(node)
            while !isempty(todo)
                x = pop!(todo)
                x in seen || (push!(seen, x); append!(todo, CX.dependencies(x)))
            end
            seen
        end
        # Whether step `j`'s signal is recorded in its graph right after it.
        function recorded_after(j)
            segment, i = r.at[j]
            g = r.segments[segment].graph
            g isa CX.Graph || return false
            record = g.signals[i]
            nodes(places) = Set(x.node for p in places for x in g.nodes[p])
            deps = CX.dependencies(record.node)
            record.kind == CUDACore.CU_GRAPH_NODE_TYPE_EVENT_RECORD && record.params == r.schedule.steps[j].signal.handle &&
                !isempty(deps) && deps ⊆ nodes(i:i) && isdisjoint(before(record.node), nodes((i + 1):length(g.nodes)))
        end
        crossing = findall(C.crosses, steps)
        waiting = filter(k -> !isempty(r.schedule.steps[k].waits), crossing)
        @test length(crossing) >= n * (n - 1) && !isempty(waiting)
        @test filter(k -> r.schedule.steps[k].waits != [r.schedule.steps[writer(k)].signal], waiting) == []
        @test filter(k -> !recorded_after(writer(k)), waiting) == []
    end
end

# A crossing between two devices that reach each other's memory goes over
# the link they share, one copy by the driver, and between two that do not,
# through the host; either way copies the bytes, a span of the source's from
# an offset, on the lane of the two, and the way through the host is one a
# pair that shares a link can take too.
@testset "a crossing is direct between devices that reach each other, staged between others" begin
    if CUDACore.ndevices() < 2
        @test_skip "needs two devices"
    else
        CX = Base.get_extension(Tempest, :CUDAExt)
        target = T.CUDATarget()
        n, offset = 5 << 20, 1 << 20
        for from in 0:(CUDACore.ndevices() - 1), to in 0:(CUDACore.ndevices() - 1)
            from == to && continue
            # What the task's streams launch is done before a lane copies, as a call's beginning has it.
            ready(f, d) = CUDACore.device!(() -> (x = f(); CUDACore.synchronize(); x), d)
            source = ready(() -> CuArray(rand(Xoshiro(from), Float32, n)), from)
            reaches = CUDACore.can_access_peer(CUDACore.CuDevice(to), CUDACore.CuDevice(from))
            out = CUDACore.device!(() -> C.link(target, from), from)
            lane = CUDACore.device!(() -> C.lane(target, from, to, out), to)
            copy = ready(() -> CUDACore.zeros(Float32, n - offset), to)
            built = CUDACore.device!(() -> C.crossing(target, copy, source, 2 => 1, offset), to)
            @test built.way isa (reaches ? CX.Direct : CX.Staging)
            staged = CUDACore.device!(() -> CX.Crossing(copy, source, built.to_address, built.from_address, 2 => 1, built.offset, CX.Staging(sizeof(copy), from)), to)
            for crossing in (built, staged)
                ready(() -> fill!(copy, 0), to)
                C.on_stream(() -> crossing(), target, lane)
                CUDACore.device!(() -> CUDACore.synchronize(lane.upload), to)
                @test Array(copy) == Array(source)[(offset + 1):end]
            end
        end
    end
end

# The link a decider fits for an ordered pair of devices prices a crossing of
# a size it was not fitted on within a factor of 1.5 of the crossing timed
# alone, whichever way the pair copies.
@testset "a link fitted for a pair of devices prices a crossing as it is timed" begin
    if CUDACore.ndevices() < 2
        @test_skip "needs two devices"
    else
        bytes = 16 << 20
        for from in 0:(CUDACore.ndevices() - 1), to in 0:(CUDACore.ndevices() - 1)
            from == to && continue
            target = T.CUDATarget(; device = from)
            d = C.Decider(target, C.lower(trace(identity, zeros(Float32, 1))); measure = :none, devices = Union{Nothing, Int}[from, to])
            ratio = C.crossing_seconds(C.link!(d, 1, 2), bytes) / C.timed_crossing(target, from, to, bytes)
            @test 1 / 1.5 < ratio < 1.5
        end
    end
end

# A crossing's page-locked staging is on the NUMA node of its source device,
# whatever node the thread runs on, as /proc/self/numa_maps counts its pages:
# over devices of two nodes, some staging is on a node the thread is not on.
# Where no device reports a node (one node, as GB10), none is chosen, nor
# where the system refuses the thread a memory policy, as a container does.
@testset "a crossing's staging is on its source's node" begin
    CX = Base.get_extension(Tempest, :CUDAExt)
    devices = 0:(CUDACore.ndevices() - 1)
    nodes = Dict(d => CX.host_node(d) for d in devices)
    if all(isnothing, values(nodes))
        @test_skip "needs devices on NUMA nodes the system reports"
    elseif CX.thread_policy() != 0
        @test_skip "needs a thread the system lets choose its memory's node"
    else
        # The nodes of the pages of the mapping at `p`, from its `N<node>=<pages>` counts.
        function page_nodes(p)
            at = string(UInt(p); base = 16) * " "
            line = only(l for l in eachline("/proc/self/numa_maps") if startswith(l, at))
            Set(parse(Int, m[1]) for m in eachmatch(r"\bN(\d+)=\d+", line))
        end
        for from in devices, to in devices
            (from == to || nodes[from] === nothing) && continue
            s = CUDACore.device!(() -> CX.Staging(2CX.CHUNK, from), to)
            @test all(h -> page_nodes(pointer(h)) == Set(nodes[from]), s.host)
        end
    end
end
