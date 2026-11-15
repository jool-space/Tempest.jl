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

        @testset "a target on another device than the arrays' is home" begin
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
                @test [s.device for s in segments] == (f === placed_half ? [0, 1, 1] : [0, 1, 1, 0, 0])
                @test [s.graph isa Vector{C.Step} for s in segments] == (f === placed_half ? [false, true, false] : [false, true, false, true, false])
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
            # Large enough that a read not ordered after the call would find the result unwritten.
            large = placed_model(Float32; blocks = 4, width = 256, tokens = 4096)
            first_arrays = placed_over(large..., 2)
            second_arrays = placed_over(map(w -> map(a -> 0.5f0 .* a, w), large[1]), 2 .* large[2], large[3] .- 1, 2)
            program = trace(placed_step, large...)
            ex = prepare(program, first_arrays...)
            expected = host_leaves(outputs(execute!(prepare(program, second_arrays...))))
            read_elsewhere() = fetch(Threads.@spawn host_leaves(outputs(ex)))
            execute!(ex)
            bind!(ex, second_arrays...)
            execute!(ex)
            @test all(splat(≈), zip(read_elsewhere(), expected))
            capture!(ex)
            bind!(ex, first_arrays...)
            replay!(ex)
            bind!(ex, second_arrays...)
            replay!(ex)
            @test all(splat(≈), zip(read_elsewhere(), expected))
            c = compile(placed_step, first_arrays...)
            c(first_arrays...)
            @test all(splat(≈), zip(fetch(Threads.@spawn host_leaves(c(second_arrays...))), expected))
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
                @test Set(s.device for s in ex.captured.segments) == Set([0, 1, 2])
            end
        end
    end
end
