# A program is prepared for, and runs on, the device its arrays are on,
# whatever device the calling task is on, and leaves the task's device and
# stream as they were. What needs two devices is skipped on a machine with one.

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
    end
end
