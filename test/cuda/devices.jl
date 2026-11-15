# A program is prepared for, and runs on, the device its arrays are on,
# whatever device the calling task is on, and leaves the task's device and
# stream as they were. Arrays on two devices are refused: a program over
# several devices is not planned yet. What needs two devices is skipped on a
# machine with one.

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

    @testset "a target on another device than the arrays' is refused before anything is read" begin
        err = try
            compile(x -> x .+ 1, x; target = T.CUDATarget(; device = 1))
        catch err
            err
        end
        @test err isa ArgumentError
        @test contains(err.msg, "argument 1 on device 0") && contains(err.msg, "the target on device 1") && contains(err.msg, "not planned yet")
        @test_throws ArgumentError prepare(trace(x -> x .+ 1, x), x; target = T.CUDATarget(; device = 1))
    end

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

        @testset "arrays on two devices are refused" begin
            b0 = cu32(data(64))
            err = try
                compile(f, a, b0)
            catch err
                err
            end
            @test err isa ArgumentError
            @test contains(err.msg, "argument 1 on device 1") && contains(err.msg, "argument 2 on device 0") && contains(err.msg, "not planned yet")
            @test_throws ArgumentError prepare(trace(f, a, b0), a, b0)
            @test task_device() == before
        end

        @testset "a call bringing arrays of another device than the program's is refused" begin
            c = compile(f, a, b)
            a0, b0 = CuArray(Array(a)), CuArray(Array(b))
            err = try
                c(a0, b0)
            catch err
                err
            end
            @test err isa ArgumentError
            @test contains(err.msg, "the target on device 1") && contains(err.msg, "not planned yet")
            # One array of the call on the program's device and one not.
            @test_throws ArgumentError c(a, b0)
            ex = prepare(trace(f, a, b), a, b)
            @test_throws ArgumentError C.point!(ex, a0, b0)
            @test_throws ArgumentError bind!(ex, a0, b0)
            @test_throws ArgumentError bind!(ex, 2 => b0)
            # The program still runs on its own device's arrays.
            @test Array(c(a, b)) ≈ f(Array(a), Array(b))
            @test task_device() == before
        end
    end
end
