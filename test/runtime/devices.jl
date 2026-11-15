# A program whose arguments are on two devices, by label, prepared and run on
# the CPU target, a crossing between them a copy on the host; CUDA's target
# refuses such arguments still.
include("../fixtures/devices.jl")

@testset "the placed model run over two devices on the CPU, $E" for E in (Float32, Float64)
    ws, x, t = placed_model(E)
    lws, lx, lt = labelled_model(ws, x, t)
    for (f, args, labelled) in ((placed_half, (ws, x), (lws, lx)), (placed_step, (ws, x, t), (lws, lx, lt)))
        program = trace(f, args...)
        whole = prepare(program, args...; target = T.CPUTarget(), verify = true)
        placed = prepare(program, labelled...; target = T.CPUTarget(), verify = true)
        @test crossed_bytes(placed.plan) == fill(sizeof(x), f === placed_half ? 1 : 2)
        # Every gradient the step returns is on its argument's device.
        f === placed_step && @test gradients_off(placed.plan, program) == 0
        execute!(whole)
        execute!(placed)
        @test map(copy, C.leaves(outputs(placed))) == map(copy, C.leaves(outputs(whole)))
        @test occursin("crossing over", sprint(C.explain, placed))
    end
end

@testset "a target that does not plan across devices refuses them" begin
    @test C.plans_across_devices(T.CPUTarget())
    @test !C.plans_across_devices(T.CUDATarget())
    x = (Labelled(rand(Float32, 3), 0), Labelled(rand(Float32, 3), 1))
    @test_throws "a program over several devices is not planned yet" C.refuse_devices(T.CUDATarget(), x, C.input_paths)
end
