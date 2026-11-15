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

# The fuzzer's cases, as shrunk, that placing on two devices failed:
# recurrences whose derivative accumulates the gradients of arguments on two
# devices, the cycle pinned to both when a gradient was pinned on the value it
# returns (seeds 107 and 355), and a step of a softmax's cumulative sum (255).
# Every labelling of their arrays over two devices plans, verified, computes
# what one device computes, and returns each gradient on its argument's.
module Faults
σ(x) = 1 / (1 + exp(-x))
const U107 = [0.0123]
function seed107(xs, a1, a3, U)
    s1 = a1
    s3 = a3
    for t in 1:3
        x = xs[:, :, t]
        z1 = s1 .* (sum((U * x)) * 0.1)
        z3 = tanh.(0.5 .* s3 .+ U107)
        s3 = σ.(z3) .* s3 .+ (1 .- σ.(z3)) .* tanh.(z1)
        s1 = σ.(z3) .* s1 .+ (1 .- σ.(z3)) .* tanh.(z3)
    end
    sum(s3) * 0.3
end
const U255 = [0.31 -0.12; 0.05 0.44; -0.27 0.18]
function seed255(xs, a1, a2)
    s2 = a2
    for t in 1:1
        x = xs[:, :, t]
        z1 = exp.((U255 * x) .- maximum((U255 * x); dims = 1))
        z2 = cumsum(z1; dims = 1) .* 0.3
        z3 = z2 .- a1
        s2 = σ.(z3) .* s2 .+ (1 .- σ.(z3)) .* tanh.(z2)
    end
    sum(s2) * 0.3
end
const U355 = [0.37;;]
function seed355(xs, a1, a3)
    s1 = a1
    s3 = a3
    acc = []
    for t in 1:3
        x = xs[:, t]
        z1 = (U355 * x) .* (sum(s3) * 0.1)
        z3 = z1 ./ (1 .+ abs.(s3)) ./ (1 .+ abs.((U355 * x)))
        s1, s2, s3 = tanh.(z3), 0.5 .* s3 .+ 0.5 .* tanh.(z3), 0.5 .* s1 .+ 0.5 .* tanh.(z3)
        push!(acc, s1)
    end
    sum(acc[end])
end
end

@testset "the fuzzer's faults across devices: seed $seed" for (seed, f, args) in (
        (107, Faults.seed107, (randn(Xoshiro(1), 1, 1, 3), 0.5 .* randn(Xoshiro(2), 1, 1), 0.5 .* randn(Xoshiro(3), 1, 1), 0.3 .* randn(Xoshiro(4), 1, 1))),
        (255, Faults.seed255, (randn(Xoshiro(5), 2, 1, 1), 0.5 .* randn(Xoshiro(6), 3, 1), 0.5 .* randn(Xoshiro(7), 3, 1))),
        (355, Faults.seed355, (randn(Xoshiro(8), 1, 3), 0.5 .* randn(Xoshiro(9), 1), 0.5 .* randn(Xoshiro(10), 1))))
    program = trace((xs...) -> T.withgradient(f, xs...), args...)
    whole = map(copy, C.leaves(outputs(execute!(prepare(program, args...; target = T.CPUTarget())))))
    for labels in Iterators.product(ntuple(_ -> 0:1, length(args))...)
        allequal(labels) && continue
        ex = prepare(program, map(Labelled, map(copy, args), labels)...; target = T.CPUTarget(), verify = true)
        @test map(copy, C.leaves(outputs(execute!(ex)))) == whole
        @test gradients_off(ex.plan, program) == 0
    end
end
