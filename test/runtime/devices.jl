# A program whose arguments are on two devices, by label, prepared and run on
# the CPU target, a crossing between them a copy on the host; CUDA's target
# refuses such arguments still.
include("../fixtures/devices.jl")

# The leaves of the outputs of `ex`, copied.
leaf_copies(ex) = map(copy, C.leaves(outputs(ex)))

# The leaves of a compiled function's results.
result_leaves(r) = collect(C.leaves(r))

# The model's arrays again, other numbers, as a later call brings them.
again(ws, x, t) = (map(w -> map(a -> a .+ 1, w), ws), 2 .* x, t .- 1)

@testset "the placed model run over two devices on the CPU, $E" for E in (Float32, Float64)
    ws, x, t = placed_model(E)
    lws, lx, lt = labelled_model(ws, x, t)
    ws2, x2, t2 = again(ws, x, t)
    lws2, lx2, lt2 = labelled_model(ws2, x2, t2)
    for (f, args, labelled, args2, labelled2) in ((placed_half, (ws, x), (lws, lx), (ws2, x2), (lws2, lx2)),
            (placed_step, (ws, x, t), (lws, lx, lt), (ws2, x2, t2), (lws2, lx2, lt2)))
        program = trace(f, args...)
        whole = prepare(program, args...; target = T.CPUTarget(), verify = true)
        placed = prepare(program, labelled...; target = T.CPUTarget(), verify = true)
        @test crossed_bytes(placed.plan) == fill(sizeof(x), f === placed_half ? 1 : 2)
        # Every gradient the step returns is on its argument's device.
        f === placed_step && @test gradients_off(placed.plan, program) == 0
        execute!(whole)
        execute!(placed)
        @test leaf_copies(placed) == leaf_copies(whole)
        # The labels stand for the devices the arguments are on, the first home.
        @test placed.devices == [0, 1]
        # Each crossing is a step of its own from the device its source is on
        # to its reader's, which the steps of a recording are segments around.
        crossing = [s for s in placed.steps if C.crosses(s)]
        @test [(s.from, s.device) for s in crossing] == (f === placed_half ? [(0, 1)] : [(0, 1), (1, 0)])
        @test all(s -> s.run isa C.Copy, crossing)
        capture!(placed)
        segments = placed.captured.segments
        @test [s.device for s in segments] == (f === placed_half ? [0, 1, 1] : [0, 1, 1, 0, 0])
        @test all(s -> allequal((step.device, step.from) for step in placed.steps[s.steps]), segments)
        @test only.(unique.([[(step.from, step.device) for step in placed.steps[s.steps]] for s in segments])) ==
            (f === placed_half ? [(0, 0), (0, 1), (1, 1)] : [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)])
        fill!(first(C.leaves(outputs(placed))), 0)
        replay!(placed)
        @test leaf_copies(placed) == leaf_copies(whole)
        text = sprint(C.explain, placed)
        @test occursin("on device 2", text) && occursin("devices by label: 1 is device 0, 2 is device 1", text)
        @test occursin(Regex(string("the target's copy of ", Base.format_bytes(sizeof(x)), raw" between devices \(estimated [0-9.]+ μs\)")), text)
        # Compiled, each call's arrays bound again, on their devices.
        c, w = compile(f, labelled...), compile(f, args...)
        @test result_leaves(c(labelled...)) == result_leaves(w(args...))
        @test result_leaves(c(labelled2...)) == result_leaves(w(args2...))
        @test result_leaves(c(labelled...)) == result_leaves(w(args...))
    end
end

@testset "an argument on another device than planned is refused by name" begin
    ws, x, t = placed_model(Float32)
    lws, lx, lt = labelled_model(ws, x, t)
    ex = prepare(trace(placed_half, ws, x), lws, lx; target = T.CPUTarget())
    moved = Labelled(x, 1)
    @test_throws "input 2 is on device 1, where the program was prepared with it on device 0" bind!(ex, lws, moved)
    @test_throws "input 2 is on device 1, where the program was prepared with it on device 0" bind!(ex, 2 => moved)
    @test_throws "input 2 is on device 1, where the program was prepared with it on device 0" C.point!(ex, lws, moved)
    c = compile(placed_half, lws, lx)
    @test_throws "argument 2 is on device 1, where the program was prepared with it on device 0" c(lws, moved)
    # On the host, an array is copied to whichever device its argument is on.
    @test result_leaves(c(lws, x)) == result_leaves(c(lws, lx))
end

@testset "a program that writes an argument on the second device" begin
    x, w = rand(Float32, 8), rand(Float32, 8)
    step!(x, w) = (w .= w .- 0.5f0 .* x; sum(w))
    expected = copy(w) .- 0.5f0 .* x
    lx, lw = Labelled(copy(x), 0), Labelled(copy(w), 1)
    c = compile(step!, lx, lw)
    @test c.executable.argument_devices == [0, 1]
    @test c(lx, lw) ≈ sum(expected)
    @test lw.data ≈ expected && lx.data == x
end

@testset "the placed model over three devices" begin
    ws, x, t = placed_model(Float64; blocks = 6)
    lws, lx, lt = labelled_model(ws, x, t; devices = 3)
    program = trace(placed_step, ws, x, t)
    whole = execute!(prepare(program, ws, x, t; target = T.CPUTarget()))
    placed = prepare(program, lws, lx, lt; target = T.CPUTarget(), verify = true)
    @test placed.devices == [0, 1, 2]
    execute!(placed)
    @test leaf_copies(placed) == leaf_copies(whole)
    replay!(capture!(placed))
    @test leaf_copies(placed) == leaf_copies(whole)
    @test Set(s.device for s in placed.captured.segments) == Set([0, 1, 2])
end

# A host target that notes the device the task is on as each buffer is
# allocated or uploaded, each launch built and run, each crossing built and
# each segment recorded, to show that what is made for a device is made on
# it: the seams a target of devices fills.
struct Watched <: C.Target
    at::Vector{Union{Nothing, Int}}
    log::Vector{Any}
end

Watched() = Watched(Union{Nothing, Int}[], Any[])
here(t::Watched) = isempty(t.at) ? nothing : last(t.at)
note!(t::Watched, what...) = (push!(t.log, (what..., here(t))); nothing)

C.device_name(::Watched) = "watched"
C.plans_across_devices(::Watched) = true
C.on_device(f, t::Watched, device) = (push!(t.at, device); try f() finally pop!(t.at) end)
C.allocate(t::Watched, T, n) = (note!(t, :allocate, T); Vector{T}(undef, n))
C.upload(t::Watched, x) = (note!(t, :upload); x)
C.crossing(t::Watched, to, from, values) = (note!(t, :crossing, values); Watching(C.Copy(to, from, values), t, values))
C.floor_launch(t::Watched, k::C.ScalarKernel, args) = (note!(t, :build); Watching(C.floor_launch(T.CPUTarget(), k, args), t, nothing))
C.capture(t::Watched, steps, keep) = (note!(t, :segment, [(s.from, s.device) for s in steps]); steps)

# A launch that notes where it runs.
struct Watching
    run::Any
    target::Watched
    values::Any
end

# The launches a step runs that note where they run.
launches(r::C.FloorRun) = length(r.launches)
launches(::Watching) = 1
launches(_) = 0

(w::Watching)() = (note!(w.target, :run, w.values); w.run())
C.Implement.repoints(w::Watching) = C.Implement.repoints(w.run)
C.Implement.repoint!(w::Watching, value, buffer) = (C.Implement.repoint!(w.run, value, buffer); w)

@testset "what is made for a device is made on it" begin
    ws, x, t = placed_model(Float32)
    lws, lx, lt = labelled_model(ws, x, t)
    # A constant read on both devices, made again on each.
    scale = Float32.(1:16)
    f(ws, x, t) = placed_loss(ws, x .* scale, t .* scale)
    target = Watched()
    ex = prepare(trace(f, ws, x, t), lws, lx, lt; target)
    p, s = ex.plan, ex.storage
    devices(what) = [e[end] for e in target.log if e[1] === what]
    # Each slot on its pool's device, each constant on its own.
    @test devices(:allocate) ⊇ Set([0, 1])
    @test count(e -> e[1] === :allocate && e[2] === UInt8, target.log) == length(s.sizes)
    @test sort([e[end] for e in target.log if e[1] === :allocate && e[2] === UInt8]) == sort([ex.devices[d] for d in s.devices])
    constants = [v for (v, val) in enumerate(p.graph.values) if val.definition isa C.ConstantArray]
    @test sort(devices(:upload)) == sort([ex.devices[p.graph.devices[v]] for v in constants]) == [0, 1]
    # Each kernel built on its group's device, each crossing on its reader's.
    floors = [(ex.devices[C.group_device(p, g)], length(k.kernels)) for (g, k) in zip(p.groups, ex.kernels) if k isa C.FloorKernels]
    @test sort(devices(:build)) == sort(reduce(vcat, [fill(d, n) for (d, n) in floors]))
    crossings = [g for g in p.groups if g.kind isa C.CrossingKind]
    @test [(e[2], e[end]) for e in target.log if e[1] === :crossing] ==
        [(only(g.outputs) => C.crossed_from(p.graph, only(g.outputs)), ex.devices[C.group_device(p, g)]) for g in crossings]
    # Each step run on its device, and each segment recorded on its own.
    empty!(target.log)
    execute!(ex)
    @test [e[end] for e in target.log if e[1] === :run] == reduce(vcat, [fill(s.device, launches(s.run)) for s in ex.steps])
    empty!(target.log)
    capture!(ex)
    recorded = [e for e in target.log if e[1] === :segment]
    @test length(recorded) == length(ex.captured.segments) > 1
    @test all(((_, steps, at),) -> all(==(at), last.(steps)), recorded)
    replay!(ex)
    whole = execute!(prepare(trace(f, ws, x, t), ws, x, t; target = T.CPUTarget()))
    @test leaf_copies(ex) == leaf_copies(whole)
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
