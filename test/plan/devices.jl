# Plans over several devices: a device for each value, a crossing where a
# value is read on another device, and the device invariant of the verifier.
include("../fixtures/devices.jl")

# The plan of `f` on `args`, its arguments on the devices `labels` gives.
function placed_plan(f, args, labels)
    graph = C.rewrite(C.lower(trace(f, args...)))
    C.plan(C.with_devices(graph, labels))
end

function invalid_by(f)
    try
        f()
        nothing
    catch err
        err isa C.InvalidPlan || rethrow()
        err.invariant
    end
end

@testset "the placed model planned over $n devices, $T" for T in (Float32, Float64), n in (2, 3)
    ws, x, t = placed_model(T; blocks = 2n)
    for (f, args) in ((placed_half, (ws, x)), (placed_step, (ws, x, t)))
        program = trace(f, args...)
        labels = C.argument_devices(program, labelled_model(ws, x, t; devices = n)[1:length(args)])
        p = C.verify(placed_plan(f, args, labels))
        whole = C.plan(C.rewrite(C.lower(program)))
        @test C.execute(p, args...) == C.execute(whole, args...)
        q, _, s = C.storage(p)
        @test C.verify(q, s) === q
        # A minimum cut copies the activation alone where the weights change
        # device, forward and, in the step, its gradient back: no value
        # between two runs of blocks is smaller, a block's norm reading it
        # whole and its residual adding it.
        @test crossed_bytes(p) == fill(sizeof(x), f === placed_half ? n - 1 : 2(n - 1))
        @test occursin("on device $n", sprint(C.explain, p))
        @test occursin(r"crossing over \(16, 32\) on device 2, reads %\d+\n  copies %\d+ \w+\(16, 32\) on device 1", sprint(C.explain, p))
        # The gradient of each argument, the target's too, is computed where
        # the argument is, by the cut alone.
        f === placed_step && @test gradients_off(p, program) == 0
    end
end

@testset "placement is a minimum cut of bytes" begin
    # A sum of a product added to an array of the second device: copying the
    # sum's 3 numbers there costs less than copying the 4×3 factor of the
    # first, or the 5×3 addend the other way.
    f(a, b, c) = sum(a .* b; dims = 1) .+ c
    args = (rand(Float32, 4, 3), rand(Float32, 1, 3), rand(Float32, 5, 3))
    p = C.verify(placed_plan(f, args, [1, 1, 2]))
    @test crossed_bytes(p) == [12]
    # A value read on the other device is copied once, however many read it there.
    g(a, b) = sum(abs2, a; dims = 1) .* b .+ sum(a; dims = 1) .* b
    p = C.verify(placed_plan(g, (rand(Float32, 4, 3), rand(Float32, 1, 3)), [2, 1]))
    @test length(crossed_bytes(p)) == 1
    # The same program and labels place alike.
    @test placed_plan(g, (rand(Float32, 4, 3), rand(Float32, 1, 3)), [2, 1]).graph.devices == p.graph.devices
end

@testset "a constant is made on each device that reads it" begin
    w = rand(Float32, 3, 3)
    f(a, b) = sum(w * a) + sum(w * b)
    args = (rand(Float32, 3, 2), rand(Float32, 3, 2))
    p = C.verify(placed_plan(f, args, [1, 2]))
    constants = [v for v in eachindex(p.graph.values) if p.graph.values[v].definition isa C.ConstantArray]
    @test sort(p.graph.devices[constants]) == [1, 2]
    @test all(v -> !(p.graph.values[v].definition isa C.ConstantArray), (p.graph.values[c].definition.operand.value for c in eachindex(p.graph.values) if C.crossing(p.graph, c)))
    @test C.execute(p, args...) == C.execute(C.plan(C.rewrite(C.lower(trace(f, args...)))), args...)
end

@testset "a recurrence's cycle is on one device" begin
    cell(Wh, xs, h0) = sum(last(accumulate((h, x) -> tanh.(transpose(Wh) * h .+ x), eachslice(xs; dims = 2); init = h0)))
    args = (rand(Float32, 4, 4), rand(Float32, 4, 6), rand(Float32, 4))
    for labels in ([1, 2, 2], [2, 1, 2], [2, 2, 1])
        program = trace(cell, args...)
        p = C.verify(placed_plan(cell, args, labels))
        @test C.execute(p, args...) == C.execute(C.plan(C.rewrite(C.lower(program))), args...)
        step = trace((xs...) -> T.withgradient(cell, xs...), args...)
        graph = C.rewrite(C.lower(step))
        q = C.verify(C.plan(C.with_devices(graph, labels)))
        @test C.execute(q, args...) == C.execute(C.plan(graph), args...)
    end
end

@testset "two requirements of one value that differ are refused" begin
    f(a, b) = a .+ b
    graph = C.rewrite(C.lower(trace(f, rand(Float32, 3), rand(Float32, 3))))
    input = only(v for v in eachindex(graph.values) if graph.values[v].definition == C.Input(2))
    @test_throws "must be on device 1 and value %$input on device 2" C.with_devices(graph, [1, 2]; pins = Dict(input => 1))
end

@testset "a program on one device plans as it did" begin
    ws, x, _ = placed_model(Float32)
    graph = C.rewrite(C.lower(trace(placed_half, ws, x)))
    placed = C.with_devices(graph, ones(Int, length(graph.arguments)))
    @test placed.values == graph.values
    @test sprint(C.explain, C.plan(placed)) == sprint(C.explain, C.plan(graph))
    @test !occursin("device", sprint(C.explain, C.plan(placed)))
end

@testset "invariant 5: devices" begin
    f(a, b) = sum(a; dims = 1) .* b
    args = (rand(Float32, 4, 3), rand(Float32, 1, 3))
    p = placed_plan(f, args, [1, 2])
    crossing = only(filter(g -> g.kind isa C.CrossingKind, p.groups))
    c = only(crossing.outputs)
    source = p.graph.values[c].definition.operand.value
    relabelled(devices) = C.Plan(C.IndexGraph(p.graph.values, p.graph.arguments, p.graph.results, p.graph.result_dims,
        p.graph.result_scalar, p.graph.result_eltype, p.graph.rewrites, p.graph.backward, devices), p.stored, p.groups, p.owner)
    @test C.verify(p) === p
    # A read joining two devices but through a crossing: the crossing's reader moved to its source's device.
    reader = only(v for v in eachindex(p.graph.values) if any(o -> c in C.read_values(o), C.operands(p.graph.values[v].definition)))
    devices = copy(p.graph.devices)
    devices[reader] = p.graph.devices[source]
    @test invalid_by(() -> C.verify(relabelled(devices))) == 5
    # A crossing onto its own device.
    devices = copy(p.graph.devices)
    devices[c] = devices[source]
    @test invalid_by(() -> C.verify(relabelled(devices))) == 5
    # A crossing not stored: what reads it reads what nothing stores.
    stored = copy(p.stored)
    stored[c] = false
    @test invalid_by(() -> C.verify(C.Plan(p.graph, stored, p.groups, p.owner))) == 1
    # A kernel computing a member on another device than its own.
    g = only(filter(g -> !(g.kind isa C.CrossingKind) && length(C.members(g)) > 1, p.groups))
    devices = copy(p.graph.devices)
    devices[last(C.members(g))] = 3 - devices[first(C.members(g))]
    @test invalid_by(() -> C.verify(relabelled(devices))) == 5
    # A crossing computed by a kernel that is not a crossing's.
    k = findfirst(==(crossing), p.groups)
    groups = copy(p.groups)
    groups[k] = C.Group(C.PointwiseKind(), crossing.domain, crossing.reductions, crossing.outputs, crossing.terms, crossing.phase)
    @test invalid_by(() -> C.verify(C.Plan(p.graph, p.stored, groups, p.owner))) == 5
    # A slot holding a value of another device.
    q, _, s = C.storage(p)
    with(s; kwargs...) = C.Storage((get(kwargs, f, getfield(s, f)) for f in fieldnames(C.Storage))...)
    slot = s.slots[c]
    @test slot > 0
    slot_devices = copy(s.devices)
    slot_devices[slot] = 3 - slot_devices[slot]
    @test invalid_by(() -> C.verify(q, with(s; devices = slot_devices))) == 5
    # A result that replaces an argument on another device.
    r = q.graph.results[1].value
    a = only(v for v in eachindex(q.graph.values) if q.graph.values[v].definition isa C.Input && q.graph.devices[v] != q.graph.devices[r])
    @test invalid_by(() -> C.verify(q, with(s; staged = [r => a]))) == 5
end
