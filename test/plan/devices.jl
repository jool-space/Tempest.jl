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

@testset "the placed model planned over two devices, $T" for T in (Float32, Float64)
    ws, x, t = placed_model(T)
    for (f, args, crossed) in ((placed_half, (ws, x), 2), (placed_step, (ws, x, t), 5))
        program = trace(f, args...)
        labels = C.argument_devices(program, labelled_model(ws, x, t)[1:length(args)])
        p = C.verify(placed_plan(f, args, labels))
        whole = C.plan(C.rewrite(C.lower(program)))
        @test C.execute(p, args...) == C.execute(whole, args...)
        q, _, s = C.storage(p)
        @test C.verify(q, s) === q
        # The rule copies the activation and its norm, which the next block
        # reads, to the second device, and in the step the norm's sums too
        # and two gradients back: a minimum cut copies the activation alone,
        # once each way.
        @test crossings(p) == crossed
        @test_broken crossings(p) == (f === placed_half ? 1 : 2)
        @test occursin("on device 2", sprint(C.explain, p))
        @test occursin(r"crossing over \(16, 32\) on device 2, reads %\d+\n  copies %\d+ \w+\(16, 32\) on device 1", sprint(C.explain, p))
    end
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
