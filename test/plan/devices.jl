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
        labels = C.argument_devices(C.CPUTarget(), program, labelled_model(ws, x, t; devices = n)[1:length(args)]).labels
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

# A write into a middle range of columns is selected by a mask that is the
# conjunction of two comparisons of positions, made from values made
# anywhere: each device that reads it makes it, and only the addends cross.
@testset "a value made from values made anywhere is made on each device that reads it" begin
    f(a, b, c, d) = (s = c[:, 2:3] .+ d[:, 2:3]; x = copy(a); x[:, 2:3] .-= s; y = copy(b); y[:, 2:3] .-= s; (x, y))
    args = Tuple(rand(Float32, 4, 5) for _ in 1:4)
    p = C.verify(placed_plan(f, args, [1, 2, 1, 2]))
    g = p.graph
    copies = [c for c in eachindex(g.values) if C.crossing(g, c)]
    @test all(c -> g.values[g.values[c].definition.operand.value].definition isa C.Input, copies)
    masks = [v for v in eachindex(g.values) if g.values[v].definition isa C.Pointwise && g.values[v].definition.f === (&)]
    @test sort(g.devices[masks]) == [1, 2]
    @test C.execute(p, args...) == C.execute(C.plan(C.rewrite(C.lower(trace(f, args...)))), args...)
end

# A sum of what several devices computed, which each of them reads, is
# computed on each from the others' terms by the same arithmetic in the same
# order; nothing crosses but the terms. The goal's spelling, over two and
# three replicas.
@testset "a reduction every device reads is computed on each of them, $n devices" for n in (2, 3)
    ws, x, t = placed_model(Float32; tokens = 96)
    args = replicas_of(ws, x, t, n)
    ex = prepare(trace(replicated_sgd!, args...), labelled_replicas(args...)...; target = T.CPUTarget(), build = false, verify = true)
    g = ex.plan.graph
    copies = [v for v in eachindex(g.values) if C.crossing(g, v)]
    # No crossing copies a value whose terms lie on several devices.
    @test all(c -> length(term_devices(g, g.values[c].definition.operand.value)) == 1, copies)
    # Each device receives each other's gradient of each weight once, and the loss's terms.
    weight = C.value_bytes(g.values[only(v for v in eachindex(g.values) if g.values[v].definition isa C.Input && g.values[v].definition.argument == 1)])
    lanes = [(g.devices[g.values[c].definition.operand.value], g.devices[c]) for c in copies if C.value_bytes(g.values[c]) == weight]
    @test length(lanes) == n * (n - 1) * 2length(ws) && all(l -> count(==(l), lanes) == 2length(ws), unique(lanes))
    # Each replica's update reads its gradient reduced on its own device, by the same arithmetic as every other's.
    results = [o.value for o in g.results if o isa C.Read]
    updates = [r for r in results if C.value_bytes(g.values[r]) == weight && length(term_devices(g, r)) == n]
    @test length(updates) == n * 2length(ws)
    for r in updates
        @test all(v -> g.devices[v] == g.devices[r] && !C.crossing(g, v), [u for o in C.operands(g.values[r].definition) for u in C.read_values(o)])
    end
    # The replicas' updates of one weight, the replica's own weight left out, alike.
    written = [computed(g, r, k -> :own) for r in updates]
    @test length(unique(written)) == 2length(ws)
    @test all(w -> count(==(w), written) == n, unique(written))
end

# Terms all on one device are joined there once, and the result copied.
@testset "a sum of one device's terms is computed once and copied" begin
    f(a, b, c) = (s = a .+ b; (sum(s), sum(s .* c; dims = 3)))
    args = (rand(Float32, 4, 3), rand(Float32, 4, 3), rand(Float32, 4, 3, 5))
    p = C.verify(placed_plan(f, args, [1, 1, 2]))
    g = p.graph
    sums = [v for v in eachindex(g.values) if g.values[v].definition isa C.Pointwise && g.values[v].definition.f === (+)]
    @test length(sums) == 1
    @test crossed_bytes(p) == [sizeof(args[1])]
    @test C.execute(p, args...) == C.execute(C.plan(C.rewrite(C.lower(trace(f, args...)))), args...)
    # With its terms on two devices and read on both, it is computed on both, and only the terms cross.
    h(a, b, c, d) = (s = a .+ b; (sum(s .* c; dims = 3), sum(s .* d; dims = 3)))
    args = (rand(Float32, 4, 3), rand(Float32, 4, 3), rand(Float32, 4, 3, 5), rand(Float32, 4, 3, 5))
    p = C.verify(placed_plan(h, args, [1, 2, 1, 2]))
    g = p.graph
    sums = [v for v in eachindex(g.values) if g.values[v].definition isa C.Pointwise && g.values[v].definition.f === (+)]
    @test sort(g.devices[sums]) == [1, 2]
    @test sort([g.values[g.values[c].definition.operand.value].definition isa C.Input for c in eachindex(g.values) if C.crossing(g, c)]) == [true, true]
    @test C.execute(p, args...) == C.execute(C.plan(C.rewrite(C.lower(trace(h, args...)))), args...)
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

# The plan of program `program`, prepared from its arguments on the devices
# `labels` gives, its results pinned as a preparation pins them.
function pinned_plan(program, labels)
    graph = C.rewrite(C.lower(program))
    pins = [k => labels[a] for (k, a) in [program.writes; program.gradients]]
    C.verify(C.plan(C.with_devices(graph, labels; pins)))
end

@testset "a gradient is returned on its argument's device" begin
    # The small weight is read where the large input is, copied there; its
    # gradient is summed there and its 3 numbers copied back, the input's
    # gradient computed where the input is.
    f(w, x) = sum(abs2, w .* x)
    args = (rand(Float32, 1, 3), rand(Float32, 4, 3))
    program = trace((xs...) -> T.withgradient(f, xs...), args...)
    @test program.gradients == gradient_pairs(program) == [2 => 1, 3 => 2]
    for labels in ([2, 1], [1, 2])
        p = pinned_plan(program, labels)
        @test gradients_off(p, program) == 0
        @test crossed_bytes(p) == [12, 12]
        @test C.execute(p, args...) == C.execute(C.plan(C.rewrite(C.lower(program))), args...)
    end
    # Prepared, the gradients are pinned so.
    ex = prepare(program, Labelled(args[1], 1), Labelled(args[2], 0); target = T.CPUTarget(), verify = true)
    @test gradients_off(ex.plan, program) == 0
    # A gradient that reads no value, the input's of a sum, is made where
    # the input is.
    g(w, x) = sum(abs2, w) + sum(x)
    program = trace((xs...) -> T.withgradient(g, xs...), args...)
    p = pinned_plan(program, [1, 2])
    @test gradients_off(p, program) == 0
    # Only the sum of the weight's squares crosses, to be added to the input's.
    @test crossed_bytes(p) == [4]
    @test C.execute(p, args...) == C.execute(C.plan(C.rewrite(C.lower(program))), args...)
end

@testset "a result that replaces an argument is written on its device" begin
    # The result reads only the second argument, and is copied to the
    # first's device, whose array it replaces.
    f!(a, b) = (a .= 2 .* b; nothing)
    args = (rand(Float32, 4), rand(Float32, 4))
    program = trace(f!, args...)
    p = pinned_plan(program, [1, 2])
    @test crossed_bytes(p) == [16]
    @test C.execute(p, args...) == C.execute(C.plan(C.rewrite(C.lower(program))), args...)
end

@testset "a result pinned to two devices is refused" begin
    graph = C.rewrite(C.lower(trace((a, b) -> a .+ b, rand(Float32, 3), rand(Float32, 3))))
    @test_throws "result 1 must be on device 2 and on device 1" C.with_devices(graph, [1, 2]; pins = [1 => 1, 1 => 2])
end

@testset "what two devices compute apart is a run of kernels on each, $(nameof(f))" for f in (towers_loss, towers_step)
    a, b = placed_model(Float32; blocks = 3), again(placed_model(Float32; blocks = 3)...)
    args = (a..., b...)
    program = trace(f, args...)
    labels = C.argument_devices(C.CPUTarget(), program, (on_label(0, a)..., on_label(1, b)...)).labels
    p = C.verify(placed_plan(f, args, labels))
    @test C.execute(p, args...) == C.execute(C.plan(C.rewrite(C.lower(program))), args...)
    # The towers meet where their losses are added, a number crossing: the
    # first's kernels, the second's, and the sum where the first is.
    @test crossed_bytes(p) == [sizeof(Float32)]
    @test device_runs(p) == [1, 2, 1]
    f === towers_step && @test gradients_off(p, program) == 0
    q, _, s = C.storage(p)
    @test C.verify(q, s) === q
end

@testset "kernels two devices interleave are ordered in runs" begin
    # Two chains of two kernels, one on each device, and a fifth reading both.
    needs = [Set{Int}(), Set{Int}(), Set([1]), Set([2]), Set([3, 4])]
    @test C.in_runs([1, 2, 3, 4, 5], needs, [1, 2, 1, 2, 1]) == [1, 3, 2, 4, 5]
    # A device's kernel that reads another's waits for it, and no longer.
    @test C.in_runs([1, 2, 3, 4, 5], [Set{Int}(), Set{Int}(), Set([2]), Set([2]), Set([3, 4])], [1, 2, 1, 2, 1]) == [1, 2, 4, 3, 5]
    # One that waits holds back none of its device's that does not.
    @test C.in_runs([1, 2, 3, 4], [Set{Int}(), Set{Int}(), Set([2]), Set([1])], [1, 2, 1, 1]) == [1, 4, 2, 3]
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
    # A kernel computing a member on another device than its own: of the
    # placed step's, as the sum here and its product with what crosses are
    # two kernels, the product waiting for the crossing and the sum not.
    @test all(g -> length(C.members(g)) == 1, p.groups)
    ws, x, t = placed_model(Float32)
    step = placed_plan(placed_step, (ws, x, t), C.argument_devices(C.CPUTarget(), trace(placed_step, ws, x, t), labelled_model(ws, x, t)).labels)
    g = first(filter(g -> length(C.members(g)) > 1, step.groups))
    devices = copy(step.graph.devices)
    devices[last(C.members(g))] = 3 - devices[first(C.members(g))]
    moved = C.IndexGraph(step.graph.values, step.graph.arguments, step.graph.results, step.graph.result_dims,
        step.graph.result_scalar, step.graph.result_eltype, step.graph.rewrites, step.graph.backward, devices)
    @test invalid_by(() -> C.verify(C.Plan(moved, step.stored, step.groups, step.owner))) == 5
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
