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

# A crossing copies only the range of its source's last axis that its readers
# read, one span of the source's storage: the remade sum of a middle range of
# columns crosses those columns of each addend, and a product reading one
# column of a matrix on another device crosses the column.
@testset "a crossing copies the span its readers read" begin
    f(a, b, c, d) = (s = c[:, 2:3] .+ d[:, 2:3]; x = copy(a); x[:, 2:3] .-= s; y = copy(b); y[:, 2:3] .-= s; (x, y))
    args = Tuple(rand(Float32, 4, 5) for _ in 1:4)
    p = C.verify(placed_plan(f, args, [1, 2, 1, 2]))
    @test crossed_bytes(p) == [32, 32]
    @test occursin("from element 4 to %", sprint(C.explain, p))
    @test C.execute(p, args...) == C.execute(C.plan(C.rewrite(C.lower(trace(f, args...)))), args...)
    g(a, b) = a[:, 3] .* b
    args = (rand(Float32, 4, 5), rand(Float32, 4, 100))
    p = C.verify(placed_plan(g, args, [1, 2]))
    @test crossed_bytes(p) == [16]
    @test [p.graph.values[v].dims for v in eachindex(p.graph.values) if C.crossing(p.graph, v)] == [(4,)]
    @test C.execute(p, args...) == C.execute(C.plan(C.rewrite(C.lower(trace(g, args...)))), args...)
    # A crossing of every other column is no one span, and is refused.
    graph = C.with_devices(C.rewrite(C.lower(trace(f, (rand(Float32, 4, 5) for _ in 1:4)...))), [1, 2, 1, 2])
    c = findfirst(v -> C.crossing(graph, v), eachindex(graph.values))
    copied = graph.values[c]
    strided = C.Read(copied.definition.operand.value, (C.IR.var(1), C.IR.scale(C.IR.var(2), 2)))
    graph.values[c] = C.Value(C.Crossed(strided), copied.dims, copied.eltype, copied.scalar)
    @test invalid_by(() -> C.verify(C.plan(graph))) == 5
end

# A sum of what two devices computed, which each of them reads, is computed
# on each from the other's terms by the same arithmetic in the same order;
# nothing crosses but the terms. The goal's spelling over two replicas.
@testset "a reduction every device reads is computed on each of them, $n devices" for n in (2,)
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

# Over three devices or more, a sum of what each computed, which each reads,
# is made in pieces: a range of its last axis on each device, from that
# range of each device's term, each piece copied to the others, which put
# the pieces together. Each element is summed on one device, so that every
# replica's update reads the same values, and each device's link carries in
# each direction its share of the others' terms and its piece to each other,
# 2(n - 1)/n of a gradient, where made again on each it carried n - 1. The
# goal's spelling, over three and four replicas, of weights whose last axes
# share out evenly.
@testset "a reduction every device reads is made in pieces, $n devices" for n in (3, 4)
    ws, x, t = placed_model(Float32; width = 12, tokens = 96)
    args = replicas_of(ws, x, t, n)
    ex = prepare(trace(replicated_sgd!, args...), labelled_replicas(args...)...; target = T.CPUTarget(), build = false, verify = true)
    g = ex.plan.graph
    copies = planned_crossings(ex.plan)
    source(c) = g.values[c].definition.operand.value
    # A number is never made in pieces: a partial sum of the losses, read on
    # one device that holds none of its terms, is copied there once.
    pieces = unique!([source(c) for c in copies if !isempty(g.values[source(c)].dims) && length(term_devices(g, source(c))) > 1])
    weights = 2length(ws)
    # Each weight's reduced gradient is in a piece on each device, no piece
    # computed twice, nor any range of one.
    @test length(pieces) == n * weights
    @test all(d -> count(v -> g.devices[v] == d, pieces) == weights, 1:n)
    @test allunique([computed(g, v, identity) for v in pieces])
    # A piece is read by its crossings and the selects that put the pieces
    # together, and a crossing of one only by those: none is made again, made
    # in pieces again, or added to.
    users = C.readers(g)
    selects(u) = g.values[u].definition isa C.Pointwise && g.values[u].definition.f === ifelse
    @test all(v -> all(u -> C.crossing(g, u) || selects(u), users[v]), pieces)
    @test all(c -> all(selects, users[c]), [c for c in copies if source(c) in pieces])
    # Each device's link in each direction carries its share of the terms and
    # of the pieces, and the losses' numbers.
    gradient = sum(w -> sum(sizeof, w), ws)
    @test maximum(values(link_bytes(ex.plan))) <= 2(n - 1) * gradient ÷ n + 16n
    # The replicas' updates of one weight, the replica's own weight left out,
    # read the same pieces put together alike.
    written = [computed(g, g.results[k].value, a -> :own) for (k, a) in ex.program.writes]
    @test length(unique(written)) == weights
    @test all(w -> count(==(w), written) == n, unique(written))
end

# The rule's small cases: a sum of three devices' terms read on two, and of
# two devices' terms read on three, are made in pieces over the three, each
# device's link carrying its share, two thirds of what made again or copied
# would carry in the heaviest direction.
@testset "a sum whose terms or readers are on three devices is made in pieces" begin
    f(a, b, c, d, e) = (s = a .+ b .+ c; (sum(s .* d; dims = 3), sum(s .* e; dims = 3)))
    g(a, b, c, d, e) = (s = a .+ b; (sum(s .* c; dims = 3), sum(s .* d; dims = 3), sum(s .* e; dims = 3)))
    matrices(k) = Tuple(rand(Float32, 4, 3) for _ in 1:k)
    cubes(k) = Tuple(rand(Float32, 4, 3, 5) for _ in 1:k)
    for (h, args, labels) in ((f, (matrices(3)..., cubes(2)...), [1, 2, 3, 1, 2]), (g, (matrices(2)..., cubes(3)...), [1, 2, 1, 2, 3]))
        p = C.verify(placed_plan(h, args, labels))
        @test maximum(values(link_bytes(p))) == 64
        @test C.execute(p, args...) == C.execute(C.plan(C.rewrite(C.lower(trace(h, args...)))), args...)
    end
end

# A sum of three devices' terms read on the three whose last axis is shorter
# than the three is made in as many pieces as the axis is long, on the first
# devices, the third sending its terms and receiving the pieces, 48 bytes in
# the heaviest direction where made again it would be 64; of numbers it is
# made again on each, each device sending its number to the two others.
@testset "a sum of a last axis shorter than its devices, or of numbers, by the same rule" begin
    f(a, b, c, d, e, h) = (s = a .+ b .+ c; (sum(s .* d; dims = 3), sum(s .* e; dims = 3), sum(s .* h; dims = 3)))
    n(a, b, c, d, e, h) = (s = sum(a) + sum(b) + sum(c); (s .* d, s .* e, s .* h))
    short = (Tuple(rand(Float32, 4, 2) for _ in 1:3)..., Tuple(rand(Float32, 4, 2, 5) for _ in 1:3)...)
    numbers = Tuple(rand(Float32, 4, 3) for _ in 1:6)
    for (h, args, heaviest, shapes) in ((f, short, 48, [(4,)]), (n, numbers, 8, [()]))
        p = C.verify(placed_plan(h, args, [1, 2, 3, 1, 2, 3]))
        g = p.graph
        @test maximum(values(link_bytes(p))) == heaviest
        @test unique([g.values[c].dims for c in planned_crossings(p)]) == shapes
        @test C.execute(p, args...) == C.execute(C.plan(C.rewrite(C.lower(trace(h, args...)))), args...)
    end
end

# A chain is realized as a whole. A sum of two devices' terms, added to a
# third's, the whole read on the third device and a fourth: copied, each sum
# computed where the cut placed it and copied where it is read, each device
# and direction carries one matrix, 64 bytes, where made again or in pieces
# the third would receive both of the first two's terms. Read on a fourth and
# a fifth, it is made in pieces over the five.
@testset "a chain is copied as the cut placed its values, or made in pieces" begin
    f(a, b, c, d, e) = (s = (a .+ b) .+ c; (sum(s .* d; dims = 3), sum(s .* e; dims = 3)))
    args = (rand(Float32, 4, 4), rand(Float32, 4, 4), rand(Float32, 4, 4), rand(Float32, 4, 4, 5), rand(Float32, 4, 4, 5))
    for (labels, heaviest, crossings) in (([1, 2, 3, 3, 4], 64, 3), ([1, 2, 3, 4, 5], 96, 16))
        p = C.verify(placed_plan(f, args, labels))
        @test maximum(values(link_bytes(p))) == heaviest
        @test length(planned_crossings(p)) == crossings
        @test C.execute(p, args...) == C.execute(C.plan(C.rewrite(C.lower(trace(f, args...)))), args...)
    end
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

# A sum of two devices' terms, read on its own device and on a third that
# holds neither, is copied to the third once: made again there, both terms
# would cross to it, and the second device's link would carry its term to
# the first and the third, 96 bytes in the heaviest direction rather than 48.
@testset "a sum read on a device that holds none of its terms is copied there" begin
    f(a, b, c, d) = (s = a .+ b; (sum(s .* c; dims = 3), sum(s .* d; dims = 3)))
    args = (rand(Float32, 4, 3), rand(Float32, 4, 3), rand(Float32, 4, 3, 5), rand(Float32, 4, 3, 5))
    p = C.verify(placed_plan(f, args, [1, 2, 1, 3]))
    @test crossed_bytes(p) == [48, 48]
    @test maximum(values(link_bytes(p))) == 48
    @test C.execute(p, args...) == C.execute(C.plan(C.rewrite(C.lower(trace(f, args...)))), args...)
end

# A kernel's values come after the same crossings, however the kernel is
# formed: a set fixed by the cover, as a group grown from a seed, holds no
# value that waits for another device beside one that does not, which would
# hold the second back behind the crossing.
@testset "a fixed set mixing values that wait for another device and values that do not is no group" begin
    f(a, b, c) = (sum(a .* c; dims = 3), a .+ b)
    args = (rand(Float32, 4, 3), rand(Float32, 4, 3), rand(Float32, 4, 3, 5))
    graph = C.rewrite(C.lower(trace(f, args...)))
    # The fold of the first device's arrays, and the sum that reads the second's through a crossing.
    pair(g) = (only(v for v in eachindex(g.values) if g.values[v].definition isa C.Reduction),
        only(v for v in eachindex(g.values) if g.values[v].definition isa C.Pointwise && g.values[v].definition.f === (+)))
    placed = C.with_devices(graph, [1, 2, 1])
    fold, sum_ = pair(placed)
    after_crossing(v) = any(u -> C.crossing(placed, u), findall(C.ancestors(placed)[v]))
    @test placed.devices[fold] == placed.devices[sum_] && !after_crossing(fold) && after_crossing(sum_)
    @test_throws C.NotAGroup C.plan(placed; fixed = [[fold, sum_]])
    # The same set on one device, where nothing crosses, is one kernel.
    one = C.with_devices(graph, [1, 1, 1])
    @test Set(pair(one)) in [Set(C.members(g)) for g in C.plan(one; fixed = [collect(pair(one))]).groups]
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

# Devices run at once: each runs a kernel that waits for a crossing once the
# crossing is estimated there, and what it can before it, so that none holds
# its own work behind a crossing it could have run beside. What waits and
# no other device waits for, each replica's update, runs after what waits
# for nothing; what another device waits for runs as soon as it can, before
# or after. Two replicas of data parallelism run their kernels in the same
# order. Three need not: each sends its piece of a sum to the other two at
# once, one at a time over its link out, and which it sends to first is a
# tie-break; unless each sends to the same neighbour first, some device's
# link in takes two at once, so that one replica's pieces arrive later than
# the others' and it runs other kernels first.
@testset "each device runs what waits for no crossing first, $n devices" for n in (2, 3)
    # Weights whose last axes share out evenly, so that pieces are alike.
    ws, x, t = placed_model(Float32; width = 12, tokens = 96)
    args = replicas_of(ws, x, t, n)
    p = prepare(trace(replicated_sgd!, args...), labelled_replicas(args...)...; target = T.CPUTarget(), build = false, verify = true).plan
    g = p.graph
    up = C.ancestors(g)
    waits(k) = any(m -> any(u -> C.crossing(g, u), findall(up[m])), C.members(p.groups[k]))
    kernels(d) = [k for (k, group) in enumerate(p.groups) if C.group_device(p, group) == d && !(group.kind isa C.CrossingKind)]
    crossed = Set(C.crossed_from(g, only(group.outputs)) for group in p.groups if group.kind isa C.CrossingKind)
    awaited(k) = any(in(crossed), C.members(p.groups[k]))
    for d in 1:n
        w = [waits(k) for k in kernels(d) if !(waits(k) && awaited(k))]
        @test w == sort(w)
    end
    # Each kernel as any device runs it: its kind, whether it waits, and its members' definitions.
    summary(v) = (d = g.values[v].definition; (typeof(d), d isa C.Pointwise ? d.f : d isa C.Reduction ? d.fold : nothing, g.values[v].dims))
    alike(k) = (C.kind_name(p.groups[k].kind), waits(k), sort!(map(summary, C.members(p.groups[k])); by = repr))
    replica(d) = [alike(k) for k in kernels(d) if !(any(m -> g.values[m].scalar, C.members(p.groups[k])))]
    n == 2 && @test replica(2) == replica(1)
end

# Kernels by stream and estimated seconds: each device's own stream `(d, d)`,
# a crossing's the lane `(a, b)` from its source's device to its reader's,
# its seconds those of each of its two hops, out of `a` and into `b`.
@testset "kernels run as they are estimated to start" begin
    run(needs, streams, seconds) = C.run_order(map(Set{Int}, needs), streams, seconds)
    # A chain on each device, the second's crossing to the first, read there
    # last: the second's first kernel starts first, as the first device
    # begins on what follows it sooner, then each device's kernel after.
    @test run([[], [], [1], [2], [4], [3, 5]], [(1, 1), (2, 2), (1, 1), (2, 2), (2, 1), (1, 1)], ones(6)) == [2, 1, 4, 3, 5, 6]
    # A device's sum of its own term (1) and one another device crosses to it
    # (6 on lane (2, 1), estimated there at 2.5), its sum (5) crossing back
    # (8) to be read there (9): the device runs what nothing waits for (2, 3)
    # while the crossing is under way, the sum once it is there and the device
    # free, before the work left (4), and so does the other device.
    needs = [[], [], [], [], [1, 6], [7], [], [5], [8]]
    streams = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1), (2, 1), (2, 2), (1, 2), (2, 2)]
    order = run(needs, streams, [1, 1, 1, 1, 1, 0.75, 1, 1, 1])
    @test filter(k -> streams[k] == (1, 1), order) == [1, 2, 3, 5, 4]
    # Not before it is there: estimated to arrive later, the sum runs last.
    order = run(needs, streams, [1, 1, 1, 1, 1, 1.75, 1, 1, 1])
    @test filter(k -> streams[k] == (1, 1), order) == [1, 2, 3, 4, 5]
    # Every estimate twice as long, crossings with them: the same order.
    @test run(needs, streams, 2 .* [1, 1, 1, 1, 1, 0.75, 1, 1, 1]) == run(needs, streams, [1, 1, 1, 1, 1, 0.75, 1, 1, 1])
    # Two lanes into one device share its link in: the crossing whose source
    # is written first goes first, though later in the plan's order, and the
    # other after it.
    @test run([[], [1], [], [3], [2, 4]], [(1, 1), (1, 3), (2, 2), (2, 3), (3, 3)], [2, 1, 1, 1, 1]) == [1, 3, 4, 2, 5]
    # Two lanes out of one device share its link out, which takes them in
    # the order their sources are written: the crossing to the third device
    # (6), whose source is written first, goes out first though the third's
    # link in is held by the second's crossing (4) and the other lane's (5)
    # is free to land, its upload waiting until that link is free.
    needs = [[], [1], [], [3], [2], [1], [4, 6], [5]]
    streams = [(1, 1), (1, 1), (2, 2), (2, 3), (1, 2), (1, 3), (3, 3), (2, 2)]
    order = run(needs, streams, [1, 1, 0.5, 3, 1, 1, 1, 1])
    @test filter(k -> first(streams[k]) == 1 && last(streams[k]) != 1, order) == [6, 5]
    @test last(order) == 7
    # On one stream, the plan's order wherever it lets them, whatever the estimates.
    first_ready(needs) = (order = Int[]; while length(order) < length(needs)
        push!(order, findfirst(k -> !(k in order) && all(in(order), needs[k]), eachindex(needs)))
    end; order)
    rng = Random.Xoshiro(3)
    for _ in 1:50
        n = rand(rng, 2:30)
        needs = [Set(filter(_ -> rand(rng) < 0.2, 1:(k - 1))) for k in 1:n]
        shuffled = randperm(rng, n)
        permuted = [Set(shuffled[j] for j in needs[k]) for k in invperm(shuffled)]
        @test C.run_order(permuted, fill((1, 1), n), rand(rng, n)) == first_ready(permuted)
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
