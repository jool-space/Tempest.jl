# The order an execution's steps need between streams (`stream_schedule`):
# the waits, from what each step touches, and the runs a recording launches.
# Steps by hand first, each touching storage numbered by hand, then the
# steps of programs placed over devices by label on the CPU target. Each
# order is checked by a derivation of its own (fixtures/schedules.jl).
include("../fixtures/devices.jl")
include("../fixtures/schedules.jl")

# A kernel on device `d`'s own stream, and a crossing from `from` to `to`.
kernel(d) = C.Step(nothing, d)
across(from, to) = C.Step(nothing, to, from)

touching(reads, writes) = (reads = collect(Int, reads), writes = collect(Int, writes))

# The order of `steps` touching `touches`, checked: every two steps that
# touch one storage, one writing it, run in plan order however the streams
# run, and in orders the streams may run them in, drawn at random.
function checked(steps, touches)
    s = C.stream_schedule(steps, touches)
    @test unordered(steps, touches, s.waits, s.runs) == []
    rng = Xoshiro(1)
    @test all(_ -> in_plan_order(interleaving(rng, steps, s.waits, s.runs), touches), 1:20)
    s
end

@testset "a crossing waits only for what wrote its source, and only its first reader waits for it" begin
    steps = [kernel(0), kernel(0), across(0, 1), kernel(1), kernel(1), kernel(1)]
    touches = [touching([], [1]), touching([], [2]), touching([1], [3]), touching([4], [5]), touching([3], [6]), touching([3], [7])]
    s = checked(steps, touches)
    @test s.waits == [[], [], [1], [], [3], []]
    # The second device's step before the reader runs with it, in one run.
    @test s.runs == [[1, 2], [3], [4, 5, 6]]
end

@testset "a step that writes over a crossing's source waits for the crossing" begin
    # The source's slot taken again by a later value on its device.
    steps = [kernel(0), across(0, 1), kernel(0), kernel(1)]
    touches = [touching([], [1]), touching([1], [2]), touching([], [1]), touching([2], [])]
    s = checked(steps, touches)
    @test s.waits == [[], [1], [2], [2]]
    @test s.runs == [[1], [2], [3], [4]]
end

@testset "a crossing into a slot waits for the readers of what it held" begin
    steps = [kernel(1), kernel(1), kernel(0), across(0, 1), kernel(1)]
    touches = [touching([], [5]), touching([5], [6]), touching([], [1]), touching([1], [5]), touching([5], [])]
    s = checked(steps, touches)
    # The writer of the source, and the last reader of the slot's value
    # before, after which its writer need not be waited for.
    @test sort(s.waits[4]) == [2, 3]
    @test s.waits[5] == [4]
end

@testset "crossings in the two directions wait for nothing of each other" begin
    steps = [kernel(0), kernel(1), across(0, 1), across(1, 0), kernel(1), kernel(0)]
    touches = [touching([], [1]), touching([], [2]), touching([1], [3]), touching([2], [4]), touching([3], []), touching([4], [])]
    s = checked(steps, touches)
    @test s.waits == [[], [], [1], [2], [3], [4]]
end

@testset "a wait the stream is ordered after already is left out" begin
    # Of the same stream: the later of two writers on one device.
    steps = [kernel(0), kernel(0), across(0, 1)]
    s = checked(steps, [touching([], [1]), touching([], [2]), touching([1, 2], [3])])
    @test s.waits[3] == [2]
    # Across streams: the last step reads the copy that a crossing back
    # made after the first crossing's reader, which waited for that crossing,
    # and overwrites the first crossing's source.
    steps = [kernel(0), across(0, 1), kernel(1), across(1, 0), kernel(0)]
    touches = [touching([], [1]), touching([1], [2]), touching([2], [3]), touching([3], [4]), touching([4], [1])]
    s = checked(steps, touches)
    @test s.waits == [[], [1], [2], [3], [4]]
end

@testset "a device that waits for a crossing fed by itself runs twice" begin
    steps = [kernel(0), across(0, 1), kernel(1), across(1, 0), kernel(0)]
    touches = [touching([], [1]), touching([1], [2]), touching([2], [3]), touching([3], [4]), touching([4], [5])]
    s = checked(steps, touches)
    @test s.runs == [[1], [2], [3], [4], [5]]
    @test count(r -> C.stream_of(steps[first(r)]) == (0, 0), s.runs) == 2
end

@testset "a step that waits for nothing begun later joins its stream's run" begin
    steps = [kernel(1), kernel(0), across(0, 1), kernel(1), kernel(1)]
    touches = [touching([], [9]), touching([], [1]), touching([1], [2]), touching([9], [10]), touching([2], [])]
    s = checked(steps, touches)
    @test s.runs == [[1, 4], [2], [3], [5]]
end

@testset "the check finds a wait left out" begin
    steps = [kernel(0), across(0, 1), kernel(0)]
    touches = [touching([], [1]), touching([1], [2]), touching([], [1])]
    @test unordered(steps, touches, [[], [1], []], [[1, 3], [2]]) == [(2, 3)]
    @test unordered(steps, touches, [[], [], [2]], [[1], [2], [3]]) == [(1, 2)]
    @test_throws "waits for step 2, launched with or after it" unordered(steps, touches, [[], [1], [2]], [[1, 3], [2]])
end

@testset "one device: one run, no waits, and a recording of one segment" begin
    ws, x, t = placed_model(Float32)
    ex = prepare(trace(placed_step, ws, x, t), ws, x, t; target = T.CPUTarget())
    s = C.stream_schedule(ex.steps, ex.touches)
    @test all(isempty, s.waits)
    @test s.runs == [collect(eachindex(ex.steps))]
    @test ex.schedule.runs == s.runs
    @test isempty(ex.schedule.lanes)
    @test all(k -> isempty(ex.schedule.steps[k].waits) && ex.schedule.steps[k].signal === nothing, eachindex(ex.steps))
    capture!(ex)
    @test length(ex.captured.segments) == 1
    @test only(ex.captured.segments).steps == collect(eachindex(ex.steps))
end

# A program that crosses a value to the second device and back: the copy
# back takes the slot of the value first crossed, once that crossing has
# read it.
there_and_back(w0, x, w1, w2) = (h = tanh.(w0 * x); y = tanh.(w1 * h); tanh.(w2 * y) .+ x)

# The leaves of the outputs of `ex` after its steps ran in `order`, copied.
function ran_in(ex, order)
    foreach(k -> ex.steps[k].run(), order)
    map(copy, C.leaves(outputs(ex)))
end

# The arrays of a call of `f`, and the same labelled with their devices.
function placed_call(f)
    if f === there_and_back
        xs = (randn(Xoshiro(2), 16, 16), randn(Xoshiro(3), 16, 8), randn(Xoshiro(4), 64, 16), randn(Xoshiro(5), 16, 64))
        return xs, map(Labelled, xs, (0, 0, 1, 0))
    end
    ws, x, t = placed_model(Float64)
    (ws, x, t), labelled_model(ws, x, t)
end

@testset "steps run in any order the schedule allows compute what plan order computes: $(nameof(f))" for f in (there_and_back, placed_step)
    xs, labelled = placed_call(f)
    ex = prepare(trace(f, xs...), labelled...; target = T.CPUTarget(), verify = true)
    s = C.stream_schedule(ex.steps, ex.touches)
    if f === there_and_back
        # The copy back is written over the storage the first crossing read.
        crossing = findall(C.crosses, ex.steps)
        @test length(crossing) == 2
        @test !isempty(intersect(ex.touches[crossing[1]].reads, ex.touches[crossing[2]].writes))
    end
    expected = map(copy, C.leaves(outputs(execute!(ex))))
    rng = Xoshiro(7)
    @test all(_ -> ran_in(ex, interleaving(rng, ex.steps, s.waits, s.runs)) == expected, 1:30)
    # Without the waits, an order the streams alone allow computes otherwise.
    none = [Int[] for _ in ex.steps]
    @test any(_ -> ran_in(ex, interleaving(rng, ex.steps, none, s.runs)) != expected, 1:30)
end

@testset "the placed model over $n devices" for n in (2, 3)
    ws, x, t = placed_model(Float64; blocks = 6)
    for (f, args) in ((placed_half, (ws, x)), (placed_step, (ws, x, t)))
        labelled = labelled_model(ws, x, t; devices = n)[1:length(args)]
        ex = prepare(trace(f, args...), labelled...; target = T.CPUTarget(), verify = true)
        s = checked(ex.steps, ex.touches)
        @test any(C.crosses, ex.steps)
        # A reader of each copy the program reads waits for its crossing.
        @test any(!isempty, s.waits)
        # Executions one after another: the next one's steps wait for this one's crossings.
        twice = checked(repeat(ex.steps, 2), repeat(ex.touches, 2))
        @test length(twice.runs) <= 2length(s.runs)
    end
end

@testset "the gradients of two towers on two devices, joined by their losses" begin
    ws, x, t = placed_model(Float32)
    vs, y, u = again(ws, x, t)
    program = trace(towers_step, ws, x, t, vs, y, u)
    ex = prepare(program, on_label(0, (ws, x, t))..., on_label(1, (vs, y, u))...; target = T.CPUTarget(), verify = true)
    checked(ex.steps, ex.touches)
end

# Arguments that share memory, one of them written, are two storages to the
# steps: a crossing reading the one and a kernel writing the other would not
# be ordered. Every binding of a call's arrays refuses them where it locates
# them (see `located!`): `compile`'s call, `point!` and so `prepare` with
# `point`. Arrays bound by reference become the storage as `prepare` plans,
# before any binding of a call sees them; refusing them needs that binding
# to locate them as the others do.
@testset "arguments sharing memory, one written, are refused by every binding" begin
    f!(x, y) = (x .= 2 .* y; sum(x))
    a = rand(Float32, 8)
    program = trace(f!, copy(a), copy(a))
    c = T.compile(f!, copy(a), copy(a))
    @test_throws "argument 2 shares memory with argument 1, which f! writes" c(a, a)
    # Compiled on them, as its preparation binds them.
    @test_throws "shares memory with" T.compile(f!, a, a)
    ex = prepare(program, copy(a), copy(a); target = T.CPUTarget(), point = true)
    @test_throws "input 2 shares memory with input 1, which the program writes" C.point!(ex, a, a)
    @test_throws "input 2 shares memory with input 1" prepare(program, a, a; target = T.CPUTarget(), point = true)
    refused(f) = try
        f()
        false
    catch err
        err isa ArgumentError
    end
    @test_broken refused(() -> prepare(program, a, a; target = T.CPUTarget(), by_reference = (1, 2)))
end

# Data parallelism, the trial's SGD step over two replicas: each crossing
# starts once what it copies is written, runs beside both devices' kernels,
# and holds back only what needs it. Storage shares a slot only where the
# program orders its users, so no crossing waits for a kernel over what its
# copy overwrites, nor a kernel for a crossing over what it reads.
@testset "data parallelism: a crossing waits only for what it copies, and only what needs it waits for it" begin
    ws, x, t = placed_model(Float32)
    args = shards_of(ws, x, t)
    ex = prepare(trace(shards_sgd!, args...), labelled_shards(args...)...; target = T.CPUTarget(), verify = true)
    steps, touches = ex.steps, ex.touches
    (; waits) = C.stream_schedule(steps, touches)
    crossing = findall(C.crosses, steps)
    lanes = unique(map(k -> C.stream_of(steps[k]), crossing))
    @test Set(lanes) == Set([(0, 1), (1, 0)])
    # The step before `k` that last wrote `key`.
    writer(k, key) = findlast(j -> key in touches[j].writes, 1:(k - 1))
    # A crossing waits only for what wrote its source.
    for k in crossing
        @test waits[k] ⊆ [writer(k, key) for key in touches[k].reads]
    end
    # What waits for a crossing reads its copy, or writes over its source.
    for j in eachindex(steps), k in waits[j]
        C.crosses(steps[k]) || continue
        reads_copy = !isempty(intersect(touches[j].reads, touches[k].writes))
        overwrites = !isempty(intersect(touches[j].writes, touches[k].reads))
        @test reads_copy || overwrites
    end
    # And every two steps that touch one storage, one writing it, are ordered.
    @test unordered(ex) == []
end

# Storage shares a slot only where the program orders its users: a crossing
# of the data-parallel step, which the program orders after the kernels that
# wrote what it copies alone, never takes a slot a kernel of its reader's
# device used before it, which it would wait for; nor a kernel one that a
# crossing still copies from.
@testset "a slot is shared only where the program orders its users" begin
    ws, x, t = placed_model(Float32)
    args = shards_of(ws, x, t)
    ex = prepare(trace(shards_sgd!, args...), labelled_shards(args...)...; target = T.CPUTarget(), build = false)
    q, s = ex.plan, ex.storage
    @test C.verify(q, s) === q
    order = C.plan_order(q)
    reads = [C.group_reads(q.graph, g) for g in q.groups]
    users(u) = [q.owner[u]; [j for j in eachindex(reads) if u in reads[j]]]
    # Each slot's values in the order written, each one's users before the next's kernel.
    for slot in eachindex(s.sizes)
        held = sort!([v for v in eachindex(s.slots) if s.slots[v] == slot]; by = v -> q.owner[v])
        for (u, v) in zip(held, held[2:end])
            @test all(j -> j == q.owner[v] || C.ordered(order, j, q.owner[v]), users(u))
        end
    end
    # A crossing's copy put by hand in a slot whose value a kernel used
    # earlier in plan order, but which the crossing need not run after.
    across = [k for (k, g) in enumerate(q.groups) if g.kind isa C.CrossingKind]
    # What slot `slot` holds as kernel `k` writes: the value written into it last before.
    holding(slot, k) = argmax(v -> s.slots[v] == slot && q.owner[v] < k ? q.owner[v] : 0, eachindex(s.slots))
    pairs = [(k, u) for k in across for u in (holding(slot, k) for slot in eachindex(s.sizes))
        if s.slots[u] > 0 && q.owner[u] < k && s.devices[s.slots[u]] == q.graph.devices[only(q.groups[k].outputs)] &&
            maximum(users(u)) < k && !all(j -> C.ordered(order, j, k), users(u))]
    @test !isempty(pairs)
    k, u = first(pairs)
    c = only(q.groups[k].outputs)
    slots = copy(s.slots)
    slots[c] = s.slots[u]
    sizes = copy(s.sizes)
    sizes[s.slots[u]] = max(sizes[s.slots[u]], s.sizes[s.slots[c]])
    forged = C.Storage(s.lifetimes, slots, sizes, s.devices, s.donated, s.staged, s.overwrites, s.reused)
    err = try
        C.verify(q, forged)
        nothing
    catch err
        err
    end
    @test err isa C.InvalidPlan && err.invariant == 2 && occursin("does not run before it", err.message)
end
