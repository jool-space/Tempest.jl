# The order an execution's steps need between streams. A device runs its own
# steps on one stream, in order; a crossing runs on a lane of its own, the
# stream of its ordered pair of devices, so that it copies beside the kernels
# of both. Whatever two streams touch of one storage is ordered between them
# by a wait (see plan/order.jl). What a step touches is known where the steps
# are made (see `step_touches`); the waits and the runs a recording launches
# are derived from it alone (see `stream_schedule`), and an execution runs by
# them (see `Schedule` and `walk`).

"""
    stream_of(step) -> (from, device)

The stream `step` runs on: a device's own, `(d, d)`, or the lane of the
ordered pair of devices a crossing copies between, `(from, to)`.
"""
stream_of(s::Step) = (s.from, s.device)

"""
    step_touches(p, saves, copies, storage_of) -> Vector{Touches}

What each step of an execution of plan `p` reads and writes, in the order
the steps are made (see `planned`): each input saved apart, reading the
input and writing the buffer it is saved in; each kernel, reading what its
group reads and writing what it stores; and each staged result, reading
the result and writing the argument it replaces. `storage_of(v)` names the
storage value `v` lives in, so that values sharing storage, in a slot or
over an argument's, are one key.
"""
function step_touches(p::Plan, saves, copies, storage_of)
    reads = Vector{Any}[]
    writes = Vector{Any}[]
    for save in saves
        input = last(save.copy.values)
        push!(reads, [storage_of(input)])
        # The buffer an input is saved in is no value's: it is named by the input.
        push!(writes, [(:saved, input)])
    end
    for g in p.groups
        push!(reads, map(storage_of, sort!(collect(group_reads(p.graph, g)))))
        push!(writes, [storage_of(v) for v in members(g) if p.stored[v]])
    end
    for c in copies
        argument, result = c.values
        push!(reads, [storage_of(result)])
        push!(writes, [storage_of(argument)])
    end
    numbered(reads, writes)
end

# Storage named in any way, as `reads[k]` and `writes[k]` name what step
# `k` touches, numbered in the order first named.
function numbered(reads, writes)
    keys = Dict{Any, Int}()
    key(x) = get!(keys, x, length(keys) + 1)
    [(reads = unique!(map(key, r)), writes = unique!(map(key, w))) for (r, w) in zip(reads, writes)]
end

"""
    stream_schedule(steps, touches) -> (; waits, runs)

The waits between streams that `steps`, in plan order, need, and the runs a
recording launches them in, from the storage each touches (`touches[k]`,
see `Touches`).

`waits[k]` holds the earlier steps on other streams that step `k` waits for,
as `stream_order` derives them: the last that wrote what it reads, and the
last that wrote and those that read since what it writes, less those its
stream is ordered after already. Storage shares a slot only where the
program orders its users (see `storage`), so these are the waits of the
values each step reads and the arguments it writes; one device's steps wait
for none.

`runs` holds the positions of each run, in the order they are launched, each
of one stream's steps in order: a step joins its stream's last run unless it
waits for a step of a run launched after that one began, so that what it
waits for is launched before it. A wait binds to what was launched before
it: launched later, a graph that waits for a crossing a graph of its own
device feeds would wait for the crossing of the execution before. One
device's steps are one run.
"""
function stream_schedule(steps::Vector{Step}, touches::Vector{Touches})
    (; on, waits) = stream_order(map(stream_of, steps), touches)
    (; waits, runs = stream_runs(on, waits))
end

# The runs of steps on streams `on`, waiting for `waits`: each step joins its
# stream's last run unless it waits for one of a run begun after it.
function stream_runs(on, waits)
    runs = Vector{Int}[]
    starts = Int[]
    run_of = zeros(Int, length(on))
    current = Dict{Int, Int}()
    for k in eachindex(on)
        r = get(current, on[k], 0)
        if r == 0 || any(j -> starts[run_of[j]] > starts[r], waits[k])
            push!(runs, Int[])
            push!(starts, k)
            r = current[on[k]] = length(runs)
        end
        push!(runs[r], k)
        run_of[k] = r
    end
    runs
end

## Running.

"""
    Scheduled(step, stream, waits, signal)

A step as its schedule runs it (see `Schedule`): on `stream` (see
`stream`), after waiting for `waits`, the events of steps on other streams,
and recording `signal` once it has run, where a step on another stream
waits for it, else `nothing`. What a target captures and replays (see
`capture`).
"""
struct Scheduled
    step::Step
    stream::Any
    waits::Vector{Any}
    signal::Any
end

# Run step `s` where the task is: its waits, its run, its record.
function run!(target::Target, s::Scheduled)
    foreach(e -> wait!(target, s.stream, e), s.waits)
    s.step.run()
    s.signal === nothing || record!(target, s.stream, s.signal)
    nothing
end

# A lane and the own streams of the two devices it copies between, with the
# events that order a call's crossings after what was launched on either
# device before the call, and what is launched on either after it after them.
struct LaneEnds
    lane::Any
    from::Any
    to::Any
    begun_from::Any
    begun_to::Any
    ended::Any
end

"""
An execution's steps as they run, each on its stream (see `stream`), in plan
order, with the waits that order it after steps on other streams (see
`stream_schedule`); the runs of each stream's steps, in the order they are
launched; and per lane its ends, which order a call's crossings after what
was launched before the call on either device, and what is launched after
the call on either after the crossings (see `walk`).
"""
struct Schedule
    steps::Vector{Scheduled}
    runs::Vector{Vector{Int}}
    lanes::Vector{LaneEnds}
end

# The schedule of `steps`, which touch `touches`, on `target`, its streams
# and events made where their steps run.
function Schedule(target::Target, steps::Vector{Step}, touches::Vector{Touches})
    (; waits, runs) = stream_schedule(steps, touches)
    pairs = filter(σ -> first(σ) != last(σ), unique(map(stream_of, steps)))
    devices = unique([map(last ∘ stream_of, steps); map(first, pairs)])
    streams = Dict{Any, Any}()
    foreach(d -> streams[(d, d)] = on_device(() -> stream(target, d, d), target, d), devices)
    foreach(σ -> streams[σ] = on_device(() -> stream(target, σ...), target, last(σ)), pairs)
    made(σ) = on_stream(() -> event(target, streams[σ]), target, streams[σ])
    waited = falses(length(steps))
    foreach(w -> waited[w] .= true, waits)
    signals = Any[waited[k] ? made(stream_of(s)) : nothing for (k, s) in enumerate(steps)]
    scheduled = [Scheduled(s, streams[stream_of(s)], signals[waits[k]], signals[k]) for (k, s) in enumerate(steps)]
    ends = [LaneEnds(streams[(a, b)], streams[(a, a)], streams[(b, b)], made((a, a)), made((b, b)), made((a, b))) for (a, b) in pairs]
    Schedule(scheduled, runs, ends)
end

"""
    walk(target, schedule)

Run an execution's steps, each with the calling task where its stream runs
(see `on_stream`): the one place they are put on their devices. The runs
are walked in the order they are launched, each in one entry of its stream,
each step after its waits and before its record (see `Scheduled`). An
execution on one device is one run, entered once.

The call's crossings are ordered by its two ends, on each lane: as it
begins, each lane waits for what was launched before the call on its two
devices, which wrote its sources and read what its copies overwrite; as it
ends, each device waits for its lanes, so that what reads the call's
results, writes its arguments or frees its storage next waits for them.

A run launched before a crossing that comes earlier in plan order holds
nothing that conflicts with it: every two steps on two streams that touch
one storage, one writing it, are ordered by the waits, and a step that
waits for one launched with a run begun later than its stream's last
begins a run of its own (see `stream_schedule`). So every two that conflict
are launched in plan order, and a target whose events order nothing, whose
crossings run on the streams of the two devices they copy between, runs
them in that order on the stream they share.
"""
function walk(target::Target, schedule::Schedule)
    fork!(target, schedule)
    for run in schedule.runs
        on_stream(() -> foreach(k -> run!(target, schedule.steps[k]), run), target, schedule.steps[first(run)].stream)
    end
    rejoin!(target, schedule)
end

# A call's beginning, on each lane: what was launched before on its devices first.
function fork!(target::Target, schedule::Schedule)
    for e in schedule.lanes
        on_stream(() -> record!(target, e.from, e.begun_from), target, e.from)
        on_stream(() -> record!(target, e.to, e.begun_to), target, e.to)
        on_stream(() -> (wait!(target, e.lane, e.begun_from); wait!(target, e.lane, e.begun_to)), target, e.lane)
    end
    nothing
end

# A call's end, on each lane: what is launched after on its devices waits for it.
function rejoin!(target::Target, schedule::Schedule)
    for e in schedule.lanes
        on_stream(() -> record!(target, e.lane, e.ended), target, e.lane)
        on_stream(() -> wait!(target, e.from, e.ended), target, e.from)
        on_stream(() -> wait!(target, e.to, e.ended), target, e.to)
    end
    nothing
end

"""
A run of a recording's steps on one stream: the stream, where its graph is
made, launched and changed (see `on_stream`), the positions of its steps
among the recording's, and the graph the target captured of them (see
`capture`), recorded again in place where its launches move (see
`recapture`).
"""
mutable struct Segment
    const stream::Any
    const steps::Vector{Int}
    graph::Any
end

"""
What `capture!` records of an executable: the schedule of the executions a
replay runs, and a segment per run of it, each captured as its stream's own
graph, launched in the order of the runs, between the call's two ends (see
`walk`); and per step, its segment and its place there. An execution on one
device is one segment, its graph the target's capture of every step.
"""
struct Captured
    schedule::Schedule
    segments::Vector{Segment}
    at::Vector{Tuple{Int, Int}}
end

# The schedule `s` recorded on `target`, a segment per run, each graph keeping `keep` alive.
function record(target::Target, s::Schedule, keep)
    segments = Segment[]
    at = Vector{Tuple{Int, Int}}(undef, length(s.steps))
    for run in s.runs
        stream = s.steps[first(run)].stream
        push!(segments, Segment(stream, run, on_stream(() -> capture(target, s.steps[run], keep), target, stream)))
        foreach(((i, k),) -> at[k] = (length(segments), i), enumerate(run))
    end
    Captured(s, segments, at)
end

# Launch what `r` recorded, segment by segment, between the call's two ends.
function replay_recording(target::Target, r::Captured)
    fork!(target, r.schedule)
    for s in r.segments
        on_stream(() -> replay(target, s.graph), target, s.stream)
    end
    rejoin!(target, r.schedule)
end

# Segment `s` of what `r` recorded, recorded again after its launches moved.
recapture_segment!(target::Target, r::Captured, s::Segment, keep) =
    s.graph = on_stream(() -> recapture(target, s.graph, r.schedule.steps[s.steps], keep), target, s.stream)

# Destroy every segment's graph of what `r` recorded.
release_recording(target::Target, r::Captured) = foreach(s -> on_stream(() -> release(target, s.graph), target, s.stream), r.segments)
