# The order steps have between the streams they run on. A device runs its own
# steps on one stream, in order; a crossing runs on a lane of its own, the
# stream of its ordered pair of devices, so that it copies beside the kernels
# of both. Whatever two streams touch of one thing is ordered between them by
# a wait: the later step waits for the earlier, one on each stream, as plan
# order has them. One derivation serves two ends. Over the values each kernel
# of a plan reads and stores, it is the order the program itself gives its
# kernels (see `plan_order`), which storage keeps to, so that sharing a slot
# never orders two streams the program leaves apart; over the storage each
# step of an execution touches, it is the waits the execution runs by (see
# `stream_schedule`).

"""
What a step reads and writes, as keys: values, or storage (a slot, an
argument's storage, or a buffer of a value's own, each numbered; see
`numbered`).
"""
const Touches = @NamedTuple{reads::Vector{Int}, writes::Vector{Int}}

"""
    StreamOrder

The order of steps in plan order, each on a stream (`streams`, numbered as
first met; `on[k]` the number of step `k`'s): `waits[k]` holds the earlier
steps on other streams that step `k` waits for, and `after[k]`, for each
stream, the last of its steps that step `k` runs after, its own stream's
`k`. See `stream_order` and `ordered`.
"""
struct StreamOrder
    streams::Vector{Any}
    on::Vector{Int}
    waits::Vector{Vector{Int}}
    after::Vector{Vector{Int}}
end

"""
    ordered(order, j, k)

Whether step `j` runs before step `k` by `order`: earlier on `k`'s stream,
or on another one before a step `k` waits for, itself or through what that
waits for.
"""
ordered(o::StreamOrder, j, k) = j < k && o.after[k][o.on[j]] >= j

"""
    stream_order(streams, touches) -> StreamOrder

The order steps in plan order, step `k` on `streams[k]`, need between their
streams, from what each touches (`touches[k]`, see `Touches`).

Step `k` waits for the earlier steps on other streams that last wrote what it
reads, and that last wrote and since read what it writes. A stream runs its
own in order, so a step that waits for one waits for everything before it on
its stream, and for everything that one waited for: a wait the stream is
already ordered after is left out. One stream's steps wait for none.
"""
function stream_order(streams::AbstractVector, touches::Vector{Touches})
    numbers = unique(streams)
    index = Dict(σ => i for (i, σ) in enumerate(numbers))
    on = [index[σ] for σ in streams]
    n, m = length(streams), length(numbers)
    # Per stream, the last step of each stream it is ordered after; and per
    # step, its stream's as the step ends.
    clocks = [zeros(Int, m) for _ in 1:m]
    after = Vector{Vector{Int}}(undef, n)
    # Per key, the step that wrote it last, and per stream the last that read it since.
    writer = Dict{Int, Int}()
    readers = Dict{Int, Vector{Int}}()
    waits = [Int[] for _ in 1:n]
    for k in 1:n
        σ = on[k]
        earlier = Int[]
        for key in touches[k].reads
            haskey(writer, key) && push!(earlier, writer[key])
        end
        for key in touches[k].writes
            haskey(writer, key) && push!(earlier, writer[key])
            haskey(readers, key) && append!(earlier, filter(>(0), readers[key]))
        end
        # The latest first: what it was ordered after need not be waited for.
        for j in sort!(unique!(earlier); rev = true)
            on[j] == σ && continue
            clocks[σ][on[j]] >= j && continue
            push!(waits[k], j)
            clocks[σ] .= max.(clocks[σ], after[j])
        end
        clocks[σ][σ] = k
        after[k] = copy(clocks[σ])
        for key in touches[k].reads
            get!(() -> zeros(Int, m), readers, key)[σ] = k
        end
        for key in touches[k].writes
            writer[key] = k
            readers[key] = zeros(Int, m)
        end
    end
    StreamOrder(numbers, on, waits, after)
end

"The stream kernel `g` of `p` runs on: its device's own, or for a crossing the lane from its source's."
group_stream(p::Plan, g::Group) = (source_device(p, g), group_device(p, g))

"""
    plan_order(p) -> StreamOrder

The order the program gives the kernels of plan `p`, in the order they run,
each on its stream (see `group_stream`): each stream's in order, and between
streams through the values each kernel reads and stores, whatever storage
they are given.
"""
function plan_order(p::Plan)
    touches = Touches[(reads = sort!(collect(group_reads(p.graph, g))), writes = [v for v in members(g) if p.stored[v]]) for g in p.groups]
    stream_order([group_stream(p, g) for g in p.groups], touches)
end

"""
    run_order(needs, streams, seconds) -> order

Kernels in the order they run (`order[i]` runs `i`th), kernel `k` needing
those of `needs[k]`, on stream `streams[k]` (see `group_stream`), estimated
to take `seconds[k]`: as they would start if the streams ran each its own
in that order, at once, every kernel taking its time.

A device runs one kernel at a time. A crossing takes its time on its
source's link out and its reader's link in, one crossing at a time on each:
two lanes that share a direction share its rate, each crossing after the
last on it. A kernel can start once those it needs have ended and its
device, or a crossing its two directions, are free. The next taken is the
one that can start first; of two that can at once, the one after which
another stream can begin on what follows it soonest (see `awaited`); and
then the first in the plan's order.

So a kernel reading a crossing runs once the crossing is estimated there,
and its device runs what it can before it; once it can run, it runs before
what nothing on another stream waits for, and what a crossing copies is
computed as soon as its device can, the link kept busy beside the work. On
one stream every kernel whose needs have run can start as the one before it
ends, and none is awaited by another stream: the kernels run in the plan's
order wherever it lets them. The estimates order the kernels only; the
waits between streams follow from the order whatever it is (see
`stream_order`), so an estimate that is wrong costs time, never a result.
"""
function run_order(needs, streams, seconds)
    n = length(needs)
    after = [Int[] for _ in 1:n]
    foreach(k -> foreach(j -> push!(after[j], k), needs[k]), 1:n)
    soon = awaited(after, streams, seconds)
    # What each stream holds: its device, or a crossing's two directions.
    holds(σ) = first(σ) == last(σ) ? ((first(σ), :run),) : ((first(σ), :out), (last(σ), :in))
    free = Dict{Tuple{Any, Symbol}, Float64}()
    numbers = unique(streams)
    on = indexin(streams, numbers)
    held = [holds(σ) for σ in numbers]
    # Each stream's kernels whose needs have run: by the time they can
    # start, those that can as soon as the stream is free by `soon`.
    later = [Tuple{Float64, Float64, Int}[] for _ in numbers]
    now = [Tuple{Float64, Int}[] for _ in numbers]
    ready = zeros(n)
    waiting = map(length, needs)
    foreach(k -> waiting[k] == 0 && heap_push!(later[on[k]], (0.0, soon[k], k)), 1:n)
    order = Int[]
    while length(order) < n
        best, at = (Inf, Inf, typemax(Int)), 0
        for s in eachindex(numbers)
            isempty(later[s]) && isempty(now[s]) && continue
            t = maximum(r -> get(free, r, 0.0), held[s])
            while !isempty(later[s]) && first(first(later[s])) <= t
                _, l, k = heap_pop!(later[s])
                heap_push!(now[s], (l, k))
            end
            key = isempty(now[s]) ? first(later[s]) : (t, first(now[s])...)
            key < best && ((best, at) = (key, s))
        end
        at == 0 && error("kernels depend on each other in a cycle")
        start, _, k = best
        isempty(now[at]) ? heap_pop!(later[at]) : heap_pop!(now[at])
        push!(order, k)
        ends = start + seconds[k]
        foreach(r -> free[r] = ends, held[at])
        for j in after[k]
            ready[j] = max(ready[j], ends)
            waiting[j] -= 1
            waiting[j] == 0 && heap_push!(later[on[j]], (ready[j], soon[j], j))
        end
    end
    order
end

# How soon after each kernel ends another stream can begin on what follows
# it, by the estimates: none for one a kernel on another stream reads, the
# time its own stream takes on the way for one whose followers lead to such
# a kernel, the least of its followers', and Inf for one nothing on another
# stream follows (see `run_order`).
function awaited(after, streams, seconds)
    n = length(after)
    # The kernels in an order where each comes after what it needs.
    waiting = zeros(Int, n)
    foreach(js -> foreach(j -> waiting[j] += 1, js), after)
    order = filter(k -> waiting[k] == 0, collect(1:n))
    for k in order, j in after[k]
        waiting[j] -= 1
        waiting[j] == 0 && push!(order, j)
    end
    length(order) == n || error("kernels depend on each other in a cycle")
    soon = fill(Inf, n)
    for k in Iterators.reverse(order)
        soon[k] = minimum(j -> streams[j] == streams[k] ? seconds[j] + soon[j] : 0.0, after[k]; init = Inf)
    end
    soon
end
