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
