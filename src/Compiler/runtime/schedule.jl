# The order an execution's steps need between streams. A device runs its own
# steps on one stream, in order; a crossing runs on a lane of its own, the
# stream of its ordered pair of devices, so that it copies beside the kernels
# of both. Whatever two streams touch of one storage is ordered between them
# by a wait: the later step waits for the earlier, one on each stream, as
# plan order has them. What a step touches is known where the steps are made
# (see `step_touches`); the waits and the runs a recording launches are
# derived from it alone (see `stream_schedule`).

"""
What a step reads and writes, as storage keys: a slot, an argument's
storage, or a buffer of a value's own, each numbered (see `numbered`).
"""
const Touches = @NamedTuple{reads::Vector{Int}, writes::Vector{Int}}

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
recording launches them in, from what each touches (`touches[k]`, see
`Touches`).

`waits[k]` holds the earlier steps on other streams that step `k` waits for:
the last that wrote what it reads, and the last that wrote and those that
read since what it writes, each of another stream. A stream runs its own in
order, so a step that waits for one waits for everything before it on its
stream, and for everything that one waited for: a wait the stream is already
ordered after is left out. One device's steps touch only its own storage,
and wait for none.

`runs` holds the positions of each run, in the order they are launched, each
of one stream's steps in order: a step joins its stream's last run unless it
waits for a step of a run launched after that one began, so that what it
waits for is launched before it. A wait binds to what was launched before
it: launched later, a graph that waits for a crossing a graph of its own
device feeds would wait for the crossing of the execution before. One
device's steps are one run.
"""
function stream_schedule(steps::Vector{Step}, touches::Vector{Touches})
    streams = unique(map(stream_of, steps))
    index = Dict(σ => i for (i, σ) in enumerate(streams))
    on = [index[stream_of(s)] for s in steps]
    n, m = length(steps), length(streams)
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
