# Storage, the last step of the decision procedure: where each stored value
# lives. Inputs, constants and results have storage of their own. Every other
# stored value lives in a slot, and a value takes a slot only where every
# kernel that used what it held runs before the one that writes it by the
# program's own order (see `plan_order`): on one stream earlier, or before it
# through the values it reads. Storage so never orders two streams the
# program leaves apart, which would hold a crossing back until the kernels of
# a device it does not need have run, or a device until a crossing has; on
# one device, whose kernels are one stream, values whose lifetimes do not
# overlap share a slot.
# A result that replaces an argument is written into the argument's storage
# where the kernel that writes it reads the argument only at the address it
# writes; it is staged otherwise, in storage of its own and copied after the
# last kernel. A fold by key that starts from contents nothing reads after it
# takes their storage, so that its kernel writes only the elements it
# combines, as an append to a cache writes one position: a result written
# from an argument it replaces, through such folds, is the argument's storage.
# A value its kernel writes where it reads, for the last time, a value of its
# size takes that value's slot, as each sum of a gradient accumulating over a
# value's uses takes the last one's.

"""
    Storage

Where a plan's stored values live: for each value, the kernels from the one
that writes it to the last that reads it (`lifetimes`) and its slot (`slots`,
zero for a value with storage of its own); the bytes of each slot (`sizes`)
and the device it is on (`devices`, see `with_devices`), whose values alone
it holds.
`donated` holds the results written into the storage of the arguments they
replace and `staged` those copied there after the last kernel, each as
`result value => argument value`; `overwrites` the folds by key written over
the storage of the contents they start from, as `fold value => value`; and
`reused` the values written over the slot of a value their kernel reads last,
as `value => value`.
"""
struct Storage
    lifetimes::Vector{UnitRange{Int}}
    slots::Vector{Int}
    sizes::Vector{Int}
    devices::Vector{Int}
    donated::Vector{Pair{Int, Int}}
    staged::Vector{Pair{Int, Int}}
    overwrites::Vector{Pair{Int, Int}}
    reused::Vector{Pair{Int, Int}}
end

"""
    storage(plan, replaces = []; in_place = k -> true, seconds = k -> 0.0) -> (plan, order, storage)

The plan's storage, and the plan with its kernels in the order they run (the
plan's kernel `order[k]` runs `k`th), kernel `k` estimated to take
`seconds(k)` (see `run_order`). `replaces` pairs result values with the
argument values they replace; `in_place(k)` says whether the realization of
kernel `k` reads each element a set of outputs needs before it writes any of
them. A kernel that writes a result into an argument's storage, or a fold by
key over the contents it starts from, runs after the other kernels that read
them, when the plan allows it and a kernel it would so run after is on its
stream or before it already. Slots are assigned in the order of kernels:
each value takes the smallest slot free for its kernel that fits it, or else
grows the largest free slot, or else a new one; a slot is free for a kernel
once every kernel using what it holds runs before it (see `plan_order`).
"""
function storage(p::Plan, replaces = Pair{Int, Int}[]; in_place = k -> true, seconds = k -> 0.0)
    graph = p.graph
    reads = [group_reads(graph, g) for g in p.groups]
    needs = [Set(p.owner[r] for r in rs if p.owner[r] > 0 && p.owner[r] != k) for (k, rs) in enumerate(reads)]
    # The kernels that read each value, and those that need each kernel.
    readers = Dict{Int, Vector{Int}}()
    for (j, rs) in enumerate(reads), r in rs
        push!(get!(() -> Int[], readers, r), j)
    end
    after = [Int[] for _ in p.groups]
    foreach(k -> foreach(j -> push!(after[j], k), needs[k]), eachindex(needs))
    results = Set(o.value for o in graph.results if o isa Read)
    streams = [group_stream(p, g) for g in p.groups]
    base(v) = p.owner[v] > 0 ? overwritable(graph, p.groups[p.owner[v]], v) : nothing
    overwrites, overwritten, bases = Pair{Int, Int}[], Set{Int}(), Set{Int}()
    # Kernel `k` made to run after every other kernel that reads value `a`,
    # unless one of them runs after it already, which would make kernels need
    # each other in a cycle: whether it was. Each kernel it now needs is in
    # `added`, for `undo!`. The needs are acyclic, so a cycle would pass
    # through `k`, and only what runs after `k` is walked, once per result
    # an argument's storage takes, where ordering every kernel again took
    # each a pass over every kernel for each kernel. A kernel on another
    # stream it is not made to run after, which would hold one stream for
    # another by storage alone: it must run before `k` already.
    function follow!(added, k, a)
        readers_apart = Int[]
        js = Int[]
        for j in get(readers, a, Int[])
            j == k || j in needs[k] || push!(streams[j] == streams[k] ? js : readers_apart, j)
        end
        all(j -> runs_after(after, j, [k]), readers_apart) || return false
        isempty(js) && return true
        runs_after(after, k, js) && return false
        foreach(j -> (push!(needs[k], j); push!(after[j], k); push!(added, k => j)), js)
        true
    end
    undo!(added) = foreach(((k, j),) -> (delete!(needs[k], j); pop!(after[j])), Iterators.reverse(added))
    # Each fold by key of `pairs` writes over its value, if the kernels can run
    # in an order where it does so after every other kernel reading the value.
    function overwrite!(pairs)
        any(((v, b),) -> v in overwritten || b in bases, pairs) && return false
        added = Pair{Int, Int}[]
        for (v, b) in pairs
            follow!(added, p.owner[v], b) && continue
            undo!(added)
            return false
        end
        append!(overwrites, pairs)
        foreach(((v, b),) -> (push!(overwritten, v); push!(bases, b)), pairs)
        true
    end
    donated, staged, taking = Pair{Int, Int}[], Pair{Int, Int}[], Set{Int}()
    for (r, a) in replaces
        r == a && continue
        # Folds by key from the argument to the result, each over the last.
        chain, v, b = Pair{Int, Int}[], r, base(r)
        while b !== nothing && b != a && !(b in results)
            push!(chain, v => b)
            v, b = b, base(b)
        end
        if b == a && overwrite!([chain; v => a])
            push!(donated, r => a)
            push!(taking, r)
            continue
        end
        k = p.owner[r]
        if k > 0 && !(r in taking) && in_place(k) && writes_over(graph, p.groups[k], r, a) && follow!(Pair{Int, Int}[], k, a)
            push!(donated, r => a)
            push!(taking, r)
            continue
        end
        push!(staged, r => a)
    end
    for v in eachindex(graph.values)
        b = base(v)
        b === nothing || v in results || b in results || p.owner[b] == 0 || overwrite!([v => b])
    end
    order = run_order(needs, streams, map(seconds, eachindex(p.groups)))
    position = invperm(order)
    p = Plan(graph, p.stored, p.groups[order], map(o -> o == 0 ? 0 : position[o], p.owner))
    n = length(graph.values)
    # What each kernel reads, by its place in the order they run, in order.
    reading = [sort!(collect(reads[j])) for j in order]
    ordering = plan_order(p)
    # Each value's last kernel, and on each stream its last that uses it.
    last = copy(p.owner)
    users = zeros(Int, length(ordering.streams), n)
    foreach(v -> p.owner[v] > 0 && (users[ordering.on[p.owner[v]], v] = p.owner[v]), 1:n)
    for (k, rs) in enumerate(reading), r in rs
        last[r] = max(last[r], k)
        users[ordering.on[k], r] = k
    end
    lifetimes = [p.owner[v] > 0 ? (p.owner[v]:last[v]) : (1:0) for v in 1:n]
    slots = zeros(Int, n)
    pool = SlotPool(ordering)
    sizes = pool.sizes
    over = Dict(overwrites)
    reused, taken = Pair{Int, Int}[], Set{Tuple{Int, Int}}()
    # A value its kernel reads for the last time, at the address the kernel
    # writes `v`, a value of `v`'s size in a slot no other value of the kernel
    # took so: its storage, which `v` may take; nothing otherwise.
    function dying(v)
        k = p.owner[v]
        g = p.groups[k]
        # A scan's or a recurrence's kernel writes a step before it reads the next.
        g.kind isa Union{ReductionKind, PointwiseKind} && in_place(order[k]) || return nothing
        for b in reading[k]
            slots[b] > 0 && vacated(pool, slots[b], k) && !((slots[b], k) in taken) && last[b] == k && p.owner[b] != k && !(b in results) || continue
            value_bytes(graph.values[b]) == value_bytes(graph.values[v]) && graph.values[b].eltype == graph.values[v].eltype || continue
            writes_over(graph, g, v, b) && return b
        end
        nothing
    end
    for v in sort!([v for v in 1:n if p.stored[v] && p.owner[v] > 0 && !(v in results)]; by = v -> (p.owner[v], v))
        if haskey(over, v)
            # The slot of the contents it writes over; none over an argument's storage.
            s = slots[over[v]]
            s > 0 && occupy!(pool, s, view(users, :, v))
            slots[v] = s
            continue
        end
        b = dying(v)
        if b !== nothing
            s = slots[b]
            push!(taken, (s, p.owner[v]))
            occupy!(pool, s, view(users, :, v))
            slots[v] = s
            push!(reused, v => b)
            continue
        end
        slots[v] = place!(pool, value_bytes(graph.values[v]), p.owner[v], view(users, :, v), graph.devices[v])
    end
    p, order, Storage(lifetimes, slots, sizes, pool.devices, donated, staged, overwrites, reused)
end

"""
The slots stored values take: each slot's bytes (`sizes`), on each stream
the last kernel that uses what it holds (`busy`), and its device
(`devices`), by the order the slots are free in (`order`, see
`plan_order`). The slots free for the kernels of each stream are kept by
size, the least of each first, and the busy ones by the kernel of each
stream that each awaits, so that a value finds its slot without a pass
over every slot: with one, the backward of a stack of layers, whose
activations live from the forward on, took a pass over thousands of slots
for each value it stored. A slot is free for a stream's kernels once they
run after its kernel on every stream, which, the streams' kernels running
in order, each kernel of a stream finds in its turn: once for a slot,
counted by the streams it awaits (`pending`). An entry for a slot from
before it was taken or held longer (`held`) is stale.
"""
struct SlotPool
    order::StreamOrder
    sizes::Vector{Int}
    busy::Vector{Vector{Int}}
    devices::Vector{Int}
    held::Vector{Int}
    # The streams that write on each device.
    writers::Dict{Int, Vector{Int}}
    # For each stream, a heap of the slots free for its kernels of each
    # size, and the sizes, in order.
    free::Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}
    free_sizes::Dict{Int, Vector{Int}}
    # For each stream and each it awaits, a heap of the slots by the kernel
    # awaited; and for each stream and slot, the streams it awaits yet.
    freeing::Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int, Int}}}
    pending::Dict{Tuple{Int, Int}, Int}
end

function SlotPool(order::StreamOrder)
    writers = Dict{Int, Vector{Int}}()
    foreach(((σ, stream),) -> push!(get!(() -> Int[], writers, last(stream)), σ), enumerate(order.streams))
    SlotPool(order, Int[], Vector{Int}[], Int[], Int[], writers, Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}(), Dict{Int, Vector{Int}}(),
        Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int, Int}}}(), Dict{Tuple{Int, Int}, Int}())
end

# Slot `s` held until the kernels `until`, on each stream, at least: awaited
# by each stream that writes on its device until they have run.
function occupy!(pool::SlotPool, s, until)
    busy = pool.busy[s]
    all(τ -> until[τ] <= busy[τ], eachindex(busy)) && return
    busy .= max.(busy, until)
    held = pool.held[s] += 1
    for σ in pool.writers[pool.devices[s]]
        awaited = findall(>(0), busy)
        pool.pending[(σ, s)] = length(awaited)
        foreach(τ -> heap_push!(get!(() -> Tuple{Int, Int, Int}[], pool.freeing, (σ, τ)), (busy[τ], s, held)), awaited)
    end
end

"Whether kernel `k` uses what slot `s` holds, and every other kernel that does runs before it."
function vacated(pool::SlotPool, s, k)
    after = pool.order.after[k]
    busy = pool.busy[s]
    busy[pool.order.on[k]] == k && all(τ -> busy[τ] <= after[τ], eachindex(busy))
end

"""
    place!(pool, bytes, k, until, device) -> slot

The slot of a value of `bytes` on `device`, written by kernel `k` and used
last on each stream by kernel `until[τ]`: of the device's slots free for
`k`, the smallest that fits it, or else the largest, grown to fit it, or
else a new one; the least slot of those alike.
"""
function place!(pool::SlotPool, bytes, k, until, device)
    (; order, sizes, busy, devices, held, free, freeing, pending) = pool
    σ = order.on[k]
    after = order.after[k]
    for τ in eachindex(order.streams)
        heap = get(freeing, (σ, τ), nothing)
        heap === nothing && continue
        # A kernel of its own stream before `k`, or of another that `k` runs after.
        reached = min(k - 1, after[τ])
        while !isempty(heap) && first(heap[1]) <= reached
            _, s, h = heap_pop!(heap)
            h == held[s] || continue
            (pending[(σ, s)] -= 1) == 0 || continue
            each = get!(() -> Int[], pool.free_sizes, σ)
            haskey(free, (σ, sizes[s])) || insert!(each, searchsortedfirst(each, sizes[s]), sizes[s])
            heap_push!(get!(() -> Tuple{Int, Int}[], free, (σ, sizes[s])), (s, h))
        end
    end
    s = take_free!(pool, σ, bytes)
    if s == 0
        push!(sizes, 0)
        push!(busy, zeros(Int, length(order.streams)))
        push!(devices, device)
        push!(held, 0)
        s = length(sizes)
    end
    sizes[s] = max(sizes[s], bytes)
    occupy!(pool, s, until)
    s
end

# The slot free for the kernels of stream `σ` that a value of `bytes` takes,
# the smallest that fits it or else the largest; zero for none. A slot
# another stream took since it was freed is stale and passed over.
function take_free!(pool::SlotPool, σ, bytes)
    (; free, held) = pool
    each = get!(() -> Int[], pool.free_sizes, σ)
    while !isempty(each)
        i = min(searchsortedfirst(each, bytes), length(each))
        heap = free[(σ, each[i])]
        while !isempty(heap)
            s, h = heap_pop!(heap)
            h == held[s] && (isempty(heap) && (delete!(free, (σ, each[i])); deleteat!(each, i)); return s)
        end
        delete!(free, (σ, each[i]))
        deleteat!(each, i)
    end
    0
end

# Whether any kernel of `js` runs after kernel `k`, by the kernels that need
# each (`after`): what runs after `k` walked once.
function runs_after(after, k, js)
    targets = Set(js)
    seen, pending = Set(k), [k]
    while !isempty(pending)
        for j in after[pop!(pending)]
            j in targets && return true
            j in seen && continue
            push!(seen, j)
            push!(pending, j)
        end
    end
    false
end

# A heap, the least on top.
function heap_push!(h, x)
    push!(h, x)
    i = length(h)
    while i > 1 && h[i >> 1] > h[i]
        h[i], h[i >> 1] = h[i >> 1], h[i]
        i >>= 1
    end
    h
end

function heap_pop!(h)
    top, last = h[1], pop!(h)
    isempty(h) && return top
    h[1] = last
    i, n = 1, length(h)
    while 2i <= n
        c = 2i < n && h[2i + 1] < h[2i] ? 2i + 1 : 2i
        h[c] < h[i] || break
        h[i], h[c] = h[c], h[i]
        i = c
    end
    top
end

"""
The value whose storage fold by key `r`, in kernel `g`, may take: the contents
it starts from, where it reads them as they are stored and nothing else in the
kernel reads them; nothing otherwise. The kernel then need not copy them.
"""
function overwritable(graph, g::Group, r)
    keyed(graph, r) || return nothing
    val = graph.values[r]
    o = val.definition.into
    o isa Read && isempty(gathered_values(o)) || return nothing
    b = o.value
    graph.values[b].eltype == val.eltype && reads_storage(graph.values[b], o.index, val.dims) || return nothing
    others = Group(g.kind, g.domain, filter(!=(r), g.reductions), g.outputs, g.terms, g.phase)
    b in group_reads(graph, others) || b in stored_reads(g.terms[r]) || b in key_reads(graph, r) ? nothing : b
end

"""
Whether kernel `g` may write output `r` into the storage of `a`: it reads `a`
only at the address where it writes `r`, and only in folds over its primary
axes, which complete before any output is written, or in outputs of `r`'s
shape, which a kernel computes together before it stores any of them. A fold
by key reads what it starts from and its key before it writes, but not at its
address.
"""
function writes_over(graph, g::Group, r, a)
    r in g.outputs || return false
    any(m -> a in keyed_reads(graph, m), members(g)) && return false
    D = Domain(g.domain)
    at = dense_address(graph.values[r].dims, placement(graph, g, r), D)
    for m in members(g), t in subterms((g.terms[m],))
        t isa Recomputed && a in stored_reads(t) && return false
        t isa Union{Load, Ix} && a in stored_reads(t) || continue
        t isa Load && t.value == a && !(a in indirect_values(t.index)) || return false
        dense_address(graph.values[a].dims, t.index, D) == at || return false
        (m in g.reductions ? primary(graph, g, m) : graph.values[m].dims == graph.values[r].dims) || return false
    end
    true
end

function explain(io::IO, p::Plan, s::Storage)
    slotted = [v for v in eachindex(s.slots) if s.slots[v] > 0]
    apart = sum(v -> value_bytes(p.graph.values[v]), slotted; init = 0)
    println(io, "storage: ", length(slotted), " values in ", length(s.sizes), " slots (", Base.format_bytes(sum(s.sizes; init = 0)),
        ", against ", Base.format_bytes(apart), " stored apart)")
    for (how, pairs) in (("in place of", s.donated), ("staged, then copied into", s.staged)), (r, a) in pairs
        println(io, "  result ", describe_value(p.graph, r), " ", how, " input ", p.graph.values[a].definition.argument)
    end
    for (v, b) in s.overwrites
        println(io, "  fold by key ", describe_value(p.graph, v), " written over ", describe_value(p.graph, b))
    end
    for (v, b) in s.reused
        println(io, "  ", describe_value(p.graph, v), " written over ", describe_value(p.graph, b), ", which its kernel reads last")
    end
end
