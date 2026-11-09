# Storage, the last step of the decision procedure: where each stored value
# lives. Inputs, constants and results have storage of their own. Every other
# stored value lives in a slot from the kernel that writes it to the last
# kernel that reads it, and values whose lifetimes do not overlap share a slot.
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
zero for a value with storage of its own); the bytes of each slot (`sizes`).
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
    donated::Vector{Pair{Int, Int}}
    staged::Vector{Pair{Int, Int}}
    overwrites::Vector{Pair{Int, Int}}
    reused::Vector{Pair{Int, Int}}
end

"""
    storage(plan, replaces = []; in_place = k -> true) -> (plan, order, storage)

The plan's storage, and the plan with its kernels in the order they run (the
plan's kernel `order[k]` runs `k`th). `replaces` pairs result values with the
argument values they replace; `in_place(k)` says whether the realization of
kernel `k` reads each element a set of outputs needs before it writes any of
them. A kernel that writes a result into an argument's storage, or a fold by
key over the contents it starts from, runs after the other kernels that read
them, when the plan allows it. Slots are
assigned in the order of kernels: each value takes the smallest free slot that
fits it, or else grows the largest free slot, or else a new one.
"""
function storage(p::Plan, replaces = Pair{Int, Int}[]; in_place = k -> true)
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
    base(v) = p.owner[v] > 0 ? overwritable(graph, p.groups[p.owner[v]], v) : nothing
    overwrites, overwritten, bases = Pair{Int, Int}[], Set{Int}(), Set{Int}()
    # Kernel `k` made to run after every other kernel that reads value `a`,
    # unless one of them runs after it already, which would make kernels need
    # each other in a cycle: whether it was. Each kernel it now needs is in
    # `added`, for `undo!`. The needs are acyclic, so a cycle would pass
    # through `k`, and only what runs after `k` is walked, once per result
    # an argument's storage takes, where ordering every kernel again took
    # each a pass over every kernel for each kernel.
    function follow!(added, k, a)
        js = [j for j in get(readers, a, Int[]) if j != k && !(j in needs[k])]
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
    order = kernel_order(needs)
    position = invperm(order)
    p = Plan(graph, p.stored, p.groups[order], map(o -> o == 0 ? 0 : position[o], p.owner))
    n = length(graph.values)
    # What each kernel reads, by its place in the order they run, in order.
    reading = [sort!(collect(reads[j])) for j in order]
    last = copy(p.owner)
    for (k, rs) in enumerate(reading), r in rs
        last[r] = max(last[r], k)
    end
    lifetimes = [p.owner[v] > 0 ? (p.owner[v]:last[v]) : (1:0) for v in 1:n]
    slots = zeros(Int, n)
    pool = SlotPool()
    (; sizes, busy) = pool
    over = Dict(overwrites)
    reused, taken = Pair{Int, Int}[], Set{Tuple{Int, Int}}()
    # A value its kernel reads for the last time, at the address the kernel
    # writes `v`, a value of `v`'s size in a slot no other value of the kernel
    # took so: its storage, which `v` may take; nothing otherwise.
    function dying(v)
        k = p.owner[v]
        g = p.groups[k]
        # A scan's or a recurrence's kernel writes a step before it reads the next.
        g.kind in (:reduction, :pointwise) && in_place(order[k]) || return nothing
        for b in reading[k]
            slots[b] > 0 && busy[slots[b]] == k && !((slots[b], k) in taken) && last[b] == k && p.owner[b] != k && !(b in results) || continue
            value_bytes(graph.values[b]) == value_bytes(graph.values[v]) && graph.values[b].eltype == graph.values[v].eltype || continue
            writes_over(graph, g, v, b) && return b
        end
        nothing
    end
    for v in sort!([v for v in 1:n if p.stored[v] && p.owner[v] > 0 && !(v in results)]; by = v -> (p.owner[v], v))
        if haskey(over, v)
            # The slot of the contents it writes over; none over an argument's storage.
            s = slots[over[v]]
            s > 0 && occupy!(pool, s, last[v])
            slots[v] = s
            continue
        end
        b = dying(v)
        if b !== nothing
            s = slots[b]
            push!(taken, (s, p.owner[v]))
            occupy!(pool, s, last[v])
            slots[v] = s
            push!(reused, v => b)
            continue
        end
        slots[v] = place!(pool, value_bytes(graph.values[v]), p.owner[v], last[v])
    end
    p, order, Storage(lifetimes, slots, sizes, donated, staged, overwrites, reused)
end

"""
The slots stored values take: each slot's bytes (`sizes`) and the last
kernel of the values in it (`busy`). The slots free before a kernel are
kept by size, the least of each size first, and the busy ones by the kernel
after which they free, so that a value finds its slot without a pass over
every slot: with one, the backward of a stack of layers, whose activations
live from the forward on, took a pass over thousands of slots for each
value it stored.
"""
struct SlotPool
    sizes::Vector{Int}
    busy::Vector{Int}
    # A heap of the free slots of each size, and the sizes, in order.
    free::Dict{Int, Vector{Int}}
    free_sizes::Vector{Int}
    # A heap of the busy slots by `busy`, an entry gone stale once its slot's
    # `busy` moved past it.
    freeing::Vector{Tuple{Int, Int}}
end

SlotPool() = SlotPool(Int[], Int[], Dict{Int, Vector{Int}}(), Int[], Tuple{Int, Int}[])

# Slot `s` busy until kernel `until` at least.
function occupy!(pool::SlotPool, s, until)
    until > pool.busy[s] || return
    pool.busy[s] = until
    heap_push!(pool.freeing, (until, s))
end

"""
    place!(pool, bytes, k, until) -> slot

The slot of a value of `bytes` written by kernel `k` and read last by
kernel `until`: of the slots free before `k`, the smallest that fits it,
or else the largest, grown to fit it, or else a new one; the least slot of
those alike.
"""
function place!(pool::SlotPool, bytes, k, until)
    (; sizes, busy, free, free_sizes, freeing) = pool
    while !isempty(freeing) && first(freeing[1]) < k
        b, s = heap_pop!(freeing)
        b == busy[s] || continue
        size = sizes[s]
        haskey(free, size) || insert!(free_sizes, searchsortedfirst(free_sizes, size), size)
        heap_push!(get!(() -> Int[], free, size), s)
    end
    i = min(searchsortedfirst(free_sizes, bytes), length(free_sizes))
    s = if i == 0
        push!(sizes, 0)
        push!(busy, 0)
        length(sizes)
    else
        size = free_sizes[i]
        s = heap_pop!(free[size])
        isempty(free[size]) && (delete!(free, size); deleteat!(free_sizes, i))
        s
    end
    sizes[s] = max(sizes[s], bytes)
    occupy!(pool, s, until)
    s
end

# Kernels in an order where each runs after those it needs, otherwise in their
# own order: each time the first kernel whose needs have run, the kernels
# ready kept in a heap. Nothing for a cycle.
function kernel_order(needs)
    waiting = map(length, needs)
    after = [Int[] for _ in needs]
    foreach(k -> foreach(j -> push!(after[j], k), needs[k]), eachindex(needs))
    ready = Int[]
    foreach(k -> waiting[k] == 0 && heap_push!(ready, k), eachindex(needs))
    order = Int[]
    while !isempty(ready)
        k = heap_pop!(ready)
        push!(order, k)
        for j in after[k]
            waiting[j] -= 1
            waiting[j] == 0 && heap_push!(ready, j)
        end
    end
    length(order) == length(needs) ? order : nothing
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
