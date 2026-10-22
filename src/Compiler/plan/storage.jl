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

"""
    Storage

Where a plan's stored values live: for each value, the kernels from the one
that writes it to the last that reads it (`lifetimes`) and its slot (`slots`,
zero for a value with storage of its own); the bytes of each slot (`sizes`).
`donated` holds the results written into the storage of the arguments they
replace and `staged` those copied there after the last kernel, each as
`result value => argument value`; `overwrites` the folds by key written over
the storage of the contents they start from, as `fold value => value`.
"""
struct Storage
    lifetimes::Vector{UnitRange{Int}}
    slots::Vector{Int}
    sizes::Vector{Int}
    donated::Vector{Pair{Int, Int}}
    staged::Vector{Pair{Int, Int}}
    overwrites::Vector{Pair{Int, Int}}
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
    results = Set(o.value for o in graph.results if o isa Read)
    base(v) = p.owner[v] > 0 ? overwritable(graph, p.groups[p.owner[v]], v) : nothing
    overwrites = Pair{Int, Int}[]
    # Each fold by key of `pairs` writes over its value, if the kernels can run
    # in an order where it does so after every other kernel reading the value.
    function overwrite!(pairs)
        any(((v, b),) -> any(((w, c),) -> w == v || c == b, overwrites), pairs) && return false
        trial = copy(needs)
        for (v, b) in pairs
            k = p.owner[v]
            trial[k] = union(trial[k], (j for j in eachindex(reads) if j != k && b in reads[j]))
        end
        kernel_order(trial) === nothing && return false
        needs = trial
        append!(overwrites, pairs)
        true
    end
    donated, staged = Pair{Int, Int}[], Pair{Int, Int}[]
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
            continue
        end
        k = p.owner[r]
        if k > 0 && !(r in first.(donated)) && in_place(k) && writes_over(graph, p.groups[k], r, a)
            trial = copy(needs)
            trial[k] = union(needs[k], (j for j in eachindex(reads) if j != k && a in reads[j]))
            if kernel_order(trial) !== nothing
                needs = trial
                push!(donated, r => a)
                continue
            end
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
    last = copy(p.owner)
    for (k, g) in enumerate(p.groups), r in group_reads(graph, g)
        last[r] = max(last[r], k)
    end
    lifetimes = [p.owner[v] > 0 ? (p.owner[v]:last[v]) : (1:0) for v in 1:n]
    slots = zeros(Int, n)
    sizes, busy = Int[], Int[] # per slot: its bytes, and the last kernel of the value in it
    over = Dict(overwrites)
    for v in sort!([v for v in 1:n if p.stored[v] && p.owner[v] > 0 && !(v in results)]; by = v -> (p.owner[v], v))
        if haskey(over, v)
            # The slot of the contents it writes over; none over an argument's storage.
            s = slots[over[v]]
            s > 0 && (busy[s] = max(busy[s], last[v]))
            slots[v] = s
            continue
        end
        bytes = value_bytes(graph.values[v])
        free = [s for s in eachindex(sizes) if busy[s] < p.owner[v]]
        fits = filter(s -> sizes[s] >= bytes, free)
        s = !isempty(fits) ? fits[argmin(sizes[fits])] : !isempty(free) ? free[argmax(sizes[free])] :
            (push!(sizes, 0); push!(busy, 0); length(sizes))
        sizes[s] = max(sizes[s], bytes)
        busy[s] = last[v]
        slots[v] = s
    end
    p, order, Storage(lifetimes, slots, sizes, donated, staged, overwrites)
end

# Kernels in an order where each runs after those it needs, otherwise in their
# own order; nothing for a cycle.
function kernel_order(needs)
    order, placed = Int[], falses(length(needs))
    while length(order) < length(needs)
        k = findfirst(j -> !placed[j] && all(i -> placed[i], needs[j]), eachindex(needs))
        k === nothing && return nothing
        push!(order, k)
        placed[k] = true
    end
    order
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
        t isa Load && t.value == a && !(a in reduce(vcat, map(indirect_values, t.index); init = Int[])) || return false
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
end
