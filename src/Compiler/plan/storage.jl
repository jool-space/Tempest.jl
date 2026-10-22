# Storage, the last step of the decision procedure: where each stored value
# lives. Inputs, constants and results have storage of their own. Every other
# stored value lives in a slot from the kernel that writes it to the last
# kernel that reads it, and values whose lifetimes do not overlap share a slot.
# A result that replaces an argument is written into the argument's storage
# where the kernel that writes it reads the argument only at the address it
# writes; it is staged otherwise, in storage of its own and copied after the
# last kernel.

"""
    Storage

Where a plan's stored values live: for each value, the kernels from the one
that writes it to the last that reads it (`lifetimes`) and its slot (`slots`,
zero for a value with storage of its own); the bytes of each slot (`sizes`).
`donated` holds the results written into the storage of the arguments they
replace and `staged` those copied there after the last kernel, each as
`result value => argument value`.
"""
struct Storage
    lifetimes::Vector{UnitRange{Int}}
    slots::Vector{Int}
    sizes::Vector{Int}
    donated::Vector{Pair{Int, Int}}
    staged::Vector{Pair{Int, Int}}
end

"""
    storage(plan, replaces = []; in_place = k -> true) -> (plan, order, storage)

The plan's storage, and the plan with its kernels in the order they run (the
plan's kernel `order[k]` runs `k`th). `replaces` pairs result values with the
argument values they replace; `in_place(k)` says whether the realization of
kernel `k` reads each element a set of outputs needs before it writes any of
them. A kernel that writes a result into an argument's storage runs after the
other kernels that read the argument, when the plan allows it. Slots are
assigned in the order of kernels: each value takes the smallest free slot that
fits it, or else grows the largest free slot, or else a new one.
"""
function storage(p::Plan, replaces = Pair{Int, Int}[]; in_place = k -> true)
    graph = p.graph
    reads = [group_reads(graph, g) for g in p.groups]
    needs = [Set(p.owner[r] for r in rs if p.owner[r] > 0 && p.owner[r] != k) for (k, rs) in enumerate(reads)]
    donated, staged = Pair{Int, Int}[], Pair{Int, Int}[]
    for (r, a) in replaces
        r == a && continue
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
    order = kernel_order(needs)
    position = invperm(order)
    p = Plan(graph, p.stored, p.groups[order], map(o -> o == 0 ? 0 : position[o], p.owner))
    n = length(graph.values)
    results = Set(o.value for o in graph.results if o isa Read)
    last = copy(p.owner)
    for (k, g) in enumerate(p.groups), r in group_reads(graph, g)
        last[r] = max(last[r], k)
    end
    lifetimes = [p.owner[v] > 0 ? (p.owner[v]:last[v]) : (1:0) for v in 1:n]
    slots = zeros(Int, n)
    sizes, busy = Int[], Int[] # per slot: its bytes, and the last kernel of the value in it
    for v in sort!([v for v in 1:n if p.stored[v] && p.owner[v] > 0 && !(v in results)]; by = v -> (p.owner[v], v))
        bytes = value_bytes(graph.values[v])
        free = [s for s in eachindex(sizes) if busy[s] < p.owner[v]]
        fits = filter(s -> sizes[s] >= bytes, free)
        s = !isempty(fits) ? fits[argmin(sizes[fits])] : !isempty(free) ? free[argmax(sizes[free])] :
            (push!(sizes, 0); push!(busy, 0); length(sizes))
        sizes[s] = max(sizes[s], bytes)
        busy[s] = last[v]
        slots[v] = s
    end
    p, order, Storage(lifetimes, slots, sizes, donated, staged)
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
Whether kernel `g` may write output `r` into the storage of `a`: it reads `a`
only at the address where it writes `r`, and only in folds over its primary
axes, which complete before any output is written, or in outputs of `r`'s
shape, which a kernel computes together before it stores any of them.
"""
function writes_over(graph, g::Group, r, a)
    r in g.outputs || return false
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
end
