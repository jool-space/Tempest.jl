# Storage, the last step of the decision procedure: where each stored value
# lives. Inputs, constants and results have storage of their own. Every other
# stored value lives in a slot from the kernel that writes it to the last
# kernel that reads it, and values whose lifetimes do not overlap share a slot.

"""
    Storage

Where a plan's stored values live: for each value, the kernels from the one
that writes it to the last that reads it (`lifetimes`) and its slot (`slots`,
zero for a value with storage of its own); the bytes of each slot (`sizes`).
"""
struct Storage
    lifetimes::Vector{UnitRange{Int}}
    slots::Vector{Int}
    sizes::Vector{Int}
end

"""
    storage(plan) -> Storage

Slots for the plan's stored values, assigned in the order of its kernels: each
value takes the smallest free slot that fits it, or else grows the largest
free slot, or else a new one.
"""
function storage(p::Plan)
    graph = p.graph
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
    Storage(lifetimes, slots, sizes)
end

function explain(io::IO, p::Plan, s::Storage)
    slotted = [v for v in eachindex(s.slots) if s.slots[v] > 0]
    apart = sum(v -> value_bytes(p.graph.values[v]), slotted; init = 0)
    println(io, "storage: ", length(slotted), " values in ", length(s.sizes), " slots (", Base.format_bytes(sum(s.sizes; init = 0)),
        ", against ", Base.format_bytes(apart), " stored apart)")
end
