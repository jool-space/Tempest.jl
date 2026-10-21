# What implementations see of a group: a `GroupView`, made from the plan when
# the cover asks for claims, estimates and builds.

"""
    group_view(plan, group) -> GroupView

The view of `group` implementations claim: its members and their terms, every
value it computes or reads, where it keeps its members, the bytes it moves,
and its reference kernels.
"""
function group_view(p::Plan, g::Group)
    graph = p.graph
    roots = [g.terms[m] for m in members(g) if haskey(g.terms, m)]
    seen = Set{Int}(members(g))
    for t in subterms(roots; nested = true)
        t isa Union{Load, Stat, Recomputed} && push!(seen, t.value)
    end
    foreach(t -> union!(seen, stored_reads(t)), roots)
    for m in members(g)
        d = graph.values[m].definition
        d isa Opaque && foreach(o -> o isa Read && push!(seen, o.value), d.operands)
    end
    values = Dict(v => value_view(p, v) for v in seen)
    places = Dict{Int, Union{Nothing, Tuple{Vararg{Affine}}}}(m => placement(graph, g, m) for m in members(g))
    reference(target, buffers) = build_floor(target, p, g, buffers)
    GroupView(g.kind, g.domain, copy(g.reductions), copy(g.outputs), g.terms, g.phase,
        primary_axes(graph, g), values, places, group_bytes(p, g), reference)
end

function value_view(p::Plan, v)
    val = p.graph.values[v]
    d = val.definition
    fold, reduced = d isa Reduction ? (d.fold, d.reduced) : (nothing, ())
    ValueView(val.eltype, val.dims, val.scalar, p.stored[v], fold, reduced, masked(p.graph, v))
end
