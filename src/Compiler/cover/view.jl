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
    foreach(m -> union!(seen, keyed_reads(graph, m)), members(g))
    # The integers a value's mask reads, as a length the device holds, where
    # they are inputs: a kernel may read them without running after another.
    for v in collect(seen)
        mask = masked(graph, v)
        mask === nothing || union!(seen, (u for e in first(mask) for u in indirect_values(e) if p.owner[u] == 0))
    end
    values = Dict(v => value_view(p, v) for v in seen)
    places = Dict{Int, Union{Nothing, Tuple{Vararg{Affine}}}}(m => placement(graph, g, m) for m in members(g))
    reference(target, buffers) = build_floor(target, p, g, buffers)
    GroupView(g.kind, g.domain, copy(g.reductions), copy(g.outputs), g.terms, g.phase,
        primary_axes(graph, g), values, places, group_bytes(p, g), reference)
end

"""
    group_views(plan) -> Vector{GroupView}
    group_views(prepared) -> Vector{GroupView}

The views of a plan's groups, or of the groups of the plan a prepared program
runs, in the plan's order: what a proposal reads of the plan it is given, and
what a package's precompile workload claims.
"""
group_views(p::Plan) = [group_view(p, g) for g in p.groups]
group_views(ex::Executable) = group_views(ex.plan)

function value_view(p::Plan, v)
    val = p.graph.values[v]
    d = val.definition
    fold, reduced = d isa Union{Reduction, Keyed} ? (d.fold, folded_axes(d)) : d isa Scan ? (d.fold, ()) : (nothing, ())
    scan = d isa Scan ? (; d.reverse, d.exclusive) : nothing
    into = d isa Keyed && d.into !== nothing ? stored_term(d.into) : nothing
    carried = d isa Carried ? (; d.axis, from = stored_term(d.from)) : nothing
    ValueView(val.eltype, val.dims, val.scalar, p.stored[v], fold, reduced, d isa Keyed, into, scan, masked(p.graph, v), carried)
end
