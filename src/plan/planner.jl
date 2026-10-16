# The decision procedure: which values are stored, which are recomputed inside
# their readers, and how the work groups into kernels.
#
# A kernel computes a group over one domain. A reduction group's domain is the
# iteration domain of its reductions: it holds reductions over the same axes,
# later ones reading earlier ones at the group's position along the kept axes
# (phases), and pointwise outputs that read those reductions there.

struct Group
    kind::Symbol # :reduction, :pointwise, :contraction or :opaque
    domain::Tuple{Vararg{Int}}
    reduced::Tuple{Vararg{Int}}
    reductions::Vector{Int}
    outputs::Vector{Int}
    terms::Dict{Int, Term}
    phase::Dict{Int, Int}
end

Group(kind, domain, reduced = ()) =
    Group(kind, domain, reduced, Int[], Int[], Dict{Int, Term}(), Dict{Int, Int}())

members(g::Group) = [g.reductions; g.outputs]

struct Plan
    graph::IndexGraph
    stored::Vector{Bool}
    groups::Vector{Group}
    owner::Vector{Int}
end

# Where a group keeps a value it computes: a full-domain value at every index,
# a reduction (or a value shaped like one) at the position along the kept axes.
function placement(graph, g::Group, v)
    val = graph.values[v]
    (val.scalar || isempty(val.dims)) && return ()
    val.dims == g.domain && return identity_index(g.domain)
    kept = ntuple(k -> k in g.reduced ? 1 : g.domain[k], length(g.domain))
    val.dims == kept && return ntuple(k -> k in g.reduced ? constant(0) : var(k), length(g.domain))
    nothing
end

struct OutsideGroup <: Exception end

"The term computing value `v` at `index`, an index over the group's domain."
function value_term(graph, stored, g::Group, v, index)
    val = graph.values[v]
    d = val.definition
    if v in g.reductions
        index == placement(graph, g, v) || throw(OutsideGroup())
        return Stat(v)
    end
    if v in g.outputs
        index == placement(graph, g, v) || throw(OutsideGroup())
        return definition_term(graph, stored, g, d, val, index)
    end
    stored[v] && return Load(v, index)
    d isa Pointwise || return Load(v, index)
    definition_term(graph, stored, g, d, val, index)
end

function definition_term(graph, stored, g, d::Pointwise, val, index)
    domain = Domain(g.domain)
    args = map(d.operands) do o
        operand_term(graph, stored, g, o, index, domain)
    end
    Apply(d.f, args, val.eltype)
end

operand_term(graph, stored, g, o::Read, index, domain) =
    value_term(graph, stored, g, o.value, map(e -> substitute(e, index, domain), o.index))
operand_term(graph, stored, g, o::Literal, index, domain) = Lit(o.value)
operand_term(graph, stored, g, o::IndexOperand{T}, index, domain) where {T} =
    Ix{T}(substitute(o.expr, index, domain))

# The values an operand reads: its own, and those its formulas gather from.
indirect_values(e::Affine) = reduce(vcat, (indirect_values(a) for (a, _) in e.terms); init = Int[])
indirect_values(::Var) = Int[]
indirect_values(a::Union{Div, Mod}) = indirect_values(a.e)
indirect_values(a::Indirect) = [a.value; reduce(vcat, map(indirect_values, a.index); init = Int[])]
read_values(o::Read) = [o.value; reduce(vcat, map(indirect_values, o.index); init = Int[])]
read_values(o::IndexOperand) = indirect_values(o.expr)
read_values(o::Literal) = Int[]

# The values each value reads, directly or through others.
function ancestors(graph)
    up = [Set{Int}() for _ in graph.values]
    for (id, v) in enumerate(graph.values), o in operands(v.definition), r in read_values(o)
        push!(up[id], r)
        union!(up[id], up[r])
    end
    up
end

# Try to add a reduction or an output to a group: its term must stay inside the
# group's domain, and nothing it loads may depend on the group.
function try_add!(graph, stored, up, g::Group, v; reduction)
    val = graph.values[v]
    index = reduction ? identity_index(g.domain) : placement(graph, g, v)
    index === nothing && return false
    candidate = Group(g.kind, g.domain, g.reduced, copy(g.reductions), copy(g.outputs), copy(g.terms), copy(g.phase))
    reduction || push!(candidate.outputs, v)
    t = try
        reduction ? operand_term(graph, stored, candidate, val.definition.operand, index, Domain(g.domain)) :
        value_term(graph, stored, candidate, v, index)
    catch e
        e isa OutsideGroup || rethrow()
        return false
    end
    group = Set(members(g))
    for l in loads(t)
        (l.value in group || !isempty(intersect(up[l.value], group))) && return false
    end
    # A member reads the group's reductions, or something the group reads.
    shared = Set(l.value for m in members(g) for l in loads(g.terms[m]))
    connected = isempty(g.reductions) || !isempty(stats(t)) || any(l -> l.value in shared, loads(t))
    connected || return false
    reduction ? push!(g.reductions, v) : push!(g.outputs, v)
    g.terms[v] = t
    g.phase[v] = maximum(s -> g.phase[s], stats(t); init = 0) + (reduction ? 1 : 0)
    true
end

"""
    plan(graph::IndexGraph) -> Plan

Decide what is stored and form the kernels.
"""
function plan(graph::IndexGraph)
    values = graph.values
    n = length(values)
    readers = zeros(Int, n)
    for v in values, o in operands(v.definition)
        o isa Read && (readers[o.value] += 1)
    end
    results = Set(o.value for o in graph.results if o isa Read)
    for r in results
        readers[r] += 1
    end
    # Products and opaque operations read their operands from storage.
    materialized = falses(n)
    for v in values
        v.definition isa Union{Contraction, Opaque} || continue
        foreach(o -> materialized[o.value] = true, operands(v.definition))
    end
    stored = map(eachindex(values)) do id
        d = values[id].definition
        d isa Pointwise ? (readers[id] != 1 || id in results || materialized[id]) : true
    end
    # A pointwise value that reads nothing stored is cheaper to recompute.
    for id in eachindex(values)
        d = values[id].definition
        d isa Pointwise && stored[id] && !(id in results) && !materialized[id] || continue
        all(o -> !(o isa Read) || !stored[o.value] && values[o.value].definition isa Pointwise, d.operands) || continue
        all(o -> o isa Literal || o isa IndexOperand, d.operands) && (stored[id] = false)
    end
    live = falses(n)
    for o in graph.results, r in read_values(o)
        live[r] = true
    end
    for id in n:-1:1
        live[id] || continue
        for o in operands(values[id].definition), r in read_values(o)
            live[r] = true
        end
    end
    up = ancestors(graph)
    owner = zeros(Int, n)
    groups = Group[]
    for id in 1:n
        live[id] && owner[id] == 0 || continue
        d = values[id].definition
        d isa Reduction || continue
        g = Group(:reduction, d.iteration, d.reduced)
        try_add!(graph, stored, up, g, id; reduction = true)
        changed = true
        while changed
            changed = false
            for other in (id + 1):n
                live[other] && owner[other] == 0 && !(other in members(g)) || continue
                od = values[other].definition
                if od isa Reduction && od.iteration == g.domain && od.reduced == g.reduced
                    changed |= try_add!(graph, stored, up, g, other; reduction = true)
                elseif od isa Pointwise && stored[other]
                    changed |= try_add!(graph, stored, up, g, other; reduction = false)
                end
            end
        end
        push!(groups, g)
        foreach(m -> owner[m] = length(groups), members(g))
    end
    for id in 1:n
        live[id] && owner[id] == 0 && stored[id] || continue
        d = values[id].definition
        d isa Union{Input, ConstantArray} && continue
        g = if d isa Pointwise
            g = Group(:pointwise, values[id].dims)
            push!(g.outputs, id)
            g.terms[id] = value_term(graph, stored, g, id, identity_index(values[id].dims))
            g.phase[id] = 0
            g
        elseif d isa Contraction
            g = Group(:contraction, d.iteration)
            push!(g.outputs, id)
            g
        else
            g = Group(:opaque, values[id].dims)
            push!(g.outputs, id)
            g
        end
        push!(groups, g)
        owner[id] = length(groups)
    end
    # A value only its own kernel reads need not be written.
    external = falses(n)
    for o in graph.results, r in read_values(o)
        external[r] = true
    end
    for (k, g) in enumerate(groups), r in group_reads(graph, g)
        owner[r] != k && (external[r] = true)
    end
    for g in groups
        g.kind === :reduction || continue
        for m in members(g)
            external[m] && continue
            stored[m] = false
            filter!(!=(m), g.outputs)
        end
    end
    order = order_groups(graph, groups, owner)
    position = invperm(order)
    Plan(graph, stored, groups[order], map(o -> o == 0 ? 0 : position[o], owner))
end

# The values a group reads from storage.
function group_reads(graph, g::Group)
    reads = Set{Int}()
    for v in members(g)
        if haskey(g.terms, v)
            for l in loads(g.terms[v])
                push!(reads, l.value)
                foreach(e -> union!(reads, indirect_values(e)), l.index)
            end
            for t in subterms((g.terms[v],))
                t isa Ix && union!(reads, indirect_values(t.expr))
            end
        else
            for o in operands(graph.values[v].definition)
                union!(reads, read_values(o))
            end
        end
    end
    reads
end

# Kernels in an order where every stored value is written before it is read.
function order_groups(graph, groups, owner)
    needs = map(eachindex(groups)) do k
        deps = Set(owner[r] for r in group_reads(graph, groups[k]) if owner[r] > 0)
        delete!(deps, k)
        deps
    end
    order = Int[]
    placed = falses(length(groups))
    while length(order) < length(groups)
        progress = false
        for k in eachindex(groups)
            placed[k] && continue
            all(d -> placed[d], needs[k]) || continue
            push!(order, k)
            placed[k] = true
            progress = true
        end
        progress || error("kernels depend on each other in a cycle")
    end
    order
end
