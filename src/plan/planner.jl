# The decision procedure: which values are stored, which are recomputed inside
# their readers, and how the work groups into kernels.
#
# A kernel computes a group: maps and folds over one shared index domain. Each
# fold reduces its own set of the domain's axes; a fold's value sits at the
# domain's index with its reduced axes at zero. Folds over the group's first
# set of axes (its primary axes) can be read by later members at that
# position: a schedule completes them before what follows (phases). A fold
# over other axes is completed by merging partial states, so within the group
# nothing reads it; its operand, often a value the group already computes, is
# never written. Which axes a schedule sweeps, and which folds it completes in
# a block and which by merging partial states, is the schedule's choice.

struct Group
    kind::Symbol # :reduction, :pointwise or :opaque
    domain::Tuple{Vararg{Int}}
    reductions::Vector{Int}
    outputs::Vector{Int}
    terms::Dict{Int, Term}
    phase::Dict{Int, Int}
    # An opaque group's epilogues: a result, and the value its kernel writes
    # in its place, computed by `terms` from the result's tile (`Stat`).
    epilogues::Dict{Int, Int}
end

Group(kind, domain) = Group(kind, domain, Int[], Int[], Dict{Int, Term}(), Dict{Int, Int}(), Dict{Int, Int}())

members(g::Group) = [g.reductions; g.outputs]

"The axes a fold reduces."
fold_axes(graph, r) = graph.values[r].definition.reduced

"The axes of the group's first fold: folds over them can be read within the group."
primary_axes(graph, g::Group) = isempty(g.reductions) ? () : fold_axes(graph, first(g.reductions))

kept_dims(domain, axes) = ntuple(k -> k in axes ? 1 : domain[k], length(domain))

"Whether fold `r` reduces the group's primary axes."
primary(graph, g::Group, r) = fold_axes(graph, r) == primary_axes(graph, g)

struct Plan
    graph::IndexGraph
    stored::Vector{Bool}
    groups::Vector{Group}
    owner::Vector{Int}
end

# Where a group keeps a value it computes: at the domain's index with each axis
# the value lacks (extent one) at zero. A fold lacks its reduced axes.
function placement(graph, g::Group, v)
    val = graph.values[v]
    (val.scalar || isempty(val.dims)) && return ()
    # A value with fewer axes than the domain has the missing trailing ones at
    # extent one, as a product's value lacks its summed axes.
    n = length(val.dims)
    n <= length(g.domain) || return nothing
    padded = ntuple(k -> k <= n ? val.dims[k] : 1, length(g.domain))
    all(k -> padded[k] == g.domain[k] || padded[k] == 1, eachindex(padded)) || return nothing
    identity_index(padded)[1:n]
end

struct OutsideGroup <: Exception end

"The term computing value `v` at `index`, an index over the group's domain."
function value_term(graph, stored, g::Group, v, index)
    val = graph.values[v]
    d = val.definition
    if v in g.reductions
        primary(graph, g, v) && index == placement(graph, g, v) || throw(OutsideGroup())
        return Stat(v)
    end
    if v in g.outputs
        index == placement(graph, g, v) || throw(OutsideGroup())
        # A value the kernel computes itself, such as a claimed call's
        # result, is read as the kernel's tile of it.
        d isa Opaque && return Stat(v)
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

# Try to add a fold or an output to a group. Its term must stay inside the
# group's domain, nothing it loads may depend on the group, and no member may
# already load it.
function try_add!(graph, stored, up, g::Group, v; fold)
    val = graph.values[v]
    index = fold ? identity_index(g.domain) : placement(graph, g, v)
    index === nothing && return false
    # An output covers the domain, or the positions of the primary folds
    # (missing trailing axes read as extent one).
    padded = ntuple(k -> get(val.dims, k, 1), length(g.domain))
    fold || padded == g.domain || padded == kept_dims(g.domain, primary_axes(graph, g)) || return false
    any(m -> any(l -> l.value == v, loads(g.terms[m])), members(g)) && return false
    candidate = Group(g.kind, g.domain, copy(g.reductions), copy(g.outputs), copy(g.terms), copy(g.phase), copy(g.epilogues))
    fold || push!(candidate.outputs, v)
    t = try
        fold ? operand_term(graph, stored, candidate, val.definition.operand, index, Domain(g.domain)) :
        value_term(graph, stored, candidate, v, index)
    catch e
        e isa OutsideGroup || rethrow()
        return false
    end
    group = Set(members(g))
    for l in loads(t)
        (l.value in group || !isempty(intersect(up[l.value], group))) && return false
    end
    # A member reads the group's folds, or something the group reads.
    shared = Set(l.value for m in members(g) for l in loads(g.terms[m]))
    connected = isempty(g.reductions) || !isempty(stats(t)) || any(l -> l.value in shared, loads(t))
    connected || return false
    push!(fold ? g.reductions : g.outputs, v)
    g.terms[v] = t
    g.phase[v] = maximum(s -> g.phase[s], stats(t); init = 0) + (fold ? 1 : 0)
    true
end

"""
    plan(graph::IndexGraph; fixed = []) -> Plan

Decide what is stored and form the kernels. Each set of values in `fixed`, a
candidate an implementation proposed, forms a group of its own, its folds
stored; the planner groups the rest around them.
"""
function plan(graph::IndexGraph; fixed = Vector{Int}[])
    graph = with_stored_results(graph)
    values = graph.values
    n = length(values)
    # Reads of a value, one per reader and formula: a reader that reads it
    # several times at one index computes it once.
    counts = zeros(Int, n)
    for (id, v) in enumerate(values), o in unique(operands(v.definition))
        o isa Read && (counts[o.value] += 1)
    end
    results = Set(o.value for o in graph.results if o isa Read)
    for r in results
        counts[r] += 1
    end
    # Opaque operations read their operands from storage.
    materialized = falses(n)
    for v in values
        v.definition isa Opaque || continue
        foreach(o -> materialized[o.value] = true, operands(v.definition))
    end
    stored = map(eachindex(values)) do id
        d = values[id].definition
        d isa Pointwise ? (counts[id] != 1 || id in results || materialized[id]) : true
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
    # Opaque work first, one group per operation (all results of a claimed
    # call together), which may absorb the pointwise work after a result as an
    # epilogue its kernel applies over the result's output tile.
    instances = Dict{Int, Int}()
    for id in 1:n
        d = values[id].definition
        d isa Opaque || continue
        key = d.node.op isa Call ? d.node.op.instance : -id
        k = get!(instances, key) do
            push!(groups, Group(:opaque, values[id].dims))
            length(groups)
        end
        push!(groups[k].outputs, id)
        owner[id] = k
    end
    users = map(unique, readers(graph))
    for (k, g) in enumerate(groups), o in copy(g.outputs)
        values[o].definition.epilogue && absorb_epilogue!(graph, stored, users, results, owner, g, k, o)
    end
    for set in fixed
        push!(groups, fixed_group(graph, stored, up, owner, set))
        foreach(m -> owner[m] = length(groups), set)
    end
    for id in 1:n
        live[id] && owner[id] == 0 || continue
        d = values[id].definition
        d isa Reduction || continue
        g = Group(:reduction, d.iteration)
        try_add!(graph, stored, up, g, id; fold = true)
        changed = true
        while changed
            changed = false
            for other in (id + 1):n
                live[other] && owner[other] == 0 && !(other in members(g)) || continue
                od = values[other].definition
                if od isa Reduction && od.iteration == g.domain
                    changed |= try_add!(graph, stored, up, g, other; fold = true)
                elseif od isa Pointwise && stored[other]
                    changed |= try_add!(graph, stored, up, g, other; fold = false)
                end
            end
        end
        push!(groups, g)
        foreach(m -> owner[m] = length(groups), members(g))
    end
    # Stored pointwise values left: a group from each, joined by the later
    # ones over its domain that read it.
    for id in 1:n
        live[id] && owner[id] == 0 && stored[id] || continue
        d = values[id].definition
        d isa Union{Input, ConstantArray} && continue
        d isa Pointwise || error("value %$id has no kernel")
        g = Group(:pointwise, values[id].dims)
        try_add!(graph, stored, up, g, id; fold = false)
        for other in (id + 1):n
            live[other] && owner[other] == 0 && stored[other] || continue
            values[other].definition isa Pointwise && values[other].dims == g.domain || continue
            any(m -> m in up[other], members(g)) && try_add!(graph, stored, up, g, other; fold = false)
        end
        push!(groups, g)
        foreach(m -> owner[m] = length(groups), members(g))
    end
    # A value only its own kernel reads need not be written.
    external = falses(n)
    for o in graph.results, r in read_values(o)
        external[r] = true
    end
    for (k, g) in enumerate(groups), r in group_reads(graph, g)
        owner[r] != k && (external[r] = true)
    end
    # The folds a fold over other than the primary axes reads stay stored: a
    # realization that completes it in a launch of its own loads them.
    for g in groups, r in g.reductions, s in stats(g.terms[r])
        primary(graph, g, r) || (external[s] = true)
    end
    for g in groups
        g.kind in (:reduction, :pointwise) || continue
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

"A candidate whose values do not form one group."
struct NotAGroup <: Exception
    values::Vector{Int}
end

Base.showerror(io::IO, e::NotAGroup) = print(io, "the values ", join(map(v -> "%$v", e.values), " "), " do not form one group")

# The group of exactly the values in `set`: its folds over one domain, then
# stored pointwise values over it.
function fixed_group(graph, stored, up, owner, set)
    values = graph.values
    all(v -> owner[v] == 0, set) || throw(NotAGroup(set))
    d = values[first(set)].definition
    g = d isa Reduction ? Group(:reduction, d.iteration) : Group(:pointwise, values[first(set)].dims)
    for v in set
        dv = values[v].definition
        added = dv isa Reduction ? dv.iteration == g.domain && try_add!(graph, stored, up, g, v; fold = true) :
            dv isa Pointwise && stored[v] && try_add!(graph, stored, up, g, v; fold = false)
        added || throw(NotAGroup(set))
    end
    g
end

# An epilogue: the chain of pointwise values after an opaque result, each read
# only by the next, over the result's domain, whose other reads align with the
# result's tile (the same index, or zero along an axis of extent one).
# Limits: pointwise work only, aligned reads only, no positions; the last value
# of the chain is written in the result's place.
function absorb_epilogue!(graph, stored, users, results, owner, g::Group, k, o)
    values = graph.values
    D = values[o].dims
    chain = Int[]
    current, final, final_term = o, nothing, nothing
    while !(current in results) && length(users[current]) == 1
        c = only(users[current])
        values[c].definition isa Pointwise && values[c].dims == D && owner[c] == 0 || break
        trial = Group(:opaque, D)
        push!(trial.outputs, o)
        append!(trial.outputs, chain)
        push!(trial.outputs, c)
        t = try
            value_term(graph, stored, trial, c, identity_index(D))
        catch e
            e isa OutsideGroup || rethrow()
            break
        end
        aligned(graph, t, D) || break
        push!(chain, c)
        current, final, final_term = c, c, t
    end
    final === nothing && return
    stored[o] = false
    foreach(c -> stored[c] = false, chain)
    stored[final] = true
    g.epilogues[o] = final
    g.terms[final] = final_term
    push!(g.outputs, final)
    foreach(c -> owner[c] = k, chain)
end

function aligned(graph, t, D)
    whole = identity_index(D)
    for s in subterms((t,))
        s isa Ix && return false
        s isa Load || continue
        Dv = graph.values[s.value].dims
        length(s.index) == length(Dv) <= length(D) || return false
        for k in eachindex(Dv)
            e = s.index[k]
            (Dv[k] == D[k] && e == whole[k]) || (Dv[k] == 1 && e == constant(0)) || return false
        end
    end
    true
end

# Every result becomes a value read whole, so results are buffers: a result
# that is a layout of a value, or a constant, gets a value of its own.
function with_stored_results(graph::IndexGraph)
    values = copy(graph.values)
    results = copy(graph.results)
    for (k, r) in enumerate(results)
        dims = graph.result_dims[k]
        if r isa Read && r.index == identity_index(dims) && values[r.value].dims == dims &&
           values[r.value].scalar == graph.result_scalar[k]
            continue
        end
        # A reshape of a value's dense storage presents that storage as it is.
        if r isa Read && !graph.result_scalar[k] && !values[r.value].scalar &&
           prod(values[r.value].dims; init = 1) == prod(dims; init = 1) &&
           dense_address(values[r.value].dims, r.index, Domain(dims)) == dense_address(dims, identity_index(dims), Domain(dims))
            continue
        end
        push!(values, Value(Pointwise(identity, (r,)), dims, graph.result_eltype[k], graph.result_scalar[k]))
        results[k] = Read(length(values), identity_index(dims))
    end
    IndexGraph(values, graph.arguments, results, graph.result_dims, graph.result_scalar, graph.result_eltype)
end

function dense_address(dims, index, d::Domain)
    total = constant(0)
    stride = 1
    for (e, n) in zip(index, dims)
        total = add(total, scale(e, stride), d)
        stride *= n
    end
    total
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
