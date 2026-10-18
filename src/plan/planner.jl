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

# Where a group keeps a value it computes, as the value's index over the
# domain: a fold of the group at the variables of the axes it keeps, any other
# value over the whole domain or at the positions of the primary folds; or
# nothing.
function placement(graph, g::Group, v)
    val = graph.values[v]
    # A result of the group's own opaque work, at its formula.
    d = val.definition
    g.kind === :opaque && d isa Opaque && return opaque_dims(d) == g.domain ? d.index : nothing
    axes = v in g.reductions ? fold_axes(graph, v) : val.dims == g.domain ? () : primary_axes(graph, g)
    kept = Tuple(k for k in eachindex(g.domain) if !(k in axes))
    map(k -> g.domain[k], kept) == val.dims ? map(var, kept) : nothing
end

"The iteration index of fold `d` read at `index`, its value's index, with `inner[j]` along its `j`th reduced axis."
iteration_index(d::Reduction, index, inner) = ntuple(length(d.iteration)) do k
    j = findfirst(==(k), d.reduced)
    j === nothing ? index[k - count(<(k), d.reduced)] : inner[j]
end

struct OutsideGroup <: Exception end

"""
The term computing value `v` at `index`, an index over `domain`: the group's
domain, or inside a recomputed fold that domain extended by the fold's axes.
"""
function value_term(graph, stored, g::Group, v, index, domain = Domain(g.domain))
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
        return definition_term(graph, stored, g, d, val, index, domain)
    end
    stored[v] && return Load(v, index)
    d isa Reduction && return fold_term(graph, stored, g, v, index, domain)
    d isa Pointwise || return Load(v, index)
    definition_term(graph, stored, g, d, val, index, domain)
end

function definition_term(graph, stored, g, d::Pointwise, val, index, domain)
    args = map(d.operands) do o
        operand_term(graph, stored, g, o, index, domain)
    end
    Apply(d.f, args, val.eltype)
end

# A fold recomputed where it is read: its operand over the domain extended by
# its reduced axes.
function fold_term(graph, stored, g, v, index, domain)
    d = graph.values[v].definition
    base = length(domain)
    extents = Tuple(d.iteration[k] for k in d.reduced)
    iteration = iteration_index(d, index, ntuple(j -> var(base + j), length(d.reduced)))
    extended = Domain((domain.extents..., extents...))
    Fold(v, index, extents, operand_term(graph, stored, g, d.operand, iteration, extended))
end

operand_term(graph, stored, g, o::Read, index, domain) =
    value_term(graph, stored, g, o.value, map(e -> substitute(e, index, domain), o.index), domain)
operand_term(graph, stored, g, o::Literal, index, domain) = Lit(o.value)
operand_term(graph, stored, g, o::IndexOperand{T}, index, domain) where {T} =
    Ix{T}(substitute(o.expr, index, domain))

# The values an operand reads: its own, and those its formulas gather from.
indirect_values(e::Affine) = reduce(vcat, (indirect_values(a) for (a, _) in e.terms); init = Int[])
indirect_values(::Var) = Int[]
indirect_values(a::Union{Div, Mod}) = indirect_values(a.e)
indirect_values(a::Indirect) = [a.value; reduce(vcat, map(indirect_values, a.index); init = Int[])]
gathered_values(o::Read) = reduce(vcat, map(indirect_values, o.index); init = Int[])
gathered_values(o::IndexOperand) = indirect_values(o.expr)
gathered_values(o::Literal) = Int[]
read_values(o::Read) = [o.value; gathered_values(o)]
read_values(o::Operand) = gathered_values(o)

# The values each value reads, directly or through others.
function ancestors(graph)
    up = [Set{Int}() for _ in graph.values]
    for (id, v) in enumerate(graph.values), o in operands(v.definition), r in read_values(o)
        push!(up[id], r)
        union!(up[id], up[r])
    end
    up
end

"""
The kernels formed so far, the kernel that owns each value (zero for none
yet), each value's ancestors, and what each kernel reads, once asked.
"""
struct Dependencies
    owner::Vector{Int}
    groups::Vector{Group}
    up::Vector{Set{Int}}
    reads::Dict{Int, Set{Int}}
end

Dependencies(owner, groups, up) = Dependencies(owner, groups, up, Dict{Int, Set{Int}}())

# Whether any of the values `reads` depends on a member of `group`: through
# its ancestors, and through the kernels that own any of them, each of which
# runs after everything its members read. A kernel that read such a value and
# joined such a member would make kernels depend on each other in a cycle.
function depends(graph, deps::Dependencies, reads, group)
    pending, seen, kernels = collect(reads), Set{Int}(), Set{Int}()
    while !isempty(pending)
        r = pop!(pending)
        r in seen && continue
        push!(seen, r)
        (r in group || !isdisjoint(deps.up[r], group)) && return true
        for u in Iterators.flatten(((r,), deps.up[r]))
            k = deps.owner[u]
            (k == 0 || k in kernels) && continue
            push!(kernels, k)
            append!(pending, get!(() -> group_reads(graph, deps.groups[k]), deps.reads, k))
        end
    end
    false
end

# Try to add a fold or an output to a group. Its term must stay inside the
# group's domain, nothing it loads may depend on the group, nothing the group
# loads may depend on it, and no member may already load it.
function try_add!(graph, stored, deps::Dependencies, g::Group, v; fold)
    val = graph.values[v]
    # An output covers the domain, or the positions of the primary folds.
    index = fold ? identity_index(g.domain) : placement(graph, g, v)
    index === nothing && return false
    any(m -> v in stored_reads(g.terms[m]), members(g)) && return false
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
    reads = stored_reads(t)
    depends(graph, deps, reads, group) && return false
    # A member reads the group's folds, or something the group reads.
    shared = Set(r for m in members(g) for r in stored_reads(g.terms[m]))
    # Nor may what the group reads from other kernels depend on `v`.
    depends(graph, deps, filter(!in(group), shared), Set((v,))) && return false
    connected = isempty(g.reductions) || !isempty(stats(t)) || any(in(shared), reads)
    connected || return false
    push!(fold ? g.reductions : g.outputs, v)
    g.terms[v] = t
    g.phase[v] = maximum(s -> g.phase[s], stats(t); init = 0) + (fold ? 1 : 0)
    true
end

"""
    plan(graph::IndexGraph; fixed = [], recompute_folds = false, keep = ()) -> Plan

Decide what is stored and form the kernels. Each set of values in `fixed`, a
candidate an implementation proposed, forms a group of its own, its folds
stored; the planner groups the rest around them. With `recompute_folds`, folds
that `recomputed_folds` selects, apart from those in `keep` and `fixed`, are
recomputed where they are read instead of stored.
"""
function plan(graph::IndexGraph; fixed = Vector{Int}[], recompute_folds = false, keep = ())
    graph = with_stored_results(graph)
    values = graph.values
    n = length(values)
    # What the results read, directly or through others: nothing else runs.
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
    # Reads of a value, one per running reader and formula: a reader that
    # reads it several times at one index computes it once.
    counts = zeros(Int, n)
    for (id, v) in enumerate(values), o in unique(operands(v.definition))
        live[id] && o isa Read && (counts[o.value] += 1)
    end
    results = Set(o.value for o in graph.results if o isa Read)
    for r in results
        counts[r] += 1
    end
    # An opaque operation runs when any of its results is read (a call's
    # results run together), and reads its operands from storage, as a gather
    # reads its indices.
    running = Set(instance(graph, id) for id in 1:n if live[id] && values[id].definition isa Opaque)
    materialized = falses(n)
    for id in 1:n
        live[id] || continue
        for o in operands(values[id].definition)
            foreach(r -> materialized[r] = true, gathered_values(o))
        end
        values[id].definition isa Opaque && instance(graph, id) in running || continue
        foreach(o -> o isa Read && (materialized[o.value] = true), operands(values[id].definition))
    end
    recomputed = recompute_folds ? recomputed_folds(graph, results, materialized, union(Set{Int}(keep), fixed...)) : Set{Int}()
    # Maps of recomputed folds are recomputed with them.
    after = falses(n)
    for id in 1:n
        d = values[id].definition
        d isa Pointwise || continue
        after[id] = any(o -> o isa Read && (o.value in recomputed || after[o.value]), d.operands)
    end
    # A result of an opaque operation that nobody reads is not stored: its
    # kernel leaves it unwritten.
    # Work nothing reads, such as a call's result whose reference a rewrite
    # replaced, is not stored.
    stored = map(eachindex(values)) do id
        d = values[id].definition
        id in results || materialized[id] || !(id in recomputed || after[id]) || return false
        d isa Union{Pointwise, Reduction, Opaque} && !live[id] && return false
        d isa Pointwise ? (counts[id] != 1 || id in results || materialized[id]) : true
    end
    # A pointwise value that reads nothing stored is cheaper to recompute.
    for id in eachindex(values)
        d = values[id].definition
        d isa Pointwise && stored[id] && !(id in results) && !materialized[id] || continue
        all(o -> !(o isa Read) || !stored[o.value] && values[o.value].definition isa Pointwise, d.operands) || continue
        all(o -> o isa Literal || o isa IndexOperand, d.operands) && (stored[id] = false)
    end
    # A pointwise value read only by one group over its domain, at the same
    # index, is recomputed by that group: grouped again without it.
    p = group_values(graph, copy(stored), live, results, running, fixed)
    users = readers(graph)
    shared = filter(id -> !any(s -> id in s, fixed) && recomputable(p, users, id, live, results, materialized), 1:n)
    isempty(shared) && return p
    stored[shared] .= false
    group_values(graph, stored, live, results, running, fixed)
end

# An opaque operation's instance: a call's results run together.
instance(graph, id) = (op = graph.values[id].definition.node.op; op isa Call ? op.instance : -id)

# Whether stored pointwise value `id` is read only by the members of one group
# over its domain, each reading it at the domain's index.
function recomputable(p::Plan, users, id, live, results, materialized)
    v = p.graph.values[id]
    v.definition isa Pointwise && p.stored[id] && live[id] && !(id in results) && !materialized[id] || return false
    readers = filter(u -> live[u], users[id])
    k = isempty(readers) ? 0 : p.owner[first(readers)]
    k != 0 && k != p.owner[id] && p.groups[k].kind in (:reduction, :pointwise) && p.groups[k].domain == v.dims || return false
    all(readers) do u
        p.owner[u] == k && all(operands(p.graph.values[u].definition)) do o
            !(id in read_values(o)) || o isa Read && o.value == id && o.index == identity_index(v.dims)
        end
    end
end

# The kernels, given what is stored.
function group_values(graph, stored, live, results, running, fixed)
    values = graph.values
    n = length(values)
    up = ancestors(graph)
    owner = zeros(Int, n)
    groups = Group[]
    deps = Dependencies(owner, groups, up)
    # Opaque work first, one group per operation over its first result's
    # domain (all results of a claimed call together, in order), which may
    # absorb the pointwise work after a result as an epilogue its kernel
    # applies over the result's output tile.
    output(id) = (op = values[id].definition.node.op; op isa Call ? op.output : 1)
    instances = Dict{Int, Int}()
    for id in sort([id for id in 1:n if values[id].definition isa Opaque && instance(graph, id) in running]; by = output)
        k = get!(instances, instance(graph, id)) do
            push!(groups, Group(:opaque, opaque_dims(values[id].definition)))
            length(groups)
        end
        push!(groups[k].outputs, id)
        owner[id] = k
    end
    users = map(u -> unique(filter(i -> live[i], u)), readers(graph))
    for (k, g) in enumerate(groups), o in copy(g.outputs)
        values[o].definition.epilogue && absorb_epilogue!(graph, stored, users, results, owner, g, k, o)
    end
    for set in fixed
        push!(groups, fixed_group(graph, stored, deps, set))
        foreach(m -> owner[m] = length(groups), set)
    end
    for id in 1:n
        live[id] && owner[id] == 0 || continue
        d = values[id].definition
        d isa Reduction && stored[id] || continue
        g = Group(:reduction, d.iteration)
        try_add!(graph, stored, deps, g, id; fold = true)
        changed = true
        while changed
            changed = false
            for other in (id + 1):n
                live[other] && owner[other] == 0 && !(other in members(g)) || continue
                od = values[other].definition
                if od isa Reduction && od.iteration == g.domain && stored[other]
                    changed |= try_add!(graph, stored, deps, g, other; fold = true)
                elseif od isa Pointwise && stored[other]
                    changed |= try_add!(graph, stored, deps, g, other; fold = false)
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
        try_add!(graph, stored, deps, g, id; fold = false)
        for other in (id + 1):n
            live[other] && owner[other] == 0 && stored[other] || continue
            values[other].definition isa Pointwise && values[other].dims == g.domain || continue
            any(m -> m in up[other], members(g)) && try_add!(graph, stored, deps, g, other; fold = false)
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
# stored pointwise values over it. A candidate spanning several domains waits
# for groups that span them: a kernel is maps and folds over one domain.
function fixed_group(graph, stored, deps, set)
    values = graph.values
    all(v -> deps.owner[v] == 0, set) || throw(NotAGroup(set))
    d = values[first(set)].definition
    g = d isa Reduction ? Group(:reduction, d.iteration) : Group(:pointwise, values[first(set)].dims)
    for v in set
        dv = values[v].definition
        added = dv isa Reduction ? dv.iteration == g.domain && try_add!(graph, stored, deps, g, v; fold = true) :
            dv isa Pointwise && stored[v] && try_add!(graph, stored, deps, g, v; fold = false)
        added || throw(NotAGroup(set))
    end
    g
end

# A fold recomputed where it is read saves its storage and a kernel. Selected
# when every value it reaches through maps is a fold over another domain (a
# fold over the same domain joins its group as a later phase instead), each
# reading fold computes it at most once per block (it moves along every kept
# axis of the reader wider than a tile), and either the fold is a product over
# few elements, recomputed as one product per tile, or no reading fold reads
# it at positions that move along its own reduced axes, so a kernel computes
# it once per block, before or during its sweep. A fold that recomputes one
# selected before it is selected only in the last case: inside a reader's
# sweep, the folds it recomputes would be recomputed at every step, for every
# position they do not depend on. Results, the operands of opaque work, the
# indices of gathers and the folds `kept` stay stored.
const RECOMPUTED_PRODUCT = 256
const RECOMPUTED_SPAN = 128

function recomputed_folds(graph, results, materialized, kept = Set{Int}())
    values = graph.values
    users = readers(graph)
    selected = Set{Int}()
    for v in eachindex(values)
        d = values[v].definition
        d isa Reduction && !(v in results) && !materialized[v] && !(v in kept) || continue
        # The folds v reaches through maps, with the index at which each reads v.
        consumers = Tuple{Int, Tuple{Vararg{Affine}}}[]
        pending = [(u, v, identity_index(values[v].dims)) for u in unique(users[v])]
        ok = true
        while ok && !isempty(pending)
            u, w, index = pop!(pending)
            du = values[u].definition
            domain = iteration_domain(values[u])
            for o in reads(du)
                o.value == w || continue
                at = map(e -> substitute(e, o.index, domain), index)
                if du isa Reduction
                    push!(consumers, (u, at))
                elseif du isa Pointwise && !(u in results) && !materialized[u]
                    append!(pending, [(x, u, at) for x in unique(users[u])])
                else
                    ok = false
                end
            end
        end
        ok && !isempty(consumers) || continue
        all(consumers) do (u, at)
            du = values[u].definition
            du.iteration != d.iteration &&
                all(k -> k in du.reduced || du.iteration[k] <= RECOMPUTED_SPAN || any(e -> mentions(e, k), at), eachindex(du.iteration))
        end || continue
        o = d.operand
        product = prod(k -> d.iteration[k], d.reduced; init = 1) <= RECOMPUTED_PRODUCT && o isa Read &&
            values[o.value].definition isa Pointwise{typeof(*)} && length(values[o.value].definition.operands) == 2
        hoisted = all(((u, at),) -> !any(k -> any(e -> mentions(e, k), at), values[u].definition.reduced), consumers)
        hoisted || product && !recomputes(values, v, selected, results, materialized) || continue
        push!(selected, v)
    end
    selected
end

# Whether fold `v` reads a fold of `selected` through maps that stay unstored.
function recomputes(values, v, selected, results, materialized)
    pending, seen = [v], Set([v])
    while !isempty(pending)
        for o in reads(values[pop!(pending)].definition)
            u = o.value
            u in selected && return true
            values[u].definition isa Pointwise && !(u in results) && !materialized[u] && !(u in seen) || continue
            push!(seen, u)
            push!(pending, u)
        end
    end
    false
end

# An epilogue: the chain of pointwise values after an opaque result, each read
# only by the next, over the result's domain (`opaque_dims`), whose other reads
# align with the result's tile (see `aligned_axes`).
# Limits: pointwise work only, aligned reads only, no positions; the last value
# of the chain is written in the result's place.
function absorb_epilogue!(graph, stored, users, results, owner, g::Group, k, o)
    values = graph.values
    D = opaque_dims(values[o].definition)
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

aligned(graph, t, D) =
    all(s -> !(s isa Union{Ix, Fold}) && !(s isa Load && aligned_axes(graph, s, D) === nothing), subterms((t,)))

"""
    aligned_axes(graph, load, domain) -> axes or nothing

The axes of `domain` along which a load reads, one per axis of the loaded
value, when each axis of the value is read at one variable of the domain of
the same extent, in the domain's order: the value is the domain's tile with
the other axes broadcast.
"""
function aligned_axes(graph, s::Load, D)
    dims = graph.values[s.value].dims
    axes = Int[]
    for (k, e) in enumerate(s.index)
        e.constant == 0 && length(e.terms) == 1 && e.terms[1].first isa Var && e.terms[1].second == 1 || return nothing
        n = e.terms[1].first.n
        D[n] == dims[k] && (isempty(axes) || n > last(axes)) || return nothing
        push!(axes, n)
    end
    axes
end

# Every result becomes a value read whole, so results are buffers: a result
# that is a layout of a value, or a constant, gets a value of its own.
function with_stored_results(graph::IndexGraph)
    values = copy(graph.values)
    results = copy(graph.results)
    for (k, r) in enumerate(results)
        dims = graph.result_dims[k]
        # A value read whole, or a reshape of it, presents its storage as it is.
        r isa Read && values[r.value].scalar == graph.result_scalar[k] && reads_storage(values[r.value], r.index, dims) && continue
        push!(values, Value(narrow(Pointwise(identity, (r,)), dims), map(k -> dims[k], wide_axes(dims)),
            graph.result_eltype[k], graph.result_scalar[k]))
        results[k] = Read(length(values), wide_index(dims))
    end
    IndexGraph(values, graph.arguments, results, graph.result_dims, graph.result_scalar, graph.result_eltype, graph.calls, graph.rewrites)
end

# The values a group reads from storage, its own excepted.
function group_reads(graph, g::Group)
    reads = Set{Int}()
    for v in members(g)
        if haskey(g.terms, v)
            union!(reads, stored_reads(g.terms[v]))
        else
            for o in operands(graph.values[v].definition)
                union!(reads, read_values(o))
            end
        end
    end
    setdiff!(reads, members(g))
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
