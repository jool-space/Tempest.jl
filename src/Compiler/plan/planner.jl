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
#
# A scan's group sweeps the axes it scans along, in its direction: its scans
# alike, and the maps over its domain. A map before a scan is its operand's
# term, as Futhark's `scanomap` fuses one into its scan; a map after reads
# each scan at the position the sweep has reached. A fold along those axes
# rides the sweep, in either direction, since it need not take its elements
# in order; it is complete only after the sweep, so nothing in the group
# reads it.
#
# A fold by key joins a reduction's group over its iteration, as a fold over
# other than the primary axes does: it combines each element at its key, is
# complete only when the group's work is, and nothing in the group reads it.
#
# A fold over fewer positions along axes it keeps may join a reduction's
# group over a domain that holds them, where it shares a fold it recomputes
# with the group's folds, as MLA's dV, over 64 features, shares the scores
# with dK and dQ, over 96 (see `joint_domain`): it is computed over its own
# positions, and nothing in the group reads it.

struct Group
    kind::Symbol # :reduction, :scan or :pointwise
    domain::Tuple{Vararg{Int}}
    reductions::Vector{Int}
    outputs::Vector{Int}
    terms::Dict{Int, Term}
    phase::Dict{Int, Int}
end

Group(kind, domain) = Group(kind, domain, Int[], Int[], Dict{Int, Term}(), Dict{Int, Int}())

members(g::Group) = [g.reductions; g.outputs]

"The axes a fold reduces, or a scan folds along."
fold_axes(graph, r) = folded_axes(graph.values[r].definition)
folded_axes(d::Reduction) = d.reduced
folded_axes(d::Scan) = d.axes
folded_axes(d::Keyed) = Tuple(eachindex(d.iteration))

keyed(graph, v) = graph.values[v].definition isa Keyed

"The axes of the group's first fold not by key: folds over them can be read within the group."
function primary_axes(graph, g::Group)
    k = findfirst(r -> !keyed(graph, r), g.reductions)
    k === nothing ? () : fold_axes(graph, g.reductions[k])
end

kept_dims(domain, axes) = ntuple(k -> k in axes ? 1 : domain[k], length(domain))

"Whether fold `r` reduces the group's primary axes."
primary(graph, g::Group, r) = !keyed(graph, r) && fold_axes(graph, r) == primary_axes(graph, g)

"The values the key of fold by key `v` gathers from; none for any other value."
key_reads(graph, v) = keyed(graph, v) ? reduce(vcat, map(indirect_values, graph.values[v].definition.key); init = Int[]) : Int[]

"""
The values fold by key `v` reads besides its elements: those its key gathers
from, and those of the contents it starts from; none for any other value.
"""
function keyed_reads(graph, v)
    keyed(graph, v) || return Int[]
    into = graph.values[v].definition.into
    [key_reads(graph, v); into === nothing ? Int[] : read_values(into)]
end

"""
A program's plan: which values are stored, and the groups its kernels
compute, in order. A proposal is handed one, and reads its groups through
`group_views`.
"""
struct Plan
    graph::IndexGraph
    stored::Vector{Bool}
    groups::Vector{Group}
    owner::Vector{Int}
end

# Where a group keeps a value it computes, as the value's index over the
# domain: a fold of the group at the variables of the axes it keeps, any other
# value over the whole domain or at the positions of the primary folds, or a
# slice of one of its folds where the fold keeps it; or nothing. A scan's
# group keeps everything but its folds over its whole domain. A fold by key
# keeps each element at its key. A fold over fewer positions than the domain
# (see `restricted`) keeps its own.
function placement(graph, g::Group, v)
    g.kind === :recurrence && return v in g.outputs ? recurrence_places(graph, g.outputs).places[v] : nothing
    val = graph.values[v]
    d = val.definition
    d isa Keyed && return v in g.reductions ? d.key : nothing
    g.kind === :scan && !(d isa Reduction) && return val.dims == g.domain ? identity_index(g.domain) : nothing
    axes = v in g.reductions ? fold_axes(graph, v) : val.dims == g.domain ? () : primary_axes(graph, g)
    kept = Tuple(k for k in eachindex(g.domain) if !(k in axes))
    extents = v in g.reductions && d isa Reduction ? d.iteration : g.domain
    map(k -> extents[k], kept) == val.dims && return map(var, kept)
    v in g.reductions ? nothing : slice_placement(graph, g, v)
end

"""
Whether fold `r` of reduction group `g` covers fewer positions than the
group's domain: its iteration lies within the domain, shorter along axes it
keeps (see `joint_domain`).
"""
restricted(graph, g::Group, r) =
    g.kind === :reduction && r in g.reductions && (d = graph.values[r].definition) isa Reduction && d.iteration != g.domain

# A slice of a fold of the group: the fold read at a variable of the slice
# along some of its axes and at constants along the others, kept where the
# fold keeps those variables. A fold whose term does not vary along the
# constants' axes is the same there as anywhere along them (see `kept_at`),
# as a log-sum-exp is along the features of attention's output.
function slice_placement(graph, g::Group, v)
    d = graph.values[v].definition
    d isa Pointwise && d.f === identity && only(d.operands) isa Read || return nothing
    o = only(d.operands)
    o.value in g.reductions || return nothing
    at = placement(graph, g, o.value)
    at === nothing && return nothing
    n = length(graph.values[v].dims)
    found = map(j -> findall(==(var(j)), o.index), 1:n)
    all(f -> length(f) == 1, found) && all(e -> e in map(var, 1:n) || isempty(e.terms), o.index) || return nothing
    Tuple(map(f -> at[only(f)], found))
end

# Whether fold `v` of the group, read at `index`, is what the group keeps: at
# its placement, or at constants along axes its term does not vary along.
function kept_at(graph, g::Group, v, index)
    at = placement(graph, g, v)
    (at === nothing || length(at) != length(index)) && return false
    all(zip(index, at)) do (e, p)
        e == p && return true
        a = findfirst(k -> p == var(k), eachindex(g.domain))
        a !== nothing && isempty(e.terms) && 0 <= e.constant < g.domain[a] && !varies(graph, g, g.terms[v], a)
    end
end

# Whether term `t` of group `g` varies along axis `a` of its domain: through
# what it reads, and through the folds of the group it reads.
varies(graph, g::Group, t::Term, a) = any(subterms((t,); nested = true)) do s
    s isa Stat && return any(e -> mentions(e, a), something(placement(graph, g, s.value), ()))
    s isa Union{Load, Ix} && mentions(s, a)
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
Each value's term at an index is built once, in `memo`: terms are interned,
so a value read twice is one subterm, and built again at every path to it,
a chain of maps each reading the last twice would take time doubling with
each map.
"""
function value_term(graph, stored, g::Group, v, index, domain = Domain(g.domain), memo = Dict{Any, Term}())
    key = (v, index, domain)
    haskey(memo, key) && return memo[key]
    memo[key] = built_term(graph, stored, g, v, index, domain, memo)
end

function built_term(graph, stored, g::Group, v, index, domain, memo)
    val = graph.values[v]
    d = val.definition
    # A recurrence's group reads what it computes wherever this step stored it.
    g.kind === :recurrence && v in g.outputs && return Load(v, index)
    if v in g.reductions
        g.kind === :scan && d isa Reduction && throw(OutsideGroup())
        # A fold over fewer positions than the domain has no value elsewhere.
        restricted(graph, g, v) && throw(OutsideGroup())
        primary(graph, g, v) && kept_at(graph, g, v, index) || throw(OutsideGroup())
        return Stat(v)
    end
    if v in g.outputs
        index == placement(graph, g, v) || throw(OutsideGroup())
        return definition_term(graph, stored, g, d, val, index, domain, memo)
    end
    stored[v] && return Load(v, index)
    d isa Reduction && return fold_term(graph, stored, g, v, index, domain, memo)
    d isa Pointwise || return Load(v, index)
    definition_term(graph, stored, g, d, val, index, domain, memo)
end

function definition_term(graph, stored, g, d::Pointwise, val, index, domain, memo = Dict{Any, Term}())
    args = map(d.operands) do o
        operand_term(graph, stored, g, o, index, domain, memo)
    end
    Apply(d.f, args, val.eltype)
end

# A fold recomputed where it is read: its operand over the domain extended by
# its reduced axes.
function fold_term(graph, stored, g, v, index, domain, memo = Dict{Any, Term}())
    d = graph.values[v].definition::Reduction
    base = length(domain)
    extents = Tuple(d.iteration[k] for k in d.reduced)
    iteration = iteration_index(d, index, ntuple(j -> var(base + j), length(d.reduced)))
    extended = Domain((domain.extents..., extents...))
    Recomputed(v, index, extents, operand_term(graph, stored, g, d.operand, iteration, extended, memo))
end

# The values `value_term` and `operand_term` would load for `v` or `o`, the
# term's `stored_reads`, added to `reads`: found by the same rules from the
# graph alone, each value visited once where the term repeats it at every
# reader. Its formulas' gathers are the same at any index.
function value_reads!(reads, seen, graph, stored, g::Group, v)
    v in seen && return reads
    push!(seen, v)
    d = graph.values[v].definition
    g.kind === :recurrence && v in g.outputs && return push!(reads, v)
    v in g.reductions && return reads
    v in g.outputs && return foreach(o -> operand_reads!(reads, seen, graph, stored, g, o), operands(d))
    stored[v] && return push!(reads, v)
    d isa Reduction && return operand_reads!(reads, seen, graph, stored, g, d.operand)
    d isa Pointwise || return push!(reads, v)
    foreach(o -> operand_reads!(reads, seen, graph, stored, g, o), operands(d))
end

function operand_reads!(reads, seen, graph, stored, g, o)
    union!(reads, gathered_values(o))
    o isa Read && value_reads!(reads, seen, graph, stored, g, o.value)
    reads
end

operand_term(graph, stored, g, o::Read, index, domain, memo = Dict{Any, Term}()) =
    value_term(graph, stored, g, o.value, map(e -> substitute(e, index, domain), o.index), domain, memo)
operand_term(graph, stored, g, o::Literal, index, domain, memo = nothing) = Lit(o.value)
operand_term(graph, stored, g, o::IndexOperand{T}, index, domain, memo = nothing) where {T} =
    Ix{T}(substitute(o.expr, index, domain))

# An operand read from storage, as a term over its reader's domain: the
# contents a fold by key starts from.
stored_term(o::Read) = Load(o.value, o.index)
stored_term(o::Literal) = Lit(o.value)
stored_term(o::IndexOperand{T}) where {T} = Ix{T}(o.expr)

# The values an operand reads: its own, and those its formulas gather from.
gathered_values(o::Read) = reduce(vcat, map(indirect_values, o.index); init = Int[])
gathered_values(o::IndexOperand) = indirect_values(o.expr)
gathered_values(o::Literal) = Int[]
read_values(o::Read) = [o.value; gathered_values(o)]
read_values(o::Operand) = gathered_values(o)

# The values each value reads, directly or through others, as a mask over
# the values. A recurrence's state reads, after its first step, the value it
# is carried from and what that reads, though it comes after: a kernel that
# reads the state runs after the recurrence's, which reads all of it. As
# values after the state read it, the reads are taken again until no value
# reads more.
function ancestors(graph)
    n = length(graph.values)
    up = [falses(n) for _ in 1:n]
    carried = [(c, v.definition.from.value) for (c, v) in enumerate(graph.values) if v.definition isa Carried && v.definition.from isa Read]
    grown = true
    while grown
        grown = false
        for (id, v) in enumerate(graph.values), o in operands(v.definition), r in read_values(o)
            up[id][r] = true
            up[id] .|= up[r]
        end
        for (c, f) in carried
            before = count(up[c])
            up[c][f] = true
            up[c] .|= up[f]
            grown |= count(up[c]) > before
        end
    end
    up
end

"""
The kernels formed so far, the kernel that owns each value (zero for none
yet), each value's ancestors, what each kernel reads, once asked, and what
each member's term loads, once asked: a group tries many candidates against
the same members.
"""
struct Dependencies
    owner::Vector{Int}
    groups::Vector{Group}
    up::Vector{BitVector}
    reads::Dict{Int, Set{Int}}
    # By the term's own hash: an identity dictionary hashes an immutable term
    # by its contents, all of it, at every lookup.
    loads::Dict{Term, Vector{Int}}
end

Dependencies(owner, groups, up) = Dependencies(owner, groups, up, Dict{Int, Set{Int}}(), Dict{Term, Vector{Int}}())

term_loads(deps::Dependencies, t::Term) = get!(() -> stored_reads(t), deps.loads, t)

# Whether any of the values `reads` depends on a member of `group`: through
# its ancestors, and through the kernels that own any of them, each of which
# runs after everything its members read. A kernel that read such a value and
# joined such a member would make kernels depend on each other in a cycle.
# A value among the ancestors of one already looked at is passed over: its
# ancestors are among them, and their kernels were looked at.
# The latest value pending is looked at first: values are numbered after
# their operands, so its ancestors cover most of the others.
function depends(graph, deps::Dependencies, reads, group)
    n = length(deps.owner)
    pending, covered, kernels = falses(n), falses(n), Set{Int}()
    top = 0
    for r in reads
        pending[r] = true
        top = max(top, r)
    end
    while top > 0 && (r = findprev(pending, top)) !== nothing
        pending[r] = false
        top = r
        covered[r] && continue
        any(m -> m == r || deps.up[r][m], group) && return true
        fresh = deps.up[r] .& .!covered
        fresh[r] = true
        covered .|= fresh
        for u in findall(fresh)
            k = deps.owner[u]
            (k == 0 || k in kernels) && continue
            push!(kernels, k)
            for x in get!(() -> group_reads(graph, deps.groups[k]), deps.reads, k)
                covered[x] || (pending[x] = true; top = max(top, x))
            end
        end
    end
    false
end

# Try to add a fold or an output to a group. Its term must stay inside the
# group's domain, nothing it loads may depend on the group, nothing the group
# loads may depend on it, and no member may already load it. With
# `recomputing`, a fold joins a group whose folds would then cover different
# positions only where it shares what they recompute (see `mixes`).
function try_add!(graph, stored, deps::Dependencies, g::Group, v; fold, recomputing = true)
    val = graph.values[v]
    # An output covers the domain, or the positions of the primary folds.
    index = fold ? identity_index(g.domain) : placement(graph, g, v)
    index === nothing && return false
    any(m -> v in term_loads(deps, g.terms[m]) || v in keyed_reads(graph, m), members(g)) && return false
    any(in(members(g)), keyed_reads(graph, v)) && return false
    candidate = fold ? g : Group(g.kind, g.domain, g.reductions, [g.outputs; v], g.terms, g.phase)
    # What the term would load, from the graph before the term is built: most
    # candidates fail here, and a term inlines every value recomputed in it.
    loads = Set{Int}()
    fold ? operand_reads!(loads, Set{Int}(), graph, stored, candidate, val.definition.operand) :
        value_reads!(loads, Set{Int}(), graph, stored, candidate, v)
    reads = [collect(loads); keyed_reads(graph, v)]
    group = Set(members(g))
    depends(graph, deps, reads, group) && return false
    t = try
        fold ? operand_term(graph, stored, candidate, val.definition.operand, index, Domain(g.domain)) :
        value_term(graph, stored, candidate, v, index)
    catch e
        e isa OutsideGroup || rethrow()
        return false
    end
    # A member reads the group's folds, or something the group reads.
    shared = Set(r for m in members(g) for r in term_loads(deps, g.terms[m]))
    isempty(g.reductions) || !isempty(stats(t)) || any(in(shared), reads) || return false
    fold && !mixes(graph, g, v, t; recomputing) && return false
    # Nor may what the group reads from other kernels depend on `v`.
    depends(graph, deps, filter(!in(group), shared), Set((v,))) && return false
    push!(fold ? g.reductions : g.outputs, v)
    g.terms[v] = t
    g.phase[v] = maximum(s -> g.phase[s], stats(t); init = 0) + (fold ? 1 : 0)
    true
end

"""
    plan(graph::IndexGraph; fixed = [], recompute = (), store = (), ride = ()) -> Plan

Decide what is stored and form the kernels. Each set of values in `fixed`, a
candidate an implementation proposed, forms a group of its own, its folds
stored; the planner groups the rest around them. Each value in `recompute` is
recomputed where it is read instead of stored, and each map in `store` is
stored rather than recomputed where it is read. Each sum in `ride` rides the
steps of the recurrence whose cycle it reads (see `riding`): a fold of its
group, the values between computed at each step and not stored.
"""
function plan(graph::IndexGraph; fixed = Vector{Int}[], recompute = (), store = (), ride = ())
    s = storing(graph; recompute, store, ride)
    graph, stored, live, results = s.graph, s.stored, s.live, s.results
    # A pointwise value read only by one group over its domain, at the same
    # index, is recomputed by that group: grouped again without it.
    p = group_values(graph, copy(stored), live, results, fixed, s.riders)
    users = readers(graph)
    n = length(graph.values)
    shared = filter(id -> !any(s -> id in s, fixed) && recomputable(p, users, id, live, results, s.materialized), 1:n)
    isempty(shared) && return p
    stored[shared] .= false
    group_values(graph, stored, live, results, fixed, s.riders)
end

# What `plan` decides before it forms kernels: the graph with its results
# stored, what is stored, what runs, and the folds recomputed where read.
# The values in `store` are stored as those a gather reads are.
# Whether group `g` may take value `v`: anything but a step of an
# accumulation, which joins the group of a fold it adds. Through the step
# before it, every step would join the first one's group, which would load
# every term.
adds_to(values, g::Group, v) = !accumulates(values, v) || any(o -> o isa Read && o.value in g.reductions, values[v].definition.operands)

# Whether value `v` sums two operands, a fold among them: a step of an accumulation.
function accumulates(values, v)
    d = values[v].definition
    d isa Pointwise && d.f === (+) && length(d.operands) == 2 && any(o -> o isa Read && values[o.value].definition isa Reduction, d.operands)
end

function storing(graph::IndexGraph; recompute = (), store = (), ride = ())
    recomputed = Set{Int}(recompute)
    graph = with_stored_results(graph)
    values = graph.values
    n = length(values)
    # What the results read, directly or through others: nothing else runs.
    # A state read keeps what it is carried from, after it, to a fixpoint: a
    # state whose next value only another state's reads, as crossing states'
    # are, was left with no kernel.
    live = falses(n)
    for o in graph.results, r in read_values(o)
        live[r] = true
    end
    while true
        for id in n:-1:1
            live[id] || continue
            for o in operands(values[id].definition), r in read_values(o)
                live[r] = true
            end
        end
        grown = false
        for (id, v) in enumerate(values)
            d = v.definition
            live[id] && d isa Carried && d.from isa Read && !live[d.from.value] && (live[d.from.value] = grown = true)
        end
        grown || break
    end
    # Each sum riding a recurrence's steps, by the recurrence's state, and
    # the values between it and the cycle, which its steps compute.
    cycles = recurrences(graph)
    riders = Dict{Int, Vector{Int}}()
    between = Dict{Int, Set{Int}}()
    for r in ride
        k = findfirst(((cs, _, cycle),) -> live[first(cs)] && riding(graph, live, cycle, r) !== nothing, cycles)
        k === nothing && throw(NotAGroup([r]))
        cs, _, cycle = cycles[k]
        c = first(cs)
        push!(get!(() -> Int[], riders, c), r)
        union!(get!(() -> Set{Int}(), between, c), riding(graph, live, cycle, r))
    end
    foreach(b -> union!(recomputed, b), Base.values(between))
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
    # A gather reads its indices from storage, and a fold by key the contents
    # it starts from.
    materialized = falses(n)
    for id in 1:n
        live[id] || continue
        d = values[id].definition
        for o in operands(d)
            foreach(r -> materialized[r] = true, gathered_values(o))
        end
        d isa Keyed && d.into !== nothing && foreach(r -> materialized[r] = true, read_values(d.into))
    end
    foreach(v -> materialized[v] = true, store)
    # Maps of recomputed folds are recomputed with them: a fold recomputed
    # where it is read has no storage, so a map of it is a map of its term. A
    # recomputed map's readers may still be stored; they recompute it inside.
    after = falses(n)
    for id in 1:n
        d = values[id].definition
        d isa Pointwise || continue
        after[id] = any(o -> o isa Read && (o.value in recomputed && values[o.value].definition isa Reduction || after[o.value]), d.operands)
    end
    # Work nothing reads is not stored.
    stored = map(eachindex(values)) do id
        d = values[id].definition
        id in results || materialized[id] || !(id in recomputed || after[id]) || return false
        d isa Union{Pointwise, Reduction, Scan, Keyed} && !live[id] && return false
        d isa Pointwise ? (counts[id] != 1 || id in results || materialized[id] || fold_slice(values, d)) : true
    end
    # A sum another sum reads, adding a fold, is stored: the sums of a value
    # used many times, as its gradient accumulates over each use. Recomputed,
    # the last one's reader would load every term, a kernel growing with the
    # uses; stored, each is written beside the fold it adds.
    users = readers(graph)
    for id in 1:n
        live[id] && !stored[id] && !(id in recomputed || after[id]) && accumulates(values, id) || continue
        any(r -> live[r] && accumulates(values, r), users[id]) && (stored[id] = true)
    end
    # A recurrence's state is stored, and the values of its cycle something
    # outside it reads, the value the state is carried from among them: its
    # group stores them at each step. The rest it recomputes at each step,
    # the value it is carried from becoming the state one step on (see
    # `recurrence_group`); so does a map of what is stored, which a reader
    # outside recomputes too.
    if !isempty(cycles)
        users = readers(graph)
        for (cs, fs, cycle) in cycles
            foreach(c -> stored[c] = true, cs)
            # A sum riding the steps, and what it computes there, are read
            # within them.
            c = first(cs)
            within = union(Set(cycle), get(riders, c, Int[]), get(between, c, Set{Int}()))
            for w in cycle
                w in cs && continue
                d = values[w].definition
                # A value a state is carried from that another of the step
                # reads, as an LSTM's hidden state reads its cell's, is
                # stored where the step reads it.
                stored[w] = w in results || materialized[w] || any(u -> live[u] && !(u in within), users[w]) ||
                    w in fs && any(u -> live[u] && u in cycle && !(u in cs), users[w])
                stored[w] && !(w in fs) && d isa Pointwise && all(o -> !(o isa Read) || stored[o.value], d.operands) && (stored[w] = false)
            end
        end
    end
    # A pointwise value that reads nothing stored is cheaper to recompute.
    for id in eachindex(values)
        d = values[id].definition
        d isa Pointwise && stored[id] && !(id in results) && !materialized[id] || continue
        all(o -> !(o isa Read) || !stored[o.value] && values[o.value].definition isa Pointwise, d.operands) || continue
        all(o -> o isa Literal || o isa IndexOperand, d.operands) && (stored[id] = false)
    end
    # A stored half-precision value widened to Float32 is widened where it is
    # read, from the half value: a reader loads half the bytes, and a product
    # takes the half value on the half tensor cores (see `widened`), where it
    # would take a stored Float32 copy at Float32's precision. BFloat16
    # attention with its output rounded, as cuDNN computes it, stored its
    # output's gradient so for the product into the values' gradient: the
    # step took 515 μs at TF32 and 900 at full precision, 412 and 430 now.
    for id in eachindex(values)
        d = values[id].definition
        stored[id] && widened(graph, d) && !(id in results) && !materialized[id] || continue
        o = only(d.operands)
        o isa Read && stored[o.value] && (stored[id] = false)
    end
    foreach(b -> foreach(u -> stored[u] = false, b), Base.values(between))
    (; graph, stored, live, results, materialized, riders)
end

# Whether `d` widens a half-precision value to Float32, as `Float32.(k)` does
# before a product: a product takes the half value as it is, its products
# exact in Float32, and the tensor cores accumulate them in Float32.
widened(graph, d) = d isa Pointwise && d.f isa Cast{Float32} && length(d.operands) == 1 &&
    eltype_of(graph, only(d.operands)) <: HalfFloat

# Whether `d` is a slice of a fold: the fold read at constants along some of
# its axes. It is stored, so that what reads it reads the slice and not the
# whole fold, which the fold's group need not write (see `slice_placement`).
function fold_slice(values, d::Pointwise)
    d.f === identity && length(d.operands) == 1 && only(d.operands) isa Read || return false
    o = only(d.operands)
    values[o.value].definition isa Reduction && any(e -> isempty(e.terms), o.index)
end

# Whether stored pointwise value `id` is read only within one group over its
# domain, at the domain's index: by the group's members, or by values the
# group recomputes where it reads them, which read it so in turn.
function recomputable(p::Plan, users, id, live, results, materialized)
    v = p.graph.values[id]
    v.definition isa Pointwise && p.stored[id] && live[id] && !(id in results) && !materialized[id] || return false
    groups = Set{Int}()
    pending, seen = [id], Set{Int}()
    while !isempty(pending)
        w = pop!(pending)
        dims = p.graph.values[w].dims
        for u in users[w]
            live[u] && !(u in seen) || continue
            push!(seen, u)
            all(operands(p.graph.values[u].definition)) do o
                !(w in read_values(o)) || o isa Read && o.value == w && o.index == identity_index(dims)
            end || return false
            p.owner[u] == 0 && !p.stored[u] ? push!(pending, u) : push!(groups, p.owner[u])
        end
    end
    length(groups) == 1 || return false
    k = only(groups)
    k != 0 && k != p.owner[id] && p.groups[k].kind in (:reduction, :scan, :pointwise) && p.groups[k].domain == v.dims
end

# Insert into the sorted `frontier` the stored values after `seed` that read
# any of `from`, directly or through values a term computes in place: the
# only values whose terms can reach a group grown from `from`. Those `skip`
# names are passed over.
function reach!(frontier, followers, stored, from, seed; skip = w -> false)
    pending, seen = collect(from), Set{Int}()
    while !isempty(pending)
        for w in followers[pop!(pending)]
            w in seen && continue
            push!(seen, w)
            if !stored[w]
                push!(pending, w)
            elseif w > seed && !skip(w)
                k = searchsortedfirst(frontier, w)
                get(frontier, k, 0) == w || insert!(frontier, k, w)
            end
        end
    end
    frontier
end

# What the planner needs to form kernels, before any is formed.
Dependencies(graph::IndexGraph) = Dependencies(zeros(Int, length(graph.values)), Group[], ancestors(graph))

# The kernels, given what is stored.
function group_values(graph, stored, live, results, fixed, riders = Dict{Int, Vector{Int}}())
    values = graph.values
    n = length(values)
    deps = Dependencies(graph)
    owner, groups, up = deps.owner, deps.groups, deps.up
    for set in fixed
        push!(groups, fixed_group(graph, stored, deps, set))
        foreach(m -> owner[m] = length(groups), set)
    end
    for (cs, fs, cycle) in recurrences(graph)
        live[first(cs)] || continue
        g = recurrence_group(graph, stored, cs, fs, cycle, get(riders, first(cs), Int[]))
        # What a recurrence computes at each step is its group's alone: a set
        # fixed apart holding any of it would be a second kernel writing it.
        any(m -> owner[m] != 0, members(g)) && throw(NotAGroup(filter(m -> owner[m] != 0, members(g))))
        push!(groups, g)
        foreach(m -> owner[m] = length(groups), members(g))
    end
    # Who reads each value, directly or through a gather: a group grows toward
    # them, so it never scans the values that cannot reach it.
    followers = [Int[] for _ in values]
    for id in 1:n, o in operands(values[id].definition), r in read_values(o)
        live[id] && push!(followers[r], id)
    end
    foreach(unique!, followers)
    # Scans first, so that a fold along a scan's axes rides its sweep, wherever
    # it is in the graph; folds by key last, so that they join a group over
    # their iteration where one forms.
    seeds = [filter(id -> values[id].definition isa Scan, 1:n); filter(id -> values[id].definition isa Reduction, 1:n);
        filter(id -> values[id].definition isa Keyed, 1:n)]
    for id in seeds
        live[id] && owner[id] == 0 && stored[id] || continue
        d = values[id].definition
        g = d isa Scan ? Group(:scan, values[id].dims) : Group(:reduction, d.iteration)
        after = d isa Scan ? 0 : id
        try_add!(graph, stored, deps, g, id; fold = true)
        # A group reaches what reads its members, and what reads what they
        # read. Of the latter, a value that depends on a member is passed
        # over: the last value stored on a path to it from the group would be
        # among its loads and depend on the group, so it joins only where
        # every such path runs through what the group computes, and then it
        # is reached from a member. The weights every step of a long program
        # reads would otherwise bring every later step to every group.
        apart(w) = any(m -> deps.up[w][m], members(g))
        frontier = reach!(Int[], followers, stored, members(g), after)
        reach!(frontier, followers, stored, [r for m in members(g) for r in term_loads(deps, g.terms[m])], after; skip = apart)
        # A pass visits the frontier in order; what a new member reaches above
        # the cursor is visited in this pass, the rest in the next.
        changed = true
        while changed
            changed = false
            k = 1
            while k <= length(frontier)
                other = frontier[k]
                k += 1
                live[other] && owner[other] == 0 && !(other in members(g)) || continue
                od = values[other].definition
                added = if g.kind === :reduction && od isa Reduction
                    # A fold over other positions widens the domain, if it joins.
                    domain = joint_domain(graph, g, other)
                    h = domain === nothing ? nothing : domain == g.domain ? g : with_domain(g, domain)
                    h !== nothing && try_add!(graph, stored, deps, h, other; fold = true) && (g = h; true)
                else
                    od isa Union{Reduction, Scan, Keyed} ? sweeps_with(graph, g, other) && try_add!(graph, stored, deps, g, other; fold = true) :
                        od isa Pointwise && other > id && !awaits_sweep(graph, deps, g, other) && adds_to(values, g, other) && try_add!(graph, stored, deps, g, other; fold = false)
                end
                added || continue
                changed = true
                reach!(frontier, followers, stored, [other], after)
                reach!(frontier, followers, stored, term_loads(deps, g.terms[other]), after; skip = apart)
                k = searchsortedfirst(frontier, other) + 1
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
        frontier = reach!(Int[], followers, stored, [id], id)
        k = 1
        while k <= length(frontier)
            other = frontier[k]
            k += 1
            live[other] && owner[other] == 0 && values[other].definition isa Pointwise && values[other].dims == g.domain || continue
            any(m -> up[other][m], members(g)) && adds_to(values, g, other) && try_add!(graph, stored, deps, g, other; fold = false) || continue
            reach!(frontier, followers, stored, [other], id)
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
        g.kind in (:reduction, :scan, :pointwise) || continue
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

"""
Whether fold or scan `v` joins the sweep of group `g`: a fold a reduction's
group over its iteration; a scan, or a fold along its axes, a scan's group
over its domain, the scan in the group's direction.
"""
function sweeps_with(graph, g::Group, v)
    d = graph.values[v].definition
    g.kind === :reduction && return d isa Union{Reduction, Keyed} && d.iteration == g.domain
    d isa Keyed && return false
    e = graph.values[first(g.reductions)].definition
    d isa Reduction && return d.iteration == g.domain && d.reduced == e.axes
    graph.values[v].dims == g.domain && d.axes == e.axes && d.reverse == e.reverse
end

"""
The domain of reduction group `g` with fold `v` among its folds: the group's
where `v` iterates over it; for a fold over other extents along axes it
keeps, alike along those it sums, the box that holds both, over which each
fold of the group is computed at its own positions (see `restricted` and
`mixes`); nothing otherwise. MLA's backward sums dV over the values' 64
features, and dK and dQ over the 96 of the queries and keys, each
recomputing the scores: one kernel recomputes them once.
"""
function joint_domain(graph, g::Group, v)
    d = graph.values[v].definition
    g.kind === :reduction && d isa Union{Reduction, Keyed} || return nothing
    d.iteration == g.domain && return g.domain
    d isa Reduction && length(d.iteration) == length(g.domain) && all(k -> d.iteration[k] == g.domain[k], d.reduced) ||
        return nothing
    domain = map(max, d.iteration, g.domain)
    # A fold by key combines each element of the whole domain.
    domain == g.domain || all(r -> graph.values[r].definition isa Reduction, g.reductions) || return nothing
    domain
end

"Group `g` over `domain`, its members as they are."
with_domain(g::Group, domain) = Group(g.kind, domain, copy(g.reductions), copy(g.outputs), copy(g.terms), copy(g.phase))

"""
Whether fold `v`, of term `t`, may join reduction group `g`, whose domain
holds its positions: always, where every fold of the group then iterates
over the domain. A group whose folds cover different positions computes
each fold's term over the whole domain, as a tile does, and keeps it at the
fold's own positions; so each fold over fewer positions (`restricted`) sums
the group's primary axes and reads none of its folds, no member reads it,
and the outputs lie over the domain. What such a fold shares with the
others is the reason it shares their kernel: with `recomputing`, each one
recomputes a fold another fold of the group recomputes, as attention's dV,
dK and dQ recompute the scores. Shared, it is the same fold at positions
that move along no axis where either fold covers fewer than the domain: so
the work a fold does past its own positions is its own product's, as dV's
over 64 features where dK and dQ have 96, and not the shared fold's. An
MLP's weight gradients recompute its hidden layer too, at its units along
the axis where the other covers fewer: they stay apart. Without
`recomputing`, the group is a proposal's, which the cover prices.
"""
function mixes(graph, g::Group, v, t; recomputing)
    d = graph.values[v].definition
    narrowed = filter(r -> restricted(graph, g, r), g.reductions)
    d isa Reduction && d.iteration != g.domain && push!(narrowed, v)
    isempty(narrowed) && return true
    terms = Dict{Int, Term}(m => g.terms[m] for m in members(g))
    terms[v] = t
    axes = isempty(g.reductions) ? d.reduced : primary_axes(graph, g)
    all(r -> graph.values[r].definition.reduced == axes && isempty(stats(terms[r])), narrowed) || return false
    all(m -> isdisjoint(stats(terms[m]), narrowed), keys(terms)) || return false
    all(m -> placement(graph, g, m) !== nothing, g.outputs) || return false
    recomputing || return true
    folds = [g.reductions; v]
    narrower(r) = (e = graph.values[r].definition.iteration; [k for k in eachindex(e) if e[k] != g.domain[k]])
    function recomputed(r, axes)
        Set(s.value for s in subterms((terms[r],); nested = true) if s isa Recomputed && !any(k -> mentions(s, k), axes))
    end
    shares(r, o) = (axes = [narrower(r); narrower(o)]; !isdisjoint(recomputed(r, axes), recomputed(o, axes)))
    all(r -> any(o -> o != r && shares(r, o), folds), narrowed)
end

# Whether map `v` waits for another sweep: it reads a scan of another
# sweep that has no kernel yet. That scan's group reads this one's values
# from storage, so a map that reads both can join it, and not this one.
awaits_sweep(graph, deps, g::Group, v) = g.kind === :scan &&
    any(s -> deps.owner[s] == 0 && graph.values[s].definition isa Scan && !(s in g.reductions), findall(deps.up[v]))

# Recurrences: their carried states, the value each is carried from, and the
# cycle between, the values that read a state and that a value a state is
# carried from reads, both through others: what its group computes at each
# step. States each carried from a value reading the other, through others,
# are one recurrence's: an LSTM's hidden state and its cell, each step
# reading both, and two states assigned at once, `s1, s2 = f(s2), g(s1)`,
# though neither state's value is carried from one that reads it.
function recurrences(graph)
    carried = Tuple{Int, Int}[]
    for (c, v) in enumerate(graph.values)
        d = v.definition
        d isa Carried || continue
        from = d.from
        from isa Read && from.value > c || continue
        push!(carried, (c, from.value))
    end
    cycles = [recurrence_cycle(graph, [c], [f]) for (c, f) in carried]
    length(carried) <= 1 && return [([c], [f], cycle) for ((c, f), cycle) in zip(carried, cycles)]
    up = ancestors(graph)
    # Which states each state's value is carried from reads, through others.
    n = length(carried)
    reach = [[up[f][c] for (c, _) in carried] for (_, f) in carried]
    for m in 1:n, k in 1:n
        reach[k][m] || continue
        for j in 1:n
            reach[m][j] && (reach[k][j] = true)
        end
    end
    sets = Vector{Int}[]
    for k in 1:n
        any(set -> k in set, sets) && continue
        push!(sets, [j for j in 1:n if j == k || reach[k][j] && reach[j][k]])
    end
    map(sort!(sets; by = minimum)) do set
        cs, fs = [carried[k][1] for k in sort(set)], [carried[k][2] for k in sort(set)]
        (cs, fs, length(cs) == 1 ? cycles[only(set)] : recurrence_cycle(graph, cs, fs))
    end
end

# The states of recurrence group `g`, its first outputs, and the values each is carried from.
recurrence_states(graph, members) = [m for m in members if graph.values[m].definition isa Carried]
carried_from(graph, c) = (graph.values[c].definition.from::Read).value

"""
    riding(graph, live, cycle, r) -> values or nothing

The values a sum `r` reads between it and a recurrence's `cycle`, which ride
the recurrence's steps with it: computed at each step from the cycle's
values there, the sum adding each step's part as the steps run, as a weight's
gradient adds each step's while its operands are fresh. Each is a map or a
fold that only `r` and the others read; nothing where `r` reads no value of
the cycle, or another reads one of them, or a state is carried from one, or
`r` sums an axis other than the steps.
"""
function riding(graph, live, cycle, r)
    values = graph.values
    d = values[r].definition
    d isa Reduction && d.fold isa SumFold && !(r in cycle) || return nothing
    inside = Set(cycle)
    # Each value that reads the cycle, through others or a state carried
    # from one that does, as a second recurrence whose step reads the
    # first's; and those `r` reads, through others.
    reaches, read = falses(length(values)), falses(r)
    read[r] = true
    carried = [(c, carried_from(graph, c)) for (c, v) in enumerate(values) if v.definition isa Carried && v.definition.from isa Read]
    while true
        for u in eachindex(values)
            reaches[u] && continue
            reaches[u] = u in inside || any(o -> any(x -> reaches[x], read_values(o)), operands(values[u].definition))
        end
        grown = false
        for (c, f) in carried
            !reaches[c] && reaches[f] && (reaches[c] = grown = true)
        end
        grown || break
    end
    reaches[r] || return nothing
    for u in r:-1:1
        read[u] || continue
        foreach(o -> foreach(x -> read[x] = true, read_values(o)), operands(values[u].definition))
    end
    between = [u for u in 1:(r - 1) if read[u] && reaches[u] && !(u in inside)]
    all(u -> values[u].definition isa Union{Pointwise, Reduction}, between) || return nothing
    set = Set([between; r])
    users = readers(graph)
    all(u -> all(w -> !live[w] || w in set, users[u]), between) || return nothing
    # A state reads the value it is carried from, which no reader counts.
    any(((_, f),) -> f in set, carried) && return nothing
    # Each value of the cycle and between at its axis along the steps, from
    # the states'; the sum adds them up where it reads them along an axis it
    # sums, not a step's alone or along another.
    along = Dict{Int, Int}(c => values[c].definition.axis for c in cycle if values[c].definition isa Carried)
    for u in sort!([[w for w in cycle if !haskey(along, w)]; between])
        e = values[u].definition
        k = steps_iteration(e, length(values[u].dims), along)
        k === nothing && return nothing
        k == 0 && continue
        if e isa Reduction
            k in e.reduced && return nothing
            k = count(j -> !(j in e.reduced), 1:k)
        end
        along[u] = k
    end
    k = steps_iteration(d, 0, along)
    k isa Int && k in d.reduced || return nothing
    between
end

# The axis of `d`'s iteration along a recurrence's steps, where it reads each
# value along them, by its axis there in `along`, a step at a time: 0 where
# it reads none, nothing where it reads one otherwise or along two. A map
# iterates over its `rank` axes, a fold over its iteration's.
function steps_iteration(d, rank, along)
    n = d isa Reduction ? length(d.iteration) : rank
    k = 0
    for o in operands(d)
        o isa Read && haskey(along, o.value) || continue
        e = o.index[along[o.value]]
        j = findfirst(i -> e == var(i), 1:n)
        j === nothing && return nothing
        k in (0, j) || return nothing
        k = j
    end
    k
end

"""
    riding_sums(plan) -> values

The sums of `plan` that could ride the steps of one of its recurrences:
computed apart from them, over the steps, reading the recurrence's cycle
through values only they read (see `riding`), each once.
"""
function riding_sums(p::Plan)
    graph = p.graph
    values = graph.values
    live = falses(length(values))
    foreach(g -> foreach(m -> live[m] = true, members(g)), p.groups)
    for id in length(values):-1:1
        live[id] || continue
        foreach(o -> foreach(r -> live[r] = true, read_values(o)), operands(values[id].definition))
    end
    found = Int[]
    for (cs, fs, cycle) in recurrences(graph)
        c = first(cs)
        live[c] || continue
        k = p.owner[c]
        for (r, v) in enumerate(values)
            d = v.definition
            d isa Reduction && d.fold isa SumFold && p.owner[r] != k && r > maximum(fs) && !(r in found) || continue
            riding(graph, live, cycle, r) === nothing || push!(found, r)
        end
    end
    found
end

function recurrence_cycle(graph, cs, fs)
    values = graph.values
    lo, hi = minimum(cs), maximum(fs)
    reaches, feeds = falses(hi), falses(hi)
    foreach(c -> reaches[c] = true, cs)
    foreach(f -> feeds[f] = true, fs)
    for w in lo:hi
        reaches[w] && continue
        reaches[w] = any(o -> any(r -> r <= hi && reaches[r], read_values(o)), operands(values[w].definition))
    end
    for w in hi:-1:lo
        feeds[w] || continue
        for o in operands(values[w].definition), r in read_values(o)
            lo <= r && (feeds[r] = true)
        end
    end
    cycle = [w for w in lo:hi if reaches[w] && feeds[w]]
    all(w -> w in cs || values[w].definition isa Union{Pointwise, Reduction}, cycle) ||
        throw(ArgumentError("a recurrence whose step scans, scatters or writes is not supported yet"))
    cycle
end

"""
    recurrence_places(graph, members) -> (; domain, places)

A recurrence's group's domain and where it keeps each of `members`, the
states first, as the member's index over it. The domain is the states', then an
axis for each axis of a member along none of the state's. A value of the
cycle lies along the state's axis `q` at its own axis `r` where it reads the
state, or a value of the cycle lying along `q` at its axis `s`, with `s`
indexed by `r`'s variable: as the value the state is carried from lies along
the state's own axes, and as flash-linear-attention's `v_new`, `Ũ - W S`,
lies along the state's columns and chunks, its rows an axis of its own.
"""
function recurrence_places(graph, members)
    values = graph.values
    cs = recurrence_states(graph, members)
    fs = [carried_from(graph, c) for c in cs]
    c = first(cs)
    d = values[c].definition::Carried
    D = values[c].dims
    along = Dict{Int, Vector{Int}}(c => collect(eachindex(D)) for c in cs)
    for w in recurrence_cycle(graph, cs, fs)
        w in cs && continue
        e = values[w].definition
        kept = e isa Reduction ? [k for k in eachindex(e.iteration) if !(k in e.reduced)] : collect(eachindex(values[w].dims))
        at = zeros(Int, length(D))
        for o in operands(e)
            o isa Read && haskey(along, o.value) || continue
            for (q, s) in enumerate(along[o.value])
                s == 0 && continue
                r = findfirst(k -> o.index[s] == var(k), kept)
                r === nothing && continue
                at[q] in (0, r) || throw(ArgumentError("a recurrence whose step reads a value along two of the state's axes is not supported yet"))
                at[q] = r
            end
        end
        along[w] = at
    end
    domain = collect(D)
    places = Dict{Int, Tuple{Vararg{Affine}}}()
    for m in members
        (m in cs || m in fs) && (places[m] = identity_index(D); continue)
        along[m][d.axis] == 0 && throw(ArgumentError("a recurrence whose step computes a value not along its steps is not supported yet"))
        places[m] = ntuple(length(values[m].dims)) do r
            q = findfirst(==(r), along[m])
            q === nothing || return var(q)
            push!(domain, values[m].dims[r])
            var(length(domain))
        end
    end
    (; domain = Tuple(domain), places)
end

# A recurrence's group, stepping along the states' axis: the states, each
# stored at each step, where it starts or else the value it is carried from
# one step back; the values of the cycle stored, in order; and the values
# the states are carried from, stored or not, in the states' order, each its
# state one step on. Their terms recompute the rest of the cycle
# and read what the group computes wherever this step stored it. The sums
# riding its steps are its folds, each over its own iteration: as a fold
# along a scan's axes rides its sweep, each is complete only after the
# steps, and nothing in the group reads it.
function recurrence_group(graph, stored, cs, fs, cycle, riders = Int[])
    values = graph.values
    c = first(cs)
    all(f -> values[f].dims == values[c].dims, fs) || throw(ArgumentError("a recurrence's state carried from a value of another shape is not supported yet"))
    d = values[c].definition
    all(k -> values[k].dims == values[c].dims && values[k].definition.axis == d.axis && values[k].definition.reverse == d.reverse, cs) ||
        throw(ArgumentError("a recurrence whose states differ in shape or direction is not supported yet"))
    allunique([cs; fs]) || throw(ArgumentError("a recurrence's state carried from another state is not supported yet"))
    # The states, then the rest in the graph's order, which a step computes
    # them in: an LSTM's hidden state one step on reads its cell's.
    members = [cs; sort!([[w for w in cycle if !(w in cs) && !(w in fs) && stored[w]]; fs])]
    (; domain, places) = recurrence_places(graph, members)
    g = Group(:recurrence, domain)
    inside = copy(stored)
    foreach(w -> w in members || (inside[w] = false), cycle)
    append!(g.outputs, members)
    whole = Domain(domain)
    memo = Dict{Any, Term}()
    foreach(c -> g.terms[c] = operand_term(graph, inside, g, values[c].definition.init, places[c], whole, memo), cs)
    for m in members[(length(cs) + 1):end]
        g.terms[m] = values[m].definition isa Reduction ? fold_term(graph, inside, g, m, places[m], whole, memo) :
            definition_term(graph, inside, g, values[m].definition, values[m], places[m], whole, memo)
    end
    foreach(m -> g.phase[m] = 0, members)
    for r in riders
        d = values[r].definition::Reduction
        push!(g.reductions, r)
        g.terms[r] = operand_term(graph, inside, g, d.operand, identity_index(d.iteration), Domain(d.iteration), memo)
        g.phase[r] = 0
    end
    g
end

"A recurrence group's axes (see `Tempest.Compiler.Implement.recurrence_axes`)."
function recurrence_axes(graph, g::Group)
    state = graph.values[first(g.outputs)]
    Implement.recurrence_axes(state.definition.axis, length(state.dims), g.domain, g.outputs, g.terms,
        recurrence_places(graph, g.outputs).places)
end

"A candidate whose values do not form one group."
struct NotAGroup <: Exception
    values::Vector{Int}
end

Base.showerror(io::IO, e::NotAGroup) = print(io, "the values ", join(map(v -> "%$v", e.values), " "), " do not form one group")

# The group of exactly the values in `set`, added in order: a reduction group
# over the domain of its folds, wherever they come in the set, or with no fold
# a pointwise group over the first value's domain. A candidate spanning
# several domains waits for groups that span them: a kernel is maps and folds
# over one domain, each fold at its own positions within it.
function fixed_group(graph, stored, deps, set)
    values = graph.values
    all(v -> deps.owner[v] == 0, set) || throw(NotAGroup(set))
    folds = filter(v -> values[v].definition isa Reduction, set)
    # Folds over other extents along axes they keep share the box that holds
    # them (see `joint_domain`).
    domain = isempty(folds) ? values[first(set)].dims : values[first(folds)].definition.iteration
    for v in folds
        extents = values[v].definition.iteration
        length(extents) == length(domain) && (domain = map(max, domain, extents))
    end
    g = isempty(folds) ? Group(:pointwise, domain) : Group(:reduction, domain)
    for v in set
        dv = values[v].definition
        added = dv isa Reduction ? joint_domain(graph, g, v) == g.domain && try_add!(graph, stored, deps, g, v; fold = true, recomputing = false) :
            dv isa Pointwise && stored[v] && try_add!(graph, stored, deps, g, v; fold = false)
        added || throw(NotAGroup(set))
    end
    g
end

# The folds that can be recomputed where they are read, as proposals for the
# cover (see `Recompute`): a fold that is not a result, an index of a gather,
# or in `kept`, whose every reader through maps is a fold over another domain,
# or over the same extents reading it elsewhere than where its group keeps it,
# as attention reads its scores when the features are as many as the keys (a
# fold that reads it there joins its group as a later phase instead). A fold
# recomputed inside a reader's sweep, at positions that move along its own
# reduced axes, does not recompute such a fold in turn: nested sweeps multiply
# the work past what a plan should price.
# Whether recomputing pays is the cover's decision, by cost.
function recomputable_folds(graph, kept = Set{Int}())
    s = storing(graph)
    graph, results, materialized = s.graph, s.results, s.materialized
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
        own = Tuple(var(k) for k in eachindex(d.iteration) if !(k in d.reduced))
        all(((u, at),) -> values[u].definition.iteration != d.iteration || at != own, consumers) || continue
        hoisted = all(((u, at),) -> !any(k -> any(e -> mentions(e, k), at), values[u].definition.reduced), consumers)
        hoisted || !recomputes(values, v, union(selected, kept), results, materialized) || continue
        push!(selected, v)
    end
    selected
end

# Whether fold `v` reads a fold of `selected` through maps that stay unstored.
# A map among `selected`, one the partition recomputes in the backward, is
# walked through like any other, since recomputing it nests no sweep. Taken
# for a fold, the widening of BFloat16 attention's rounded output, which its
# output's gradient reads, kept the sum over features of the probabilities'
# gradient from the proposals, and the training step stored that sum, as
# large as the scores: 1114 μs at TF32, against 777 proposed.
function recomputes(values, v, selected, results, materialized)
    pending, seen = [v], Set([v])
    while !isempty(pending)
        for o in reads(values[pop!(pending)].definition)
            u = o.value
            u in selected && values[u].definition isa Reduction && return true
            values[u].definition isa Pointwise && !(u in results) && !materialized[u] && !(u in seen) || continue
            push!(seen, u)
            push!(pending, u)
        end
    end
    false
end

# Every result becomes a value read whole, so results are buffers: a result
# that is a layout of a value, or a constant, gets a value of its own. A fold
# or pointwise value that only one result reads, through a permutation of its
# axes, is stored in the result's order instead of copied: its domain is
# permuted, and its id stays.
function with_stored_results(graph::IndexGraph)
    values = copy(graph.values)
    results = copy(graph.results)
    users = readers(graph)
    for (k, r) in enumerate(results)
        r isa Read || continue
        v = values[r.value]
        p = permutation(r.index)
        (p === nothing || issorted(p)) && continue
        v.definition isa Union{Pointwise, Reduction} && !v.scalar || continue
        isempty(users[r.value]) && count(o -> o isa Read && o.value == r.value, results) == 1 || continue
        values[r.value] = permuted(v, p)
        results[k] = Read(r.value, identity_index(values[r.value].dims))
    end
    for (k, r) in enumerate(results)
        dims = graph.result_dims[k]
        # A value read whole, or a reshape of it, presents its storage as it is.
        r isa Read && values[r.value].scalar == graph.result_scalar[k] && reads_storage(values[r.value], r.index, dims) && continue
        push!(values, Value(narrow(Pointwise(identity, (r,)), dims), map(k -> dims[k], wide_axes(dims)),
            graph.result_eltype[k], graph.result_scalar[k]))
        results[k] = Read(length(values), wide_index(dims))
    end
    IndexGraph(values, graph.arguments, results, graph.result_dims, graph.result_scalar, graph.result_eltype, graph.rewrites,
        vcat(graph.backward, falses(length(values) - length(graph.backward))))
end

# The axes an index reads a value at, when it is a permutation of the reader's
# variables.
function permutation(index)
    p = Int[]
    for e in index
        e.constant == 0 && length(e.terms) == 1 && e.terms[1].first isa Var && e.terms[1].second == 1 || return nothing
        push!(p, e.terms[1].first.n)
    end
    sort(p) == 1:length(p) ? p : nothing
end

# Value `v` stored with its axes permuted: `v′[x] = v[x[p[1]], …, x[p[n]]]`. A
# fold permutes the kept axes of its iteration; its reduced axes stay.
function permuted(v::Value, p)
    dims = Tuple(v.dims[findfirst(==(j), p)] for j in eachindex(p))
    d = v.definition
    if d isa Pointwise
        σ = Tuple(var(p[j]) for j in eachindex(p))
        return Value(Pointwise(d.f, map(o -> through(o, σ, Domain(dims)), d.operands)), dims, v.eltype, v.scalar)
    end
    kept = Tuple(k for k in eachindex(d.iteration) if !(k in d.reduced))
    iteration = collect(d.iteration)
    σ = Vector{Affine}(map(var, 1:length(d.iteration)))
    for (j, k) in enumerate(kept)
        iteration[kept[p[j]]] = d.iteration[k]
        σ[k] = var(kept[p[j]])
    end
    iteration = Tuple(iteration)
    operand = through(d.operand, Tuple(σ), Domain(iteration))
    Value(Reduction(d.fold, operand, iteration, d.reduced), dims, v.eltype, v.scalar)
end

# The values a group reads from storage, its own excepted.
function group_reads(graph, g::Group)
    reads = Set{Int}()
    for v in members(g)
        if haskey(g.terms, v)
            union!(reads, stored_reads(g.terms[v]), keyed_reads(graph, v))
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
