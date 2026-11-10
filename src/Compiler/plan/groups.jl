# Groups: what a kernel computes, where it keeps each value it computes, and
# what it reads; and a plan, the groups of a program's kernels in order with
# what is stored. The planner forms them (planner.jl).
#
# A kernel computes a group: maps and folds over one shared index domain. Each
# fold reduces its own set of the domain's axes; a fold's value sits at the
# domain's index with its reduced axes at zero. What else holds of a group is
# its kind's to say, and each kind says it by a method. Of a group of its
# kind: where it keeps a value it computes (`placement`); whether a member
# may read a fold of the group, and where (`reads_fold`); whether members
# read the group's other values as terms or where a step stored them
# (`stepwise`); whether a fold of it covers fewer positions than its domain
# (`restricted`); and, as it grows, the group with one more value taken
# (`taking`, with the planner). The planner tests no group's kind.

"""
A reduction's group: folds over one domain, and the maps that read them or
that they read. Folds over the group's first set of axes (its primary axes)
can be read by later members at that position: a schedule completes them
before what follows (phases). A fold over other axes is completed by merging
partial states, so within the group nothing reads it; its operand, often a
value the group already computes, is never written. Which axes a schedule
sweeps, and which folds it completes in a block and which by merging partial
states, is the schedule's choice.

A fold by key joins a reduction's group over its iteration, as a fold over
other than the primary axes does: it combines each element at its key, is
complete only when the group's work is, and nothing in the group reads it.

A fold over fewer positions along axes it keeps may join a reduction's
group over a domain that holds them, where it shares a fold it recomputes
with the group's folds, as MLA's dV, over 64 features, shares the scores
with dK and dQ, over 96 (see `joint_domain`): it is computed over its own
positions, and nothing in the group reads it.
"""
struct ReductionKind end

"""
A scan's group, which sweeps the axes it scans along, in its direction: its
scans alike, and the maps over its domain. A map before a scan is its
operand's term, as Futhark's `scanomap` fuses one into its scan; a map after
reads each scan at the position the sweep has reached. A fold along those
axes rides the sweep, in either direction, since it need not take its
elements in order; it is complete only after the sweep, so nothing in the
group reads it.
"""
struct ScanKind end

"A group of maps over one domain: a stored map, and the later ones over its domain that read it."
struct PointwiseKind end

"""
A recurrence's group, which steps along its states' axis: the values of its
cycle, each step computing in order those it stores and reading them where
it stored them, and the sums riding its steps, its folds (see
`recurrence_group`).
"""
struct RecurrenceKind end

"The kind of a group: what its kernel does over its domain."
const GroupKind = Union{ReductionKind, ScanKind, PointwiseKind, RecurrenceKind}

"A kind's name, as a group's view has it and as it is written out."
kind_name(::ReductionKind) = :reduction
kind_name(::ScanKind) = :scan
kind_name(::PointwiseKind) = :pointwise
kind_name(::RecurrenceKind) = :recurrence

Base.show(io::IO, k::GroupKind) = print(io, kind_name(k))

"""
Whether a group's members read its other values where a step stored them: a
recurrence's, whose step computes each value it stores once, in order. A
group of another kind computes a value's term again in each member that
reads it: so a value only the group reads is never written, and a stored
map only it reads is recomputed in it.
"""
stepwise(::RecurrenceKind) = true
stepwise(::GroupKind) = false

struct Group
    kind::GroupKind
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
key_reads(graph, v) = keyed(graph, v) ? indirect_values((graph.values[v].definition::Keyed).key) : Int[]

"""
The values fold by key `v` reads besides its elements: those its key gathers
from, and those of the contents it starts from; none for any other value.
"""
function keyed_reads(graph, v)
    keyed(graph, v) || return Int[]
    into = (graph.values[v].definition::Keyed).into
    append!(key_reads(graph, v), into === nothing ? Int[] : read_values(into))
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

"Where group `g` keeps value `v`, which it computes, as the value's index over the domain; or nothing."
placement(graph, g::Group, v) = placement(g.kind, graph, g, v)

placement(::Union{ReductionKind, PointwiseKind}, graph, g::Group, v) = kept_placement(graph, g, v)

# A scan's group keeps everything but its folds over its whole domain.
function placement(::ScanKind, graph, g::Group, v)
    val = graph.values[v]
    val.definition isa Union{Reduction, Keyed} && return kept_placement(graph, g, v)
    val.dims == g.domain ? identity_index(g.domain) : nothing
end

# A recurrence's group keeps each value it stores along its states' axes (see `recurrence_places`).
placement(::RecurrenceKind, graph, g::Group, v) = v in g.outputs ? recurrence_places(graph, g.outputs).places[v] : nothing

# A fold of the group at the variables of the axes it keeps, any other value
# over the whole domain or at the positions of the primary folds, or a slice
# of one of its folds where the fold keeps it; or nothing. A fold by key keeps
# each element at its key. A fold over fewer positions than the domain (see
# `restricted`) keeps its own.
function kept_placement(graph, g::Group, v)
    val = graph.values[v]
    d = val.definition
    d isa Keyed && return v in g.reductions ? d.key : nothing
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
restricted(graph, g::Group, r) = restricted(g.kind, graph, g, r)
restricted(::ReductionKind, graph, g::Group, r) = r in g.reductions && (d = graph.values[r].definition) isa Reduction && d.iteration != g.domain
restricted(::GroupKind, graph, g::Group, r) = false

# A slice of a fold of the group: the fold read at a variable of the slice
# along some of its axes and at constants along the others, kept where the
# fold keeps those variables. A fold whose term does not vary along the
# constants' axes is the same there as anywhere along them (see `kept_at`),
# as a log-sum-exp is along the features of attention's output.
function slice_placement(graph, g::Group, v)
    d = graph.values[v].definition
    d isa Pointwise && d.f === identity && only(d.operands) isa Read || return nothing
    o = only(d.operands)::Read
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

"""
Whether a member of group `g` may read its fold `r` at `index`: a fold over
the primary axes, where the group keeps it. A reduction's group reads no
fold over fewer positions than its domain, a scan's the scans of its sweep
and no fold riding it, and a recurrence's none of the sums riding its steps.
"""
function reads_fold end

reads_fold(::ReductionKind, graph, g::Group, r, index) = !restricted(graph, g, r) && primary(graph, g, r) && kept_at(graph, g, r, index)
reads_fold(::ScanKind, graph, g::Group, r, index) = !(graph.values[r].definition isa Reduction) && primary(graph, g, r) && kept_at(graph, g, r, index)
reads_fold(::RecurrenceKind, graph, g::Group, r, index) = false

# Whether term `t` of group `g` varies along axis `a` of its domain: through
# what it reads, and through the folds of the group it reads.
varies(graph, g::Group, t::Term, a) = any(subterms((t,); nested = true)) do s
    s isa Stat && return any(e -> mentions(e, a), something(placement(graph, g, s.value), ()))
    s isa Union{Load, Ix} && mentions(s, a)
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
