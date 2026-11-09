# Groups: what a kernel computes, where it keeps each value it computes, and
# what it reads; and a plan, the groups of a program's kernels in order with
# what is stored. The planner forms them (planner.jl).
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

"A reduction's group: folds over one domain, and the maps that read them or that they read."
struct ReductionKind end

"A scan's group: scans along the same axes in one direction, the maps over its domain, and the folds riding its sweep."
struct ScanKind end

"A group of maps over one domain."
struct PointwiseKind end

"A recurrence's group: the values of its cycle, computed a step at a time, and the sums riding its steps."
struct RecurrenceKind end

"The kind of a group: what its kernel does over its domain."
const GroupKind = Union{ReductionKind, ScanKind, PointwiseKind, RecurrenceKind}

"A kind's name, as a group's view has it and as it is written out."
kind_name(::ReductionKind) = :reduction
kind_name(::ScanKind) = :scan
kind_name(::PointwiseKind) = :pointwise
kind_name(::RecurrenceKind) = :recurrence

Base.show(io::IO, k::GroupKind) = print(io, kind_name(k))

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

# Where a group keeps a value it computes, as the value's index over the
# domain: a fold of the group at the variables of the axes it keeps, any other
# value over the whole domain or at the positions of the primary folds, or a
# slice of one of its folds where the fold keeps it; or nothing. A scan's
# group keeps everything but its folds over its whole domain. A fold by key
# keeps each element at its key. A fold over fewer positions than the domain
# (see `restricted`) keeps its own.
function placement(graph, g::Group, v)
    g.kind isa RecurrenceKind && return v in g.outputs ? recurrence_places(graph, g.outputs).places[v] : nothing
    val = graph.values[v]
    d = val.definition
    d isa Keyed && return v in g.reductions ? d.key : nothing
    g.kind isa ScanKind && !(d isa Reduction) && return val.dims == g.domain ? identity_index(g.domain) : nothing
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
    g.kind isa ReductionKind && r in g.reductions && (d = graph.values[r].definition) isa Reduction && d.iteration != g.domain

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
