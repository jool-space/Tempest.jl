# The kernels of a plan as they form, and the context forming reads them
# through: whether a value is stored, whether a kernel owns it, and whether
# values depend on a group's members. Everything forming a kernel reads of
# the others goes through here, so that a context can note it, and a plan
# formed again near another takes each kernel whose forming read nothing
# that differs (replan.jl).

"""
What forming kernels reads the kernels formed so far, and what is stored,
through: a context, given to every function that forms a kernel in place of
the arrays. Five functions make the interface. Three read: whether a value
is stored (`isstored`), whether a kernel formed so far owns it (`owned`),
and whether values depend on a group's members, through what they read and
the kernels that own it (`depends`). One writes: a kernel formed, its
members owned by it (`formed!`). One brackets the forming of one kernel,
which a context may note (`noted`). A plain context (`Plain`) reads the
kernels and notes nothing; a recording one (`Recording`) notes what each
kernel's forming read (see `Formed`), so that a plan formed again with a
few values changed takes each kernel whose forming read none of them as it
was (see `replanned`). A recurrence's step reads through a view of either
(`Stepping`).

A group is reused only if everything it read is unchanged, and everything
it read went through here.
"""
abstract type Forming end

# The values each value reads, directly or through others, as a mask over
# the values. A recurrence's state reads, after its first step, the value it
# is carried from and what that reads, though it comes after: a kernel that
# reads the state runs after the recurrence's, which reads all of it. As
# values after the state read it, the reads are taken again until no value
# reads more.
function ancestors(graph)
    n = length(graph.values)
    up = [falses(n) for _ in 1:n]
    carried = Tuple{Int, Int}[(c, v.definition.from.value) for (c, v) in enumerate(graph.values) if v.definition isa Carried && v.definition.from isa Read]
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
What `depends` walks with: the values pending, those covered and those a
value brings, a mask over the graph's values each, and the kernels looked
at. One is kept from one walk to the next: allocated at each, they were
most of the planner's garbage, a graph's worth of bits for each value
tried.
"""
struct DependsWalk
    pending::BitVector
    covered::BitVector
    fresh::BitVector
    kernels::Set{Int}
end

DependsWalk(n::Int) = DependsWalk(falses(n), falses(n), falses(n), Set{Int}())

"""
The kernels as they form: what is stored, the kernels formed so far, the
kernel that owns each value (zero for none yet), each value's ancestors,
what each kernel reads, once asked, and what each member's term loads, once
asked: a group tries many candidates against the same members. With them,
the walk `depends` reuses. Forming reads them through a context alone (see
`Forming`).
"""
struct Kernels
    stored::Vector{Bool}
    owner::Vector{Int}
    groups::Vector{Group}
    up::Vector{BitVector}
    reads::Dict{Int, Set{Int}}
    # By the term's own hash: an identity dictionary hashes an immutable term
    # by its contents, all of it, at every lookup.
    loads::Dict{Term, Vector{Int}}
    walk::DependsWalk
end

Kernels(stored, up) = Kernels(stored, zeros(Int, length(stored)), Group[], up, Dict{Int, Set{Int}}(), Dict{Term, Vector{Int}}(),
    DependsWalk(length(stored)))

"""
What a walk of what depends on a group found (see `depends`): no
dependence; one on a value it was asked about, which no kernel can take
away; or one found through what a kernel reads.
"""
@enum Walked apart_from direct through_kernels

"""
A walk of what depends on a group (see `depends`), as noted: what it was
asked, whether the values `reads` depend on the members `group`; what it
found; and its last root among the note's `roots`, the values it looked at
the ancestors of.
"""
struct Walk
    reads::Vector{Int}
    group::Vector{Int}
    found::Walked
    stop::Int
end

"""
What forming one kernel read, as a recording context notes it: the values
whose storage it looked at (`stores`) and those it looked at whether a
kernel owns (`owners`), each once, and its walks of what depends on the
group (see `Walk`), with the values each looked at the ancestors of, in
`roots`.
"""
struct Note
    stores::Vector{Int}
    owners::Vector{Int}
    roots::Vector{Int}
    walks::Vector{Walk}
    # The values among `stores` and `owners`, to note each once.
    stored_seen::BitVector
    owned_seen::BitVector
end

Note(n::Int) = Note(Int[], Int[], Int[], Walk[], falses(n), falses(n))

"""
Where forming starts a kernel, in the order it forms them: each set fixed as
a group (`:fixed`, at its place among them), each recurrence (`:recurrence`,
at its place among the graph's), then each fold and each map no kernel holds
yet, grown from it (`:fold` and `:map`, at the value).
"""
struct Seed
    kind::Symbol
    at::Int
end

Base.show(io::IO, s::Seed) = print(io, s.kind === :fixed ? "set fixed $(s.at)" : s.kind === :recurrence ? "recurrence $(s.at)" : "$(s.kind) %$(s.at)")

"""
A kernel as formed, from its seed (see `Seed`), before what no other kernel
reads is left unstored; what it reads; and what its forming read (see
`Note`), sorted.
"""
struct Formed
    seed::Seed
    group::Group
    reads::Set{Int}
    stores::Vector{Int}
    owners::Vector{Int}
    roots::Vector{Int}
    walks::Vector{Walk}
end

"A context that reads the kernels as they form and notes nothing."
struct Plain <: Forming
    kernels::Kernels
end

"A context that reads the kernels as they form and notes what each kernel's forming read."
struct Recording <: Forming
    kernels::Kernels
    note::Note
end

Recording(k::Kernels) = Recording(k, Note(length(k.stored)))

"""
A recurrence's step as its group forms its terms, through context `cx`: the
values of the cycle the group does not keep are computed within the step,
whether stored or not.
"""
struct Stepping <: Forming
    cx::Forming
    within::Set{Int}
end

isstored(cx::Plain, v) = cx.kernels.stored[v]
isstored(cx::Recording, v) = (note!(cx.note.stores, cx.note.stored_seen, v); cx.kernels.stored[v])
isstored(cx::Stepping, v) = !(v in cx.within) && isstored(cx.cx, v)

owned(cx::Plain, v) = cx.kernels.owner[v] != 0
owned(cx::Recording, v) = (note!(cx.note.owners, cx.note.owned_seen, v); cx.kernels.owner[v] != 0)

note!(noted, seen, v) = (seen[v] || (seen[v] = true; push!(noted, v)); nothing)

"""
    noted(f, cx, graph, seed) -> (g, formed)

The kernel `f()` forms through `cx` from `seed`, and what its forming read:
a `Formed` where `cx` records, nothing where it does not.
"""
noted(f, cx::Plain, graph, seed) = (f(), nothing)

function noted(f, cx::Recording, graph, seed)
    n = cx.note
    foreach(v -> n.stored_seen[v] = false, n.stores)
    foreach(v -> n.owned_seen[v] = false, n.owners)
    empty!(n.stores), empty!(n.owners), empty!(n.roots), empty!(n.walks)
    g = f()
    # The group the plan holds has its outputs trimmed; the one noted is kept as formed.
    kept = Group(g.kind, g.domain, g.reductions, copy(g.outputs), g.terms, g.phase)
    g, Formed(seed, kept, group_reads(graph, g), sort(n.stores), sort(n.owners), copy(n.roots), copy(n.walks))
end

"""
    formed!(cx, g, reads = nothing)

Kernel `g` formed: the next, owning its members, reading `reads` where given.
"""
function formed!(cx::Union{Plain, Recording}, g::Group, reads = nothing)
    k = cx.kernels
    push!(k.groups, g)
    foreach(m -> k.owner[m] = length(k.groups), members(g))
    reads === nothing || (k.reads[length(k.groups)] = reads)
    nothing
end

# What a walk looked at, noted where `cx` records.
walked_from!(cx::Plain, r) = nothing
walked_from!(cx::Recording, r) = (push!(cx.note.roots, r); nothing)
walk_found!(cx::Plain, reads, group, found) = nothing
walk_found!(cx::Recording, reads, group, found) =
    (push!(cx.note.walks, Walk(collect(Int, reads), collect(Int, group), found, length(cx.note.roots))); nothing)

term_loads(cx::Union{Plain, Recording}, t::Term) = get!(() -> stored_reads(t), cx.kernels.loads, t)

# Whether any of the values `reads` depends on a member of `group`: through
# its ancestors, and through the kernels that own any of them, each of which
# runs after everything its members read. A kernel that read such a value and
# joined such a member would make kernels depend on each other in a cycle.
# A value among the ancestors of one already looked at is passed over: its
# ancestors are among them, and their kernels were looked at.
# The latest value pending is looked at first: values are numbered after
# their operands, so its ancestors cover most of the others.
function depends(graph, cx::Union{Plain, Recording}, reads, group)
    tally!(:walked)
    k = cx.kernels
    (; pending, covered, fresh, kernels) = k.walk
    fill!(pending, false)
    fill!(covered, false)
    empty!(kernels)
    top = 0
    for r in reads
        pending[r] = true
        top = max(top, r)
    end
    while top > 0 && (r = findprev(pending, top)) !== nothing
        pending[r] = false
        top = r
        covered[r] && continue
        walked_from!(cx, r)
        if any(m -> m == r || k.up[r][m], group)
            walk_found!(cx, reads, group, r in reads ? direct : through_kernels)
            return true
        end
        # What `r` brings that is not covered yet, `r` among it.
        up, f, c = k.up[r].chunks, fresh.chunks, covered.chunks
        for w in eachindex(c)
            f[w] = up[w] & ~c[w]
            c[w] |= f[w]
        end
        fresh[r] = true
        covered[r] = true
        for u in findall(fresh)
            o = k.owner[u]
            (o == 0 || o in kernels) && continue
            push!(kernels, o)
            for x in get!(() -> group_reads(graph, k.groups[o]), k.reads, o)
                covered[x] || (pending[x] = true; top = max(top, x))
            end
        end
    end
    walk_found!(cx, reads, group, apart_from)
    false
end
