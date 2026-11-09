# Planning again near a plan: the cover's trials each change what the planner
# is given by a few values (see `Inputs`), and most of a plan's kernels form
# from the same seeds reading the same state. A plan kept with how it formed
# (`Planned`) is formed again with a change (`replanned`) by taking each
# kernel whose forming read nothing the change makes differ, as it was, and
# forming the rest. What a kernel's forming read is what it read through its
# context (see `Forming`), and the comparison with planning whole, within a
# preparation that verifies, is what holds the two to one plan.

"""
    Planned(graph, i::Inputs) -> Planned

A plan of `graph` given `i`, with what each grouping pass of it formed and
what forming each kernel read (see `Formation`), so that a change to what it
is given forms again only the kernels the change reaches (see `replanned`).
"""
struct Planned
    plan::Plan
    inputs::Inputs
    grounds::Grounds
    first::Formation
    second::Formation
end

function Planned(graph::IndexGraph, i::Inputs)
    tally!(:plans)
    s = storing(graph; i.recompute, i.store, i.ride)
    grounds = Grounds(graph, s)
    p, first, second = passes(grounds, s, i, Recording)
    Planned(p, i, grounds, first, second)
end

"""
    replanned(p::Planned, change::Inputs) -> Planned

The plan of what `p` was given with `change` merged in, formed near `p`:
each kernel whose forming read nothing that differs is taken as `p` formed
it, and the rest form again (see `Replay`). Crosschecking (see
`crosschecking`), it is compared with the plan formed whole, and where they
differ it throws, naming the first kernel that differs and what forming it
whole read that the kernel taken had not.
"""
function replanned(p::Planned, change::Inputs)
    tally!(:plans)
    i = merge(p.inputs, change)
    s = storing(p.grounds.source; i.recompute, i.store, i.ride)
    plan, first, second = passes(p.grounds, s, i, Recording, p)
    q = Planned(plan, i, p.grounds, first, second)
    if crosschecking()
        whole, grouped, regrouped = passes(p.grounds, s, i, Recording)
        agree(q, Planned(whole, i, p.grounds, grouped, regrouped), p)
    end
    q
end

"""
A grouping pass formed near the same pass of a plan before, as `formation`
takes it (see `taken`, `grew!` and `passed!`): the graph, that pass's
formation (`before`), the sets fixed and the sums riding here, and what differs from
it as the kernels form. The values whose storage differs (`restored`); the
members of the kernels formed so far there and not here (`lost`), and here
and not there (`gained`); all of them (`changed`, and in order `changes`);
and the members and reads of each kernel formed here unlike there
(`reformed`).
"""
struct Replay
    graph::IndexGraph
    before::Formation
    fixed::Vector{Vector{Int}}
    riders::Dict{Int, Vector{Int}}
    up::Vector{BitVector}
    restored::BitVector
    lost::BitVector
    gained::BitVector
    changed::BitVector
    changes::Vector{Int}
    reformed::Vector{Tuple{Vector{Int}, Set{Int}}}
end

function replay(p::Planned, pass, stored, fixed, riders)
    before = pass == 1 ? p.first : p.second
    n = length(stored)
    restored = BitVector(stored .!= before.stored)
    r = Replay(p.grounds.graph, before, fixed, riders, p.grounds.up, restored, falses(n), falses(n), copy(restored), findall(restored),
        Tuple{Vector{Int}, Set{Int}}[])
    # A set fixed there and not here takes its kernel away.
    for k in (length(fixed) + 1):length(before.fixed)
        j = get(before.at, Seed(:fixed, k), 0)
        j == 0 || lose!(r, before.formed[j].group)
    end
    r
end

change!(r::Replay, v) = (r.changed[v] || (r.changed[v] = true; push!(r.changes, v)); nothing)
lose!(r::Replay, g::Group) = foreach(m -> (r.lost[m] = true; change!(r, m)), members(g))

function gain!(r::Replay, f::Formed)
    foreach(m -> (r.gained[m] = true; change!(r, m)), members(f.group))
    push!(r.reformed, (members(f.group), f.reads))
end

# The kernel formed there from `seed`, where its seed is the same here: the
# same set fixed, the same sums riding the same recurrence.
function formed_before(r::Replay, seed::Seed)
    j = get(r.before.at, seed, 0)
    j == 0 && return nothing
    f = r.before.formed[j]
    seed.kind === :fixed && r.before.fixed[seed.at] != r.fixed[seed.at] && return nothing
    if seed.kind === :recurrence
        c = first(f.group.outputs)
        get(r.riders, c, Int[]) == get(r.before.riders, c, Int[]) || return nothing
    end
    f
end

"""
The kernel formed there from `seed`, where forming it here, through context
`cx`, would read nothing that differs.
"""
function taken(r::Replay, seed::Seed, cx)
    f = formed_before(r, seed)
    f !== nothing && unchanged(r, f, cx) ? f : nothing
end

"Kernel `f` formed here: where it is unlike the one formed there from its seed, what it changes."
function grew!(r::Replay, f::Formed)
    j = get(r.before.at, f.seed, 0)
    j != 0 && same_group(r.before.formed[j].group, f.group) && return
    gain!(r, f)
    j == 0 || lose!(r, r.before.formed[j].group)
    nothing
end

"No kernel formed here from `seed`: the one formed there, if any, is lost."
function passed!(r::Replay, seed::Seed)
    j = get(r.before.at, seed, 0)
    j == 0 || lose!(r, r.before.formed[j].group)
    nothing
end

"""
Whether forming `f` here, through context `cx`, reads what it read there.
Whether a kernel owns each value it looked at that of, and whether each
value it looked at that of is stored, are as they were where nothing
differs of them. Each walk of what depends on the group (see `Walk`) finds
what it found: one that found a dependence on a value it was asked about
finds it again, whatever kernels formed; one that found none finds none
where each kernel formed otherwise here, owning a value it covered, reads
only values it covered, as a kernel owning none of them, or owning them no
longer, only takes away; any other found what it did through kernels, and
finds it again where nothing it covered differs. A walk these leave in
doubt is asked again, of the kernels formed here.
"""
function unchanged(r::Replay, f::Formed, cx)
    any(v -> r.lost[v] != r.gained[v], f.owners) && return false
    any(v -> r.restored[v], f.stores) && return false
    start = 1
    for w in f.walks
        roots = view(f.roots, start:w.stop)
        start = w.stop + 1
        w.found === direct || alike(r, w, roots) || depends(r.graph, cx, w.reads, Set(w.group)) == (w.found !== apart_from) || return false
    end
    true
end

# Whether walk `w`, which looked at the ancestors of `roots`, finds here what
# it found there, as far as what differs tells without walking.
function alike(r::Replay, w::Walk, roots)
    covers(v) = any(x -> x == v || r.up[x][v], roots)
    w.found === apart_from && return !any(((ms, rs),) -> any(covers, ms) && !all(covers, rs), r.reformed)
    n = length(r.changed)
    for x in roots
        r.changed[x] && return false
        a = r.up[x]
        if 64 * length(r.changes) < n
            any(v -> a[v], r.changes) && return false
        else
            any(k -> a.chunks[k] & r.changed.chunks[k] != 0, eachindex(a.chunks)) && return false
        end
    end
    true
end

same_group(a::Group, b::Group) = a.kind === b.kind && a.domain == b.domain && a.reductions == b.reductions &&
    a.outputs == b.outputs && a.terms == b.terms && a.phase == b.phase

"""
Whether `q`, formed near plan `p`, is the plan formed whole (`whole`): each
pass's kernels as formed, in order, and the plan. Where not, an error names
the first kernel that differs, whether it was taken as `p` formed it, what
it noted that differs from when `p` formed it (whether a value is stored,
whether a kernel owns it), and what forming it whole read that it had not
noted.
"""
function agree(q::Planned, whole::Planned, p::Planned)
    for (pass, a, b, c) in ((1, q.first, whole.first, p.first), (2, q.second, whole.second, p.second))
        k = findfirst(j -> j > length(b.formed) || a.formed[j].seed != b.formed[j].seed ||
            !same_group(a.formed[j].group, b.formed[j].group), eachindex(a.formed))
        k === nothing && length(a.formed) == length(b.formed) && continue
        k === nothing && (k = length(a.formed) + 1)
        f, g = get(a.formed, k, nothing), get(b.formed, k, nothing)
        what = f === nothing ? "no kernel" : string("the kernel from ", f.seed, a.taken[k] ? ", taken as formed before" : ", formed again")
        noted = ""
        if f !== nothing && a.taken[k]
            # Whether a kernel owned each value it noted, there as it formed and here.
            owned_at(x, upto) = Set(m for j in 1:(upto - 1) for m in members(x.formed[j].group))
            there, here = owned_at(c, c.at[f.seed]), owned_at(a, k)
            noted = string("; of what it noted, the storage of ", filter(v -> a.stored[v] != c.stored[v], f.stores), " and the owner of ",
                filter(v -> (v in there) != (v in here), f.owners), " differ")
        end
        missed = f === nothing || g === nothing ? "" : string("; forming it whole read the storage of ", setdiff(g.stores, f.stores),
            ", the owners of ", setdiff(g.owners, f.owners), " and walked from ", setdiff(g.roots, f.roots), ", which it had not noted")
        error("a plan formed near another differs from the plan formed whole in grouping pass $pass at its kernel $k: ",
            what, " where whole there is ", g === nothing ? "none" : string("the kernel from ", g.seed), noted, missed)
    end
    a, w = q.plan, whole.plan
    a.stored == w.stored && a.owner == w.owner && length(a.groups) == length(w.groups) && all(splat(same_group), zip(a.groups, w.groups)) ||
        error("a plan formed near another differs from the plan formed whole, every kernel formed alike")
    nothing
end
