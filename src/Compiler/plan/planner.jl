# The decision procedure: which values are stored, which are recomputed inside
# their readers, and how the work groups into kernels.
# What is stored is decided first (storing.jl); then kernels form, one grown
# from each seed in turn: a group (groups.jl) takes a value where its term
# (terms.jl) stays inside the group's domain and no kernel would come to
# depend on itself (context.jl). Recurrences are in recurrences.jl, and
# planning again near a plan in replan.jl.

"""
What the planner is given: the sets of values fixed as groups, in order, the
folds recomputed where they are read, the maps stored apart, and the sums
riding their recurrence's steps, in order (see `plan`). Inputs compose by
`merge`, the sets and sums of each after those before: what each of the
cover's proposals changes (see `change`).
"""
struct Inputs
    fixed::Vector{Vector{Int}}
    recompute::Set{Int}
    store::Set{Int}
    ride::Vector{Int}
end

Inputs(; fixed = Vector{Int}[], recompute = (), store = (), ride = ()) =
    Inputs(collect(Vector{Int}, fixed), Set{Int}(recompute), Set{Int}(store), collect(Int, ride))

Base.merge(a::Inputs, b::Inputs) = Inputs([a.fixed; b.fixed], a.recompute ∪ b.recompute, a.store ∪ b.store, [a.ride; b.ride])

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
    tally!(:plans)
    i = Inputs(; fixed, recompute, store, ride)
    s = storing(graph; i.recompute, i.store, i.ride)
    first(passes(Grounds(graph, s), s, i, Plain))
end

"""
What forming every plan of one graph shares, whatever the planner is given:
the graph as given and with its results stored, what runs, the results, each
value's ancestors, the values that read each, and through a gather too
(`followers`, those that run only), the recurrences, and the seeds of the
kernels other than the fixed sets', in the order they form (see `Seed`).
"""
struct Grounds
    source::IndexGraph
    graph::IndexGraph
    live::BitVector
    results::Set{Int}
    up::Vector{BitVector}
    users::Vector{Vector{Int}}
    followers::Vector{Vector{Int}}
    recurrences::Vector{Cycle}
    seeds::Vector{Seed}
end

# The grounds of `source`, from what `storing` decided of it.
function Grounds(source::IndexGraph, s)
    graph, live = s.graph, s.live
    values = graph.values
    n = length(values)
    # Who reads each value, directly or through a gather: a group grows toward
    # them, so it never scans the values that cannot reach it.
    followers = [Int[] for _ in values]
    for id in 1:n, o in operands(values[id].definition), r in read_values(o)
        live[id] && push!(followers[r], id)
    end
    foreach(unique!, followers)
    recurring = recurrences(graph)
    # Scans first, so that a fold along a scan's axes rides its sweep, wherever
    # it is in the graph; folds by key last, so that they join a group over
    # their iteration where one forms. Then the maps, constants and inputs
    # aside, which no kernel computes.
    folds = [filter(id -> values[id].definition isa Scan, 1:n); filter(id -> values[id].definition isa Reduction, 1:n);
        filter(id -> values[id].definition isa Keyed, 1:n)]
    maps = filter(id -> !(values[id].definition isa Union{Input, ConstantArray}), 1:n)
    seeds = Seed[[RecurrenceSeed(k) for k in eachindex(recurring)]; [FoldSeed(id) for id in folds]; [MapSeed(id) for id in maps]]
    Grounds(source, graph, live, s.results, ancestors(graph), readers(graph), followers, recurring, seeds)
end

"""
    passes(grounds, s, i::Inputs, context, before = nothing) -> (plan, first, second)

The plan of `i`, given what `storing` decided of it (`s`), its kernels formed
through a context of type `context` (see `Forming`), near the formations of
`before` where given (see `replay`); and the formation of each grouping
pass, nothing where the context does not record, the first where there was
no second.
"""
function passes(grounds::Grounds, s, i::Inputs, context, before = nothing)
    stored = copy(s.stored)
    # A pointwise value read only by one group over its domain, at the same
    # index, is recomputed by that group: grouped again without it.
    grouped = formation(grounds, context(Kernels(copy(stored), grounds.up)), i.fixed, s.riders, replay(before, 1, stored, i.fixed, s.riders))
    p = kernels(grounds, grouped.kernels)
    n = length(grounds.graph.values)
    shared = filter(id -> !any(set -> id in set, i.fixed) && recomputable(p, grounds.users, id, grounds.live, grounds.results, s.materialized), 1:n)
    regrouped = grouped
    if !isempty(shared)
        stored[shared] .= false
        regrouped = formation(grounds, context(Kernels(stored, grounds.up)), i.fixed, s.riders, replay(before, 2, stored, i.fixed, s.riders))
        p = kernels(grounds, regrouped.kernels)
    end
    # Within a preparation that verifies, every plan the cover tries.
    VERIFYING[] && verify(p)
    p, grouped.formation, regrouped.formation
end

"""
How one grouping pass formed its kernels, kept where its context records: what
was stored as it began, the sets fixed and the sums riding each recurrence's
steps, by the recurrence's state; and each kernel as formed (see `Formed`),
in order, whether taken as it was from a formation before (`taken`), and its
place by its seed (`at`).
"""
struct Formation
    stored::Vector{Bool}
    fixed::Vector{Vector{Int}}
    riders::Dict{Int, Vector{Int}}
    formed::Vector{Formed}
    taken::Vector{Bool}
    at::Dict{Seed, Int}
end

Formation(cx::Plain, fixed, riders) = nothing
Formation(cx::Recording, fixed, riders) = Formation(copy(cx.kernels.stored), fixed, riders, Formed[], Bool[], Dict{Seed, Int}())

kept!(::Nothing, f, taken) = nothing
function kept!(record::Formation, f::Formed, taken)
    push!(record.formed, f)
    push!(record.taken, taken)
    record.at[f.seed] = length(record.formed)
    nothing
end

# Forming with no formation before: none taken, nothing to note of what
# changed (see `Replay`).
replay(::Nothing, pass, stored, fixed, riders) = nothing
taken(::Nothing, seed, cx) = nothing
grew!(::Nothing, f) = nothing
passed!(::Nothing, seed) = nothing

"""
    formation(grounds, cx, fixed, riders, near = nothing) -> (; kernels, formation)

The kernels of one grouping pass, formed through context `cx` from each seed
in turn (see `Seed`): a kernel of each set in `fixed`, of each recurrence,
with the sums in `riders` riding its steps, then one grown from each fold and
each map no kernel holds yet. Where `near` replays the formation of a plan
before (see `Replay`), a kernel it formed from the same seed is taken as it
was where nothing its forming read differs; the rest form here. With them,
how they formed, where `cx` records.
"""
function formation(grounds::Grounds, cx::Forming, fixed, riders, near = nothing)
    tally!(:groupings)
    record = Formation(cx, fixed, riders)
    for seed in Seed[[FixedSeed(k) for k in eachindex(fixed)]; grounds.seeds]
        if !starts(grounds, cx, seed)
            passed!(near, seed)
            continue
        end
        f = taken(near, seed, cx)
        if f === nothing
            tally!(:formed)
            g, f = noted(() -> grown(grounds, cx, seed, fixed, riders), cx, grounds.graph, seed)
            formed!(cx, g, f === nothing ? nothing : f.reads)
            f === nothing || grew!(near, f)
            kept!(record, f, false)
        else
            tally!(:reused)
            formed!(cx, Group(f.group.kind, f.group.domain, f.group.reductions, copy(f.group.outputs), f.group.terms, f.group.phase), f.reads)
            kept!(record, f, true)
        end
    end
    (; kernels = cx.kernels, formation = record)
end

# Whether a kernel forms from `seed`: from a fixed set always; from a
# recurrence that runs; from a stored fold or map that runs and no kernel
# holds yet.
starts(grounds::Grounds, cx::Forming, seed::FixedSeed) = true
starts(grounds::Grounds, cx::Forming, seed::RecurrenceSeed) = grounds.live[first(grounds.recurrences[seed.at].states)]
starts(grounds::Grounds, cx::Forming, seed::Union{FoldSeed, MapSeed}) = grounds.live[seed.at] && !owned(cx, seed.at) && isstored(cx, seed.at)

# The kernel formed from `seed` (see `formation`).
grown(grounds::Grounds, cx::Forming, seed::FixedSeed, fixed, riders) = fixed_group(grounds.graph, cx, fixed[seed.at])
grown(grounds::Grounds, cx::Forming, seed::FoldSeed, fixed, riders) = grown_fold(grounds, cx, seed.at)
grown(grounds::Grounds, cx::Forming, seed::MapSeed, fixed, riders) = grown_map(grounds, cx, seed.at)

function grown(grounds::Grounds, cx::Forming, seed::RecurrenceSeed, fixed, riders)
    recurrence = grounds.recurrences[seed.at]
    g = recurrence_group(grounds.graph, cx, recurrence, get(riders, first(recurrence.states), Int[]))
    # What a recurrence computes at each step is its group's alone: a set
    # fixed apart holding any of it would be a second kernel writing it.
    held = filter(m -> owned(cx, m), members(g))
    isempty(held) || throw(NotAGroup(held))
    g
end

# The kernel grown from fold or scan `id`.
function grown_fold(grounds::Grounds, cx::Forming, id)
    graph, followers = grounds.graph, grounds.followers
    values = graph.values
    d = values[id].definition
    g = d isa Scan ? Group(ScanKind(), values[id].dims) : Group(ReductionKind(), d.iteration)
    after = d isa Scan ? 0 : id
    try_add!(graph, cx, g, id; fold = true)
    # A group reaches what reads its members, and what reads what they
    # read. Of the latter, a value that depends on a member is passed
    # over: the last value stored on a path to it from the group would be
    # among its loads and depend on the group, so it joins only where
    # every such path runs through what the group computes, and then it
    # is reached from a member. The weights every step of a long program
    # reads would otherwise bring every later step to every group.
    apart(w) = any(m -> grounds.up[w][m], members(g))
    frontier = reach!(Int[], followers, cx, members(g), after)
    reach!(frontier, followers, cx, [r for m in members(g) for r in term_loads(cx, g.terms[m])], after; skip = apart)
    # A pass visits the frontier in order; what a new member reaches above
    # the cursor is visited in this pass, the rest in the next.
    changed = true
    while changed
        changed = false
        k = 1
        while k <= length(frontier)
            other = frontier[k]
            k += 1
            grounds.live[other] && !owned(cx, other) && !(other in members(g)) || continue
            od = values[other].definition
            added = if g.kind isa ReductionKind && od isa Reduction
                # A fold over other positions widens the domain, if it joins.
                domain = joint_domain(graph, g, other)
                h = domain === nothing ? nothing : domain == g.domain ? g : with_domain(g, domain)
                h !== nothing && try_add!(graph, cx, h, other; fold = true) && (g = h; true)
            else
                od isa Union{Reduction, Scan, Keyed} ? sweeps_with(graph, g, other) && try_add!(graph, cx, g, other; fold = true) :
                    od isa Pointwise && other > id && !awaits_sweep(grounds, cx, g, other) && adds_to(values, g, other) &&
                        try_add!(graph, cx, g, other; fold = false)
            end
            added || continue
            changed = true
            reach!(frontier, followers, cx, [other], after)
            reach!(frontier, followers, cx, term_loads(cx, g.terms[other]), after; skip = apart)
            k = searchsortedfirst(frontier, other) + 1
        end
    end
    g
end

# The kernel grown from stored map `id`, joined by the later ones over its
# domain that read it.
function grown_map(grounds::Grounds, cx::Forming, id)
    graph = grounds.graph
    values = graph.values
    values[id].definition isa Pointwise || error("value %$id has no kernel")
    g = Group(PointwiseKind(), values[id].dims)
    try_add!(graph, cx, g, id; fold = false)
    frontier = reach!(Int[], grounds.followers, cx, [id], id)
    k = 1
    while k <= length(frontier)
        other = frontier[k]
        k += 1
        grounds.live[other] && !owned(cx, other) && values[other].definition isa Pointwise && values[other].dims == g.domain || continue
        any(m -> grounds.up[other][m], members(g)) && adds_to(values, g, other) && try_add!(graph, cx, g, other; fold = false) || continue
        reach!(frontier, grounds.followers, cx, [other], id)
    end
    g
end

# Insert into the sorted `frontier` the stored values after `seed` that read
# any of `from`, directly or through values a term computes in place: the
# only values whose terms can reach a group grown from `from`. Those `skip`
# names are passed over, and what reads them: what depends on a value that
# depends on the group's members does too.
function reach!(frontier, followers, cx::Forming, from, seed; skip = w -> false)
    pending, seen = collect(from), Set{Int}()
    while !isempty(pending)
        for w in followers[pop!(pending)]
            w in seen && continue
            push!(seen, w)
            skip(w) && continue
            if !isstored(cx, w)
                push!(pending, w)
            elseif w > seed
                k = searchsortedfirst(frontier, w)
                get(frontier, k, 0) == w || insert!(frontier, k, w)
            end
        end
    end
    frontier
end

# Try to add a fold or an output to a group. Its term must stay inside the
# group's domain, nothing it loads may depend on the group, nothing the group
# loads may depend on it, and no member may already load it. With
# `recomputing`, a fold joins a group whose folds would then cover different
# positions only where it shares what they recompute (see `mixes`).
function try_add!(graph, cx::Forming, g::Group, v; fold, recomputing = true)
    tally!(:tries)
    val = graph.values[v]
    # An output covers the domain, or the positions of the primary folds.
    index = fold ? identity_index(g.domain) : placement(graph, g, v)
    index === nothing && return false
    any(m -> v in term_loads(cx, g.terms[m]) || v in keyed_reads(graph, m), members(g)) && return false
    any(in(members(g)), keyed_reads(graph, v)) && return false
    candidate = fold ? g : Group(g.kind, g.domain, g.reductions, push!(copy(g.outputs), v), g.terms, g.phase)
    # What the term would load, from the graph before the term is built: most
    # candidates fail here, and a term inlines every value recomputed in it.
    loads = Set{Int}()
    fold ? operand_reads!(loads, Set{Int}(), graph, cx, candidate, val.definition.operand) :
        value_reads!(loads, Set{Int}(), graph, cx, candidate, v)
    reads = append!(collect(loads), keyed_reads(graph, v))
    group = Set(members(g))
    depends(graph, cx, reads, group) && return false
    t = try
        fold ? operand_term(graph, cx, candidate, val.definition.operand, index, Domain(g.domain)) :
        value_term(graph, cx, candidate, v, index)
    catch e
        e isa OutsideGroup || rethrow()
        return false
    end
    # A member reads the group's folds, or something the group reads.
    shared = Set(r for m in members(g) for r in term_loads(cx, g.terms[m]))
    isempty(g.reductions) || !isempty(stats(t)) || any(in(shared), reads) || return false
    fold && !mixes(graph, g, v, t; recomputing) && return false
    # Nor may what the group reads from other kernels depend on `v`.
    depends(graph, cx, filter(!in(group), shared), Set((v,))) && return false
    push!(fold ? g.reductions : g.outputs, v)
    g.terms[v] = t
    g.phase[v] = maximum(s -> g.phase[s], stats(t); init = 0) + (fold ? 1 : 0)
    true
end

"""
Whether fold or scan `v` joins the sweep of group `g`: a fold a reduction's
group over its iteration; a scan, or a fold along its axes, a scan's group
over its domain, the scan in the group's direction.
"""
function sweeps_with(graph, g::Group, v)
    d = graph.values[v].definition
    g.kind isa ReductionKind && return d isa Union{Reduction, Keyed} && d.iteration == g.domain
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
    g.kind isa ReductionKind && d isa Union{Reduction, Keyed} || return nothing
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
    folds = push!(copy(g.reductions), v)
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
awaits_sweep(grounds::Grounds, cx::Forming, g::Group, v) = g.kind isa ScanKind &&
    any(s -> grounds.graph.values[s].definition isa Scan && !(s in g.reductions) && !owned(cx, s), findall(grounds.up[v]))

# Whether group `g` may take value `v`: anything but a step of an
# accumulation, which joins the group of a fold it adds. Through the step
# before it, every step would join the first one's group, which would load
# every term.
adds_to(values, g::Group, v) = !accumulates(values, v) || any(o -> o isa Read && o.value in g.reductions, values[v].definition.operands)

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
function fixed_group(graph, cx::Forming, set)
    values = graph.values
    all(v -> !owned(cx, v), set) || throw(NotAGroup(set))
    folds = filter(v -> values[v].definition isa Reduction, set)
    # Folds over other extents along axes they keep share the box that holds
    # them (see `joint_domain`).
    domain = isempty(folds) ? values[first(set)].dims : values[first(folds)].definition.iteration
    for v in folds
        extents = values[v].definition.iteration
        length(extents) == length(domain) && (domain = map(max, domain, extents))
    end
    g = isempty(folds) ? Group(PointwiseKind(), domain) : Group(ReductionKind(), domain)
    for v in set
        dv = values[v].definition
        added = dv isa Reduction ? joint_domain(graph, g, v) == g.domain && try_add!(graph, cx, g, v; fold = true, recomputing = false) :
            dv isa Pointwise && isstored(cx, v) && try_add!(graph, cx, g, v; fold = false)
        added || throw(NotAGroup(set))
    end
    g
end

"""
    kernels(grounds, k::Kernels) -> Plan

The plan of the kernels a pass formed, in an order where each runs after
what it reads, each value only its own kernel reads left unwritten.
"""
function kernels(grounds::Grounds, k::Kernels)
    graph, stored, groups, owner = grounds.graph, k.stored, k.groups, k.owner
    n = length(graph.values)
    # A value only its own kernel reads need not be written.
    external = falses(n)
    for o in graph.results, r in read_values(o)
        external[r] = true
    end
    for (j, g) in enumerate(groups), r in get!(() -> group_reads(graph, g), k.reads, j)
        owner[r] != j && (external[r] = true)
    end
    # The folds a fold over other than the primary axes reads stay stored: a
    # realization that completes it in a launch of its own loads them.
    for g in groups, r in g.reductions, s in stats(g.terms[r])
        primary(graph, g, r) || (external[s] = true)
    end
    for g in groups
        g.kind isa Union{ReductionKind, ScanKind, PointwiseKind} || continue
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

# Kernels in an order where every stored value is written before it is read:
# in passes over the kernels in their order, each pass taking every kernel
# whose writers are taken, the ones it took already among them. A kernel's
# pass is found from its writers': a writer before it taken in the same
# pass, one after it in the pass before. So each kernel is looked at once,
# where passes repeated until all were taken looked at every kernel a pass,
# as many passes as the program has layers: 22 million times for the GPT
# step's 1354 kernels at 32 layers.
function order_groups(graph, groups, owner)
    n = length(groups)
    needs = map(eachindex(groups)) do k
        deps = Set(owner[r] for r in group_reads(graph, groups[k]) if owner[r] > 0)
        delete!(deps, k)
        deps
    end
    tally!(:ordered, n)
    # Each kernel's pass once every writer's is known, in an order where
    # writers come first.
    waiting = map(length, needs)
    after = [Int[] for _ in groups]
    foreach(k -> foreach(d -> push!(after[d], k), needs[k]), eachindex(groups))
    ready = filter(k -> waiting[k] == 0, collect(eachindex(groups)))
    pass = ones(Int, n)
    done = 0
    while !isempty(ready)
        k = pop!(ready)
        done += 1
        for d in needs[k]
            pass[k] = max(pass[k], d < k ? pass[d] : pass[d] + 1)
        end
        for j in after[k]
            waiting[j] -= 1
            waiting[j] == 0 && push!(ready, j)
        end
    end
    done == n || error("kernels depend on each other in a cycle")
    sort!(collect(eachindex(groups)); by = k -> (pass[k], k))
end
