# Which groups there are: the cover. Implementations propose candidates of
# their own, sets of values grown from anchors that they accept; the planner
# offers its own joins of a stored value into the kernel that reads it, folds
# recomputed where they are read, and maps stored apart from a kernel a claim
# on which failed to build. A proposal replaces the generator's groups when
# the plan with it is cheaper, each compared on the groups that differ.

"""
    grow(graph, anchor, accept) -> Vector{Int}

A candidate grown from `anchor` along its readers: each step adds the first
reader of the set, in value order, with which `accept(set)` holds.
"""
function grow(graph, anchor, accept)
    users = readers(graph)
    set = [anchor]
    while true
        next = sort!(unique(u for v in set for u in users[v] if !(u in set)))
        k = findfirst(u -> accept([set; u]), next)
        k === nothing && return set
        push!(set, next[k])
    end
end

"A proposal the cover compared with its plan at the time, and whether it took it."
struct Considered
    proposal::Proposal
    implementation::Union{Nothing, Implementation} # nothing for the planner's own
    comparison::Comparison
    accepted::Bool
end

"""
What a plan's neighbourhoods need, found once per plan: the values each
group owns, where the planner forms each group (a fixed set's index, a fold's
seed or a pointwise seed), the groups that read each value, and the values
that read each.
"""
struct Around
    owned::Vector{Vector{Int}}
    places::Vector{Tuple{Symbol, Int}}
    reading::Vector{Vector{Int}}
    followers::Vector{Vector{Int}}
end

function Around(p::Plan, sets)
    owned = [Int[] for _ in p.groups]
    for (v, k) in enumerate(p.owner)
        k == 0 || push!(owned[k], v)
    end
    function place(j)
        g = p.groups[j]
        i = findfirst(s -> p.owner[first(s)] == j, sets)
        i === nothing || return (:fixed, i)
        seeded_at(g.kind, g, owned[j])
    end
    reading = [Int[] for _ in p.graph.values]
    for (j, g) in enumerate(p.groups), r in group_reads(p.graph, g)
        push!(reading[r], j)
    end
    followers = [Int[] for _ in p.graph.values]
    for (id, v) in enumerate(p.graph.values), o in operands(v.definition), r in read_values(o)
        push!(followers[r], id)
    end
    Around(owned, map(place, eachindex(p.groups)), reading, followers)
end

# Where group `g`, of its kind and owning `owned`, is seeded, by the value it grew from.
seeded_at(::ReductionKind, g::Group, owned) = (:fold, first(g.reductions))
seeded_at(::PointwiseKind, g::Group, owned) = (:pointwise, minimum(owned))
seeded_at(::RecurrenceKind, g::Group, owned) = (:recurrence, first(g.outputs))
seeded_at(::ScanKind, g::Group, owned) = (:scan, first(g.reductions))
seeded_at(::CrossingKind, g::Group, owned) = (:crossing, only(g.outputs))

# The stored values `v` reads, directly or through values not stored.
function loaded(p::Plan, v)
    found, pending, seen = Int[], [v], Set{Int}()
    while !isempty(pending)
        for o in operands(p.graph.values[pop!(pending)].definition), r in read_values(o)
            r in seen && continue
            push!(seen, r)
            p.stored[r] || p.owner[r] != 0 ? push!(found, r) : push!(pending, r)
        end
    end
    found
end

# The groups of `p` that value `v` reaches directly: its own, those owning
# what it loads, and those reading it, directly or through values not stored.
function reached(p::Plan, a::Around, v)
    v > length(p.owner) && return Int[]
    found = pushfirst!([p.owner[r] for r in loaded(p, v)], p.owner[v])
    pending, seen = [v], Set{Int}()
    while !isempty(pending)
        for w in a.followers[pop!(pending)]
            w in seen && continue
            push!(seen, w)
            push!(found, p.owner[w])
            p.stored[w] || push!(pending, w)
        end
    end
    filter!(!=(0), found)
end

# The groups of `p` that a change to value `v` could change: those it
# reaches, and those that could grow into it, reading what it loads; in a
# stack of layers the latter may span them all.
neighbours(p::Plan, a::Around, v) =
    v > length(p.owner) ? Int[] : [reached(p, a, v); (a.reading[r] for r in loaded(p, v))...]

# The groups of `p` that hold value `v`: its own, and those owning what it
# loads; a map no group owns is computed inside the groups that read it,
# directly or through values not stored, and is held by them. Without them a
# map's storing apart had one key in every plan, and its price in the
# starting plan passed for its price in the plan a storing taken had
# changed: in `w * log10.(1 .+ x .^ 2)`, storing `1 .+ x .^ 2` as well lost
# by 8% measured against the product computing everything, a margin a shared
# runner flipped, where against the plan storing the log10 map it loses by
# estimate.
held(p::Plan, a::Around, v) =
    v > length(p.owner) ? Int[] :
    p.owner[v] == 0 && !p.stored[v] ? reached(p, a, v) :
    filter!(!=(0), [p.owner[v]; [p.owner[r] for r in loaded(p, v)]])

"""
A proposal's key: its place in a plan, by the values it fixes, the groups
that hold them (`held`), in the order the planner forms them, and its values
among them, numbered as `canonical` numbers one group's. Proposals with one
key in the generator's plan are the same edit in identical layers, and are
planned once. The groups that read its stored values are not among them: in
a stack of layers those include a sum over every layer, such as an
optimizer's update of weights every layer reads, in which each layer's term
sits at a place of its own, and no two layers would be alike. The groups
that recompute a map no group owns are, being where it is computed.

The place is held by its number among the places of the cover's plans
(`Places`), and what the proposal does there, `edit`, as text whose values
the place names are numbered as the place numbers them, and the others after
(see `proposal_key`). Many proposals share a place, as each step of a loop
has many around the same groups, and a key that wrote its place out was as
long as those groups' text: a 64-step loop's gradient wrote 3.6 GB of keys
for 12,930 proposals in 260 places.
"""
struct ProposalKey
    place::Int
    edit::String
end

Base.:(==)(a::ProposalKey, b::ProposalKey) = a.place == b.place && a.edit == b.edit
Base.hash(k::ProposalKey, h::UInt) = hash(k.edit, hash(k.place, h))

"""
A place of proposals in a plan: its number among the cover's places, and the
number each value it mentions has in its text.
"""
struct Place
    number::Int
    numbers::Dict{Int, Int}
end

"""
The places of the proposals one cover weighs: each place's text, by its
number, alike in every plan the cover weighs; each plan's places, by the
groups around them; and each plan's groups' texts, each written once for
every place it is in.
"""
struct Places
    numbers::Dict{String, Int}
    found::IdDict{Plan, Dict{Vector{Int}, Place}}
    texts::IdDict{Plan, Dict{Int, String}}
end

Places() = Places(Dict{String, Int}(), IdDict{Plan, Dict{Vector{Int}, Place}}(), IdDict{Plan, Dict{Int, String}}())

# `text` with each value it mentions, `%v`, written `%k` for `k = number(v)`.
numbered_values(text, number) = replace(text, r"%\d+" => s -> string("%", number(parse(Int, SubString(s, 2)))))

# Each value `numbers` numbers, in the order of its number, with its type,
# shape and whether `p` stores it.
value_shapes(p::Plan, numbers) = join((string("%", k, " ", p.graph.values[v].eltype, p.graph.values[v].dims, p.stored[v] ? " stored" : "")
    for (v, k) in sort!(collect(numbers); by = last)), ", ")

# The place of groups `around` of plan `p`, in the order the planner forms
# them: their texts, each value numbered in the order they first mention it,
# and those values' shapes.
function proposal_place!(places::Places, p::Plan, around::Vector{Int})
    found = get!(() -> Dict{Vector{Int}, Place}(), places.found, p)
    get!(found, around) do
        texts = get!(() -> Dict{Int, String}(), places.texts, p)
        io = IOBuffer()
        foreach(j -> print(io, get!(() -> "| " * numbered_text(p, p.groups[j]), texts, j)), around)
        numbers = Dict{Int, Int}()
        text = numbered_values(String(take!(io)), v -> get!(numbers, v, length(numbers) + 1))
        Place(get!(places.numbers, string(text, "; ", value_shapes(p, numbers)), length(places.numbers) + 1), numbers)
    end
end

"""
    proposal_key(p, a, q, places) -> ProposalKey

The key of proposal `q` in plan `p`, whose neighbourhoods are `a`: its place
among `places`, and its edit, its values numbered as the place numbers them,
those it alone names after, with their shapes. Two proposals have one key
exactly where the text of their groups, edit and values' shapes, written
out and numbered together, is one: a place's text never holds the ` || `
each edit starts with, so where the place ends in that text is known.
"""
function proposal_key(p::Plan, a::Around, q::Proposal, places::Places)
    around = Set{Int}()
    foreach(v -> union!(around, held(p, a, v)), fixes(q))
    place = proposal_place!(places, p, sort!(collect(around); by = j -> a.places[j]))
    io = IOBuffer()
    key_text(io, p, q)
    print(io, " || ", join(map(v -> "%$v", fixes(q)), " "))
    added = Dict{Int, Int}()
    number(v) = get(() -> get!(added, v, length(place.numbers) + length(added) + 1), place.numbers, v)
    edit = numbered_values(String(take!(io)), number)
    ProposalKey(place.number, string(edit, "; ", value_shapes(p, added)))
end

"""
A proposal's edit: the groups its trial `t` forms that the plan `p` it
changes does not, and those of `p` it no longer forms, each in its
canonical form. The comparison of `t` with `p` is of these groups alone (see
`compare`), so proposals with one edit are priced alike; and a proposal in
each of a stack of identical layers has one, where the groups around it, a
sum over every layer among them, tell the layers apart. The plans' identities
are kept in `known` (see `identities!`).
"""
function edit_key(known, p::Plan, t::Plan)
    ids_p, in_p = identities!(known, p)
    ids_t, in_t = identities!(known, t)
    added = sort!([canonical(t, g) for (g, id) in zip(t.groups, ids_t) if !(id in in_p)])
    removed = sort!([canonical(p, g) for (g, id) in zip(p.groups, ids_p) if !(id in in_t)])
    string(join(added, " | "), " || ", join(removed, " | "))
end

# Every stored value of a kernel that another kernel reads, joined into the
# reader's group: a choice the planner's rules make, offered to the cover. A
# value of a recurrence's step stays its group's: joined, a second kernel
# wrote it, and a neural ODE's weights' gradient read it wrong in one
# preparation in three.
function joins(p::Plan)
    a = Around(p, Vector{Int}[])
    found = Join[]
    for (k, g) in enumerate(p.groups)
        g.kind isa Union{ReductionKind, PointwiseKind} || continue
        for v in sort!(collect(group_reads(p.graph, g)))
            j = p.owner[v]
            j == 0 || j == k || p.groups[j].kind isa RecurrenceKind || push!(found, Join(v, sort!([a.owned[k]; v])))
        end
    end
    found
end

# The stored maps of each kernel of maps that one other kernel alone reads,
# one of them at several positions: each kernel's a proposal to recompute
# them there (see `RecomputeMaps`). A map read at one position, as a
# product reads its factor, is not read shifted.
function shifted_maps(p::Plan)
    a = Around(p, Vector{Int}[])
    found = RecomputeMaps[]
    for (k, g) in enumerate(p.groups)
        g.kind isa PointwiseKind || continue
        stored = [m for m in a.owned[k] if p.stored[m]]
        all(m -> p.graph.values[m].definition isa Pointwise, stored) || continue
        readers = unique(j for m in stored for j in a.reading[m] if j != k)
        length(readers) == 1 || continue
        reader = p.groups[only(readers)]
        positions(v) = unique(Tuple{Vararg{Affine}}[l.index for m in members(reader) if haskey(reader.terms, m) for l in loads(reader.terms[m]) if l.value == v])
        any(v -> length(positions(v)) > 1, stored) && push!(found, RecomputeMaps(stored))
    end
    found
end

"""
    proposals(d, base) -> [(proposal, implementation or nothing)]

What the cover weighs against the generator's plan `base`, each with what
proposes it, nothing for the planner's own: the implementations'
candidates, the planner's joins of a stored value into a kernel that reads
it, its maps recomputed where they are read shifted, its folds recomputed
where an implementation recomputes folds, its maps stored apart from a
kernel that applies a function an implementation cannot lower or a claim
on which failed to build, and its sums riding a recurrence's steps.
"""
function proposals(d::Decider, base::Plan)
    found = Tuple{Proposal, Union{Nothing, Implementation}}[]
    # Each proposal once, by what it is and by the values it fixes, found by
    # a lookup rather than by a pass over the proposals before it, as many as
    # a program has layers.
    proposed, fixing = Set{Proposal}(), Set{Vector{Int}}()
    for i in implementations(d.target), x in Base.invokelatest(candidates, i, d.target, base)
        q = proposal(x)
        q in proposed && continue
        push!(found, (q, i))
        push!(proposed, q)
        push!(fixing, fixes(q))
    end
    for q in joins(base)
        q.group in fixing && continue
        push!(found, (q, nothing))
        push!(fixing, q.group)
    end
    foreach(q -> push!(found, (q, nothing)), shifted_maps(base))
    if any_recomputes_folds(d.target)
        for v in sort!(collect(recomputable_folds(d.graph, d.kept; pinned = d.pinned)))
            push!(found, (Recompute(v), nothing))
        end
    end
    for g in base.groups
        isempty(failures(d, base, g)) && !unlowered(d, base, g) && continue
        for v in recomputed_maps(base, g)
            Store(v) in proposed && continue
            push!(found, (Store(v), nothing))
            push!(proposed, Store(v))
        end
    end
    measures(d) && foreach(((v, within, how),) -> push!(found, (Ride(v, within, how), nothing)), rides(base))
    found
end

"""
    cover(decider) -> (plan, considered)

The planner's groups, with every fold stored, and the proposals: those
implementations make, the planner's own joins of a stored value into the
kernel that reads it, the planner's own recomputing of a fold where it is
read (see `recomputable_folds`), where an implementation recomputes folds,
and its own storing apart of each map of a kernel that applies a function an
implementation cannot lower, or that a claim failed to build (see `Store`).
Proposals are accepted greedily by estimated gain over the plan, then by
their values, each when its proposer claims the group it forms (any claim,
or the floor, for a join; a claim of an implementation that recomputes folds,
for a fold recomputed). Proposals alike in the starting plan (`proposal_key`)
are planned once. Proposals whose trials there change groups alike
(`edit_key`) are one edit in identical layers: priced once, and taken
together. A proposal
is priced again in the current plan when an edit taken already changed a
group around it. Each trial forms again only the kernels its proposals reach
in the plan it changes (see `replanned`), and is compared with that plan on
the groups that differ. Every proposal compared is
returned, taken or not. Folds the decider pins are recomputed in every plan,
and those it keeps are not proposed.
"""
function cover(d::Decider)
    clock = mark()
    # The generator's plan (see `generator_plan`), with how it formed its
    # kernels: a trial forms again only those its proposals reach.
    based = Planned(d.graph, Inputs(d))
    base = based.plan
    clock = lap!(:cover_start, clock)
    search!(d, base)
    clock = lap!(:cover_search, clock)
    proposed = proposals(d, base)
    clock = lap!(:cover_proposals, clock)
    # What the starting plan stores: a set that does not form one group
    # there, where a trial would first form it, is not tried. A set holding a
    # pinned fold changes what is stored.
    start = storing(d.graph; recompute = d.pinned, ride = d.rides)
    unowned = zeros(Int, length(start.stored))
    function forms(set)
        isdisjoint(set, d.pinned) || return true
        try
            # Trials run on several threads at once, and forming fills the
            # kernels' caches as it goes: each its own, the ancestors
            # shared. One shared had its dictionaries written at once.
            k = Kernels(start.stored, unowned, Group[], based.grounds.up, Dict{Int, Set{Int}}(), Dict{Term, Vector{Int}}(),
                DependsWalk(length(unowned)))
            fixed_group(start.graph, based.grounds.awaits, Plain(k), set)
            true
        catch e
            e isa NotAGroup || rethrow()
            false
        end
    end
    arounds = IdDict{Plan, Around}()
    around(p, taken) = get!(() -> Around(p, inputs(d, taken).fixed), arounds, p)
    function trial(current::Planned, taken, qs)
        tally!(:trials)
        if isempty(taken)
            sets = inputs(d, qs).fixed
            length(sets) == 1 && !forms(only(sets)) && return nothing
        end
        replanned_taking(current, taken, qs)
    end
    # A trial's plan alone, which is what is priced: how it formed is kept
    # only for the plans taken, which trials form near.
    plan_of(p) = p === nothing ? nothing : p.plan
    # A proposal's comparison with the plan it is priced in, and the
    # implementation that claims its group: by key, once, where its kind
    # can be priced (`priceable`).
    priced = Dict{ProposalKey, Union{Nothing, Tuple{Comparison, Union{Nothing, Implementation}}}}()
    known = IdDict{Plan, Tuple{Vector{GroupIdentity}, Set{GroupIdentity}}}()
    function can_price(p, q, i)
        p === nothing && return false
        i === nothing || claimant(d, p, q, i) !== nothing || return false
        priceable(d, p, q)
    end
    function priced_plan(p, current, q, i, can = can_price(p, q, i))
        can || return nothing
        (compare(d, p, current, known), i === nothing ? nothing : claimant(d, p, q, i))
    end
    a = around(base, Proposal[])
    # Every proposal's trial against the starting plan, planned side by side
    # on the threads there are: a trial is pure, and planning is most of the
    # cover's time. Priced in order, once per key, as before.
    places = Places()
    keyed = [(q, i, proposal_key(base, a, q, places)) for (q, i) in proposed]
    firsts = Dict{ProposalKey, Proposal}()
    foreach(((q, _, k),) -> get!(firsts, k, q), keyed)
    clock = lap!(:cover_keys, clock)
    planned = Dict(k => Threads.@spawn(plan_of(trial(based, Proposal[], Proposal[q]))) for (k, q) in firsts)
    trials = Dict{ProposalKey, Union{Nothing, Plan}}(k => fetch_result(t) for (k, t) in planned)
    # Whether each trial can be priced, by its key's first proposal as it is
    # priced below; and what the implementations offer for the groups of
    # those that can, found side by side as the starting plan's are (see
    # `offer!`), as a trial is priced by the groups it differs in.
    pricing = Dict{ProposalKey, Bool}()
    foreach(((q, i, k),) -> haskey(pricing, k) || (pricing[k] = can_price(trials[k], q, i)), keyed)
    offer!(d, [(trials[k], g) for k in unique(k for (_, _, k) in keyed) if pricing[k] for g in trials[k].groups])
    clock = lap!(:cover_trials, clock)
    search_ahead!(d, base, keyed, trials, pricing)
    clock = lap!(:cover_search_ahead, clock)
    ranked = Tuple{Float64, Proposal, Union{Nothing, Implementation}, ProposalKey}[]
    for (q, i, k) in keyed
        x = get!(() -> priced_plan(trials[k], base, q, i, pricing[k]), priced, k)
        x === nothing || push!(ranked, (x[1].old - x[1].new, q, i, k))
    end
    sort!(ranked; by = r -> (-r[1], fixes(r[2])))
    clock = lap!(:cover_pricing, clock)
    # Each ranked proposal's edit, by which proposals alike are one class.
    edits = Dict{ProposalKey, String}()
    edit = [get!(() -> edit_key(known, base, trials[r[4]]), edits, r[4]) for r in ranked]
    clock = lap!(:cover_edits, clock)
    near(q) = Set(j for v in fixes(q) for j in neighbours(base, a, v))
    current, considered, taken, done = based, Considered[], Proposal[], Set{Int}()
    kept = Set(group_identity(base, g) for g in base.groups)
    # Those around which an edit taken already changed a group are priced
    # again in the current plan, by the first of them; the rest keep their
    # price.
    moved(m) = any(j -> !(group_identity(base, base.groups[j]) in kept), near(ranked[m][2]))
    # The classes in order, each where its first proposal is ranked. The
    # current plan changes only when a class is taken, and few are: the
    # trials of the moved classes ahead are planned side by side against it,
    # as the starting plan's are, and priced in order as they come. Those
    # ahead of a class taken were planned against a plan no longer current,
    # and are not used.
    classes, first_of = Dict{Int, Vector{Int}}(), Dict{String, Int}()
    for (m, r) in enumerate(ranked)
        push!(get!(() -> Int[], classes, get!(first_of, edit[m], m)), m)
    end
    order = sort!(collect(keys(classes)))
    ahead, spawned = Dict{ProposalKey, Task}(), Task[]
    function plan_ahead!(n)
        c = searchsortedfirst(order, n)
        for later in order[c:min(end, c + Threads.nthreads() - 1)]
            part = filter(moved, classes[later])
            isempty(part) && continue
            q = ranked[first(part)][2]
            key = proposal_key(current.plan, around(current.plan, taken), q, places)
            (haskey(priced, key) || haskey(ahead, key)) && continue
            t = let p = current, before = copy(taken)
                Threads.@spawn plan_of(trial(p, before, Proposal[q]))
            end
            ahead[key] = t
            push!(spawned, t)
        end
    end
    price_ahead(q, i, key) = get!(priced, key) do
        t = pop!(ahead, key, nothing)
        priced_plan(t === nothing ? plan_of(trial(current, taken, Proposal[q])) : fetch_result(t), current.plan, q, i)
    end
    for n in eachindex(ranked)
        n in done && continue
        class = [m for m in n:length(ranked) if !(m in done) && edit[m] == edit[n]]
        union!(done, class)
        plan_ahead!(n)
        was = current
        taking = Considered[]
        for part in (filter(!moved, class), filter(moved, class))
            isempty(part) && continue
            _, q, i, key = ranked[first(part)]
            x = moved(first(part)) ? price_ahead(q, i, proposal_key(current.plan, around(current.plan, taken), q, places)) : priced[key]
            x === nothing && continue
            c, j = x
            cs = [Considered(ranked[m][2], j, c, cheaper(c)) for m in part]
            cheaper(c) ? append!(taking, cs) : append!(considered, cs)
        end
        isempty(taking) && continue
        p = trial(current, taken, Proposal[c.proposal for c in taking])
        if p === nothing
            # Proposals alike that share values are taken one at a time, each
            # compared with the plan as it then is; one that no longer forms
            # a group is not compared.
            for t in taking
                p = trial(current, taken, Proposal[t.proposal])
                p === nothing && continue
                c = compare(d, p.plan, current.plan, known)
                push!(considered, Considered(t.proposal, t.implementation, c, cheaper(c)))
                cheaper(c) && (current = p; push!(taken, t.proposal))
            end
        else
            current = p
            append!(considered, taking)
            append!(taken, [c.proposal for c in taking])
        end
        kept = Set(group_identity(current.plan, g) for g in current.plan.groups)
        current === was || empty!(ahead)
    end
    # A trial planned ahead and not used is waited for, so none outlives the
    # cover; one that failed was of a plan the cover did not price.
    foreach(t -> try wait(t) catch end, spawned)
    lap!(:cover_loop, clock)
    current.plan, considered
end

# The groups by which the trials of the starting plan `base` that are priced
# (`pricing`, by key) differ from it, searched together as the starting
# plan's are, by a decider that measures: each stage's kernels compile side
# by side for every trial, rather than group by group as each trial is
# priced. `keyed` holds each proposal with its key, `trials` each key's plan.
function search_ahead!(d::Decider, base::Plan, keyed, trials, pricing)
    d.measure in (:close, :all) || return nothing
    shared = Set(group_identity(base, g) for g in base.groups)
    ahead, seen, searched = Tuple{Plan, Group, Vector{Choice}}[], Set{GroupIdentity}(), Set{ProposalKey}()
    for (_, _, k) in keyed
        k in searched && continue
        push!(searched, k)
        p = trials[k]
        pricing[k] || continue
        for g in p.groups
            id = group_identity(p, g)
            (id in shared || id in seen || haskey(d.decided, canonical(p, g))) && continue
            push!(seen, id)
            push!(ahead, (p, g, options!(d, p, g)))
        end
    end
    search!(d, ahead)
    nothing
end

"""
    plan_taking(d, proposals) -> Plan or nothing

The plan with `proposals` taken, in order (see `inputs`): the cover's trial
of them, and the plan it reached when they are those it took. Nothing where
two share a value or a set does not form one group.
"""
function plan_taking(d::Decider, proposals)
    # A value belongs to one proposal.
    allunique(Iterators.flatten(fixes(x) for x in proposals)) || return nothing
    i = inputs(d, proposals)
    try
        plan(d.graph; fixed = i.fixed, i.recompute, i.store, i.ride)
    catch e
        e isa NotAGroup || rethrow()
        nothing
    end
end

"""
    replanned_taking(current::Planned, taken, proposals) -> Planned or nothing

The plan `current`, which has `taken` taken, with `proposals` taken after
them, formed near it (see `replanned`); as `plan_taking`, nothing where two
share a value or a set does not form one group.
"""
function replanned_taking(current::Planned, taken, proposals)
    allunique(Iterators.flatten(fixes(x) for x in Iterators.flatten((taken, proposals)))) || return nothing
    try
        replanned(current, change(proposals))
    catch e
        e isa NotAGroup || rethrow()
        nothing
    end
end

function explain(io::IO, p::Plan, c::Considered)
    t = c.comparison
    μs(x) = string(round(1.0e6 * x; sigdigits = 3), " μs")
    times = c.accepted ? (t.new, t.old) : (t.old, t.new)
    why = string(" (the kernels that differ: ", μs(times[1]), " against ", μs(times[2]), ", ", t.how, ")")
    explain(io, c.proposal, c, why)
    nothing
end

# The maps a group recomputes where it reads them: the unstored maps its
# members read, directly or through other values it recomputes.
function recomputed_maps(p::Plan, g::Group)
    found, pending, seen = Int[], members(g), Set(members(g))
    while !isempty(pending)
        for o in operands(p.graph.values[pop!(pending)].definition), r in read_values(o)
            r in seen && continue
            push!(seen, r)
            p.stored[r] && continue
            p.graph.values[r].definition isa Pointwise && push!(found, r)
            push!(pending, r)
        end
    end
    sort!(found)
end

# Whether `g` applies a function an implementation the target uses cannot lower.
function unlowered(d::Decider, p::Plan, g::Group)
    view = group_view(p, g)
    any(subterms(Term[view.terms[m] for m in members(view)]; nested = true)) do t
        t isa Apply && any(i -> !Base.invokelatest(lowers, i, d.target, t, view), implementations(d.target))
    end
end

# Whether any implementation the target uses recomputes folds where it reads them.
any_recomputes_folds(target) = any(i -> Base.invokelatest(recomputes_folds, i, target), implementations(target))

recomputed_in(g; nested = true) =
    Set(t.value for m in members(g) if haskey(g.terms, m) for t in subterms((g.terms[m],); nested) if t isa Recomputed)

# The starting plan: the program as written, every fold stored but those
# the decider pins to be recomputed.
generator_plan(d::Decider) = plan(d.graph; recompute = d.pinned, ride = d.rides)

# Whether every kernel of `p` that recomputes fold `v` is claimed by an
# implementation that recomputes folds on the decider's target: among the
# claims the implementations offer for its group (see `Offered`), which the
# plan's price is found from too.
function recomputed_claimed(d::Decider, p::Plan, v)
    recomputing = [i for i in implementations(d.target) if Base.invokelatest(recomputes_folds, i, d.target)]
    all(p.groups) do g
        v in recomputed_in(g) || return true
        any(((i, _, _, _),) -> i in recomputing, offered!(d, p, g).said)
    end
end

# The implementation that claims the groups a proposal of `i` forms (see
# `claimed`): `i`, or nothing.
function claimant(d::Decider, p::Plan, q, i)
    for set in claimed(q)
        v = first(set)
        p.owner[v] == 0 && return nothing
        isempty(Base.invokelatest(claims, i, d.target, group_view(p, p.groups[p.owner[v]]))) && return nothing
    end
    i
end
