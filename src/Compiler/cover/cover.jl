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
        g.kind === :reduction ? (:fold, first(g.reductions)) : g.kind === :pointwise ? (:pointwise, minimum(owned[j])) :
            g.kind === :recurrence ? (:recurrence, first(g.outputs)) : (:scan, first(g.reductions))
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
A proposal's place in a plan, by the values it fixes: the groups that hold
them (`held`), in the order the planner forms them, and its values among
them, numbered as `canonical` numbers one group's. Proposals with one key in
the generator's plan are the same edit in identical layers, and are planned
once. The groups that read its stored values are not among them: in a stack
of layers those include a sum over every layer, such as an optimizer's update
of weights every layer reads, in which each layer's term sits at a place of
its own, and no two layers would be alike. The groups that recompute a map
no group owns are, being where it is computed. Each group's text is written
once into `texts`, which the keys of one plan's proposals share.
"""
function proposal_key(p::Plan, a::Around, q::Proposal, texts = Dict{Int, String}())
    around = Set{Int}()
    foreach(v -> union!(around, held(p, a, v)), fixes(q))
    io = IOBuffer()
    for j in sort!(collect(around); by = j -> a.places[j])
        print(io, get!(() -> group_text(p, p.groups[j]), texts, j))
    end
    key_text(io, p, q)
    print(io, " || ", join(map(v -> "%$v", fixes(q)), " "))
    numbers = Dict{String, Int}()
    text = replace(String(take!(io)), r"%\d+" => s -> string("%", get!(numbers, s, length(numbers) + 1)))
    shapes = map(sort!(collect(numbers); by = last)) do (s, k)
        v = parse(Int, s[2:end])
        string("%", k, " ", p.graph.values[v].eltype, p.graph.values[v].dims, p.stored[v] ? " stored" : "")
    end
    string(text, "; ", join(shapes, ", "))
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

group_text(p::Plan, g::Group) = sprint() do io
    print(io, "| ", g.kind, " ", g.domain)
    for m in members(g)
        v = p.graph.values[m]
        what = m in g.reductions ? fold_key(v.definition) : "value"
        print(io, "; %", m, " ", what, " ", v.eltype, v.dims, p.stored[m] ? " stored" : "", " = ", g.terms[m])
    end
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
        g.kind in (:reduction, :pointwise) || continue
        for v in sort!(collect(group_reads(p.graph, g)))
            j = p.owner[v]
            j == 0 || j == k || p.groups[j].kind === :recurrence || push!(found, Join(v, sort!([a.owned[k]; v])))
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
        g.kind === :pointwise || continue
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
group around it. Each trial plans the whole graph, and is compared with the
plan it changes on the groups that differ. Every proposal compared is
returned, taken or not. Folds the decider pins are recomputed in every plan,
and those it keeps are not proposed.
"""
function cover(d::Decider)
    clock = mark()
    base = generator_plan(d)
    clock = lap!(:cover_start, clock)
    search!(d, base)
    clock = lap!(:cover_search, clock)
    proposals = Tuple{Proposal, Union{Nothing, Implementation}}[]
    # Each proposal once, by what it is and by the values it fixes, found by
    # a lookup rather than by a pass over the proposals before it, as many as
    # a program has layers.
    proposed, fixing = Set{Proposal}(), Set{Vector{Int}}()
    for i in implementations(d.target), x in Base.invokelatest(candidates, i, d.target, base)
        q = proposal(x)
        q in proposed && continue
        push!(proposals, (q, i))
        push!(proposed, q)
        push!(fixing, fixes(q))
    end
    for q in joins(base)
        q.group in fixing && continue
        push!(proposals, (q, nothing))
        push!(fixing, q.group)
    end
    foreach(q -> push!(proposals, (q, nothing)), shifted_maps(base))
    if any_recomputes_folds(d.target)
        for v in sort!(collect(recomputable_folds(d.graph, d.kept; pinned = d.pinned)))
            push!(proposals, (Recompute(v), nothing))
        end
    end
    for g in base.groups
        isempty(failures(d, base, g)) && !unlowered(d, base, g) && continue
        for v in recomputed_maps(base, g)
            Store(v) in proposed && continue
            push!(proposals, (Store(v), nothing))
            push!(proposed, Store(v))
        end
    end
    measures(d) && foreach(v -> push!(proposals, (Ride(v), nothing)), riding_sums(base))
    clock = lap!(:cover_proposals, clock)
    # What the starting plan stores: a set that does not form one group
    # there, where a trial would first form it, is not tried. A set holding a
    # pinned fold changes what is stored.
    start = storing(d.graph; recompute = d.pinned, ride = d.rides)
    deps = Dependencies(start.graph)
    function forms(set)
        isdisjoint(set, d.pinned) || return true
        try
            # Trials run on several threads at once, and grouping fills the
            # dependencies' caches as it goes: each its own, the ancestors
            # shared. One shared had its dictionaries written at once.
            fixed_group(start.graph, start.stored, Dependencies(deps.owner, deps.groups, deps.up), set)
            true
        catch e
            e isa NotAGroup || rethrow()
            false
        end
    end
    arounds = IdDict{Plan, Around}()
    around(p, taken) = get!(() -> Around(p, inputs(d, taken).fixed), arounds, p)
    function trial(current, taken, qs)
        tally!(:trials)
        all = Proposal[taken; qs]
        if isempty(taken)
            sets = inputs(d, all).fixed
            length(sets) == 1 && !forms(only(sets)) && return nothing
        end
        plan_taking(d, all)
    end
    # A proposal's comparison with the plan it is priced in, and the
    # implementation that claims its group: by key, once, where its kind
    # can be priced (`priceable`).
    priced = Dict{String, Any}()
    known = IdDict{Plan, Any}()
    function can_price(p, q, i)
        p === nothing && return false
        i === nothing || claimant(d, p, q, i) !== nothing || return false
        priceable(d, p, q)
    end
    function priced_plan(p, current, q, i)
        can_price(p, q, i) || return nothing
        (compare(d, p, current, known), i === nothing ? nothing : claimant(d, p, q, i))
    end
    a = around(base, Proposal[])
    # Every proposal's trial against the starting plan, planned side by side
    # on the threads there are: a trial is pure, and planning is most of the
    # cover's time. Priced in order, once per key, as before.
    texts = Dict{Int, String}()
    keyed = [(q, i, proposal_key(base, a, q, texts)) for (q, i) in proposals]
    firsts = Dict{String, Any}()
    foreach(((q, _, k),) -> get!(firsts, k, q), keyed)
    clock = lap!(:cover_keys, clock)
    planned = Dict(k => Threads.@spawn(trial(base, Proposal[], Proposal[q])) for (k, q) in firsts)
    trials = Dict(k => fetch_result(t) for (k, t) in planned)
    clock = lap!(:cover_trials, clock)
    # The groups by which the trials priced below differ from the starting
    # plan, searched together as the starting plan's are: each stage's kernels
    # compile side by side for every trial, rather than group by group as
    # each trial is priced.
    if d.measure in (:close, :all)
        shared = Set(group_identity(base, g) for g in base.groups)
        ahead, seen, searched = Any[], Set{GroupIdentity}(), Set{String}()
        for (q, i, k) in keyed
            k in searched && continue
            push!(searched, k)
            p = trials[k]
            can_price(p, q, i) || continue
            for g in p.groups
                id = group_identity(p, g)
                (id in shared || id in seen || haskey(d.decided, canonical(p, g))) && continue
                push!(seen, id)
                push!(ahead, (p, g, options!(d, p, g)))
            end
        end
        search!(d, ahead)
    end
    clock = lap!(:cover_search_ahead, clock)
    ranked = Tuple{Float64, Any, Union{Nothing, Implementation}, String}[]
    for (q, i, k) in keyed
        x = get!(() -> priced_plan(trials[k], base, q, i), priced, k)
        x === nothing || push!(ranked, (x[1].old - x[1].new, q, i, k))
    end
    sort!(ranked; by = r -> (-r[1], fixes(r[2])))
    clock = lap!(:cover_pricing, clock)
    # Each ranked proposal's edit, by which proposals alike are one class.
    edits = Dict{String, String}()
    edit = [get!(() -> edit_key(known, base, trials[r[4]]), edits, r[4]) for r in ranked]
    clock = lap!(:cover_edits, clock)
    near(q) = Set(j for v in fixes(q) for j in neighbours(base, a, v))
    current, considered, taken, done = base, Considered[], Proposal[], Set{Int}()
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
    ahead, spawned = Dict{String, Task}(), Task[]
    function plan_ahead!(n)
        c = searchsortedfirst(order, n)
        for later in order[c:min(end, c + Threads.nthreads() - 1)]
            part = filter(moved, classes[later])
            isempty(part) && continue
            q = ranked[first(part)][2]
            key = proposal_key(current, around(current, taken), q)
            (haskey(priced, key) || haskey(ahead, key)) && continue
            t = let p = current, before = copy(taken)
                Threads.@spawn trial(p, before, Proposal[q])
            end
            ahead[key] = t
            push!(spawned, t)
        end
    end
    price_ahead(q, i, key) = get!(priced, key) do
        t = pop!(ahead, key, nothing)
        priced_plan(t === nothing ? trial(current, taken, Proposal[q]) : fetch_result(t), current, q, i)
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
            x = moved(first(part)) ? price_ahead(q, i, proposal_key(current, around(current, taken), q)) : priced[key]
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
                c = compare(d, p, current, known)
                push!(considered, Considered(t.proposal, t.implementation, c, cheaper(c)))
                cheaper(c) && (current = p; push!(taken, t.proposal))
            end
        else
            current = p
            append!(considered, taking)
            append!(taken, [c.proposal for c in taking])
        end
        kept = Set(group_identity(current, g) for g in current.groups)
        current === was || empty!(ahead)
    end
    # A trial planned ahead and not used is waited for, so none outlives the
    # cover; one that failed was of a plan the cover did not price.
    foreach(t -> try wait(t) catch end, spawned)
    lap!(:cover_loop, clock)
    current, considered
end

"""
    plan_taking(d, proposals; near, formed) -> Plan or nothing

The plan with `proposals` taken, in order (see `inputs`): the cover's trial
of them, and the plan it reached when they are those it took. Nothing where two share a
value or a set does not form one group. `near` and `formed`, where given,
pass to `plan`: the kernels of a plan of the same graph to form only what
the proposals change, and where to note how this one formed its own.
"""
function plan_taking(d::Decider, proposals; near = nothing, formed = nothing)
    # A value belongs to one proposal.
    allunique(Iterators.flatten(fixes(x) for x in proposals)) || return nothing
    i = inputs(d, proposals)
    try
        forming = near === nothing && formed === nothing ? (;) : (; near, formed)
        plan(d.graph; fixed = i.fixed, i.recompute, i.store, i.ride, forming...)
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
# implementation that recomputes folds on the decider's target.
function recomputed_claimed(d::Decider, p::Plan, v)
    recomputing = [i for i in implementations(d.target) if Base.invokelatest(recomputes_folds, i, d.target)]
    all(p.groups) do g
        v in recomputed_in(g) || return true
        any(i -> !isempty(Base.invokelatest(claims, i, d.target, group_view(p, g))), recomputing)
    end
end

# The implementation that claims the groups a proposal of `i` forms: `i`, or nothing.
function claimant(d::Decider, p::Plan, q, i)
    for set in change(q).fixed
        v = first(set)
        p.owner[v] == 0 && return nothing
        isempty(Base.invokelatest(claims, i, d.target, group_view(p, p.groups[p.owner[v]]))) && return nothing
    end
    i
end
