# Which groups there are: the cover. Implementations propose candidates of
# their own, sets of values grown from anchors that they accept; the planner
# offers its own joins of a stored value into the kernel that reads it, and
# folds recomputed where they are read. A proposal replaces the generator's
# groups when the plan with it is cheaper, each compared on the groups that
# differ.

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
    proposal::Any # a set of values, or a `Join`
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
        g.kind === :reduction ? (:fold, first(g.reductions)) : g.kind === :pointwise ? (:pointwise, minimum(owned[j])) : (:scan, first(g.reductions))
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
    found = [p.owner[v]; [p.owner[r] for r in loaded(p, v)]]
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

"A stored value joined into the group of a kernel that reads it: a proposal of the planner's own."
struct Join
    value::Int
    group::Vector{Int}
end

"A fold recomputed where it is read instead of stored: a proposal of the planner's own."
struct Recompute
    value::Int
end

"The values a proposal is about: those it fixes as one group, or the fold it recomputes."
fixes(q::Vector{Int}) = q
fixes(q::Join) = q.group
fixes(q::Recompute) = [q.value]

"""
A proposal's place in a plan, by the values it fixes: the groups its values
reach (`reached`), in the order the planner forms them, and its values among
them, numbered as `canonical` numbers one group's. Proposals with one key in
the generator's plan are the same edit in identical layers.
"""
function proposal_key(p::Plan, a::Around, q)
    around = Set{Int}()
    foreach(v -> union!(around, reached(p, a, v)), fixes(q))
    io = IOBuffer()
    for j in sort!(collect(around); by = j -> a.places[j])
        g = p.groups[j]
        print(io, "| ", g.kind, " ", g.domain)
        for m in members(g)
            v = p.graph.values[m]
            what = m in g.reductions ? fold_key(v.definition) : "value"
            print(io, "; %", m, " ", what, " ", v.eltype, v.dims, p.stored[m] ? " stored" : "", " = ", g.terms[m])
        end
    end
    q isa Join && print(io, " || join %", q.value)
    q isa Recompute && print(io, " || recompute")
    print(io, " || ", join(map(v -> "%$v", fixes(q)), " "))
    numbers = Dict{String, Int}()
    text = replace(String(take!(io)), r"%\d+" => s -> string("%", get!(numbers, s, length(numbers) + 1)))
    shapes = map(sort!(collect(numbers); by = last)) do (s, k)
        v = parse(Int, s[2:end])
        string("%", k, " ", p.graph.values[v].eltype, p.graph.values[v].dims, p.stored[v] ? " stored" : "")
    end
    string(text, "; ", join(shapes, ", "))
end

# Every stored value of a kernel that another kernel reads, joined into the
# reader's group: a choice the planner's rules make, offered to the cover.
function joins(p::Plan)
    a = Around(p, Vector{Int}[])
    found = Join[]
    for (k, g) in enumerate(p.groups)
        g.kind in (:reduction, :pointwise) || continue
        for v in sort!(collect(group_reads(p.graph, g)))
            j = p.owner[v]
            j == 0 || j == k || push!(found, Join(v, sort!([a.owned[k]; v])))
        end
    end
    found
end

"""
    cover(decider) -> (plan, considered)

The planner's groups, with every fold stored, and the proposals: those
implementations make, the planner's own joins of a stored value into the
kernel that reads it, and the planner's own recomputing of a fold where it is
read (see `recomputable_folds`), where an implementation recomputes folds.
Proposals are accepted greedily by estimated gain over the plan, then by
their values, each when its proposer claims the group it forms (any claim,
or the floor, for a join; a claim of an implementation that recomputes folds,
for a fold recomputed). Proposals alike in the starting plan (`proposal_key`)
are one edit in identical layers: priced once, and taken together. A proposal
is priced again in the current plan when an edit taken already changed a
group around it. Each trial plans the whole graph, and is compared with the
plan it changes on the groups that differ. Every proposal compared is
returned, taken or not. Folds the decider pins are recomputed in every plan,
and those it keeps are not proposed.
"""
function cover(d::Decider)
    base = generator_plan(d)
    search!(d, base)
    proposals = Tuple{Any, Union{Nothing, Implementation}}[]
    for i in implementations(d.target), q in Base.invokelatest(candidates, i, d.target, base)
        any(x -> x[1] == q, proposals) || push!(proposals, (q, i))
    end
    for q in joins(base)
        any(x -> fixes(x[1]) == q.group, proposals) || push!(proposals, (q, nothing))
    end
    if any_recomputes_folds(d.target)
        for v in sort!(collect(recomputable_folds(d.graph, union(d.pinned, d.kept))))
            push!(proposals, (Recompute(v), nothing))
        end
    end
    # What the starting plan stores: a set that does not form one group
    # there, where a trial would first form it, is not tried. A set holding a
    # pinned fold changes what is stored.
    start = storing(d.graph; recompute = d.pinned)
    deps = Dependencies(start.graph)
    function forms(set)
        isdisjoint(set, d.pinned) || return true
        try
            fixed_group(start.graph, start.stored, deps, set)
            true
        catch e
            e isa NotAGroup || rethrow()
            false
        end
    end
    arounds = IdDict{Plan, Around}()
    around(p, taken) = get!(() -> Around(p, Vector{Int}[fixes(x) for x in taken if !(x isa Recompute)]), arounds, p)
    function trial(current, taken, qs)
        all = Any[taken; qs]
        sets = Vector{Int}[fixes(x) for x in all if !(x isa Recompute)]
        recompute = Set{Int}(d.pinned)
        foreach(x -> x isa Recompute && push!(recompute, x.value), all)
        # A value belongs to one proposal.
        allunique(Iterators.flatten(fixes(x) for x in all)) || return nothing
        isempty(taken) && length(sets) == 1 && !forms(only(sets)) && return nothing
        try
            plan(d.graph; fixed = sets, recompute = recompute)
        catch e
            e isa NotAGroup || rethrow()
            nothing
        end
    end
    # A proposal's comparison with the plan it is priced in, and the
    # implementation that claims its group: by key, once. A fold recomputed
    # is priced when every kernel that recomputes it is claimed by an
    # implementation that recomputes folds.
    priced = Dict{String, Any}()
    function priced_plan(p, current, q, i)
        p === nothing && return nothing
        j = i === nothing ? nothing : claimant(d, p, q, i)
        i === nothing || j !== nothing || return nothing
        q isa Recompute && !recomputed_claimed(d, p, q.value) && return nothing
        (compare(d, p, current), j)
    end
    price(current, taken, q, i, key) = get!(() -> priced_plan(trial(current, taken, Any[q]), current, q, i), priced, key)
    a = around(base, Any[])
    # Every proposal's trial against the starting plan, planned side by side
    # on the threads there are: a trial is pure, and planning is most of the
    # cover's time. Priced in order, once per key, as before.
    keyed = [(q, i, proposal_key(base, a, q)) for (q, i) in proposals]
    firsts = Dict{String, Any}()
    foreach(((q, _, k),) -> get!(firsts, k, q), keyed)
    planned = Dict(k => Threads.@spawn(trial(base, Any[], Any[q])) for (k, q) in firsts)
    ranked = Tuple{Float64, Any, Union{Nothing, Implementation}, String}[]
    for (q, i, k) in keyed
        x = get!(() -> priced_plan(fetch_result(planned[k]), base, q, i), priced, k)
        x === nothing || push!(ranked, (x[1].old - x[1].new, q, i, k))
    end
    sort!(ranked; by = r -> (-r[1], fixes(r[2])))
    near(q) = Set(j for v in fixes(q) for j in neighbours(base, a, v))
    current, considered, taken, done = base, Considered[], Any[], Set{Int}()
    kept = Set(group_identity(base, g) for g in base.groups)
    for (n, (_, _, _, k)) in enumerate(ranked)
        n in done && continue
        class = [m for m in n:length(ranked) if !(m in done) && ranked[m][4] == k]
        union!(done, class)
        # Those around which an edit taken already changed a group are priced
        # again in the current plan, by the first of them; the rest keep
        # their price.
        moved(m) = any(j -> !(group_identity(base, base.groups[j]) in kept), near(ranked[m][2]))
        taking = Considered[]
        for part in (filter(!moved, class), filter(moved, class))
            isempty(part) && continue
            _, q, i, key = ranked[first(part)]
            x = moved(first(part)) ? price(current, taken, q, i, proposal_key(current, around(current, taken), q)) : priced[key]
            x === nothing && continue
            c, j = x
            cs = [Considered(ranked[m][2], j, c, cheaper(c)) for m in part]
            cheaper(c) ? append!(taking, cs) : append!(considered, cs)
        end
        isempty(taking) && continue
        p = trial(current, taken, Any[c.proposal for c in taking])
        if p === nothing
            # Proposals alike that share values are taken one at a time, each
            # compared with the plan as it then is; one that no longer forms
            # a group is not compared.
            for t in taking
                p = trial(current, taken, Any[t.proposal])
                p === nothing && continue
                c = compare(d, p, current)
                push!(considered, Considered(t.proposal, t.implementation, c, cheaper(c)))
                cheaper(c) && (current = p; push!(taken, t.proposal))
            end
        else
            current = p
            append!(considered, taking)
            append!(taken, [c.proposal for c in taking])
        end
        kept = Set(group_identity(current, g) for g in current.groups)
    end
    current, considered
end

function explain(io::IO, p::Plan, c::Considered)
    q, t = c.proposal, c.comparison
    μs(x) = string(round(1.0e6 * x; sigdigits = 3), " μs")
    times = c.accepted ? (t.new, t.old) : (t.old, t.new)
    why = string(" (the kernels that differ: ", μs(times[1]), " against ", μs(times[2]), ", ", t.how, ")")
    if q isa Join
        c.accepted && println(io, "%", q.value, " joins the kernel of %", first(v for v in q.group if v != q.value), ", which reads it", why)
        return
    elseif q isa Recompute
        c.accepted && println(io, "%", q.value, " is recomputed where it is read", why)
        return
    end
    c.accepted && println(io, "candidate ", join(map(v -> "%$v", q), " "), " from ", label(c.implementation), " replaces the generator's grouping", why)
end

# Whether any implementation the target uses recomputes folds where it reads them.
any_recomputes_folds(target) = any(i -> Base.invokelatest(recomputes_folds, i, target), implementations(target))

recomputed_in(g; nested = true) =
    Set(t.value for m in members(g) if haskey(g.terms, m) for t in subterms((g.terms[m],); nested) if t isa Recomputed)

# The starting plan: the program as written, every fold stored but those
# the decider pins to be recomputed.
generator_plan(d::Decider) = plan(d.graph; recompute = d.pinned)

# Whether every kernel of `p` that recomputes fold `v` is claimed by an
# implementation that recomputes folds on the decider's target.
function recomputed_claimed(d::Decider, p::Plan, v)
    recomputing = [i for i in implementations(d.target) if Base.invokelatest(recomputes_folds, i, d.target)]
    all(p.groups) do g
        v in recomputed_in(g) || return true
        any(i -> !isempty(Base.invokelatest(claims, i, d.target, group_view(p, g))), recomputing)
    end
end

# The implementation that claims the group a proposal of `i` forms: `i`, or nothing.
function claimant(d::Decider, p::Plan, q, i)
    v = first(q)
    p.owner[v] == 0 && return nothing
    isempty(Base.invokelatest(claims, i, d.target, group_view(p, p.groups[p.owner[v]]))) ? nothing : i
end
