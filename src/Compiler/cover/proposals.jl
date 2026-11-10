# What the cover proposes: changes to what the planner is given, each kind
# defined here with all the cover and its decisions ask of it. An
# implementation's candidate is one or several sets of values, a group each
# (`Candidate`); the planner's own proposals join a stored value into the
# group of a kernel that reads it (`Join`), recompute a fold where it is
# read (`Recompute`) or maps in the one kernel that reads them
# (`RecomputeMaps`), store a map apart (`Store`), and sum as a recurrence's
# steps run what they compute (`Ride`). A proposal is data, the values it is
# about by their places in the program's index graph, equal and hashed by
# what it holds.
#
# Each kind answers: what it changes of what the planner is given
# (`change`), from which the plan with it taken follows and nothing in the
# planner names a kind; the values it is about, which no other proposal
# taken with it may be about (`fixes`); its part of the key that tells
# proposals alike in identical layers (`key_text`); whether a plan with it
# taken can be priced (`priceable`); and how `explain` says it was taken
# (`explain`).

"A change the cover proposes to what the planner is given (see the kinds above)."
abstract type Proposal end

Base.:(==)(a::P, b::P) where {P <: Proposal} = all(f -> getfield(a, f) == getfield(b, f), fieldnames(P))
Base.hash(q::Proposal, h::UInt) = foldl((h, f) -> hash(getfield(q, f), h), fieldnames(typeof(q)); init = hash(typeof(q), h))

# What the planner is given (see `Inputs`) as decider `d` gives it: the folds
# it pins recomputed, the sums it was given riding.
Inputs(d::Decider) = Inputs(; recompute = d.pinned, ride = d.rides)

"What proposals taken in order change of what the planner is given."
change(proposals::AbstractVector) = foldl((i, q) -> merge(i, change(q)), proposals; init = Inputs())

"What decider `d`'s planner is given with `proposals` taken, in order."
inputs(d::Decider, proposals) = merge(Inputs(d), change(proposals))

"""
The values a proposal is about: the sets it fixes as groups, the fold or the
maps it recomputes, the map it stores apart, or the sum riding steps. No two
proposals taken together are about one value.
"""
function fixes end

# A plan with a proposal taken is priced unless its kind says otherwise.
priceable(::Decider, ::Plan, ::Proposal) = true

"""
An implementation's candidate: sets of values, each to form a group, one
set or several apart, proposed as one (see `candidates`).
"""
struct Candidate <: Proposal
    sets::Vector{Vector{Int}}
end

change(q::Candidate) = Inputs(; fixed = q.sets)
fixes(q::Candidate) = reduce(vcat, q.sets)
key_text(io, ::Plan, q::Candidate) =
    length(q.sets) > 1 && print(io, " || apart ", join((join(map(v -> "%$v", set), " ") for set in q.sets), " | "))
function explain(io, q::Candidate, c, why)
    sets = join((join(map(v -> "%$v", set), " ") for set in q.sets), ", ")
    what = length(q.sets) > 1 ? "candidates " * sets * ", one kernel each," : "candidate " * sets
    c.accepted && println(io, what, " from ", label(c.implementation), " replaces the generator's grouping", why)
end

# An implementation's candidate as `candidates` returns it: a set of values,
# or several sets, or a proposal.
proposal(q::Proposal) = q
proposal(set::AbstractVector{<:Integer}) = Candidate([collect(Int, set)])
proposal(sets::AbstractVector) = Candidate([collect(Int, set) for set in sets])

"A stored value joined into the group of a kernel that reads it: a proposal of the planner's own."
struct Join <: Proposal
    value::Int
    group::Vector{Int}
end

change(q::Join) = Inputs(; fixed = [q.group])
fixes(q::Join) = q.group
key_text(io, ::Plan, q::Join) = print(io, " || join %", q.value)
explain(io, q::Join, c, why) =
    c.accepted && println(io, "%", q.value, " joins the kernel of %", first(v for v in q.group if v != q.value), ", which reads it", why)

"""
A fold recomputed where it is read instead of stored: a proposal of the
planner's own, priced where every kernel that recomputes it is claimed by an
implementation that recomputes folds.
"""
struct Recompute <: Proposal
    value::Int
end

change(q::Recompute) = Inputs(; recompute = (q.value,))
fixes(q::Recompute) = [q.value]
key_text(io, ::Plan, ::Recompute) = print(io, " || recompute")
priceable(d::Decider, p::Plan, q::Recompute) = recomputed_claimed(d, p, q.value)
explain(io, q::Recompute, c, why) = c.accepted && println(io, "%", q.value, " is recomputed where it is read", why)

# The folds that can be recomputed where they are read, as proposals for the
# cover (see `Recompute`): a fold that is not a result, an index of a gather,
# or in `kept`, whose every reader through maps is a fold over another domain,
# or over the same extents reading it elsewhere than where its group keeps it,
# as attention reads its scores when the features are as many as the keys (a
# fold that reads it there joins its group as a later phase instead). A fold
# recomputed inside a reader's sweep, at positions that move along its own
# reduced axes, does not recompute such a fold in turn: nested sweeps multiply
# the work past what a plan should price. Folds in `pinned`, recomputed
# already, are not proposed and nest as the selected do; folds in `kept`,
# stored, are not proposed and nest nothing. Counted as recomputed, the
# projections a training step's update replaces in place, which stay stored,
# kept attention's scores and their gradient from the proposals, and GPT-2's
# layer stored both, 201 MB each: 19.1 ms a layer's step, against 12.4 with
# both recomputed.
# Whether recomputing pays is the cover's decision, by cost.
function recomputable_folds(graph, kept = Set{Int}(); pinned = Set{Int}())
    s = storing(graph)
    graph, results, materialized = s.graph, s.results, s.materialized
    values = graph.values
    users = readers(graph)
    selected = Set{Int}()
    for v in eachindex(values)
        d = values[v].definition
        d isa Reduction && !(v in results) && !materialized[v] && !(v in kept) && !(v in pinned) || continue
        # The folds v reaches through maps, with the index at which each reads v.
        consumers = Tuple{Int, Tuple{Vararg{Affine}}}[]
        pending = [(u, v, identity_index(values[v].dims)) for u in unique(users[v])]
        # Each reader at each index once: maps reading v along many paths,
        # as a gated update's step reads its state thrice, are as many.
        seen = Set(pending)
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
                    for x in unique(users[u])
                        (x, u, at) in seen || (push!(seen, (x, u, at)); push!(pending, (x, u, at)))
                    end
                else
                    ok = false
                end
            end
        end
        ok && !isempty(consumers) || continue
        own = Tuple(var(k) for k in eachindex(d.iteration) if !(k in d.reduced))
        all(((u, at),) -> values[u].definition.iteration != d.iteration || at != own, consumers) || continue
        hoisted = all(((u, at),) -> !any(k -> any(e -> mentions(e, k), at), values[u].definition.reduced), consumers)
        hoisted || !recomputes(values, v, union(selected, pinned), results, materialized) || continue
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

"""
The maps a kernel stores, recomputed in the one kernel that reads them: a
proposal of the planner's own, for a kernel of maps another reads at shifted
positions, as a stencil reads its neighbours. A map read once at each
position is recomputed by its reader already (see `recomputable`); read at
several, it is stored, and the kernel that stores it costs a launch and a
round trip through memory where recomputing it costs a few operations at
each position read.
"""
struct RecomputeMaps <: Proposal
    values::Vector{Int}
end

change(q::RecomputeMaps) = Inputs(; recompute = q.values)
fixes(q::RecomputeMaps) = q.values
key_text(io, ::Plan, ::RecomputeMaps) = print(io, " || recompute maps")
explain(io, q::RecomputeMaps, c, why) =
    c.accepted && println(io, join(map(v -> "%$v", q.values), " "), " are recomputed in the kernel that reads them", why)

"""
A map stored apart rather than recomputed in the kernels that read it: a
proposal of the planner's own, for each map of a kernel that applies a scalar
function an implementation cannot lower (see `lowers`), or that a claim
failed to build. Stored apart, the map runs in a kernel of its own, and the
rest of the kernel may be claimed.
"""
struct Store <: Proposal
    value::Int
end

change(q::Store) = Inputs(; store = (q.value,))
fixes(q::Store) = [q.value]
# A stored map is inlined in the terms of the groups it reaches: its
# definition tells it from the group's other maps.
function key_text(io, p::Plan, q::Store)
    d = p.graph.values[q.value].definition
    print(io, " || store ", describe(d), " ")
    join(io, (sprint(show_operand, o) for o in operands(d)), ", ")
end
explain(io, q::Store, c, why) = c.accepted && println(io, "%", q.value, " is stored apart, in a kernel of its own", why)

"""
A value computed from a recurrence's steps, computed as they run rather
than after them from what they store (see `riding`): a proposal of the
planner's own. A sum over the steps adds each step's part as the steps run,
priced where the recurrence's step can be planned, since only the steps run
a step at a time add it: the steps compute each part where its operands are
fresh, and store only what else reads them; the sum after the steps reads
every step's, as a neural ODE's weight gradients read every evaluation's
activations. Small steps run fastest summed after, in a few kernels; large
ones, read once. A value along the steps, or of their last, is written by
the steps as they compute it, as a recurrent network's output at each step
and its state after the last are: what it reads of the steps need not be
stored for it, the states of every step among them where nothing else reads
them. A product a step computes once a step may run fastest after the steps
instead, over all of them at once. It is about what its steps compute for
the value too (`within`), which nothing else reads: a proposal taken with it
that made a kernel of one of them would make one whose work nothing reads,
beside the steps computing it. How it meets the steps is `how` (see
`riding`).
"""
struct Ride <: Proposal
    value::Int
    within::Vector{Int}
    how::Symbol
end

change(q::Ride) = Inputs(; ride = (q.value,))
fixes(q::Ride) = [q.value; q.within]
key_text(io, ::Plan, q::Ride) = print(io, " || ride ", q.how)
# A recurrence a sum rides runs only a step at a time.
function priceable(d::Decider, p::Plan, q::Ride)
    g = p.groups[p.owner[q.value]]
    isempty(g.reductions) || stepped_plan(d, p, g) !== nothing
end
explain(io, q::Ride, c, why) = println(io, "%", q.value, ridden(q.how, c.accepted), why)

ridden(how, accepted) = how === :summed ?
    (accepted ? " is summed as the steps of the recurrence it reads run" : " is summed after the steps of the recurrence it reads") :
    how === :each ? (accepted ? " is computed at each step of the recurrence it reads, as the steps run" :
        " is computed after the steps of the recurrence it reads, from what they store") :
    (accepted ? " is computed at the last step of the recurrence it reads, as the steps run" :
        " is computed after the steps of the recurrence it reads, from what the last stores")

"""
    riding_values(plan) -> values
    riding_sums(plan) -> values

The values of `plan` that could ride the steps of one of its recurrences:
computed apart from them, along them, at their last or over them, reading
the recurrence's cycle through values only they read (see `riding`), each
once; and of those, the sums over the steps.
"""
riding_values(p::Plan) = [r for (r, _, _) in rides(p)]
riding_sums(p::Plan) = [r for (r, _, how) in rides(p) if how === :summed]

# Each value of `plan` that could ride the steps of one of its recurrences,
# with the values the steps would compute for it, which nothing else reads,
# and how it meets the steps. A value along them or of their last is one a
# kernel stores; a sum, any.
function rides(p::Plan)
    graph = p.graph
    values = graph.values
    live = falses(length(values))
    foreach(g -> foreach(m -> live[m] = true, members(g)), p.groups)
    for id in length(values):-1:1
        live[id] || continue
        foreach(o -> foreach(r -> live[r] = true, read_values(o)), operands(values[id].definition))
    end
    found = Tuple{Int, Vector{Int}, Symbol}[]
    taken = Set{Int}()
    for recurrence in recurrences(graph)
        c = first(recurrence.states)
        live[c] || continue
        k = p.owner[c]
        reaching = Reaching(graph, recurrence.values)
        # What a value reads between it and the cycle only it reads: a value
        # one of whose operands a later one reads too rides nothing, as each
        # layer after a recurrence's reads the one before.
        (; inside, reaches, users) = reaching
        apart(r) = all(o -> all(x -> !reaches[x] || x in inside || all(u -> u <= r || !live[u], users[x]), read_values(o)),
            operands(values[r].definition))
        for r in (maximum(recurrence.from) + 1):length(values)
            p.owner[r] != k && !(r in taken) && reaches[r] && apart(r) || continue
            rider = riding(graph, live, recurrence.values, r, reaching)
            rider !== nothing && (rider.how === :summed || p.owner[r] > 0 && p.stored[r]) || continue
            push!(found, (r, rider.between, rider.how))
            push!(taken, r)
        end
    end
    found
end
