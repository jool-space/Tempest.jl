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

"""
What the planner is given, or what a proposal changes of it: the sets of
values fixed as groups, in order, the folds recomputed where they are read,
the maps stored apart, and the sums riding their recurrence's steps, in
order. Changes compose by `merge`, the sets and sums of each after those
before. A decider's own (`Inputs(d)`) recompute the folds it pins and ride
the sums it was given.
"""
struct Inputs
    fixed::Vector{Vector{Int}}
    recompute::Set{Int}
    store::Set{Int}
    ride::Vector{Int}
end

Inputs(; fixed = Vector{Int}[], recompute = (), store = (), ride = ()) =
    Inputs(collect(Vector{Int}, fixed), Set{Int}(recompute), Set{Int}(store), collect(Int, ride))
Inputs(d::Decider) = Inputs(; recompute = d.pinned, ride = d.rides)

Base.merge(a::Inputs, b::Inputs) = Inputs([a.fixed; b.fixed], a.recompute ∪ b.recompute, a.store ∪ b.store, [a.ride; b.ride])

"What decider `d`'s planner is given with `proposals` taken, in order."
inputs(d::Decider, proposals) = foldl((i, q) -> merge(i, change(q)), proposals; init = Inputs(d))

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
A sum over a recurrence's steps of what they compute, added as they run
rather than after them from what they store (see `riding`): a proposal of
the planner's own, priced where the recurrence's step can be planned. The
steps compute each part where its operands are fresh, and store only what
else reads them; the sum after the steps reads every step's, as a neural
ODE's weight gradients read every evaluation's activations. Small steps run
fastest summed after, in a few kernels; large ones, read once.
"""
struct Ride <: Proposal
    value::Int
end

change(q::Ride) = Inputs(; ride = (q.value,))
fixes(q::Ride) = [q.value]
key_text(io, ::Plan, ::Ride) = print(io, " || ride")
priceable(d::Decider, p::Plan, q::Ride) = stepped_plan(d, p, p.groups[p.owner[q.value]]) !== nothing
explain(io, q::Ride, c, why) =
    println(io, "%", q.value, c.accepted ? " is summed as the steps of the recurrence it reads run" :
        " is summed after the steps of the recurrence it reads", why)
