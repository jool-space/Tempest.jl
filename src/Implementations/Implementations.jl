"""
The interface kernel packages implement. An implementation realizes groups of a
plan as kernels: it claims the groups it can realize, estimates the launches of
each claim, and builds them. It may also propose groups of its own and claim
calls of operations. A package makes its implementations available with
`register!` when it loads, and a target decides which of them it uses.
Implementations read plans through the public names of `Tempest.Compiler`.
"""
module Implementations

public Implementation, Work, register!, claims, estimate, build_claim, candidates, record_name,
    recomputes_folds, in_place, claims_call, epilogue_results, build_call

"""
An implementation realizes groups of a plan as kernels. `claims` returns the
ways it can realize a group, `estimate` the launches of each, and `build_claim`
builds them. A target's floor realizes any group, so a plan never depends on
an implementation.
"""
abstract type Implementation end

"One launch's work: bytes moved, flops at the rate the launch reaches, and its blocks."
struct Work
    bytes::Float64
    flops::Float64
    rate::Float64
    blocks::Float64
end

Work(bytes; flops = 0.0, rate = Inf, blocks = Inf) = Work(bytes, flops, rate, blocks)

# Implementations by the package that provides them, ordered by its name.
const REGISTERED = Pair{Module, Vector{Implementation}}[]

"""
    register!(package::Module, implementations...)

Make `implementations` available under `package`, replacing what it registered
before; extensions call this when their packages load.
"""
function register!(package::Module, implementations::Implementation...)
    filter!(p -> p.first !== package, REGISTERED)
    isempty(implementations) || push!(REGISTERED, package => Implementation[implementations...])
    sort!(REGISTERED; by = p -> string(nameof(p.first)))
    nothing
end

"""
    claims(implementation, target, plan, group) -> claims

The ways `implementation` can realize `group` of `plan` on `target`, each a
value of its own type that `estimate` and `build_claim` take; empty when it
cannot, as by default.
"""
claims(::Implementation, target, plan, group) = ()

"""
    estimate(implementation, claim, target, plan, group) -> Vector{Work}

The launches of a claim. By default one launch moving the group's bytes.
"""
function estimate end

"""
    build_claim(implementation, claim, target, plan, group, buffers) -> (launch, description)

Build a claim: `launch()` runs its kernels, and `description` says what they
are. `buffers[v]` is the storage of value `v`, for each value the group reads
or stores.
"""
function build_claim end

"""
    candidates(implementation, target, plan) -> Vector

What the implementation proposes in place of the generator's groups: sets of
values to realize as one group each, grown from anchors in `plan` (a product)
and accepted by the implementation, or `Tempest.Compiler.ClaimedCall`s of
computations it recognizes as calls it claims. None by default.
"""
candidates(::Implementation, target, plan) = Vector{Int}[]

"A claim's name in the tuning record, without value numbers."
record_name(implementation::Implementation, claim) = string(nameof(typeof(implementation)), " ", claim)

"Whether `implementation` realizes groups that recompute folds where they read them, on `target`."
recomputes_folds(::Implementation, target) = false

"""
    in_place(implementation, claim) -> Bool

Whether the kernels of `claim` read every element that the outputs of one
shape need before they write any of them, so that an output may be written
into the storage of a value they read only at the address they write it (see
`storage`). The floor's kernels do, and so do the generator's.
"""
in_place(::Implementation, claim) = false

"""
    claims_call(implementation, target, f, kwargs, inputs) -> Bool

Whether `implementation` realizes calls of the operation `f` with these
keyword arguments on `target`; `inputs` holds the `(dims, eltype)` of each
argument. A claimed call is one opaque group whose kernel computes all its
results, estimated by `estimate` like any claim (the claim is a `CallClaim`).
It replaces the call's reference only where it is cheaper than the reference
expanded and fused into its neighbours; an opaque call runs only where it is
claimed.
"""
claims_call(::Implementation, target, f, kwargs, inputs) = false

"""
    epilogue_results(implementation, f, kwargs) -> results

The results of a claimed call over whose output tiles the implementation's
kernel applies an epilogue: pointwise work after the result, absorbed by the
planner when its reads align with the result's tile.
"""
epilogue_results(::Implementation, f, kwargs) = ()

"""
    build_call(implementation, plan, group, buffers, target) -> (launch, description)

Build the kernels of a claimed call: the group's opaque results, each written
densely in its program shape (its definition's `node.dims`) to its buffer or,
where `group.epilogues` names a value, that value computed from the result's
tile by the group's terms. Arguments are read densely in their program shapes
(the definition's `shapes`). A result whose buffer is `nothing` is read by
nobody and may be left unwritten.
"""
function build_call end

end
