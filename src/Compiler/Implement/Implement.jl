"""
The interface kernel packages implement. An implementation realizes groups of a
plan as kernels: it claims the groups it can realize, estimates the launches of
each claim, and builds them. It may also propose groups of its own. A package
makes its implementations available with `register!` when it loads, and a
target decides which of them it uses. Claims read a `GroupView` of the group,
in the vocabulary of `Tempest.Compiler.IR`.
"""
module Implement

using Republic: @republic
@republic republic=false inherit=:public using ..IR

public Implementation, Work, register!, claims, variants, estimate, build_claim, candidates, record_name,
    recomputes_folds, in_place, lowers, compiling

include("groups.jl")

"""
An implementation realizes groups of a plan as kernels. `claims` returns the
ways it can realize a group, `estimate` the launches of each, and `build_claim`
builds them. A target's floor realizes any group, so a plan never depends on
an implementation.
"""
abstract type Implementation end

"""
One launch's work: bytes moved, bytes it reads again (what its blocks read of
an argument beyond the first time), flops of products whose factors are of type
`factor`, flops of folds it recomputes inside them, its blocks, whether they
split the sums among them, and the scalar functions it applies elementwise,
each with the number of positions it is applied at. Tempest prices the
bytes on its package's memory, those read again from the cache, the flops on
the line its package reaches for products of that class, and each function on
the time it measured for it on the device (see `seconds`). Functions applied
inside a product's sweep, to its factors, are `swept`: each step's small tile
of them stalls the sweep, and they cost many times what they cost streamed;
those applied inside a fold the sweep recomputes, to that fold's tiles, are
`inner`, at a factor of their own.
"""
struct Work
    bytes::Float64
    reread::Float64
    flops::Float64
    recomputed::Float64
    factor::DataType
    blocks::Float64
    split::Bool
    applied::Vector{Pair{Any, Float64}}
    swept::Vector{Pair{Any, Float64}}
    inner::Vector{Pair{Any, Float64}}
end

Work(bytes; reread = 0.0, flops = 0.0, recomputed = 0.0, factor = Float32, blocks = Inf, split = false,
    applied = Pair{Any, Float64}[], swept = Pair{Any, Float64}[], inner = Pair{Any, Float64}[]) =
    Work(bytes, reread, flops, recomputed, factor, blocks, split, applied, swept, inner)

# Implementations by the package that provides them, ordered by its name.
const REGISTERED = Pair{Module, Vector{Implementation}}[]

"""
    register!(package::Module, items...)

Make `items`, implementations, available under `package`, replacing what it
registered before; extensions call this when their packages load.
"""
function register!(package::Module, items::Implementation...)
    filter!(p -> p.first !== package, REGISTERED)
    isempty(items) || push!(REGISTERED, package => Implementation[items...])
    sort!(REGISTERED; by = p -> string(nameof(p.first)))
    nothing
end

"""
    claims(implementation, target, group) -> claims

The ways `implementation` can realize `group`, a `GroupView`, on `target`, each
a value of its own type that `estimate` and `build_claim` take; empty when it
cannot, as by default.
"""
claims(::Implementation, target, group) = ()

"""
    variants(implementation, target, group, claim) -> claims

The claims that vary `claim`, offered once it has measured best among the
claims tried on `group` so far: the next stage of a search the implementation
stages, as the generator stages its product tiles (every output tile at one
depth, then the depths of the best and its split). The cover tries them in
turn until the best has none new. None by default: then `claims` is the whole
search.
"""
variants(::Implementation, target, group, claim) = ()

"""
    estimate(implementation, claim, target, group) -> Vector{Work}

The launches of a claim. By default one launch moving the group's bytes.
"""
estimate(::Implementation, claim, target, group) = [Work(group.bytes)]

"""
    build_claim(implementation, claim, target, group, buffers) -> (launch, description)

Build a claim: `launch()` runs its kernels, and `description` says what they
are. `buffers[v]` is the storage of value `v`, for each value the group reads
or stores.
"""
function build_claim end

"""
    candidates(implementation, target, plan) -> Vector

What the implementation proposes in place of the generator's groups: sets of
values to realize as one group each, grown from anchors in `plan` (a product)
and accepted by the implementation. None by default. Proposals read the
compiler's plan, its groups through `Tempest.Compiler.group_views`.
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
    lowers(implementation, target, term, group) -> Bool

Whether `implementation` can realize a group that applies `term`, an `Apply`
in `group`, a `GroupView`: false for a generator without a lowering of the
scalar function. What an implementation cannot do is declared here, before it
claims, rather than learned from a build that fails; the cover stores a map
the target's implementations cannot all lower apart, in a kernel of its own
(see `Tempest.Compiler.Store`). True by default.
"""
lowers(::Implementation, target, term, group) = true

"""
    compiling() -> Bool

Whether an implementation may compile on the device while it claims, as a
generator does to learn what it lowers (see `lowers`): not while a
preparation that builds nothing plans, which touches no device, nor while a
package precompiles.
"""
compiling() = COMPILING[] && ccall(:jl_generating_output, Cint, ()) == 0

const COMPILING = Base.ScopedValues.ScopedValue(true)

end
