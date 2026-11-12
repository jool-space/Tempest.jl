"""
The interface kernel packages implement. An implementation realizes groups of a
plan as kernels: it claims the groups it can realize, estimates the launches of
each claim, and builds them. It may also propose groups of its own. An
implementation is a value, and a target holds those it uses (see
`CUDATarget`). Claims read a `GroupView` of the group, in the vocabulary of
`Tempest.Compiler.IR`.
"""
module Implement

using Republic: @republic
@republic republic=false inherit=:public using ..IR

public Implementation, Work, package_of, loaded, claims, variants, estimate, resources, build_claim, repoints, repoint!, candidates,
    record_name, recomputes_folds, in_place, lowers, compiling, recurrence_axes

include("groups.jl")

"""
An implementation realizes groups of a plan as kernels. `claims` returns the
ways it can realize a group, `estimate` the launches of each, and `build_claim`
builds them. A target's floor realizes any group, so a plan never depends on
an implementation. `claims`, `variants` and `estimate` answer of their
arguments alone, and the cover asks them for many groups side by side, on the
threads there are: what they keep between calls, as a trial compilation or a
library's graph, they keep in a table a lock guards.
"""
abstract type Implementation end

"""
One launch's work: bytes moved, bytes it reads again (what its blocks read of
an argument beyond the first time), bytes it passes to or takes from another
of its claim's launches (partial results in a workspace, as a split sum's
parts), flops of products whose factors are of type `factor`, flops of folds
it recomputes inside them, its blocks, whether they split the sums among
them, and the scalar functions it applies elementwise, each with the number
of positions it is applied at. Tempest prices the bytes on its package's
memory, those read again from the cache, those passed within the cache where
they fit it, as the launch that takes them finds them there, the flops on
the line its package reaches for products of that class, and each function on
the time it measured for it on the device (see `seconds`). Functions applied
inside a product's sweep, to its factors, are `swept`: each step's small tile
of them stalls the sweep, and they cost many times what they cost streamed;
those applied inside a fold the sweep recomputes, to that fold's tiles, are
`inner`, at a factor of their own.

A launch whose blocks sweep tiles one after another, waiting for what each
step loads before the next, states the `steps` a block takes and the
functions the steps compute on what they loaded, each with the number of
steps it is applied in along their chain (`stepped`): a step takes as long as its loads
and that chain take, however few positions it covers, and a sweep of many
narrow steps takes that many times as long, whatever bytes it moves.
Tempest prices a step on its package's memory and each function on the time
it measured for it in a step (see `seconds`). A launch that states no steps,
or one, is priced by its bytes, flops and functions alone, as a product's
sweep is by its line.

A fold a launch recomputes where it reads it, its flops `recomputed`, a
block sweeps in steps of its own, before the sweep or inside each step the
fold moves along: a product's steps, among the sweep's, where its body is
bilinear along its axis, and otherwise `refold_steps` apart, combining its
body elementwise, which wait on their own loads one after another with the
rest of the block's work, however the sweep hides its bytes; Tempest prices
each as a step of the sweep.
"""
struct Work
    bytes::Float64
    reread::Float64
    passed::Float64
    flops::Float64
    recomputed::Float64
    factor::DataType
    blocks::Float64
    split::Bool
    applied::Vector{Pair{Any, Float64}}
    swept::Vector{Pair{Any, Float64}}
    inner::Vector{Pair{Any, Float64}}
    steps::Float64
    stepped::Vector{Pair{Any, Float64}}
    refold_steps::Float64
end

Work(bytes; reread = 0.0, passed = 0.0, flops = 0.0, recomputed = 0.0, factor = Float32, blocks = Inf, split = false,
    applied = Pair{Any, Float64}[], swept = Pair{Any, Float64}[], inner = Pair{Any, Float64}[], steps = 0.0, stepped = Pair{Any, Float64}[],
    refold_steps = 0.0) =
    Work(bytes, reread, passed, flops, recomputed, factor, blocks, split, applied, swept, inner, steps, stepped, refold_steps)

"""
    package_of(implementation) -> Symbol

The package whose kernels `implementation` launches, by name: the rates file
keeps the line and the memory it is priced on under it, shared by the
package's implementations. By default the module that defines its type.
"""
package_of(implementation::Implementation) = nameof(parentmodule(typeof(implementation)))

"""
    loaded(T) -> Bool

Whether implementations of type `T` can be made: true but for Tempest's own
implementations of a package, whose extension says so when the package loads.
"""
loaded(::Type{<:Implementation}) = true

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
    resources(implementation, target) -> resources

What the launches `implementation` builds for one executable share, and
nothing else uses: memory a launch writes only while it runs, as a library's
workspace, made on `target` (`allocate`, `upload`) as the executable's
buffers are. An executable runs one call at a time, its launches in order,
so its launches may share it; another executable's, or a decider's timing
claims beside it, have their own. Made once per executable, and once per
decider for the claims it builds to time, when the first launch is built;
`nothing` by default.
"""
resources(::Implementation, target) = nothing

"""
    build_claim(implementation, claim, target, group, buffers, resources) -> (launch, description)

Build a claim: `launch()` runs its kernels, and `description` says what they
are. `buffers[v]` is the storage of value `v`, for each value the group reads
or stores, and `resources` what `resources(implementation, target)` made for
the executable the launch is built for.
"""
function build_claim end

"""
    repoints(launch) -> Bool

Whether `launch`, as `build_claim` returned it, can read a value from other
storage than it was built on (see `repoint!`): false by default. `compile`
points the launches of an argument the program only reads at the caller's
array, rather than copying it into the program's storage, when all of the
program's launches can.
"""
repoints(launch) = false

"""
    repoint!(launch, value, buffer)

Make `launch` read value `value`, a number among the plan's values as in
`build_claim`'s `buffers`, from `buffer` from now on, where it read the
storage it was given before; a launch that does not read it is unchanged.
`buffer` is the target's storage of that value's element type and length,
aligned as the target's own storage is to the bytes its kernels assume of
an address. A graph captured from the launch is captured again before it
runs on `buffer`.
"""
function repoint! end

"""
    candidates(implementation, target, plan) -> Vector

What the implementation proposes in place of the generator's groups: sets of
values to realize as one group each, grown from anchors in `plan` (a product)
and accepted by the implementation, or several such sets, each a group of
its own, proposed as one. None by default. Proposals read the compiler's
plan, its groups through `Tempest.Compiler.group_views`.
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

include("library.jl")
include("packages.jl")

end
