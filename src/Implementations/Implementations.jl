"""
The interface kernel packages implement. An implementation realizes groups of a
plan as kernels: it claims the groups it can realize, estimates the launches of
each claim, and builds them. It may also propose groups of its own and claim
calls of operations. A package makes its implementations available with
`register!` when it loads, and a target decides which of them it uses.
Claims read a `GroupView` of the group, in the vocabulary of `Tempest.IR`.
"""
module Implementations

using Republic: @republic
@republic republic=false inherit=:public using ..IR
using ..IR: stored_reads

public Implementation, Work, register!, claims, estimate, build_claim, candidates, record_name,
    recomputes_folds, in_place, Spec, bytes, Kernel, Form, pointwise

include("groups.jl")

"""
An implementation realizes groups of a plan as kernels. `claims` returns the
ways it can realize a group, `estimate` the launches of each, and `build_claim`
builds them. A target's floor realizes any group, so a plan never depends on
an implementation.
"""
abstract type Implementation end

"""
One launch's work: bytes moved, flops of products whose factors are of type
`factor`, flops of folds it recomputes inside them, its blocks, and whether
they split the sums among them. Tempest prices the bytes on its package's
memory and the flops on the line its package reaches for products of that
class (see `seconds`).
"""
struct Work
    bytes::Float64
    flops::Float64
    recomputed::Float64
    factor::DataType
    blocks::Float64
    split::Bool
end

Work(bytes; flops = 0.0, recomputed = 0.0, factor = Float32, blocks = Inf, split = false) =
    Work(bytes, flops, recomputed, factor, blocks, split)

# Implementations by the package that provides them, ordered by its name.
const REGISTERED = Pair{Module, Vector{Implementation}}[]

"""
    register!(package::Module, items...)

Make `items` available under `package`, replacing what it registered before;
extensions call this when their packages load. An item is an `Implementation`
or a pair `operation => function`, which registers `function` as a way to run
the operation: it takes the target and the call's `Spec`s, with the call's
keywords, and returns a `Kernel`, a vector of them, or `nothing` to decline.
The package's pairs come before its implementations.
"""
function register!(package::Module, items::Union{Implementation, Pair}...)
    filter!(p -> p.first !== package, REGISTERED)
    implementations = Implementation[i for i in items if i isa Implementation]
    pairs = Pair{Any, Any}[p for p in items if p isa Pair]
    isempty(pairs) || pushfirst!(implementations, Kernels(package, pairs))
    isempty(implementations) || push!(REGISTERED, package => implementations)
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
and accepted by the implementation, or `Tempest.Compiler.ClaimedCall`s of
computations it recognizes as calls it claims. None by default. Proposals read
the compiler's plan, its groups through `Tempest.Compiler.group_view`.
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

# Kernels of named operations: a package registers a function for an operation,
# and Tempest asks it for a kernel at each call, with the call's arguments as
# `Spec`s. The kernel's body takes groups of arrays, which Tempest builds when
# the program is prepared (see `build_kernel`).

"""
    Spec(T, dims...)

An argument of a call as a kernel's function sees it, before any buffer exists:
an element type and dimensions, with `size`, `eltype` and `bytes`. It carries
no layout; a kernel reads its arguments and writes its results densely in
their program shapes.
"""
struct Spec{T, N}
    dims::NTuple{N, Int}
end

Spec(::Type{T}, dims::Integer...) where {T} = Spec{T, length(dims)}(Int.(dims))
Spec(::Type{T}, dims::Tuple) where {T} = Spec(T, dims...)

Base.size(s::Spec) = s.dims
Base.size(s::Spec, k) = s.dims[k]
Base.eltype(::Spec{T}) where {T} = T

"The bytes of an array of this spec, or of a value as a view holds it."
bytes(s::Spec) = sizeof(eltype(s)) * prod(s.dims; init = 1)
bytes(v::ValueView) = sizeof(v.eltype) * prod(v.dims; init = 1)

"""
    Form

A form of work after a result that a kernel may absorb. Only `pointwise` exists:
a function applied to each element of the result before it is stored.
"""
struct Form
    name::Symbol
end

"The form of a function applied to each element of a result before it is stored."
const pointwise = Form(:pointwise)

"""
    Kernel(body; cost = nothing, scratch = nothing, params = nothing, epilogue = nothing)

What a registered function returns for a call. When the program runs,
`body(results, arguments[, scratch][, params][, after])` runs on the target's
arrays, with the groups the kernel declares: `results` has the structure the
operation returns, and a result nobody reads is `nothing`.

`cost` is the bytes the kernel moves. Left out, it is the bytes its group reads
and writes, in one launch; a number is one launch; a `Work` is one launch that
states flops too; a vector of numbers and `Work`s has one entry per launch.
`scratch` is a tuple of `Spec`s of workspace, and `params` a named tuple of
candidate values, the first of each of which is used. `epilogue` names a `Form`
per result it absorbs work after, in the structure of the results, and `after`
holds the function to apply, in that structure.
"""
struct Kernel{F, C, S, P, E}
    body::F
    cost::C
    scratch::S
    params::P
    epilogue::E
end

Kernel(body; cost = nothing, scratch = nothing, params = nothing, epilogue = nothing) = Kernel(body, cost, scratch, params, epilogue)

# The implementation of a package's operation pairs.
struct Kernels <: Implementation
    package::Module
    pairs::Vector{Pair{Any, Any}}
end

# A kernel a registered function returned for a call: which of its kernels,
# the call's keywords, the first candidate of each parameter, and the results
# it absorbs a form after, by number.
struct KernelClaim
    package::Module
    registered::Any
    index::Int
    kernel::Kernel
    kwargs::NamedTuple
    params::NamedTuple
    absorbs::Tuple{Vararg{Int}}
end

Base.show(io::IO, c::KernelClaim) = print(io, nameof(c.package), ".", nameof(c.registered), c.index > 1 ? "[$(c.index)]" : "")

# The results, by number, that `epilogue` names a form after, in the structure `results` of the call's.
absorbed(::Nothing, results) = ()
absorbed(::Form, result::Int) = (result,)
absorbed(epilogue::Union{Tuple, NamedTuple}, results) = Tuple(r for k in keys(epilogue) for r in absorbed(epilogue[k], results[k]))

kernels(::Nothing) = ()
kernels(kernel::Kernel) = (kernel,)
kernels(list::AbstractVector) = list

"""
    call_claims(implementation, target, op, specs) -> claims

The ways `implementation` can run the call `op` (a result's `Call`) with
arguments `specs` on `target`: every kernel the package's functions for the
operation return, and none for any other implementation.
"""
call_claims(::Implementation, target, op, specs) = ()

function call_claims(k::Kernels, target, op, specs)
    claims = KernelClaim[]
    for (f, registered) in k.pairs
        f === op.f || continue
        # A method that does not apply is no method; `nothing` declines too.
        hasmethod(registered, Tuple{typeof(target), map(typeof, specs)...}, keys(op.kwargs)) || continue
        for (index, kernel) in enumerate(kernels(registered(target, specs...; op.kwargs...)))
            params = kernel.params === nothing ? (;) : map(first, kernel.params)
            push!(claims, KernelClaim(k.package, registered, index, kernel, op.kwargs, params, absorbed(kernel.epilogue, op.structure)))
        end
    end
    claims
end

launch_work(cost::Real) = Work(Float64(cost))
launch_work(work::Work) = work

# A kernel's cost counts its own passes. The loads of the terms it absorbs are
# charged to its first launch, which reads them where it stores the result.
function estimate(::Kernels, c::KernelClaim, target, g)
    c.kernel.cost === nothing && return [Work(g.bytes)]
    launches = map(launch_work, vcat(c.kernel.cost))
    absorbed = sum(bytes(g.values[v]) for e in values(g.epilogues) for v in stored_reads(g.terms[e]); init = 0.0)
    w = first(launches)
    [Work(w.bytes + absorbed, w.flops, w.recomputed, w.factor, w.blocks, w.split); launches[2:end]]
end

record_name(::Kernels, c::KernelClaim) = string(c, isempty(c.kwargs) ? "" : string(c.kwargs), isempty(c.params) ? "" : string(", ", c.params))

end
