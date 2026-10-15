# Folds: reductions described without a schedule. A fold node is one graph value;
# its state tuple exists only inside an implementation. The observer turns the
# final state into the node's value, so consumers, storage and graph boundaries
# see an ordinary array with the reduced axes kept as singletons.
#
# Fold functions are scalar Julia functions on elements and state tuples. They
# must be deterministic and free of side effects: backends regroup and reorder
# their applications, and tile backends reevaluate them per lane.

"""
    fold_init(f::Fold, T) -> Tuple

The identity state for elements of type `T`. Generated kernels also use it to
mask padded lanes, so it must be exact: combining it with any state returns that
state.
"""
function fold_init end

"""    fold_lift(f::Fold, x) -> Tuple: the state of a single element."""
function fold_lift end

"""    fold_combine(f::Fold, a::Tuple, b::Tuple) -> Tuple: merge two states."""
function fold_combine end

"""    fold_observe(f::Fold, state::Tuple): the reduction's value from a final state."""
function fold_observe end

fold_name(f::Fold) = Symbol(lowercase(string(nameof(typeof(f)))))
fold_name(::SumFold) = :sum

fold_width(f::Fold, T) = length(fold_init(f, T))

# Result and state types follow from the fold's functions, without evaluating
# them. Every function must keep the state type of `fold_init`, so kernels can
# hold the state in fixed registers.
function fold_types(f::Fold, T)
    init = fold_init(f, T)
    S = typeof(init)
    init isa Tuple && isbitstype(S) && all(x -> x isa Real, init) ||
        throw(ArgumentError("fold $f: fold_init must return a tuple of real bits values"))
    lifted = Base.promote_op(x -> fold_lift(f, x), T)
    combined = Base.promote_op((a, b) -> fold_combine(f, a, b), S, S)
    lifted === S && combined === S || throw(
        ArgumentError("fold $f: lift and combine must return the state type $S of fold_init"),
    )
    R = Base.promote_op(state -> fold_observe(f, state), S)
    isconcretetype(R) && R <: Real ||
        throw(ArgumentError("fold $f: observe must infer a concrete real type, got $R"))
    (; state = S, result = R)
end

fold_eltype(f::Fold, T) = fold_types(f, T).result

"""
    fold_accumulator(f::Fold, T) -> Type

The element type a fold over elements of type `T` accumulates in. Every
implementation converts each element to it before `fold_lift`, starts from
`fold_init(f, fold_accumulator(f, T))`, and rounds the observed value once to
the reduction's element type `fold_eltype(f, T)`. Sums, products and
logsumexps of Float16 or BFloat16 elements accumulate in Float32: rounding every
partial state to half precision loses most of a long sum, and a result should
not depend on which backend ran the fold. Maxima and minima are exact in any
type and keep `T`, as do other folds, including `GenericFold`, whose state type
its `init` chooses.
"""
fold_accumulator(::Fold, T) = T

fold_init(::SumFold, T) = (zero(T),)
fold_lift(::SumFold, x) = (x,)
fold_combine(::SumFold, a, b) = (a[1] + b[1],)
fold_observe(::SumFold, state) = state[1]

"""
    MaxFold(), MinFold()

The largest or smallest element. NaN propagates as in `max` and `min`.
"""
struct MaxFold <: Fold end
struct MinFold <: Fold end

fold_name(::MaxFold) = :maximum
fold_name(::MinFold) = :minimum
fold_init(::MaxFold, T) = (typemin(T),)
fold_init(::MinFold, T) = (typemax(T),)
fold_lift(::Union{MaxFold, MinFold}, x) = (x,)
fold_combine(::MaxFold, a, b) = (max(a[1], b[1]),)
fold_combine(::MinFold, a, b) = (min(a[1], b[1]),)
fold_observe(::Union{MaxFold, MinFold}, state) = state[1]

"""
    ProdFold()

The product of the elements.
"""
struct ProdFold <: Fold end

fold_name(::ProdFold) = :prod
fold_init(::ProdFold, T) = (one(T),)
fold_lift(::ProdFold, x) = (x,)
fold_combine(::ProdFold, a, b) = (a[1] * b[1],)
fold_observe(::ProdFold, state) = state[1]

"""
    LogSumExpFold()

`log(sum(exp, x))` in one pass. The state is the running maximum and the sum of
exponentials shifted by it; combining two states rescales the smaller one. An
empty or all `-Inf` slice gives `-Inf`, and a `+Inf` element gives `+Inf`, as in
the shifted reference.
"""
struct LogSumExpFold <: Fold end

fold_name(::LogSumExpFold) = :logsumexp
fold_init(::LogSumExpFold, T) = (T(-Inf), zero(T))
fold_lift(::LogSumExpFold, x) = (x, one(x))

# Exponentials are shifted by the maximum, or by zero when it is infinite, as in
# the reference: an empty slice keeps a zero sum instead of -Inf - -Inf = NaN,
# and a +Inf element keeps the sum infinite.
lse_shift(m) = ifelse(isfinite(m), m, zero(m))

function fold_combine(::LogSumExpFold, a, b)
    m = max(a[1], b[1])
    shift = lse_shift(m)
    (m, a[2] * exp(a[1] - shift) + b[2] * exp(b[1] - shift))
end

fold_observe(::LogSumExpFold, state) = lse_shift(state[1]) + log(state[2])

"""
    GenericFold(; init, combine, lift = tuple, observe = only, pullback = nothing)

A fold from functions. `init` is a state tuple or a function of the element type
returning one; `combine(a, b)` and `observe(state)` take state tuples. The
optional `pullback(x, y, g, axes)` is its derivative rule (see `fold_pullback`).
Folds built from functions without captures are zero-size, which cuTile graph
regions require; other folds run on the CUDA and CPU kernels.
"""
struct GenericFold{I, L, C, O, P} <: Fold
    init::I
    lift::L
    combine::C
    observe::O
    pullback::P
end

GenericFold(; init, combine, lift = tuple, observe = only, pullback = nothing) =
    GenericFold(init, lift, combine, observe, pullback)

const HalfFloat = Union{Float16, Core.BFloat16}

fold_accumulator(::Union{SumFold, ProdFold, LogSumExpFold}, ::Type{<:HalfFloat}) = Float32

fold_init(f::GenericFold, T) = f.init isa Tuple ? f.init : f.init(T)
fold_lift(f::GenericFold, x) = f.lift(x)
fold_combine(f::GenericFold, a, b) = f.combine(a, b)
fold_observe(f::GenericFold, state) = f.observe(state)

# Scalar component forms for kernels that map functions over lanes, and binary
# forms for reduction primitives that need a compile-time combiner. They take
# the fold from its type, so they are zero-size when the fold is.
struct FoldLift{K, F <: Fold} end
(::FoldLift{K, F})(x) where {K, F} = fold_lift(F.instance, x)[K]

struct FoldMerge{K, W, F <: Fold} end
function (::FoldMerge{K, W, F})(xs...) where {K, W, F}
    a = ntuple(i -> xs[i], Val(W))
    b = ntuple(i -> xs[W + i], Val(W))
    fold_combine(F.instance, a, b)[K]
end

struct FoldObserve{F <: Fold} end
(::FoldObserve{F})(state...) where {F} = fold_observe(F.instance, state)

struct FoldOperator{F <: Fold} end
(::FoldOperator{F})(a, b) where {F} = only(fold_combine(F.instance, (a,), (b,)))

struct FoldCombiner{F <: Fold} end
(::FoldCombiner{F})(a, b) where {F} = fold_combine(F.instance, a, b)

"""
    fold_resident(f::Fold, x, valid, dims)

An optional lowering for a slice that a backend holds entirely in registers:
reduce the array-like `x` over `dims`, ignoring lanes where `valid` is false,
and return the observed value. Folds that define it return true from
`has_resident_lowering`. On GB10 a resident logsumexp ran 1.2 to 1.5 times
faster as a maximum and then a shifted sum than as one tuple-state pass; with
streamed columns the single pass was 2.4 times faster.
"""
function fold_resident end

has_resident_lowering(::Fold) = false
has_resident_lowering(::LogSumExpFold) = true

function fold_resident(::LogSumExpFold, x, valid, dims)
    T = eltype(x)
    peak = maximum(ifelse.(valid, x, T(-Inf)); dims)
    shift = lse_shift.(peak)
    shift .+ log.(sum(ifelse.(valid, exp.(x .- shift), zero(T)); dims))
end

# Reference evaluation through Base's reductions with the fold's functions.
# Base may regroup terms, which the associativity contract permits. Elements
# convert to the fold's accumulator, and the result rounds once.
function fold_array(f::Fold, x::AbstractArray, axes)
    T = eltype(x)
    A = fold_accumulator(f, T)
    A === T && return fold_elements(f, x, axes)
    R = fold_eltype(f, T)
    result = fold_elements(f, A.(x), axes)
    axes isa Colon ? R(result) : R.(result)
end

function fold_elements(f::Fold, x::AbstractArray, axes)
    states = mapreduce(
        element -> fold_lift(f, element),
        (a, b) -> fold_combine(f, a, b),
        x;
        dims = axes,
        init = fold_init(f, eltype(x)),
    )
    axes isa Colon ? fold_observe(f, states) : map(state -> fold_observe(f, state), states)
end

# Sums keep Base's pairwise order and, after conversion, accumulation type.
fold_elements(::SumFold, x::AbstractArray, ::Colon) = sum(x)
fold_elements(::SumFold, x::AbstractArray, axes::Tuple) = sum(x; dims = axes)

evaluate(op::FoldReduction, node, xs) = fold_array(op.fold, only(xs), op.axes)

"""
    Tempest.fold(f::Fold, x; dims = :)

Reduce `x` over `dims` with the fold `f`. On traced arrays this records one
`FoldReduction`; on ordinary arrays it evaluates the reference left fold.
"""
fold(f::Fold, x::AbstractArray; dims = :) = fold_array(f, x, fold_axes(x, dims))

function fold(f::Fold, v::TracedArray; dims = :)
    axes = fold_axes(v, dims)
    T = fold_eltype(f, eltype(v))
    if axes isa Colon
        v = reshape_root(v)
        return emit(v.nodes, FoldReduction(f, axes), (v.id,), (), T; scalar = true)
    end
    isempty(axes) && throw(ArgumentError("a fold needs at least one reduced axis of the input"))
    source = permuted_source(v)
    if source !== nothing
        parent, perm = source
        return permutedims(fold(f, parent; dims = Tuple(perm[d] for d in axes)), perm)
    end
    shape = ntuple(d -> d in axes ? 1 : size(v, d), ndims(v))
    emit(v.nodes, FoldReduction(f, axes), (v.id,), shape, T; scalar = false)
end

fold_axes(x, ::Colon) = Colon()

function fold_axes(x, dims)
    ds = dims isa Integer ? (dims,) : Tuple(dims)
    all(d -> d isa Integer && d > 0, ds) ||
        throw(ArgumentError("reduction dimensions must be positive integers"))
    Tuple(sort!(unique!(Int[d for d in ds if d <= ndims(x)])))
end

Base.maximum(v::TracedArray; dims = :) = fold(MaxFold(), v; dims)
Base.minimum(v::TracedArray; dims = :) = fold(MinFold(), v; dims)
Base.maximum(f, v::TracedArray; dims = :) = maximum(pointwise(f, v); dims)
Base.minimum(f, v::TracedArray; dims = :) = minimum(pointwise(f, v); dims)

# Products widen small integers as Base's `prod` does.
function Base.prod(v::TracedArray; dims = :)
    T = Base.promote_op(Base.mul_prod, eltype(v), eltype(v))
    fold(ProdFold(), pointwise(T, v); dims)
end

Base.prod(f, v::TracedArray; dims = :) = prod(pointwise(f, v); dims)

# Operators with a built-in reduction, and the operator that combines `init`
# with its result. Base's own `sum` and `prod` pass `add_sum` and `mul_prod`.
builtin_reduction(::Union{typeof(+), typeof(Base.add_sum)}) = (sum, +)
builtin_reduction(::Union{typeof(*), typeof(Base.mul_prod)}) = (prod, *)
builtin_reduction(::typeof(max)) = (maximum, max)
builtin_reduction(::typeof(min)) = (minimum, min)
builtin_reduction(op) = nothing

struct NoInit end

"""
    reduce(op, v::TracedArray; dims = :, init)
    mapreduce(f, op, v::TracedArray; dims = :, init)

`+`, `*`, `max` and `min` record their built-in reductions. A given `init` is
combined once with each result, `op(init, result)`, as Base does.

Another operator becomes a `GenericFold` and needs `init`, its identity: every
slice, and every padded lane of a parallel backend, starts from it. As for every
fold, `op` must be associative and commutative up to rounding, since backends
regroup and reorder its applications. The fold is zero-size, which cuTile graph
regions require, when `op` is a singleton and `init` equals zero, one, `typemin`
or `typemax` of the element type; another `init` is captured, and the fold runs
on the CUDA kernels. Such a fold has no derivative rule.
"""
# Real elements keep this method apart from Base's `reduce(vcat or hcat, vector
# of arrays)`, which would otherwise be ambiguous with it.
Base.reduce(op, v::TracedArray{<:Real}; dims = :, init = NoInit()) =
    mapreduce(identity, op, v; dims, init)

function Base.mapreduce(f, op, v::TracedArray; dims = :, init = NoInit())
    x = pointwise(f, v)
    builtin = builtin_reduction(op)
    if builtin !== nothing
        reduction, combine = builtin
        y = reduction(x; dims)
        return init isa NoInit ? y : pointwise(combine, init, y)
    end
    init isa NoInit && throw(
        ArgumentError(
            "reducing a traced array by an operator other than +, *, max and min " *
            "needs `init`, the identity of the operator; got $op",
        ),
    )
    init isa Real || throw(ArgumentError("`init` must be a real number, got $(typeof(init))"))
    T = promote_type(eltype(x), typeof(init))
    operator_fold = GenericFold(; init = element_identity(init, T), combine = OperatorCombine(op))
    fold(operator_fold, pointwise(T, x); dims)
end

# The combine of a fold whose state is one running value.
struct OperatorCombine{O}
    op::O
end

(c::OperatorCombine)(a, b) = (c.op(a[1], b[1]),)

# An identity rebuilt from the element type keeps the fold zero-size.
struct ElementIdentity{K} end

(::ElementIdentity{:zero})(T) = (zero(T),)
(::ElementIdentity{:one})(T) = (one(T),)
(::ElementIdentity{:typemin})(T) = (typemin(T),)
(::ElementIdentity{:typemax})(T) = (typemax(T),)

function element_identity(init, T)
    value = convert(T, init)
    for kind in (:zero, :one, :typemin, :typemax)
        candidate = ElementIdentity{kind}()
        isequal(only(candidate(T)), value) && return candidate
    end
    (value,)
end

"""
    fold_pullback(f::Fold, x, y, g, axes) -> cotangent of x

The derivative rule of a fold, as ordinary graph operations on the input `x`,
the forward value `y` and its cotangent `g`. Rules form a table: register one by
defining a method for your fold type, or pass `pullback` to `GenericFold`.
There is no generic rule: the reverse of an arbitrary associative combine needs
the prefix and suffix states of every element. A fold without a rule still runs
forward; `differentiate` fails when it reaches the fold.
"""
function fold_pullback(f::Fold, x, y, g, axes)
    throw(
        ArgumentError(
            "no pullback for fold $(fold_name(f)) ($(typeof(f))); define " *
            "Tempest.fold_pullback(::$(nameof(typeof(f))), x, y, g, axes) or pass " *
            "`pullback` to GenericFold",
        ),
    )
end

function fold_pullback(f::GenericFold, x, y, g, axes)
    f.pullback === nothing && return invoke(fold_pullback, Tuple{Fold, Any, Any, Any, Any}, f, x, y, g, axes)
    f.pullback(x, y, g, axes)
end

pullback(op::FoldReduction, xs, y, g) = (fold_pullback(op.fold, only(xs), y, g, op.axes),)

fold_pullback(::SumFold, x, y, g, axes) =
    pointwise(identity, g; dims = size(x), scalar = x isa TracedScalar)

# d logsumexp(x) / dx = exp(x - y), the softmax of the slice.
fold_pullback(::LogSumExpFold, x, y, g, axes) = g .* exp.(x .- y)

"""Indicator of `x == y`, in the element type of `x`."""
struct Tie end
(::Tie)(x, y) = oftype(x, x == y)
recomputable(::Tie) = true
local_partials(::Tie, xs, y) = map(x -> constant(x.nodes, zero(eltype(x))), xs)

# Tied extrema share the cotangent equally, so the gradient of a constant
# slice's maximum is its mean's gradient.
function fold_pullback(::Union{MaxFold, MinFold}, x, y, g, axes)
    ties = pointwise(Tie(), x, y)
    count = axes isa Colon ? sum(ties) : sum(ties; dims = axes)
    ties .* (g ./ count)
end

# The derivative of a product by one element is the product of the others.
# Dividing the result by the element fails at zeros, so the rule counts each
# slice's zeros and multiplies its nonzero elements: with no zero the partial is
# that product over the element; with one zero it is that product at the zero
# and zero elsewhere; with more zeros it vanishes.
function fold_pullback(::ProdFold, x, y, g, axes)
    T = eltype(x)
    is_zero = pointwise(Tie(), x, zero(T))
    count = axes isa Colon ? sum(is_zero) : sum(is_zero; dims = axes)
    # Zeros replaced by one, so the product and quotients below stay finite.
    factors = x .+ is_zero
    nonzero_product = fold(ProdFold(), factors; dims = axes)
    no_zero = pointwise(Tie(), count, zero(T))
    one_zero = pointwise(Tie(), count, one(T))
    g .* nonzero_product .* (no_zero ./ factors .+ one_zero .* is_zero)
end
