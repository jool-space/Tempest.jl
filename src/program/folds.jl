# Folds: reductions described without a schedule. A fold maps each element to a
# state tuple (`fold_lift`), merges states (`fold_combine`) and maps the final
# state to the result (`fold_observe`); `fold_init(f, T)` is the identity state.
# Implementations group elements in any fixed order, so `fold_combine` must be
# associative and commutative up to rounding, and `fold_init` exact: it masks
# padded lanes.

abstract type Fold end

function fold_init end
function fold_lift end
function fold_combine end
function fold_observe end

fold_name(f::Fold) = Symbol(lowercase(string(nameof(typeof(f)))))

"""
    fold_accumulator(f, T) -> Type

The type a fold over elements of type `T` accumulates in. Elements convert to it
before `fold_lift`, and the observed value rounds once to `fold_eltype(f, T)`.
Sums, products and logsumexps of half-precision elements accumulate in Float32.
"""
fold_accumulator(::Fold, T) = T

function fold_types(f::Fold, T)
    A = fold_accumulator(f, T)
    init = fold_init(f, A)
    S = typeof(init)
    init isa Tuple && isbitstype(S) || throw(ArgumentError("fold $f: fold_init must return a tuple of bits values"))
    Base.promote_op(x -> fold_lift(f, x), A) === S &&
        Base.promote_op((a, b) -> fold_combine(f, a, b), S, S) === S ||
        throw(ArgumentError("fold $f: lift and combine must keep the state type $S"))
    R = Base.promote_op(s -> fold_observe(f, s), S)
    isconcretetype(R) || throw(ArgumentError("fold $f: observe must infer a concrete type, got $R"))
    # A half-precision fold rounds its result back to the element type.
    (; state = S, accumulator = A, observed = R, result = A === T ? R : T)
end

fold_eltype(f::Fold, T) = fold_types(f, T).result

struct SumFold <: Fold end
fold_name(::SumFold) = :sum
fold_init(::SumFold, T) = (zero(T),)
fold_lift(::SumFold, x) = (x,)
fold_combine(::SumFold, a, b) = (a[1] + b[1],)
fold_observe(::SumFold, s) = s[1]

struct MaxFold <: Fold end
struct MinFold <: Fold end
fold_name(::MaxFold) = :maximum
fold_name(::MinFold) = :minimum
fold_init(::MaxFold, T) = (typemin(T),)
fold_init(::MinFold, T) = (typemax(T),)
fold_lift(::Union{MaxFold, MinFold}, x) = (x,)
fold_combine(::MaxFold, a, b) = (max(a[1], b[1]),)
fold_combine(::MinFold, a, b) = (min(a[1], b[1]),)
fold_observe(::Union{MaxFold, MinFold}, s) = s[1]

struct ProdFold <: Fold end
fold_name(::ProdFold) = :prod
fold_init(::ProdFold, T) = (one(T),)
fold_lift(::ProdFold, x) = (x,)
fold_combine(::ProdFold, a, b) = (a[1] * b[1],)
fold_observe(::ProdFold, s) = s[1]

"""
`log(sum(exp, x))` in one pass. The state is the running maximum and the sum of
exponentials shifted by it. Exponentials are shifted by zero when the maximum is
infinite, so an empty or all `-Inf` slice gives `-Inf`, and a `+Inf` element
`+Inf`.
"""
struct LogSumExpFold <: Fold end
fold_name(::LogSumExpFold) = :logsumexp
fold_init(::LogSumExpFold, T) = (T(-Inf), zero(T))
fold_lift(::LogSumExpFold, x) = (x, one(x))
lse_shift(m) = ifelse(isfinite(m), m, zero(m))
fold_combine(::LogSumExpFold, a, b) = shifted_combine(a, b)
fold_observe(::LogSumExpFold, s) = lse_shift(s[1]) + log(s[2])

# States of a running maximum and sums relative to it: combining moves both
# sums to the larger maximum.
function shifted_combine(a, b)
    m = max(a[1], b[1])
    s = lse_shift(m)
    ea, eb = exp(a[1] - s), exp(b[1] - s)
    (m, map((x, y) -> x * ea + y * eb, Base.tail(a), Base.tail(b))...)
end

"""
    fold_reduce(f) -> function or nothing

How to fold arrays of states along some of their axes at once, when `f` has a
faster way than combining them in pairs: a function of the state's components,
one array each, and the axes, that returns the folded components.
Implementations call it on their own arrays, such as tiles. Logsumexp takes the
maximum first, so that each element needs one exponential.
"""
fold_reduce(::Fold) = nothing
fold_reduce(::LogSumExpFold) = shifted_sums

function shifted_sums(states, dims)
    m = first(states)
    top = maximum(m; dims)
    shift = lse_shift.(top)
    (top, map(s -> sum(s .* exp.(m .- shift); dims), Base.tail(states))...)
end

"""
    GenericFold(; init, combine, lift = tuple, observe = only, pullback = nothing)

A fold from functions. `init` is a state tuple or a function of the element type
returning one. `pullback(x, y, g, axes)`, when given, is its derivative rule.
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
fold_name(::GenericFold) = :fold
fold_init(f::GenericFold, T) = f.init isa Tuple ? f.init : f.init(T)
fold_lift(f::GenericFold, x) = f.lift(x)
fold_combine(f::GenericFold, a, b) = f.combine(a, b)
fold_observe(f::GenericFold, s) = f.observe(s)

const HalfFloat = Union{Float16, Core.BFloat16}
fold_accumulator(::Union{SumFold, ProdFold, LogSumExpFold}, ::Type{<:HalfFloat}) = Float32

# Reference evaluation, through Base's reductions. Sums keep Base's pairwise
# order; elements convert to the accumulator and the result rounds once.
function fold_array(f::Fold, x::AbstractArray, axes::Tuple)
    T = eltype(x)
    (; accumulator, result) = fold_types(f, T)
    y = fold_elements(f, accumulator === T ? x : accumulator.(x), axes)
    result.(y)
end

function fold_elements(f::Fold, x, axes)
    states = mapreduce(e -> fold_lift(f, e), (a, b) -> fold_combine(f, a, b), x; dims = axes, init = fold_init(f, eltype(x)))
    map(s -> fold_observe(f, s), states)
end

fold_elements(::SumFold, x, axes) = sum(x; dims = axes)
