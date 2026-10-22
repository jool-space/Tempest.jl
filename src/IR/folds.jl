# Folds: reductions described without a schedule. A fold maps each element to a
# state tuple (`fold_lift`), merges states (`fold_combine`) and maps the final
# state to the result (`fold_observe`); `fold_init(f, T)` is the identity state.
# Implementations group elements in any fixed order, so `fold_combine` must be
# associative and commutative up to rounding, and `fold_init` exact: it masks
# padded lanes.

public Fold, SumFold, MaxFold, MinFold, ProdFold, Relative, Law, Exponential, Square, Statistic, Sum, Composed, LOGSUMEXP, GenericFold,
    fold_init, fold_lift, fold_combine, fold_observe, fold_reduce, fold_types, lse_shift, HalfFloat

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
Sums, products and relative folds of half-precision elements accumulate in
Float32, a tuple element component by component.
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
    # A half-precision fold rounds its result back to the element type; a
    # tuple element's, to the type of its first component.
    (; state = S, accumulator = A, observed = R, result = A === T ? R : T <: Tuple ? fieldtype(T, 1) : T)
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
    Relative(law, observe)

A fold relative to a statistic of its own elements, in one pass. An element is
`(y, g...)`, or a number `y`, which is `(y, 1)`. The state is the statistic's
state over the elements so far and one sum of `g · φ(y - m)` for each `g`,
where `m` is the statistic and `φ` the law's function: for the `Exponential`
law, `exp(y - m)` against the maximum; for the `Square` law, `(y - m)^2`
against the mean. Each sum is relative to the statistic of the elements it
covers, and combining two states moves both to the statistic of their union
by the law's `rebase`: `exp` turns a difference of maxima into a factor, and a
sum of squares about a part's mean moves to the whole's by `n (μ_part - μ)^2`.
So softmax and the variance are each one pass, with no statistic computed
first. The result is `observe(law, state)`.

`logsumexp` is the exponential fold of a number observed as `m + log(sum)` (`LOGSUMEXP`);
the sums a softmax weighs, and the sums of squares about a mean, become
relative folds by rewrite (see `rewrite`).
"""
struct Relative{L, F} <: Fold
    law::L
    observe::F
end

abstract type Law end

"""
`exp(y - m)` against the maximum `m`. A sum relative to one maximum moves to a
larger one by the factor `exp(m_old - m_new)`. Exponentials are shifted by
zero when the maximum is infinite, so an empty or all `-Inf` slice sums to
zero, and a `+Inf` element to `+Inf`.
"""
struct Exponential <: Law end

"""
`(y - m)^2` against the mean `m`, whose state is the count and the sum. A sum
of squares about a part's mean moves to the mean of the whole by
`n_part (μ_part - μ)^2` (Welford's algorithm, as Chan combines it).
"""
struct Square <: Law end

law_function(::Exponential) = exp
law_function(::Square) = abs2
statistic_name(::Exponential) = "maximum"
statistic_name(::Square) = "mean"

# The statistic's own fold: its identity, one element's state, the merge of
# two, and the statistic from the state.
statistic_init(::Exponential, T) = (T(-Inf),)
statistic_lift(::Exponential, y) = (y,)
statistic_combine(::Exponential, a, b) = (max(a[1], b[1]),)
statistic_of(::Exponential, s) = s[1]
statistic_init(::Square, T) = (zero(T), zero(T))
statistic_lift(::Square, y) = (one(y), y)
statistic_combine(::Square, a, b) = (a[1] + b[1], a[2] + b[2])
statistic_of(::Square, s) = s[2] / s[1]

"A state split into the statistic's part and the sums."
split_state(::Exponential, s) = ((s[1],), Base.tail(s))
split_state(::Square, s) = ((s[1], s[2]), Base.tail(Base.tail(s)))

# One element's sums, relative to its own statistic: `exp(y - y)` is one,
# `(y - y)^2` zero.
element_sums(::Exponential, y, gs) = gs
element_sums(::Square, y, gs) = map(zero, gs)

"A sum relative to the statistic of state `s`, moved to the statistic `m` of a larger set."
rebase(::Exponential, a, s, m) = a * exp(s[1] - lse_shift(m))
rebase(::Square, a, s, m) = ifelse(iszero(s[1]), a, a + s[1] * (s[2] / s[1] - m)^2)

lse_shift(m) = ifelse(isfinite(m), m, zero(m))

fold_name(f::Relative) = f.observe == LOGSUMEXP ? :logsumexp : Symbol("sum relative to the ", statistic_name(f.law))
fold_init(f::Relative, ::Type{T}) where {T <: Number} = (statistic_init(f.law, T)..., zero(T))
fold_init(f::Relative, ::Type{E}) where {E <: Tuple} =
    (statistic_init(f.law, fieldtype(E, 1))..., ntuple(i -> zero(fieldtype(E, i + 1)), Val(fieldcount(E) - 1))...)
fold_lift(f::Relative, y::Number) = (statistic_lift(f.law, y)..., element_sums(f.law, y, (one(y),))...)
fold_lift(f::Relative, e::Tuple) = (statistic_lift(f.law, e[1])..., element_sums(f.law, e[1], Base.tail(e))...)
function fold_combine(f::Relative, a, b)
    (sa, xs), (sb, ys) = split_state(f.law, a), split_state(f.law, b)
    s = statistic_combine(f.law, sa, sb)
    m = statistic_of(f.law, s)
    (s..., map((x, y) -> rebase(f.law, x, sa, m) + rebase(f.law, y, sb, m), xs, ys)...)
end
fold_observe(f::Relative, s) = f.observe(f.law, s)

# Observes: functions of the law and the state.

"The statistic itself: the maximum, or the mean."
struct Statistic end
(::Statistic)(law, s) = statistic_of(law, first(split_state(law, s)))

"The `i`th sum."
struct Sum
    i::Int
end
(o::Sum)(law, s) = last(split_state(law, s))[o.i]

"""
`f` of other observes and numbers: `Composed(/, (Sum(1), Sum(2)))` is the
quotient of two sums. Called with a state alone, it composes the observes of a
`GenericFold`, functions of the state.
"""
struct Composed{F, A <: Tuple}
    f::F
    args::A
end
(o::Composed)(law, s) = o.f(map(a -> a isa Number ? a : a(law, s), o.args)...)
(o::Composed)(s) = o.f(map(a -> a isa Number ? a : a(s), o.args)...)

Base.show(io::IO, ::Statistic) = print(io, "statistic")
Base.show(io::IO, o::Sum) = print(io, "sum ", o.i)
Base.show(io::IO, o::Composed) = print(io, o.f, "(", join(map(repr, o.args), ", "), ")")

"The log-sum-exp: the maximum plus the log of the sum of exponentials relative to it."
const LOGSUMEXP = Composed(+, (Statistic(), Composed(log, (Sum(1),))))

"""
    fold_reduce(f) -> function or nothing

How to fold arrays of states along some of their axes at once, when `f` has a
faster way than combining them in pairs: a function of the state's components,
one array each, and the axes, that returns the folded components.
Implementations call it on their own arrays, such as tiles. A relative fold
takes the statistic first, so that each sum is moved once.
"""
fold_reduce(::Fold) = nothing
fold_reduce(f::Relative) = RebasedSums(f.law)

struct RebasedSums{L}
    law::L
end

# The statistic's state folded along `dims`, and the statistic from it, as arrays.
statistic_reduce(::Exponential, s, dims) = (maximum(s[1]; dims),)
statistic_reduce(::Square, s, dims) = (sum(s[1]; dims), sum(s[2]; dims))
statistic_array(::Exponential, s) = s[1]
statistic_array(::Square, s) = s[2] ./ s[1]

# `rebase` over arrays of states, lane by lane.
struct Rebase{L}
    law::L
end
(r::Rebase)(a, sm...) = rebase(r.law, a, Base.front(sm), last(sm))

function (r::RebasedSums)(states, dims)
    sa, xs = split_state(r.law, states)
    s = statistic_reduce(r.law, sa, dims)
    m = statistic_array(r.law, s)
    (s..., map(a -> sum(Rebase(r.law).(a, sa..., m); dims), xs)...)
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

const HalfFloat = Union{Float16, BFloat16}
fold_accumulator(::Union{SumFold, ProdFold, Relative}, ::Type{<:HalfFloat}) = Float32
fold_accumulator(::Relative, ::Type{E}) where {E <: Tuple} =
    any(S -> S <: HalfFloat, fieldtypes(E)) ? Tuple{map(S -> S <: HalfFloat ? Float32 : S, fieldtypes(E))...} : E

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
