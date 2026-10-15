# Folds shared by frontend/folds.jl and the GPU fold tests.
using Tempest: Fold, LogSumExpFold, MaxFold, MinFold, ProdFold, GenericFold

# A user-defined tuple-state fold: Welford's population variance, combined with
# Chan's update. Its derivative rule is registered by defining a method.
struct WelfordVariance <: Fold end
Tempest.fold_init(::WelfordVariance, T) = (zero(T), zero(T), zero(T))
Tempest.fold_lift(::WelfordVariance, x) = (one(x), x, zero(x))
function Tempest.fold_combine(::WelfordVariance, a, b)
    n = a[1] + b[1]
    w = b[1] / max(n, floatmin(n))
    d = b[2] - a[2]
    (n, a[2] + d * w, a[3] + b[3] + d * d * a[1] * w)
end
Tempest.fold_observe(::WelfordVariance, state) = state[3] / state[1]

function Tempest.fold_pullback(::WelfordVariance, x, y, g, axes)
    n = axes isa Colon ? length(x) : prod(size(x, d) for d in axes)
    mean = (axes isa Colon ? sum(x) : sum(x; dims = axes)) ./ n
    g .* (eltype(x)(2 / n) .* (x .- mean))
end

# A fold from functions, with no derivative rule: the range of a slice.
const range_fold = GenericFold(;
    init = T -> (T(Inf), T(-Inf)),
    lift = x -> (x, x),
    combine = (a, b) -> (min(a[1], b[1]), max(a[2], b[2])),
    observe = state -> state[2] - state[1],
)

# A fold carrying data is not zero-size: a logsumexp at a temperature.
struct TemperedLogSumExp <: Fold
    temperature::Float32
end
Tempest.fold_init(::TemperedLogSumExp, T) = (T(-Inf), zero(T))
Tempest.fold_lift(::TemperedLogSumExp, x) = (x, one(x))
function Tempest.fold_combine(f::TemperedLogSumExp, a, b)
    m = max(a[1], b[1])
    shift = Tempest.lse_shift(m)
    t = oftype(m, f.temperature)
    (m, a[2] * exp((a[1] - shift) / t) + b[2] * exp((b[1] - shift) / t))
end
Tempest.fold_observe(f::TemperedLogSumExp, state) =
    Tempest.lse_shift(state[1]) + oftype(state[1], f.temperature) * log(state[2])

lse_reference(x; dims) = (m = maximum(x; dims); m .+ log.(sum(exp.(x .- m); dims)))
variance_reference(x; dims) = sum(abs2, x .- sum(x; dims) ./ prod(size(x, d) for d in dims); dims) ./ prod(size(x, d) for d in dims)
range_reference(x; dims) = maximum(x; dims) .- minimum(x; dims)
tempered_reference(x, t; dims) = (m = maximum(x; dims); m .+ t .* log.(sum(exp.((x .- m) ./ t); dims)))

# Associative and commutative for nonnegative values, with identity zero: the
# 2-norm of a slice of absolute values. (Base's Float32 `hypot` computes in
# Float64, which cuTile cannot lower.)
const norm_combine = (a, b) -> sqrt(a * a + b * b)
# Not associative: valid only on backends that fold left in index order.
const skewed_combine = (a, b) -> a + 2b
# `min` outside the built-in table, with identity typemax.
const smaller = (a, b) -> min(a, b)

# Several folds at once, and their comparison, for the GPU fold tests.
function fold_mix(x, dims)
    l = Tempest.fold(LogSumExpFold(), x; dims)
    p = exp.(x .- l)
    (
        lse = l,
        max = maximum(x; dims),
        softmax = p,
        mean = sum(p .* x; dims),
        variance = Tempest.fold(WelfordVariance(), x; dims),
        range = Tempest.fold(range_fold, x; dims),
    )
end

function check_fold_outputs(got, expected; rtol = 2e-5, atol = 2e-5)
    for (g, r) in zip(values(got), values(expected))
        observed = r isa Number ? only(Array(g)) : Array(g)
        @test all(isfinite, observed)
        @test isapprox(observed, r; rtol, atol)
    end
end

# Half-precision folds accumulate in Float32 (`fold_accumulator`) on every
# backend and round once. Each case maps elements in the element type, then
# reduces. Element-type accumulation would lose most of these sums: adding 0.1
# 65536 times in Float16 stalls at 256.
half_fold_input(T) = T.(0.1 .+ 0.01 .* sin.(1:(1 << 16)))

const HALF_FOLD_CASES = (
    full_sum = (identity, x -> sum(x)),
    long_columns = (identity, x -> sum(reshape(x, 4096, :); dims = 1)),
    short_columns = (identity, x -> sum(reshape(x, 512, :); dims = 1)),
    rows = (identity, x -> sum(reshape(x, 512, :); dims = 2)),
    full_product = (x -> x[1:256] .+ eltype(x)(0.9), x -> prod(x)),
    products = (x -> x .+ eltype(x)(0.9), x -> prod(reshape(x, 64, :); dims = 1)),
    logsumexp = (identity, x -> Tempest.logsumexp(reshape(x, 1024, :); dims = 1)),
)

# The Float64 reduction of the mapped elements, rounded once to their type.
function half_fold_reference(case, x)
    elements, reduce = case
    mapped = elements(x)
    T = eltype(mapped)
    r = reduce(Float64.(mapped))
    r isa Number ? T(r) : T.(r)
end

# Within one unit in the last place of the rounded reference.
function check_half_fold(got, expected)
    observed = expected isa Number ? only(Array(got)) : Array(got)
    @test eltype(observed) === eltype(expected)
    @test all(abs.(Float64.(observed) .- Float64.(expected)) .<= eps.(expected))
end
