using Tempest: Fold, LogSumExpFold, MaxFold, MinFold, GenericFold

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

@testset "Fold reference and CPU execution" begin
    x = 50 .* reshape(sin.(1.0:35.0) .+ cos.(0.3 .* (1:35)), 7, 5)
    for dims in (1, 2, (1, 2))
        @test Tempest.fold(LogSumExpFold(), x; dims) ≈ lse_reference(x; dims)
        @test Tempest.fold(MaxFold(), x; dims) == maximum(x; dims)
        @test Tempest.fold(WelfordVariance(), x; dims) ≈ variance_reference(x; dims)
        @test Tempest.fold(range_fold, x; dims) == range_reference(x; dims)
        f(x) = (
            lse = Tempest.fold(LogSumExpFold(), x; dims),
            max = maximum(x; dims),
            min = minimum(abs2, x; dims),
            var = Tempest.fold(WelfordVariance(), x; dims),
            range = Tempest.fold(range_fold, x; dims),
            tempered = Tempest.fold(TemperedLogSumExp(2.0f0), x; dims),
        )
        got = execute!(prepare_forward(f, x))
        @test got.lse ≈ lse_reference(x; dims)
        @test got.max == maximum(x; dims)
        @test got.min == minimum(abs2, x; dims)
        @test got.var ≈ variance_reference(x; dims)
        @test got.range == range_reference(x; dims)
        @test got.tempered ≈ tempered_reference(x, 2; dims)
    end
    ex = prepare_forward(x -> Tempest.fold(LogSumExpFold(), x), x)
    @test only(execute!(ex)) ≈ only(lse_reference(x; dims = (1, 2)))
    @test occursin("logsumexp [cpu]", sprint(explain, ex))
    # dims = : gives a number, (1, 2) a 1×1 array, as for sum.
    @test Tempest.logsumexp(x; dims = :) ≈ only(lse_reference(x; dims = (1, 2)))
    @test Tempest.logsumexp(x; dims = (1, 2)) ≈ lse_reference(x; dims = (1, 2))
    p = trace(x -> Tempest.logsumexp(x; dims = :), x)
    @test p.nodes[p.output].scalar
    @test value_and_gradient(differentiate(p), x)[2][1] ≈ exp.(x .- Tempest.logsumexp(x; dims = :))
    @test trace(x -> Tempest.fold(MaxFold(), x; dims = (1, 2)), x).nodes[end].dims == (1, 1)
    @test Tempest.fold(LogSumExpFold(), [-Inf, -Inf]) == -Inf
    @test Tempest.fold(LogSumExpFold(), [1.0, Inf]) == Inf
    @test Tempest.fold(LogSumExpFold(), [1000.0, 1000.0]) ≈ 1000 + log(2)
    # Integer extrema start from typemin/typemax, which equal ∓Inf for floats.
    indices = Int32[3 1; 4 1; 5 9]
    for (f, g) in ((maximum, MaxFold()), (minimum, MinFold())), dims in (:, (1,), (2,))
        p = trace(i -> f(i; dims), indices)
        @test only(Tempest.interpret(Tempest.ForwardProgram(p), indices)) == f(indices; dims)
        @test Tempest.fold(g, indices; dims) == f(indices; dims)
    end
    # Without a provider, logsumexp's traced reference is the stable fold.
    big = 1000 .* x
    p = trace(x -> Tempest.reference(Tempest.logsumexp, x; dims = (2,)), big)
    @test only(Tempest.interpret(Tempest.ForwardProgram(p), big)) ≈ lse_reference(big; dims = 2)
    @test any(node -> node.op isa Tempest.FoldReduction{LogSumExpFold}, p.nodes)
    # Fold functions must keep the state type of fold_init.
    widening = GenericFold(; init = T -> (zero(T),), combine = (a, b) -> (a[1] + b[1],), lift = x -> (Float64(x),))
    @test_throws ArgumentError trace(x -> Tempest.fold(widening, x), Float32.(x))
end

@testset "Fold derivatives" begin
    y = reshape(cos.(0.7 .* (1:24)), 6, 4)
    for f in (
        y -> sum(abs2, Tempest.fold(LogSumExpFold(), y; dims = 1)),
        y -> Tempest.fold(LogSumExpFold(), y) ^ 2,
        y -> sum(sin, maximum(y; dims = 1)),
        y -> sum(abs2, y .- maximum(y; dims = 2)),
        y -> sum(abs2, Tempest.fold(WelfordVariance(), y; dims = 1)),
        y -> Tempest.fold(WelfordVariance(), y),
    )
        p = trace(f, y)
        loss, (dy,) = value_and_gradient(p, y)
        @test loss ≈ f(y)
        @test dy ≈ finite_difference(f, (copy(y),), 1) atol = 1e-6
        @test execute!(prepare(differentiate(p), y))[2] ≈ dy
    end
    # Tied extrema share the cotangent equally.
    _, (dc,) = value_and_gradient(trace(maximum, fill(0.5, 4)), fill(0.5, 4))
    @test dc ≈ fill(0.25, 4)
    # A fold without a rule runs forward and fails only at differentiate.
    loss(y) = sum(Tempest.fold(range_fold, y; dims = 1))
    p = trace(loss, y)
    error = try
        differentiate(p)
        nothing
    catch e
        e
    end
    @test error isa ArgumentError
    @test occursin("no pullback for fold genericfold", sprint(showerror, error))
    @test only(execute!(prepare_forward(loss, y))) ≈ loss(y)
    # A rule passed to GenericFold.
    ruled = GenericFold(;
        init = T -> (zero(T),),
        lift = x -> (x * x,),
        combine = (a, b) -> (a[1] + b[1],),
        pullback = (x, y, g, axes) -> g .* (2 .* x),
    )
    h(y) = sum(Tempest.fold(ruled, y; dims = 2))
    _, (dh,) = value_and_gradient(trace(h, y), y)
    @test dh ≈ 2 .* y
end
