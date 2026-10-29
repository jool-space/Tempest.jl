# The planner's rewrites: a softmax becomes one fold relative to its maximum
# and a variance one relative to its mean, a sum over a sum's product reads a
# sum computed already, and a fold Base spells with a tuple state or a
# log-sum-exp written out is the fold named, planned alike.
include("../fixtures/programs.jl")

# Rewrites of sums, which change rounding: rewritten programs plan and compute
# what they computed, and the spellings of a softmax weighting a sum plan
# alike, one shifted sum recomputing the scores in the kernel of the sum.
scores(q, k) = C.contract(k, q, (1, 2, 3, 4), (1, 2, 5, 4), (3, 2, 5, 4)) .* 0.125
causal(s) = ifelse.(reshape(1:size(s, 1), :, 1, 1) .<= reshape(1:size(s, 3), 1, 1, :), s, -Inf)
weigh(v, p) = C.contract(v, p, (1, 2, 3, 4), (3, 2, 5, 4), (1, 2, 5, 4))
by_logsumexp(q, k, v) = (s = causal(scores(q, k)); weigh(v, exp.(s .- T.logsumexp(s; dims = 1))))
function by_maximum(q, k, v)
    s = causal(scores(q, k))
    e = exp.(s .- maximum(s; dims = 1))
    weigh(v, e ./ sum(e; dims = 1))
end
# The log-sum-exp written out is a statistic once composed into one fold, and
# the weighted sum is relative to it then.
function spelled_out(q, k, v)
    s = causal(scores(q, k))
    m = maximum(s; dims = 1)
    weigh(v, exp.(s .- (m .+ log.(sum(exp.(s .- m); dims = 1)))))
end

@testset "rewritten sums" begin
    @testset "$name" for (name, f, args) in [PROGRAMS; REPEATED_WRITES]
        for p in (trace(f, args...), differentiate(trace(f, args...)))
            planned = recomputing(C.rewrite(C.lower(p)))
            @test agree(C.execute(planned, args...), filter(!isnothing, C.leaves(interpret(p, args...))))
        end
    end
    groups(f, args...) = recomputing(C.rewrite(C.lower(trace(f, args...)))).groups
    q, k, v = data(4, 2, 6, 2; scale = 0.3), data(4, 2, 6, 2; scale = 0.2), data(4, 2, 6, 2; scale = 0.1)
    a, b, c = only(groups(by_logsumexp, q, k, v)), only(groups(by_maximum, q, k, v)), only(groups(spelled_out, q, k, v))
    @test a.terms == b.terms == c.terms
    # As many features as keys and queries: the scores and the weighted sum
    # iterate over the same extents, and the scores, which the sum reads
    # transposed, are recomputed in its kernel all the same.
    qs, ks, vs = data(6, 2, 6, 2; scale = 0.3), data(6, 2, 6, 2; scale = 0.2), data(6, 2, 6, 2; scale = 0.1)
    @test length(C.recomputed_in(only(groups(by_logsumexp, qs, ks, vs)))) == 1
    graph = C.rewrite(C.lower(trace(by_maximum, q, k, v)))
    @test graph.values[only(b.reductions)].definition.fold == IR.Relative(IR.Exponential(), IR.Composed(/, (IR.Sum(1), IR.Sum(2))))
    @test length(C.recomputed_in(b)) == 1
    planned = recomputing(graph)
    @test agree(C.execute(planned, q, k, v), [interpret(trace(by_maximum, q, k, v), q, k, v)])
    @test occursin("online softmax", sprint(explain, planned))
    # In half precision the products convert their factors to Float32, and the
    # rewrite reads the exponential through its conversion: the same one fold,
    # its scores in BFloat16 and its weights and sums in Float32, rounded to
    # BFloat16 once at the end.
    function half(q, k, v)
        s = C.contract(k, q, (1, 2, 3, 4), (1, 2, 5, 4), (3, 2, 5, 4)) .* BFloat16(0.125)
        s = ifelse.(reshape(1:size(s, 1), :, 1, 1) .<= reshape(1:size(s, 3), 1, 1, :), s, typemin(BFloat16))
        weigh(v, exp.(s .- T.logsumexp(s; dims = 1)))
    end
    qh, kh, vh = map(x -> BFloat16.(single(x)), (q, k, v))
    h = C.rewrite(C.lower(trace(half, qh, kh, vh)))
    fold = h.values[only(only(groups(half, qh, kh, vh)).reductions)].definition
    @test fold.fold == IR.Relative(IR.Exponential(), IR.Composed(/, (IR.Sum(1), IR.Sum(2))))
    @test h.values[fold.operand.value].eltype == Tuple{BFloat16, Float32, Float32}
    @test IR.fold_types(fold.fold, Tuple{BFloat16, Float32, Float32}).result == BFloat16
    @test agree(C.execute(recomputing(h), qh, kh, vh), [interpret(trace(half, qh, kh, vh), qh, kh, vh)]; rtol = 1.0e-2)
    # Scores scaled by a Float32 are Float32, and the product of the values
    # in BFloat16 with the weights is one too: the same fold, with the values
    # converted where they are read.
    @test length(C.rewrite(C.lower(trace((q, k, v) -> Ops.attention(q, k, v; causal = true), qh, kh, vh))).rewrites) == 1
end

# FlashAttention-2's identity: the softmax backward's Σ_j p·dp, dp = Σ_d v·dO,
# is Σ_d dO·o, o = Σ_j p·v the forward's output. Its gradient sums over the
# head's features, reading o, and not over the keys.
@testset "a sum over a sum's product reads a sum computed already" begin
    q, k, v = data(4, 2, 6, 2; scale = 0.3), data(4, 2, 6, 2; scale = 0.2), data(4, 2, 6, 2; scale = 0.1)
    p = differentiate(trace((q, k, v) -> sum(abs2, Ops.attention(q, k, v; causal = true)), q, k, v))
    g = C.rewrite(C.lower(p))
    exchanged = [id for (id, note) in g.rewrites if startswith(note, "sums exchanged")]
    @test length(exchanged) == 1
    d = g.values[only(exchanged)].definition
    @test map(k -> d.iteration[k], d.reduced) == (4,)
    @test agree(C.execute(recomputing(g), q, k, v), filter(!isnothing, C.leaves(interpret(p, q, k, v))))
end

# The backward reads the forward's log-sum-exp, which the online rewrite no
# longer needs: the group of the weighted sum computes it beside the sum, as
# FlashAttention's forward writes it, and stores its slice over the queries;
# no group computes it apart. With as many features as keys and queries, the
# scores tell the axes apart.
@testset "a statistic still read is kept beside the relative fold" begin
    notes(g) = [id for (id, note) in g.rewrites if occursin("beside a relative fold", note)]
    lse(p, r) = (f = p.graph.values[r].definition.fold; f isa IR.Relative && f.observe == IR.LOGSUMEXP)
    weighted(p, r) = (f = p.graph.values[r].definition.fold; f isa IR.Relative && f.observe != IR.LOGSUMEXP)
    for (q, k, v) in ((data(4, 2, 6, 2; scale = 0.3), data(4, 2, 6, 2; scale = 0.2), data(4, 2, 6, 2; scale = 0.1)),
            (data(6, 2, 6, 2; scale = 0.3), data(6, 2, 6, 2; scale = 0.2), data(6, 2, 6, 2; scale = 0.1)))
        p = differentiate(trace((q, k, v) -> sum(abs2, by_logsumexp(q, k, v)), q, k, v))
        planned = recomputing(C.rewrite(C.lower(p)))
        slice = only(notes(planned.graph))
        @test planned.graph.values[slice].dims == (2, 6, 2) && planned.stored[slice]
        forward = only(g for g in planned.groups if any(r -> weighted(planned, r), g.reductions))
        @test count(r -> lse(planned, r), forward.reductions) == 1 && slice in forward.outputs
        @test !any(g -> any(r -> lse(planned, r), g.reductions), filter(!=(forward), planned.groups))
        @test agree(C.execute(planned, q, k, v), filter(!isnothing, C.leaves(interpret(p, q, k, v))))
    end
    # The forward alone reads it nowhere else.
    q, k, v = data(4, 2, 6, 2; scale = 0.3), data(4, 2, 6, 2; scale = 0.2), data(4, 2, 6, 2; scale = 0.1)
    @test isempty(notes(C.rewrite(C.lower(trace(by_logsumexp, q, k, v)))))
end

# The laws are data: the same rules that make a softmax one fold relative to
# the maximum make a variance one fold relative to the mean (Welford), and a
# log-sum-exp written by hand the fold `logsumexp` records.
relative_folds(g) = [v.definition.fold for v in g.values if v.definition isa C.Reduction && v.definition.fold isa IR.Relative]

@testset "folds relative to a statistic" begin
    x = data(8, 6)
    layernorm(x) = (m = sum(x; dims = 1) ./ 8; (x .- m) ./ sqrt.(sum(abs2, x .- m; dims = 1) ./ 8 .+ 1.0e-5))
    variance(x) = (m = sum(x; dims = 1) ./ 8; d = x .- m; sum(d .* d; dims = 1) ./ 8)
    by_hand(x) = (m = maximum(x; dims = 1); m .+ log.(sum(exp.(x .- m); dims = 1)))
    for (f, law, observe) in ((layernorm, IR.Square(), IR.Composed(sqrt, (IR.Composed(+, (IR.Composed(*, (IR.Sum(1), 0.125)), 1.0e-5)),))),
            (variance, IR.Square(), IR.Composed(*, (IR.Sum(1), 0.125))),
            (by_hand, IR.Exponential(), IR.LOGSUMEXP))
        g = C.rewrite(C.lower(trace(f, x)))
        @test relative_folds(g) == [IR.Relative(law, observe)]
        p = recomputing(g)
        @test C.structure(p).kernels == 1
        @test agree(C.execute(p, x), [f(x)])
    end
    # The mean's own fold stays beside the variance's, in the same phase: one
    # sweep, where the two-pass spelling had the variance read the mean.
    g = C.rewrite(C.lower(trace(layernorm, x)))
    p = recomputing(g)
    @test all(r -> only(p.groups).phase[r] == 1, only(p.groups).reductions)
    # The two spellings of a log-sum-exp get the same plan, the notes aside.
    planned(f) = filter(l -> !startswith(l, "rewritten:"), split(sprint(explain, recomputing(C.rewrite(C.lower(trace(f, x))))), "\n"))
    @test planned(by_hand) == planned(x -> T.logsumexp(x; dims = 1))
end

# A fold whose state is a tuple, as Base spells it: `mapreduce` with a lift to
# the state, a merge and a tuple `init`, then what the state means, which reads
# its components. It is one fold, with no tuple stored or left in the graph:
# the lift runs in the fold, and what is computed from the state is composed
# into the fold's observe. So it plans as the fold named with `GenericFold`.
@testset "a fold whose state is a tuple, as Base spells it" begin
    x = data(30, 20)
    states(x; dims = 1) = mapreduce(welford_lift, welford, x; dims, init = (0.0, 0.0, 0.0))
    spellings = [
        x -> base_variance(x; dims = 1),
        x -> (s = states(x); getindex.(s, 3) ./ getindex.(s, 1)),
        x -> (((n, m, q),) -> sqrt(q / n)).(states(x)),
        x -> (s = states(x; dims = :); s[3] / s[1]),
        x -> (s -> s[3] / s[1] - (s[2] / s[1])^2).(mapreduce(e -> (one(e), e, e^2), (a, b) -> a .+ b, x; dims = 1, init = (0.0, 0.0, 0.0))),
    ]
    for f in spellings
        p = trace(f, x)
        @test interpret(p, x) == f(x)
        g = C.rewrite(C.lower(p))
        @test count(v -> v.definition isa C.Reduction, g.values) == 1
        @test !any(v -> v.eltype <: Tuple, g.values)
        @test agree(C.execute(recomputing(g), x), [f(x)])
    end
    planned(f) = filter(l -> !startswith(l, "rewritten:"), split(sprint(explain, recomputing(C.rewrite(C.lower(trace(f, x))))), "\n"))
    @test planned(x -> base_variance(x; dims = 1)) == planned(x -> reduce(VARIANCE, x; dims = 1))
    @test planned(x -> (s = states(x; dims = :); s[3] / s[1])) == planned(x -> reduce(VARIANCE, x))
    # The state is of the type of `init`, as Base's result is: BFloat16
    # elements accumulate in Float32, to its precision. Base's own reduction
    # along `dims` merges blocks of lifts in BFloat16 first, and is off by 1%.
    h = BFloat16.(x)
    in_single(x) = (s -> s[3] / s[1]).(mapreduce(welford_lift, welford, x; dims = 1, init = (0.0f0, 0.0f0, 0.0f0)))
    @test isapprox(interpret(trace(in_single, h), h), base_variance(Float64.(h); dims = 1); rtol = 1.0e-6)
    # Its derivative comes from its combine: two exclusive scans give each
    # element the fold of the rest, and the combine's own code the partials.
    # The gradient is the variance's written with sums. Planned, the fold
    # rides the sweep of the scan before each element, and the scan after it
    # computes the gradient: four kernels, with the count and the loss, where
    # the sums spelling, a fold relative to the mean, has two.
    @testset "a derivative for a fold Base spells" begin
        loss(x) = sum(base_variance(x; dims = 1))
        @test message(() -> differentiate(trace(loss, x))) == ""
        by_sums(x) = (m = sum(x; dims = 1) ./ size(x, 1); sum(sum(abs2, x .- m; dims = 1) ./ size(x, 1)))
        @test isapprox(last(T.gradient(loss, x)), last(T.gradient(by_sums, x)); rtol = 1.0e-12)
        p = C.plan(C.rewrite(C.lower(differentiate(trace(loss, x)))))
        @test C.structure(p).kernels == 4 && count(g -> g.kind === :scan, p.groups) == 2
        @test C.structure(C.plan(C.rewrite(C.lower(differentiate(trace(by_sums, x)))))).kernels == 2
    end
end

# The log-sum-exp written out differentiates as `logsumexp` does: its maximum
# gets no cotangent, the loss being the same at every shift, and the rules
# read the log-sum-exp alone, at the shift where the sum of exponentials is
# one, so that the maximum and the sum are one fold again. A plain loss, a
# softmax-weighted sum and causal attention plan their gradients alike, the
# notes aside.
@testset "a log-sum-exp written out differentiates as the fold" begin
    planned(f, args...) = filter(l -> !startswith(l, "rewritten:"),
        split(sprint(explain, recomputing(C.rewrite(C.lower(differentiate(trace(f, args...)))))), "\n"))
    x, s = data(8, 6), data(6, 2, 5, 2; scale = 0.4)
    q, k, v = data(4, 2, 6, 2; scale = 0.3), data(4, 2, 6, 2; scale = 0.2), data(4, 2, 6, 2; scale = 0.1)
    losses = [
        (lse -> x -> sum(lse(x; dims = 1)), (x,)),
        (lse -> (v, s) -> sum(weigh(v, exp.(s .- lse(s; dims = 1)))), (v, s)),
        (lse -> (q, k, v) -> (s = causal(scores(q, k)); sum(abs2, weigh(v, exp.(s .- lse(s; dims = 1))))), (q, k, v)),
    ]
    for (loss, args) in losses
        @test planned(loss(written_logsumexp), args...) == planned(loss(T.logsumexp), args...)
    end
    # Written with the maximum, a softmax reads the maximum and the sum apart
    # in the forward, and they stay two folds.
    @testset "gap: the rewrites read a softmax's maximum and sum through their log-sum-exp" begin
        t = data(8, 6; scale = 0.7)
        by_maximum(x) = (m = maximum(x; dims = 1); e = exp.(x .- m); sum(e ./ sum(e; dims = 1) .* t))
        @test_broken planned(by_maximum, x) == planned(x -> sum(softmax_columns(x) .* t), x)
    end
end
