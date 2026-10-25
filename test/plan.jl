# Plans compute what programs compute: every program and its differentiated
# form, planned and run kernel by kernel on the CPU, agrees with the program
# interpreter.
include("fixtures/programs.jl")

@testset "plans" begin
    @testset "$name" for (name, f, args) in [PROGRAMS; REPEATED_WRITES]
        for p in (trace(f, args...), differentiate(trace(f, args...)))
            planned = C.plan(C.lower(p))
            @test agree(C.execute(planned, args...), filter(!isnothing, C.leaves(interpret(p, args...))))
        end
    end
end

# The plan that recomputes every fold that can be: what the cover proposes.
recomputing(graph) = C.plan(graph; recompute = C.recomputable_folds(graph))

# Folds recomputed where they are read, as plans for cuTile recompute them:
# the same results, and attention's scores and logsumexp inside the one
# kernel of the sum over keys.
@testset "plans recomputing folds" begin
    @testset "$name" for (name, f, args) in [PROGRAMS; REPEATED_WRITES]
        for p in (trace(f, args...), differentiate(trace(f, args...)))
            planned = recomputing(C.lower(p))
            @test agree(C.execute(planned, args...), filter(!isnothing, C.leaves(interpret(p, args...))))
        end
    end
    q, k, v = data(4, 4, 6, 2; scale = 0.3), data(4, 2, 6, 2; scale = 0.2), data(4, 2, 6, 2; scale = 0.1)
    p = trace((q, k, v) -> Ops.attention(q, k, v; causal = true), q, k, v)
    planned = recomputing(C.lower(p))
    @test length(planned.groups) == 1
    @test length(C.recomputed_in(only(planned.groups))) == 2
    @test agree(C.execute(planned, q, k, v), [interpret(p, q, k, v)])
end

# A fold that recomputes another is recomputed only where its reader computes
# it once per block. Here a sum over keys recomputing the scores is read by a
# sum over queries, along which it moves, as attention's query gradient is by
# the gradient of the weights before it: it is stored, and its kernel
# recomputes the scores.
@testset "a fold recomputed inside a sweep recomputes no other" begin
    q, k, x = data(4, 8), data(4, 8), data(6, 8)
    function f(q, k, x)
        e = exp.(C.contract(k, q, (1, 2), (1, 3), (2, 3)))
        d = C.contract(k, e, (1, 2), (2, 3), (1, 3))
        C.contract(x, d, (1, 3), (2, 3), (1, 2))
    end
    p = trace(f, q, k, x)
    planned = recomputing(C.lower(p))
    @test all(g -> C.recomputed_in(g) == C.recomputed_in(g; nested = false), planned.groups)
    @test count(g -> !isempty(C.recomputed_in(g)), planned.groups) == 1
    @test agree(C.execute(planned, q, k, x), [interpret(p, q, k, x)])
end

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

# A scan is the sweep of a kernel along the axes it scans: a map before it is
# its operand, and a map after reads it where the sweep has it, so both fuse
# into its kernel and only the result is stored. Scans along the same axes in
# the same direction share the sweep; along another axis, a scan is another.
@testset "scans fuse the maps around them" begin
    x = data(8, 6)
    planned(f, args...) = C.plan(C.lower(trace(f, args...)))
    fused(x) = exp.(cumsum(sin.(x); dims = 1)) .* 2
    for (f, args) in ((fused, (x,)), (x -> cumsum(cumprod(x; dims = 2); dims = 2) .+ x, (x,)),
            (recurrence, (0.9 .+ 0.1 .* data(30; scale = 0.3), data(30; scale = 0.7))))
        p = planned(f, args...)
        @test C.structure(p).kernels == 1 && only(p.groups).kind === :scan
        @test length(C.structure(p).stored) == 1
        @test agree(C.execute(p, args...), [f(args...)])
        args = map(single, args)
        @test agree(C.execute(planned(f, args...), args...), [f(args...)]; rtol = 1.0e-5)
    end
    @test C.structure(planned(x -> cumsum(x; dims = 1) .+ cumsum(x; dims = 2), x)).kernels == 2
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

# A map read by a fold and by a later map over its domain is recomputed in
# their kernel, as when it is written twice.
@testset "a map one kernel reads" begin
    x = data(8, 6)
    for f in (x -> (e = exp.(x); e ./ sum(e; dims = 1)), x -> exp.(x) ./ sum(exp.(x); dims = 1))
        p = trace(f, x)
        planned = C.plan(C.lower(p))
        @test length(planned.groups) == 1
        @test agree(C.execute(planned, x), [interpret(p, x)])
    end
end

# Writes into an allocation are functional updates that plan with their
# readers: padding read by a reduction, a concatenation read by a map and a
# slice written then read are one kernel each, which stores only its result.
@testset "functional updates in one kernel" begin
    x, v = data(8, 6), data(3, 6; scale = 0.3)
    for (f, args) in ((x -> sum(abs2, padded(x); dims = 1), (x,)), ((x, v) -> vcat(x, v) .* 2, (x, v)),
                      ((x, v) -> (y = copy(x); y[2:4, :] = v; tanh.(y)), (x, v)))
        p = trace(f, args...)
        planned = C.plan(C.lower(p))
        @test length(planned.groups) == 1
        @test C.structure(planned).stored == [o.value for o in planned.graph.results]
        @test agree(C.execute(planned, args...), [interpret(p, args...)])
    end
end

# No kernel waits on itself: a fold does not join a kernel that already reads,
# from another kernel, a value computed from that fold, as Adam's update of a
# parameter reads its moment, computed from the gradient's fold.
@testset "no kernel joins a fold its reads wait on" begin
    p, q, m, w = data(8, 6), data(8, 6; scale = 0.3), data(8, 1), data(8, 1; scale = 0.5)
    function f(p, q, m, w)
        a = sum(p; dims = 2)
        g = sum(q; dims = 2)
        b = p .+ q
        n = m .+ g
        (a, b, w .- n .* a, n)
    end
    planned = C.plan(C.lower(trace(f, p, q, m, w)))
    @test agree(C.execute(planned, p, q, m, w), C.leaves(f(p, q, m, w)))
end

# A kernel is maps and folds over one domain, each fold over its own axes: the
# loss and a weight gradient join the normalization's kernel, and the values
# only they read are never stored.
@testset "folds over other axes join a kernel" begin
    full_domain_stored(p, g) = [v for v in eachindex(p.graph.values) if p.stored[v] &&
        p.graph.values[v].dims == g.domain && p.graph.values[v].definition isa C.Pointwise]
    results(p) = Set(o.value for o in p.graph.results)

    softmax = C.plan(C.lower(differentiate(trace((x, t) -> sum(softmax_columns(x) .* t), data(8, 6), data(8, 6; scale = 0.7)))))
    @test length(softmax.groups) == 1
    g = only(softmax.groups)
    @test sort([C.fold_axes(softmax.graph, r) for r in g.reductions]) == [(1,), (1,), (1, 2)]
    @test issubset(full_domain_stored(softmax, g), results(softmax))

    norm = C.plan(C.lower(differentiate(trace((x, w) -> sum(abs2, rmsnorm(x, w)), data(8, 6), data(8; scale = 0.1)))))
    # The weight gradient, reshaped from (8, 1) to (8,), is the fold's own
    # buffer: no copy.
    @test length(norm.groups) == 1
    g = only(norm.groups)
    @test Set(C.fold_axes(norm.graph, r) for r in g.reductions) == Set([(1,), (2,), (1, 2)])
    @test issubset(full_domain_stored(norm, g), results(norm))
    # Within the group nothing reads a fold over other than the primary axes.
    for m in C.members(g), s in C.stats(g.terms[m])
        @test C.primary(norm.graph, g, s)
    end
end

# A scatter by index arrays is a sum by key, which joins the kernel computing
# its elements as a fold over other than the primary axes does: an embedding's
# gradient is the loss's one kernel, and the elements of a scatter that only
# it reads are not stored, a `tanh` of the rows gathered, or a positional
# embedding and layer norm's gradient at the gathered rows.
@testset "a scatter is a sum by key in its elements' kernel" begin
    tokens = Int32.(mod1.(7 .* (1:240), 1000))
    for (f, args) in (((E, t, g) -> sum(E[:, t] .* g), (data(64, 109), Int32.(mod1.(37 .* (1:1024) .^ 2, 109)), data(64, 1024))),
            ((E, t) -> sum(abs2, tanh.(E[:, t])), (data(64, 1000), tokens)),
            ((E, t, P, w) -> sum(abs2, layernorm(E[:, t] .+ P, w)), (data(64, 1000), reshape(tokens, 80, 3), data(64, 80), data(64))))
        p = C.plan(C.rewrite(C.lower(differentiate(trace(f, args...)))))
        @test length(p.groups) == 1
        g = only(p.groups)
        scatter = only(r for r in g.reductions if p.graph.values[r].definition isa C.Keyed)
        @test !C.primary(p.graph, g, scatter)
        elements = prod(p.graph.values[scatter].definition.iteration)
        results = Set(o.value for o in p.graph.results)
        @test !any(v -> p.stored[v] && length(p.graph.values[v].dims) > 1 && prod(p.graph.values[v].dims) == elements &&
            !(p.graph.values[v].definition isa C.Input) && !(v in results), eachindex(p.graph.values))
        @test agree(C.execute(p, args...), filter(!isnothing, C.leaves(interpret(differentiate(trace(f, args...)), args...))))
    end
end

# Programs that differ by axes of extent one get the same plan: reshapes that
# insert them around pointwise work, reductions, products and their gradients,
# and attention, whose grouped heads have such an axis with one query head per
# key head, against attention written out without it.
@testset "axes of extent one" begin
    explained(p) = [sprint(explain, C.plan(C.lower(p))), sprint(explain, recomputing(C.lower(p)))]
    same(f, g, args...) = explained(trace(f, args...)) == explained(trace(g, args...))
    same_gradient(f, g, args...) = explained(differentiate(trace(f, args...))) == explained(differentiate(trace(g, args...)))
    x, w, c = data(8, 6), data(8; scale = 0.1), data(5)
    a, b = data(5, 7; scale = 0.3), data(7, 6)
    pointwise(x) = tanh.(x) .* 2
    @test same(pointwise, x -> reshape(pointwise(reshape(x, 1, 8, 1, 6)), 8, 6), x)
    @test same(x -> sum(x; dims = 1), x -> dropdims(sum(x; dims = 1); dims = 1), x)
    @test same(x -> T.logsumexp(x; dims = 2), x -> reshape(T.logsumexp(reshape(x, 8, 1, 6); dims = 3), 8, 1), x)
    unit_norm(x, w) = reshape(rmsnorm(reshape(x, 8, 1, 6), reshape(w, 8, 1)), 8, 6)
    @test same(rmsnorm, unit_norm, x, w)
    @test same_gradient((x, w) -> sum(abs2, rmsnorm(x, w)), (x, w) -> sum(abs2, unit_norm(x, w)), x, w)
    unit_product(a, b) = reshape(C.contract(reshape(a, 5, 7, 1), reshape(b, 7, 6, 1), (1, 2, 4), (2, 3, 4), (1, 3, 4)), 5, 6)
    @test same((a, b, c) -> tanh.(a * b .+ c), (a, b, c) -> tanh.(unit_product(a, b) .+ reshape(c, 5, 1, 1)), a, b, c)
    @test same_gradient((a, b) -> sum(abs2, a * b), (a, b) -> sum(abs2, unit_product(a, b)), a, b)
    q, k, v, r = data(8, 3, 16, 2; scale = 0.3), data(8, 3, 16, 2; scale = 0.2), data(8, 3, 16, 2; scale = 0.1), data(8, 3, 16, 2)
    scale = Float32(1 / sqrt(8))
    scores(q, k) = C.contract(k, q, (1, 2, 3, 4), (1, 2, 5, 4), (3, 2, 5, 4)) .* scale
    causal(s) = ifelse.(reshape(1:size(s, 1), :, 1, 1) .<= reshape(1:size(s, 3), 1, 1, :), s, -Inf)
    written(q, k, v) = (s = causal(scores(q, k)); C.contract(v, exp.(s .- T.logsumexp(s; dims = 1)), (1, 2, 3, 4), (3, 2, 5, 4), (1, 2, 5, 4)))
    @test same((q, k, v, r) -> Ops.attention(q, k, v; causal = true) .+ r, (q, k, v, r) -> written(q, k, v) .+ r, q, k, v, r)
end

# A value masked by a formula over its positions carries the formula and the
# number it is where the mask fails, through the maps of it: the planner
# reads it, so that a kernel sweeping the value skips the tiles the mask
# rules out.
@testset "masked values carry their formulas" begin
    s = single(data(6, 6))
    causal(s) = ifelse.(reshape(1:size(s, 1), :, 1) .<= reshape(1:size(s, 2), 1, :), s, -Inf32)
    g = C.lower(trace(s -> exp.(causal(s) .- T.logsumexp(causal(s); dims = 1)), s))
    found = filter(!isnothing, [C.masked(g, v) for v in eachindex(g.values)])
    # The masked scores are minus infinity where the mask fails, and their
    # probabilities zero.
    @test any(((m, x),) -> x == -Inf32, found)
    @test any(((m, x),) -> x == 0.0f0, found)
    # Keys at most the query: keys − queries ≤ 0, over the value's axes.
    @test all(((m, x),) -> m == [IR.add(IR.var(1), IR.scale(IR.var(2), -1), IR.Domain((6, 6)))], found)
    # A fold of a masked value is no masked value, nor is a product of one
    # and something unmasked, which the mask makes infinite, not a number.
    @test all(v -> C.masked(g, v) === nothing, [v for v in eachindex(g.values) if g.values[v].definition isa C.Reduction])
    h = C.lower(trace(s -> causal(s) .* s, s))
    @test C.masked(h, length(h.values)) === nothing
    # A sum of one and something unmasked is minus infinity there.
    h = C.lower(trace(s -> causal(s) .+ s, s))
    @test C.masked(h, length(h.values)) !== nothing && C.masked(h, length(h.values))[2] == -Inf32
    # A position compared with an integer the program reads, a length per
    # column as decode holds on the device, masks too: the formula reads the
    # length, keys − length ≤ 0.
    upto(s, n) = ifelse.(reshape(Int32(1):Int32(size(s, 1)), :, 1) .<= reshape(n, 1, :), s, -Inf32)
    n = Int32[2, 5, 6, 1, 3, 4]
    h = C.lower(trace((s, n) -> exp.(upto(s, n) .- T.logsumexp(upto(s, n); dims = 1)), s, n))
    found = filter(!isnothing, [C.masked(h, v) for v in eachindex(h.values)])
    d = IR.Domain((6, 6))
    length_read = IR.held_formula(h.arguments[2], (IR.var(2),), Int32, d)
    @test !isempty(found) && all(((m, x),) -> m == [IR.add(IR.add(IR.var(1), 1, d), IR.scale(length_read, -1), d)], found)
    @test any(((m, x),) -> x == 0.0f0, found)
    # A length of 64 bits is no formula: its range would overflow one's.
    h = C.lower(trace((s, n) -> upto(s, n), s, Int64.(n)))
    @test C.masked(h, length(h.values)) === nothing
end
