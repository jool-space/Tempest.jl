# Plans compute what programs compute: every program and its differentiated
# form, planned and run kernel by kernel on the CPU, agrees with the program
# interpreter.
include("fixtures/programs.jl")

@testset "plans" begin
    @testset "$name" for (name, f, args) in PROGRAMS
        for p in (trace(f, args...), differentiate(trace(f, args...)))
            planned = C.plan(C.lower(p))
            @test agree(C.execute(planned, args...), filter(!isnothing, C.leaves(interpret(p, args...))))
        end
    end
end

# The plan that recomputes the folds `recomputed_folds` selects.
recomputing(graph) = C.plan(graph; recompute = C.recomputed_folds(graph))

# Folds recomputed where they are read, as plans for cuTile recompute them:
# the same results, and attention's scores and logsumexp inside the one
# kernel of the sum over keys.
@testset "plans recomputing folds" begin
    @testset "$name" for (name, f, args) in PROGRAMS
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
        e = exp.(T.contract(k, q, (1, 2), (1, 3), (2, 3)))
        d = T.contract(k, e, (1, 2), (2, 3), (1, 3))
        T.contract(x, d, (1, 3), (2, 3), (1, 2))
    end
    p = trace(f, q, k, x)
    planned = recomputing(C.lower(p))
    @test all(g -> C.recomputed_in(g) == C.recomputed_in(g; nested = false), planned.groups)
    @test count(g -> !isempty(C.recomputed_in(g)), planned.groups) == 1
    @test agree(C.execute(planned, q, k, x), [interpret(p, q, k, x)])
end

# Rewrites of sums, which change rounding: rewritten programs plan and compute
# what they computed, and the two spellings of a softmax weighting a sum plan
# alike, one shifted sum recomputing the scores in the kernel of the sum.
scores(q, k) = T.contract(k, q, (1, 2, 3, 4), (1, 2, 5, 4), (3, 2, 5, 4)) .* 0.125
causal(s) = ifelse.(reshape(1:size(s, 1), :, 1, 1) .<= reshape(1:size(s, 3), 1, 1, :), s, -Inf)
weigh(v, p) = T.contract(v, p, (1, 2, 3, 4), (3, 2, 5, 4), (1, 2, 5, 4))
by_logsumexp(q, k, v) = (s = causal(scores(q, k)); weigh(v, exp.(s .- T.logsumexp(s; dims = 1))))
function by_maximum(q, k, v)
    s = causal(scores(q, k))
    e = exp.(s .- maximum(s; dims = 1))
    weigh(v, e ./ sum(e; dims = 1))
end

@testset "rewritten sums" begin
    @testset "$name" for (name, f, args) in PROGRAMS
        for p in (trace(f, args...), differentiate(trace(f, args...)))
            planned = recomputing(C.rewrite(C.lower(p)))
            @test agree(C.execute(planned, args...), filter(!isnothing, C.leaves(interpret(p, args...))))
        end
    end
    groups(f, args...) = recomputing(C.rewrite(C.lower(trace(f, args...)))).groups
    q, k, v = data(4, 2, 6, 2; scale = 0.3), data(4, 2, 6, 2; scale = 0.2), data(4, 2, 6, 2; scale = 0.1)
    a, b = only(groups(by_logsumexp, q, k, v)), only(groups(by_maximum, q, k, v))
    @test a.terms == b.terms
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
        s = T.contract(k, q, (1, 2, 3, 4), (1, 2, 5, 4), (3, 2, 5, 4)) .* BFloat16(0.125)
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
    unit_product(a, b) = reshape(T.batched_mul(reshape(a, 5, 7, 1), reshape(b, 7, 6, 1)), 5, 6)
    @test same((a, b, c) -> tanh.(a * b .+ c), (a, b, c) -> tanh.(unit_product(a, b) .+ reshape(c, 5, 1, 1)), a, b, c)
    @test same_gradient((a, b) -> sum(abs2, a * b), (a, b) -> sum(abs2, unit_product(a, b)), a, b)
    q, k, v, r = data(8, 3, 16, 2; scale = 0.3), data(8, 3, 16, 2; scale = 0.2), data(8, 3, 16, 2; scale = 0.1), data(8, 3, 16, 2)
    scale = Float32(1 / sqrt(8))
    scores(q, k) = T.contract(k, q, (1, 2, 3, 4), (1, 2, 5, 4), (3, 2, 5, 4)) .* scale
    causal(s) = ifelse.(reshape(1:size(s, 1), :, 1, 1) .<= reshape(1:size(s, 3), 1, 1, :), s, -Inf)
    written(q, k, v) = (s = causal(scores(q, k)); T.contract(v, exp.(s .- T.logsumexp(s; dims = 1)), (1, 2, 3, 4), (3, 2, 5, 4), (1, 2, 5, 4)))
    @test same((q, k, v, r) -> Ops.attention(q, k, v; causal = true) .+ r, (q, k, v, r) -> written(q, k, v) .+ r, q, k, v, r)
end
