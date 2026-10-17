# Plans compute what programs compute: every program and its differentiated
# form, planned and run kernel by kernel on the CPU, agrees with the program
# interpreter.
using Test
include("cases.jl")

agree(a, b) = all(map((x, y) -> isapprox(x, y; rtol = 1.0e-10, atol = 1.0e-12), a, b))

@testset "plans" begin
    @testset "$name" for (name, f, args) in PROGRAMS
        for p in (trace(f, args...), differentiate(trace(f, args...)))
            planned = T.plan(T.lower(p))
            @test agree(T.execute(planned, args...), filter(!isnothing, T.leaves(interpret(p, args...))))
        end
    end
end

# Folds recomputed where they are read, as plans for cuTile recompute them:
# the same results, and attention's scores and logsumexp inside the one
# kernel of the sum over keys.
@testset "plans recomputing folds" begin
    @testset "$name" for (name, f, args) in PROGRAMS
        for p in (trace(f, args...), differentiate(trace(f, args...)))
            planned = T.plan(T.lower(p); recompute_folds = true)
            @test agree(T.execute(planned, args...), filter(!isnothing, T.leaves(interpret(p, args...))))
        end
    end
    q, k, v = data(4, 4, 6, 2; scale = 0.3), data(4, 2, 6, 2; scale = 0.2), data(4, 2, 6, 2; scale = 0.1)
    p = trace((q, k, v) -> T.attention(q, k, v; causal = true), q, k, v)
    planned = T.plan(T.lower(p); recompute_folds = true)
    @test length(planned.groups) == 1
    @test length(T.recomputed_in(only(planned.groups))) == 2
    @test agree(T.execute(planned, q, k, v), [interpret(p, q, k, v)])
end

# Rewrites of sums, which change rounding: rewritten programs plan and compute
# what they computed, and the two spellings of a softmax weighting a sum plan
# alike, one online softmax recomputing the scores in the kernel of the sum.
scores(q, k) = T.contract(k, q, (1, 2, 3, 4), (1, 2, 5, 4), (3, 2, 5, 4)) .* 0.125
causal(s) = ifelse.(T.axis_positions(s, 1) .<= T.axis_positions(s, 3), s, -Inf)
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
            planned = T.plan(T.rewrite(T.lower(p)); recompute_folds = true)
            @test agree(T.execute(planned, args...), filter(!isnothing, T.leaves(interpret(p, args...))))
        end
    end
    groups(f, args...) = T.plan(T.rewrite(T.lower(trace(f, args...))); recompute_folds = true).groups
    q, k, v = data(4, 2, 6, 2; scale = 0.3), data(4, 2, 6, 2; scale = 0.2), data(4, 2, 6, 2; scale = 0.1)
    a, b = only(groups(by_logsumexp, q, k, v)), only(groups(by_maximum, q, k, v))
    @test a.terms == b.terms
    graph = T.rewrite(T.lower(trace(by_maximum, q, k, v)))
    @test graph.values[only(b.reductions)].definition.fold == T.OnlineSoftmax(/)
    @test length(T.recomputed_in(b)) == 1
    planned = T.plan(graph; recompute_folds = true)
    @test agree(T.execute(planned, q, k, v), [interpret(trace(by_maximum, q, k, v), q, k, v)])
    @test occursin("online softmax", sprint(explain, planned))
end

# A map read by a fold and by a later map over its domain is recomputed in
# their kernel, as when it is written twice.
@testset "a map one kernel reads" begin
    x = data(8, 6)
    for f in (x -> (e = exp.(x); e ./ sum(e; dims = 1)), x -> exp.(x) ./ sum(exp.(x); dims = 1))
        p = trace(f, x)
        planned = T.plan(T.lower(p))
        @test length(planned.groups) == 1
        @test agree(T.execute(planned, x), [interpret(p, x)])
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
        planned = T.plan(T.lower(p))
        @test length(planned.groups) == 1
        @test T.structure(planned).stored == [o.value for o in planned.graph.results]
        @test agree(T.execute(planned, args...), [interpret(p, args...)])
    end
end

# A kernel is maps and folds over one domain, each fold over its own axes: the
# loss and a weight gradient join the normalization's kernel, and the values
# only they read are never stored.
@testset "folds over other axes join a kernel" begin
    full_domain_stored(p, g) = [v for v in eachindex(p.graph.values) if p.stored[v] &&
        p.graph.values[v].dims == g.domain && p.graph.values[v].definition isa T.Pointwise]
    results(p) = Set(o.value for o in p.graph.results)

    softmax = T.plan(T.lower(differentiate(trace((x, t) -> sum(softmax_columns(x) .* t), data(8, 6), data(8, 6; scale = 0.7)))))
    @test length(softmax.groups) == 1
    g = only(softmax.groups)
    @test sort([T.fold_axes(softmax.graph, r) for r in g.reductions]) == [(1,), (1,), (1, 2)]
    @test issubset(full_domain_stored(softmax, g), results(softmax))

    norm = T.plan(T.lower(differentiate(trace((x, w) -> sum(abs2, rmsnorm(x, w)), data(8, 6), data(8; scale = 0.1)))))
    # The weight gradient, reshaped from (8, 1) to (8,), is the fold's own
    # buffer: no copy.
    @test length(norm.groups) == 1
    g = only(norm.groups)
    @test Set(T.fold_axes(norm.graph, r) for r in g.reductions) == Set([(1,), (2,), (1, 2)])
    @test issubset(full_domain_stored(norm, g), results(norm))
    # Within the group nothing reads a fold over other than the primary axes.
    for m in T.members(g), s in T.stats(g.terms[m])
        @test T.primary(norm.graph, g, s)
    end
end

# Programs that differ by axes of extent one get the same plan: reshapes that
# insert them around pointwise work, reductions, products and their gradients,
# and attention, whose grouped heads have such an axis with one query head per
# key head, against attention written out without it.
@testset "axes of extent one" begin
    explained(p) = [sprint(explain, T.plan(T.lower(p); recompute_folds)) for recompute_folds in (false, true)]
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
    causal(s) = ifelse.(T.axis_positions(s, 1) .<= T.axis_positions(s, 3), s, -Inf)
    written(q, k, v) = (s = causal(scores(q, k)); T.contract(v, exp.(s .- T.logsumexp(s; dims = 1)), (1, 2, 3, 4), (3, 2, 5, 4), (1, 2, 5, 4)))
    @test same((q, k, v, r) -> T.attention(q, k, v; causal = true) .+ r, (q, k, v, r) -> written(q, k, v) .+ r, q, k, v, r)
end
