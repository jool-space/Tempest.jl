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
    g = only(filter(g -> g.kind === :reduction, norm.groups))
    @test Set(T.fold_axes(norm.graph, r) for r in g.reductions) == Set([(1,), (2,), (1, 2)])
    @test issubset(full_domain_stored(norm, g), results(norm))
    # Within the group nothing reads a fold over other than the primary axes.
    for m in T.members(g), s in T.stats(g.terms[m])
        @test T.primary(norm.graph, g, s)
    end
end
