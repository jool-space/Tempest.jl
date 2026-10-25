# What joins a kernel: maps fuse into the scans and folds that read them,
# writes into an allocation plan with their readers, folds over other axes and
# a scatter's sum by key join the kernel computing their elements, and no
# kernel joins a fold its reads wait on.
include("../fixtures/programs.jl")

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
