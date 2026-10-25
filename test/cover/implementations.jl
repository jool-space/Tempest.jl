# Implementations propose and claim, and the cover takes by cost: candidates
# of an implementation's own in place of the generator's groups, and
# recomputing a fold where it is read, priced where an implementation claims
# every kernel that recomputes it.

# An implementation that proposes each product alone, grown from its fold, in
# place of the generator's product with the work after it; it claims a lone
# product as free, so the cover takes it. A candidate over several domains
# (the product and a sum over its result) does not form a group.
struct HostProducts <: C.Implementation end
lone_product(g) = g.kind === :reduction && all(in(g.reductions), C.members(g)) && C.product_axes(g) !== nothing
C.candidates(::HostProducts, ::T.CPUTarget, p) = [C.grow(p.graph, r, set -> length(set) <= 1)
    for g in map(g -> C.group_view(p, g), p.groups) if g.kind === :reduction && C.product_axes(g) !== nothing for r in g.reductions]
C.claims(::HostProducts, ::T.CPUTarget, g) = lone_product(g) ? [:host] : []
C.estimate(::HostProducts, claim, target, g) = [C.Work(0.0)]
C.build_claim(::HostProducts, claim, target, g, buffers) = g.reference(target, buffers)

@testset "candidates from implementations" begin
    a, b = data(6, 5; scale = 0.3), data(5, 4)
    p = trace((a, b) -> tanh.(a * b), a, b)
    @test length(C.plan(C.lower(p)).groups) == 1 # the generator fuses the tanh
    C.register!(@__MODULE__, HostProducts())
    try
        ex = prepare(p, a, b)
        @test length(only(taken(ex))) == 1
        @test length(ex.plan.groups) == 2
        @test isapprox(only(C.leaves(outputs(execute!(ex)))), interpret(p, a, b); rtol = 1.0e-12)
        @test occursin("from HostProducts replaces", sprint(explain, ex))
        # The product, the tanh and a sum over the result span two domains.
        graph = C.lower(trace((a, b) -> sum(tanh.(a * b); dims = 2), a, b))
        planned = C.plan(graph)
        r = only(g.reductions[1] for g in planned.groups if C.product_axes(C.group_view(planned, g)) !== nothing)
        @test_throws C.NotAGroup C.plan(graph; fixed = [C.grow(graph, r, set -> length(set) <= 3)])
    finally
        C.register!(@__MODULE__)
    end
end

# Recomputing a fold where it is read is a proposal: priced only where an
# implementation that recomputes folds claims every kernel that recomputes
# it, and taken by cost. These implementations claim groups over three axes,
# or that recompute nothing; one finds recomputing slow, the other cheap.
# Recomputing `q` in the product of `s`, over three axes, is proposed and
# priced; recomputing `s` in the logsumexp, over two axes, is never priced,
# since nothing claims that kernel, and `s` reads `q` inside its sweep.
struct RecomputesOverThree{Cheap} <: C.Implementation end
C.recomputes_folds(::RecomputesOverThree, target) = true
C.claims(::RecomputesOverThree, target, g) = isempty(C.recomputed_in(g)) || length(g.domain) >= 3 ? (:claim,) : ()
C.estimate(::RecomputesOverThree{Cheap}, claim, target, g) where {Cheap} =
    [C.Work(0.0; flops = isempty(C.recomputed_in(g)) == Cheap ? 1.0 : 1.0e-6)]

@testset "recomputing a fold is a proposal, priced where its kernels are claimed" begin
    x, w = data(8, 6), data(8, 8)
    function f(x, w)
        q = C.contract(w, x, (1, 2), (1, 3), (2, 3))
        s = C.contract(q, q, (1, 2), (1, 3), (2, 3))
        T.logsumexp(s; dims = 1)
    end
    graph = C.lower(trace(f, x, w))
    q = only(C.recomputable_folds(graph))
    @test graph.values[q].definition isa C.Reduction && graph.values[q].dims == (8, 6)
    for cheap in (false, true)
        C.register!(@__MODULE__, RecomputesOverThree{cheap}())
        try
            d = C.Decider(T.CPUTarget(), graph; measure = false)
            p, considered = C.cover(d)
            proposed = filter(c -> c.proposal isa C.Recompute, considered)
            @test map(c -> c.proposal.value, proposed) == [q]
            @test only(proposed).accepted == cheap
            @test (q in reduce(union, map(C.recomputed_in, p.groups); init = Set{Int}())) == cheap
            @test all(g -> !isempty(C.claims(RecomputesOverThree{cheap}(), d.target, C.group_view(p, g))), p.groups)
        finally
            C.register!(@__MODULE__)
        end
    end
end
