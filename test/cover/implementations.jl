# Implementations propose and claim, and the cover takes by cost: candidates
# of an implementation's own in place of the generator's groups, recomputing
# a fold where it is read, priced where an implementation claims every kernel
# that recomputes it, and storing a map apart from a kernel that applies a
# function an implementation cannot lower, or whose claim failed to build.

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
    ex = prepare(p, a, b; target = T.CPUTarget(HostProducts()))
    @test length(only(taken(ex))) == 1
    @test length(ex.plan.groups) == 2
    @test isapprox(only(C.leaves(outputs(execute!(ex)))), interpret(p, a, b); rtol = 1.0e-12)
    @test occursin("from HostProducts replaces", sprint(explain, ex))
    # The product, the tanh and a sum over the result span two domains.
    graph = C.lower(trace((a, b) -> sum(tanh.(a * b); dims = 2), a, b))
    planned = C.plan(graph)
    r = only(g.reductions[1] for g in planned.groups if C.product_axes(C.group_view(planned, g)) !== nothing)
    @test_throws C.NotAGroup C.plan(graph; fixed = [C.grow(graph, r, set -> length(set) <= 3)])
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
        d = C.Decider(T.CPUTarget(RecomputesOverThree{cheap}()), graph; measure = false)
        p, considered = C.cover(d)
        proposed = filter(c -> c.proposal isa C.Recompute, considered)
        @test map(c -> c.proposal.value, proposed) == [q]
        @test only(proposed).accepted == cheap
        @test (q in reduce(union, map(C.recomputed_in, p.groups); init = Set{Int}())) == cheap
        @test all(g -> !isempty(C.claims(RecomputesOverThree{cheap}(), d.target, C.group_view(p, g))), p.groups)
    end
end

# A claim that fails to build is no option once it has, and the program is
# planned again: each map of the group it claimed is proposed to be stored
# apart. This implementation claims products and maps alike as free, and
# fails to build a group that applies `log10`, as cuTile does: the map is
# stored apart, run on the floor, and the product claimed. `explain` names
# the claim that failed.
struct NoLog10 <: C.Implementation end
applies_log10(g) = any(t -> t isa C.Apply && t.f === log10, C.subterms([g.terms[m] for m in C.members(g)]))
C.claims(::NoLog10, ::T.CPUTarget, g) = g.kind === :pointwise || lone_product(g) ? [:host] : []
C.estimate(::NoLog10, claim, target, g) = [C.Work(0.0)]
C.build_claim(::NoLog10, claim, target, g, buffers) = applies_log10(g) ? error("log10 has no lowering") : g.reference(target, buffers)

@testset "a map stored apart from a kernel whose claim failed to build" begin
    a, b = data(6, 5; scale = 0.3), abs.(data(5, 4)) .+ 1
    p = trace((a, b) -> a * log10.(b .^ 2), a, b)
    ex = @test_logs (:warn, r"failed to build") match_mode = :any prepare(p, a, b; target = T.CPUTarget(NoLog10()))
    @test isapprox(only(C.leaves(outputs(execute!(ex)))), interpret(p, a, b); rtol = 1.0e-12)
    stored = only(c.proposal for c in ex.considered if c.accepted)
    @test stored isa C.Store && ex.plan.stored[stored.value]
    floor, product = ex.decisions
    @test floor.choice.implementation === nothing && !isempty(floor.failed)
    @test product.choice.implementation isa NoLog10 && isempty(product.failed)
    text = sprint(explain, ex)
    @test occursin("is stored apart", text) && occursin("a claim of NoLog10 failed to build: log10 has no lowering", text)
end

# The same implementation, declaring what it cannot lower (`lowers`) rather
# than failing to build it: it claims no group that applies `log10`, and the
# map is stored apart without a build failing first.
struct DeclinesLog10 <: C.Implementation end
C.claims(::DeclinesLog10, ::T.CPUTarget, g) = (g.kind === :pointwise || lone_product(g)) && !applies_log10(g) ? [:host] : []
C.lowers(::DeclinesLog10, ::T.CPUTarget, t, g) = t.f !== log10
C.estimate(::DeclinesLog10, claim, target, g) = [C.Work(0.0)]
C.build_claim(::DeclinesLog10, claim, target, g, buffers) = g.reference(target, buffers)

@testset "a map stored apart from a kernel that applies a function an implementation cannot lower" begin
    a, b = data(6, 5; scale = 0.3), abs.(data(5, 4)) .+ 1
    p = trace((a, b) -> a * log10.(b .^ 2), a, b)
    ex = @test_nowarn prepare(p, a, b; target = T.CPUTarget(DeclinesLog10()))
    @test isapprox(only(C.leaves(outputs(execute!(ex)))), interpret(p, a, b); rtol = 1.0e-12)
    stored = only(c.proposal for c in ex.considered if c.accepted)
    @test stored isa C.Store && ex.plan.stored[stored.value]
    floor, product = ex.decisions
    @test floor.choice.implementation === nothing && isempty(floor.failed)
    @test product.choice.implementation isa DeclinesLog10 && isempty(product.failed)
    @test occursin("is stored apart", sprint(explain, ex))
end

# A variant that fails to build is no option either, though the claim it
# varies, which builds, offers it again whenever the options are searched.
# The record times the variant faster, so it is chosen and fails first.
struct FailingVariant <: C.Implementation end
C.claims(::FailingVariant, ::T.CPUTarget, g) = g.kind === :pointwise ? [:base] : []
C.variants(::FailingVariant, ::T.CPUTarget, g, claim) = claim === :base ? [:variant] : []
C.estimate(::FailingVariant, claim, target, g) = [C.Work(0.0)]
C.build_claim(::FailingVariant, claim, target, g, buffers) = claim === :variant ? error("the variant has no lowering") : g.reference(target, buffers)

@testset "a variant that fails to build" begin
    x = data(8, 8)
    p = trace(x -> exp.(x) .+ 1, x)
    withenv("TEMPEST_CACHE" => mktempdir()) do
        target = T.CPUTarget(FailingVariant())
        d = C.Decider(target, C.rewrite(C.lower(p)); measure = :none)
        plan = C.generator_plan(d)
        g = only(plan.groups)
        for (claim, seconds) in ((:base, 2.0e-6), (:variant, 1.0e-6))
            C.save_record(C.record_path(target), C.choice_key(d, plan, g, FailingVariant(), claim), seconds)
        end
        ex = @test_logs (:warn, r"failed to build") match_mode = :any prepare(p, x; target)
        decision = only(ex.decisions)
        @test decision.choice.claim === :base && !isempty(decision.failed)
        @test isapprox(only(C.leaves(outputs(execute!(ex)))), interpret(p, x); rtol = 1.0e-12)
    end
end
