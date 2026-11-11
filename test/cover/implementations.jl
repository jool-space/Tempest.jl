# Implementations propose and claim, and the cover takes by cost: candidates
# of an implementation's own in place of the generator's groups, recomputing
# a fold where it is read, priced where an implementation claims every kernel
# that recomputes it, recomputing maps read shifted in the kernel that reads
# them, and storing a map apart from a kernel that applies a function an
# implementation cannot lower, or whose claim failed to build.

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
C.build_claim(::HostProducts, claim, target, g, buffers, _) = g.reference(target, buffers)

@testset "candidates from implementations" begin
    a, b = data(6, 5; scale = 0.3), data(5, 4)
    p = trace((a, b) -> tanh.(a * b), a, b)
    @test length(C.plan(C.lower(p)).groups) == 1 # the generator fuses the tanh
    ex = prepare(p, a, b; target = T.CPUTarget(HostProducts()))
    @test length(C.fixes(only(taken(ex)))) == 1
    @test length(ex.plan.groups) == 2
    @test isapprox(only(C.leaves(outputs(execute!(ex)))), interpret(p, a, b); rtol = 1.0e-12)
    @test occursin("from HostProducts replaces", sprint(explain, ex))
    # The product, the tanh and a sum over the result span two domains.
    graph = C.lower(trace((a, b) -> sum(tanh.(a * b); dims = 2), a, b))
    planned = C.plan(graph)
    r = only(g.reductions[1] for g in planned.groups if C.product_axes(C.group_view(planned, g)) !== nothing)
    @test_throws C.NotAGroup C.plan(graph; fixed = [C.grow(graph, r, set -> length(set) <= 3)])
end

# What an executable's launches share is made once per executable, as the
# first claim that needs it is built, and given to every claim of its
# implementation the executable builds; another executable has its own, and
# one whose claims it builds none of has none made. It goes with the
# executable.
struct SharingProducts <: C.Implementation end
mutable struct Shared
    made::Int
end
struct SharingLaunch
    launch::Any
    shared::Shared
end
(l::SharingLaunch)() = l.launch()
const SHARED = Ref(0)
C.candidates(::SharingProducts, target::T.CPUTarget, p) = C.candidates(HostProducts(), target, p)
C.claims(::SharingProducts, target::T.CPUTarget, g) = C.claims(HostProducts(), target, g)
C.estimate(::SharingProducts, claim, target, g) = [C.Work(0.0)]
C.resources(::SharingProducts, ::T.CPUTarget) = Shared(SHARED[] += 1)
C.build_claim(::SharingProducts, claim, target, g, buffers, shared) =
    ((launch, description) -> (SharingLaunch(launch, shared), description))(g.reference(target, buffers)...)

@testset "what an executable's launches share" begin
    a, b, c = data(6, 5; scale = 0.3), data(5, 4), data(4, 3)
    p = trace((a, b, c) -> tanh.((a * b) * c), a, b, c)
    target = T.CPUTarget(SharingProducts())
    shared(ex) = unique(l.shared for l in ex.launches if l isa SharingLaunch)
    exs = [prepare(p, a, b, c; target) for _ in 1:2]
    @test count(l -> l isa SharingLaunch, exs[1].launches) == 2 && length(shared(exs[1])) == 1
    @test only(shared(exs[1])) !== only(shared(exs[2]))
    @test isapprox(only(C.leaves(outputs(execute!(exs[1])))), interpret(p, a, b, c); rtol = 1.0e-12)
    made = SHARED[]
    prepare(trace(a -> tanh.(a), a), a; target)
    @test SHARED[] == made
    # Prepared in a function of its own: code at the top level keeps what it
    # computed alive until it ends.
    prepared_and_let_go() = WeakRef(only(shared(prepare(p, a, b, c; target))))
    gone = prepared_and_let_go()
    foreach(_ -> GC.gc(true), 1:3)
    @test gone.value === nothing
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

# A map read at several positions is stored, and the kernel that reads it
# shifted, as a stencil reads its neighbours, reads it from memory: the cover
# proposes recomputing the maps of the kernel that stores it there, and one
# kernel is cheaper than two.
@testset "a kernel of maps read shifted is recomputed in the kernel that reads them" begin
    u = data(8, 8; scale = 0.1)
    neighbours(w) = vcat(w[1:1, :], w[1:(end - 1), :]) .+ vcat(w[2:end, :], w[end:end, :])
    p = trace(u -> (w = exp.(u) .* 2; w .+ neighbours(w)), u)
    d = C.Decider(T.CPUTarget(), C.lower(p); measure = :none)
    base = C.generator_plan(d)
    @test length(base.groups) == 2
    proposed = only(C.shifted_maps(base))
    @test proposed.values == [v for v in eachindex(base.stored) if base.stored[v] && base.owner[v] == base.owner[first(proposed.values)]]
    ex = prepare(p, u)
    @test length(ex.kernels) == 1
    @test any(c -> c.proposal isa C.RecomputeMaps && c.accepted, ex.considered)
    @test isapprox(only(C.leaves(outputs(execute!(ex)))), interpret(p, u); rtol = 1.0e-12)
end

@testset "the floor reads a value loaded at several positions once" begin
    u = data(8, 8)
    bytes(f) = (p = C.plan(C.lower(trace(f, u))); only(C.floor_estimate(p, only(p.groups))).bytes)
    @test bytes(u -> u .+ vcat(u[2:end, :], u[end:end, :])) == bytes(u -> 2 .* u)
end

# Folds recomputed one inside another multiply their extents in the floor's
# bytes: six products of 1024 columns inside a recurrence's step are past
# what an Int holds, and the floor is priced by them all the same.
@testset "a recurrence's floor counts folds recomputed one inside another" begin
    function chained(W, x, h)
        for t in axes(x, 2)
            h = tanh.(h .+ W * (W * (W * (W * (W * (W * x[:, t]))))))
        end
        h
    end
    function bytes(n)
        g = C.lower(trace(chained, fill(0.001, n, n), fill(0.1, n, 3), fill(0.0, n)))
        p = C.plan(g; recompute = [v for v in eachindex(g.values) if g.values[v].definition isa C.Reduction])
        only(C.floor_estimate(p, only(k for k in p.groups if k.kind isa C.RecurrenceKind))).bytes
    end
    @test bytes(1024) > 1.0e19
    @test isapprox(bytes(1024) / bytes(64), 16.0^7; rtol = 0.05)
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
C.build_claim(::NoLog10, claim, target, g, buffers, _) = applies_log10(g) ? error("log10 has no lowering") : g.reference(target, buffers)

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
C.build_claim(::DeclinesLog10, claim, target, g, buffers, _) = g.reference(target, buffers)

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

# Each map of such a kernel is proposed to be stored apart (the square, the
# sum, the log10 and the product's terms), the one applying the function
# first by its gain; the others, priced in the starting plan against the
# kernel computing everything, are priced again in the plan that storing
# changed. There the maps the log10 map reads (numbered before it) are its
# own kernel's, which storing one apart leaves as it is: the comparison has
# no groups on either side, where the starting plan's had the product
# computing everything against the product reading the map.
@testset "a map stored apart is priced in the plan a storing taken changed" begin
    a, b = data(6, 5; scale = 0.3), data(5, 4)
    p = trace((a, b) -> a * log10.(1 .+ b .^ 2), a, b)
    ex = prepare(p, a, b; target = T.CPUTarget(DeclinesLog10()))
    @test isapprox(only(C.leaves(outputs(execute!(ex)))), interpret(p, a, b); rtol = 1.0e-12)
    stores = filter(c -> c.proposal isa C.Store, ex.considered)
    accepted = only(filter(c -> c.accepted, stores))
    @test ex.plan.graph.values[accepted.proposal.value].definition.f === log10
    @test length(stores) == 4
    for c in filter(c -> !c.accepted, stores)
        @test all(((plan, _, _),) -> plan.stored[accepted.proposal.value], c.comparison.sides[1])
        c.proposal.value < accepted.proposal.value || continue
        @test isempty(c.comparison.sides[1]) && isempty(c.comparison.sides[2])
    end
end

# A variant that fails to build is no option either, though the claim it
# varies, which builds, offers it again whenever the options are searched.
# The record times the variant faster, so it is chosen and fails first.
struct FailingVariant <: C.Implementation end
C.claims(::FailingVariant, ::T.CPUTarget, g) = g.kind === :pointwise ? [:base] : []
C.variants(::FailingVariant, ::T.CPUTarget, g, claim) = claim === :base ? [:variant] : []
C.estimate(::FailingVariant, claim, target, g) = [C.Work(0.0)]
C.build_claim(::FailingVariant, claim, target, g, buffers, _) = claim === :variant ? error("the variant has no lowering") : g.reference(target, buffers)

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

# An implementation's own claims are measured where the estimate cannot rank
# them, within its error of the best; one estimated past the error only once
# the record refutes the best's estimate, a time past its error.
@testset "claims measured by the estimate's error" begin
    graph = C.lower(trace(x -> sum(x; dims = 1), data(4, 3)))
    d = C.Decider(T.CPUTarget(), graph; measure = :dry)
    option(estimate, key) = C.Choice(HostProducts(), :host, estimate, key, false, 1, Any[])
    os = [option(1.0e-5, "best"), option(1.4e-5, "near"), option(4.0e-5, "split")]
    @test C.measured_ranks(d, os) == [1, 2]
    d.record["best"] = 1.3e-5
    @test C.measured_ranks(d, os) == [1, 2]
    d.record["best"] = 2.0e-5
    @test C.measured_ranks(d, os) == [1, 2, 3]
end

# A side of a comparison is timed only until it has lost: the side known to be
# faster whole, the other no further once the least it can take passes that
# time beyond the noise; it is then decided against by what is known of it.
# Measuring nothing, a decider wants what one measuring close calls would time.
@testset "a comparison's side is timed until it has lost" begin
    graph = C.lower(trace(x -> sum(x; dims = 1), data(4, 3)))
    d = C.Decider(T.CPUTarget(), graph; measure = :none)
    side(groups...) = [(nothing, nothing, C.Decision(C.Choice(nothing, nothing, e, k, false, 0, Any[]), C.Choice[], :estimated, nothing, C.Failure[]))
        for (k, e) in groups]
    old = side("old" => 1.0e-5)
    d.record["old"] = 2.0e-5
    # Where the record refuted an estimate, a kernel priced past the other
    # side's time beyond its error is not begun.
    new = side("cheap" => 1.0e-6, "floor" => 1.0)
    C.measure_sides!(d, old, new, false)
    @test isempty(d.wanted)
    c = C.stopped(d, old, new, false)
    @test c.how === :estimated && c.old == 2.0e-5 && !C.cheaper(c)
    # In a close call, where estimates rank nothing, it is.
    C.measure_sides!(d, old, new, true)
    @test d.wanted == Set(["cheap", "floor"])
    @test C.stopped(d, old, new, true) === nothing
    # One priced within its error of it is begun.
    empty!(d.wanted)
    new = side("cheap" => 1.0e-6, "floor" => 2.5e-5)
    C.measure_sides!(d, old, new, false)
    @test d.wanted == Set(["cheap", "floor"])
    @test C.stopped(d, old, new, false) === nothing
    # Timed past it, the dearest first, the side has lost, and the rest is
    # not timed, in a close call too.
    d.record["floor"] = 3.0e-5
    for close in (false, true)
        empty!(d.wanted)
        C.measure_sides!(d, old, new, close)
        @test isempty(d.wanted)
        c = C.stopped(d, old, new, close)
        @test c.how === :measured && !C.cheaper(c)
    end
end
