# The promises in DESIGN.md, as tests over plans. Promise 1 (layout spelling
# does not matter) and Promise 3 (one rule decides fusion) are checked here on
# the plan the cover chooses; the others are checked where their mechanism
# lives (plan.jl, coverage.jl, runtime.jl, cache.jl). Each promise has cases
# that hold and cases that do not yet: a case not met is `@test_broken` under
# a testset that names what closes the gap, as the coverage suite does. A
# closed gap fails here as an unexpected pass, and its case moves up.
using Statistics

# The plan the cover chooses for `f`, with every implementation the target
# has and nothing measured.
chosen(f, args...; target = T.CPUTarget()) = prepare(trace(f, args...), args...; target, measure = false).plan

# Two programs get the same plan when `explain` prints them alike: the same
# kernels, terms, storage and decisions.
same_plan(f, g, args...) = sprint(explain, chosen(f, args...)) == sprint(explain, chosen(g, args...))

# What a kernel's member computes at a point of its domain, with each address
# the point stands for: the same whatever the naming of the domain's axes.
at(t::IR.Load, x, p, g, D) = string("%", t.value, "@", IR.evaluate_index(IR.address(p.graph.values[t.value], t.index, D), x))
at(t::IR.Stat, x, p, g, D) =
    string("⟨%", t.value, "@", IR.evaluate_index(IR.address(p.graph.values[t.value], C.placement(p.graph, g, t.value), D), x), "⟩")
at(t::IR.Lit, x, p, g, D) = repr(t.value)
at(t::IR.Ix, x, p, g, D) = string(IR.evaluate_index(t.expr, x))
at(t::IR.Apply, x, p, g, D) = string(t.f, "(", join((at(a, x, p, g, D) for a in t.args), ", "), ")")

# A kernel's work whatever its domain's spelling: at each point, each member's
# address and what it computes there. Small domains only; no recomputed folds
# or gathers.
function work(p::C.Plan, g)
    D = IR.Domain(g.domain)
    sort!(vec([join((string(m in g.reductions ? "fold " : "store ", at(IR.Stat(m), x, p, g, D), " = ", at(g.terms[m], x, p, g, D))
                     for m in C.members(g) if haskey(g.terms, m)), "; ") for x in Iterators.product(map(n -> 0:(n - 1), g.domain)...)]))
end

"Plans alike up to a bijection of each kernel's domain that keeps every address."
function same_work(f, h, args...)
    p, q = chosen(f, args...), chosen(h, args...)
    C.structure(p) == C.structure(q) && length(p.groups) == length(q.groups) &&
        all(((g, k),) -> g.kind == k.kind && work(p, g) == work(q, k), zip(p.groups, q.groups))
end

const CANONICAL_DOMAIN = "one canonical axis order for a group's domain"
const SCALING_FORMS = "x / n and x · (1 / n) one form under the numerics policy where 1 / n is inexact"

@testset "promise 1: layout spelling does not matter" begin
    a, b = data(8, 6), data(8, 6; scale = 0.3)
    # Spellings that plan alike today.
    @test same_plan(a -> sum(a; dims = 1), a -> reduce(+, a; dims = 1), a)
    @test same_plan(a -> sum(abs2, a; dims = 1), a -> mapreduce(abs2, +, a; dims = 1), a)
    @test same_plan(a -> sum(a[1:4, :]; dims = 1), a -> sum(view(a, 1:4, :); dims = 1), a)
    @test same_plan(a -> mean(a; dims = 1), a -> sum(a; dims = 1) ./ 8, a)
    @test same_plan((a, b) -> a .+ b, (a, b) -> b .+ a, a, b)
    @test same_plan(a -> a .* 2, a -> 2 .* a, a)
    @test same_plan((a, b) -> sum(a .* b; dims = 1), (a, b) -> sum(b .* a; dims = 1), a, b)
    # A fold a result reads through a permutation is stored in the result's
    # order, so the transpose of a product costs no kernel of its own.
    @test same_plan((a, b) -> (a' * b)', (a, b) -> b' * a, a, b)
    @test C.structure(chosen((a, b) -> (a' * b)', a, b)).kernels == 1
    # Stored in another order, a value is still the one the program computes.
    transposed_product(a, b) = (a' * b)'
    permuted_sum(x) = permutedims(sum(reshape(x, 2, 4, 6); dims = 2), (3, 1, 2))
    @test agree(C.execute(chosen(transposed_product, a, b), a, b), C.leaves(transposed_product(a, b)))
    @test agree(C.execute(chosen(permuted_sum, a), a), C.leaves(permuted_sum(a)))
    # Reshapes around size-1 axes are checked in plan.jl, "axes of extent one".
    # Spellings that differ in the order of a kernel's domain do the same work.
    @test same_work(a -> sum(permutedims(a, (2, 1)); dims = 2), a -> permutedims(sum(a; dims = 1), (2, 1)), a)
    @test same_work(a -> tanh.(reshape(a, 48)), a -> reshape(tanh.(a), 48), a)
    @test !same_work(a -> sum(a; dims = 1), a -> sum(a; dims = 2), a)
    # The two spellings do the same work at the same addresses; whether the
    # promise is "the same plan" or "the same work" is open.
    @testset "gap: $CANONICAL_DOMAIN" begin
        @test_broken same_plan(a -> sum(permutedims(a, (2, 1)); dims = 2), a -> permutedims(sum(a; dims = 1), (2, 1)), a)
        @test_broken same_plan(a -> tanh.(reshape(a, 48)), a -> reshape(tanh.(a), 48), a)
    end
    # A quotient by a power of two is the product by its reciprocal, which
    # rounds alike.
    @test same_plan(a -> sum(a; dims = 1) ./ 8, a -> sum(a; dims = 1) .* (1 / 8), a)
    @test same_plan(a -> a ./ 4, a -> a .* 0.25, a)
    @testset "gap: $SCALING_FORMS" begin
        # This rewrite changes rounding: it belongs behind `permits_rewrites`,
        # with a note that `explain` names.
        @test_broken same_plan(a -> sum(a; dims = 1) ./ 6, a -> sum(a; dims = 1) .* (1 / 6), a)
    end
end

# A map read by a fold after it and by a product after that. The planner
# stores the map and gives it a kernel of its own, since a fold's group grows
# only through values after the fold. Computing the map inside the fold's
# kernel, which also writes it for the product, is one kernel and one pass
# over the map fewer; where the map is wide enough the estimate prefers it,
# and the cover never sees it.
map_fold_product(x, w) = (y = exp.(x); (sum(y; dims = 1), y * w))

# The estimated time of a plan: the chosen option of each group.
function estimated(p::C.Plan, target)
    d = C.Decider(target, p.graph; measure = false)
    sum(g -> C.decide(d, p, g).choice.estimate, filter(g -> C.chosen(p, g), p.groups); init = 0.0)
end

@testset "promise 3: one rule decides fusion" begin
    x, w = data(64, 32), data(32, 16)
    p = chosen(map_fold_product, x, w)
    graph = p.graph
    y = findfirst(v -> v.definition isa C.Pointwise && v.dims == (64, 32), graph.values)
    s = findfirst(v -> v.definition isa C.Reduction && v.dims == (32,), graph.values)
    # The alternative, as a proposal would fix it: the map and the fold in one group.
    alternative = C.plan(graph; fixed = [[y, s]])
    @test C.structure(alternative).kernels == 2
    @test first(alternative.groups).kind === :reduction
    @test agree(C.execute(alternative, x, w), C.leaves(map_fold_product(x, w)))
    # At this size the fold's 32 items leave the fused kernel little parallelism.
    @test estimated(p, T.CPUTarget()) <= estimated(alternative, T.CPUTarget())
    # So the cover keeps the rules' three kernels. Wide enough, the fused
    # kernel is the cheaper by estimate, and the planner's own proposal to join
    # the map into the fold's group finds it.
    @test C.structure(p).kernels == 3
    xw, ww = data(64, 512), data(512, 16)
    pw = chosen(map_fold_product, xw, ww)
    @test C.structure(pw).kernels == 2
    @test estimated(pw, T.CPUTarget()) < estimated(C.plan(pw.graph), T.CPUTarget())
    if CUDA_FUNCTIONAL
        xf, wf = CuArray(single(x)), CuArray(single(w))
        @test C.structure(chosen(map_fold_product, xf, wf; target = T.CUDATarget())).kernels == 2
    end
end

# Identical layers are identical groups, canonically, however many of them a
# program stacks. Preparing then costs the same per layer, and a cold prepare
# builds one kernel per distinct group.

stack_layer(x, (w, b)) = tanh.(w * x .+ b)
stack_loss(x, layers) = sum(abs2, foldl(stack_layer, layers; init = x))
stack(L, dev; width = 64) =
    (Tuple((dev(single(data(width, width; scale = 0.1k))), dev(single(data(width; scale = 0.2k)))) for k in 1:L), dev(single(data(width, 16))))
# The fastest of three runs after one that compiles the Julia side: the
# tests run in parallel, and a timing shares the host.
seconds(f) = (f(); minimum(_ -> (t = time(); f(); time() - t), 1:3))

@testset "promise 6: identical layers cost the same each" begin
    distinct(p) = length(unique(C.canonical(p, g) for g in p.groups))
    function planned(L)
        layers, x = stack(L, identity)
        C.plan(C.lower(differentiate(trace(stack_loss, x, layers))); recompute_folds = true)
    end
    @test distinct(planned(32)) == distinct(planned(8))
    # Two groups alike generate the same floor kernels, so each has one content
    # key and the kernel is built once.
    layered = planned(8)
    floor_keys(g) = [C.content_key("floor", k.body) for k in C.scalar_kernels(layered.graph, layered.stored, g, C.kernel_slots(layered, g))]
    alike = Dict{String, Vector{Any}}()
    foreach(g -> push!(get!(alike, C.canonical(layered, g), []), g), layered.groups)
    repeated = filter(gs -> length(gs) > 1, collect(values(alike)))
    @test !isempty(repeated)
    @test all(gs -> floor_keys(gs[1]) == floor_keys(gs[2]), repeated)
    if CUDA_FUNCTIONAL
        # A kernel is loaded or compiled once per key, however many groups
        # share it: a cold prepare builds at most one per distinct group.
        layers, x = stack(8, CuArray)
        p = differentiate(trace(stack_loss, x, layers))
        before = C.cache_counts()
        ex = prepare(p, x, layers; target = T.CUDATarget(), measure = false)
        after = C.cache_counts()
        built = (after.compiled - before.compiled) + (after.loaded - before.loaded)
        @test built > 0
        @test built <= distinct(ex.plan)
    end
    function per_layer_plan(L)
        layers, x = stack(L, identity)
        g = C.lower(differentiate(trace(stack_loss, x, layers)))
        seconds(() -> C.plan(g; recompute_folds = true)) / L
    end
    # The planner grows each group from the values that read it, so a layer
    # costs the same however many precede it.
    @test per_layer_plan(32) < 2 * per_layer_plan(8)
    # Groups alike are decided once, however many layers repeat them.
    function decided(L)
        layers, x = stack(L, identity)
        d = C.Decider(T.CPUTarget(), C.lower(differentiate(trace(stack_loss, x, layers))); measure = false)
        C.decisions(d, first(C.cover(d)))
        length(d.decided)
    end
    @test decided(32) == decided(8)
    # A call's group names the call and what it reads: attention over 6 keys
    # and over 9 are different groups.
    function call_key(n)
        q, k, v = data(4, 2, 6, 2; scale = 0.3), data(4, 2, n, 2; scale = 0.2), data(4, 2, n, 2; scale = 0.1)
        graph = C.lower(trace((q, k, v) -> Ops.attention(q, k, v), q, k, v))
        p = C.plan(C.claimed(graph, [C.ClaimedCall(s, ()) for s in graph.calls]))
        C.canonical(p, only(g for g in p.groups if g.kind === :opaque))
    end
    @test call_key(6) != call_key(9)
    if CUDA_FUNCTIONAL
        # Options rank alike in groups alike, so a rank chosen for one is the
        # same choice in each.
        layers, x = stack(8, CuArray)
        d = C.Decider(T.CUDATarget(), C.lower(differentiate(trace(stack_loss, x, layers))); measure = false)
        p = first(C.generator_plan(d))
        ranked = Dict{String, Any}()
        @test all(filter(g -> C.chosen(p, g), p.groups)) do g
            os = [(o.key, o.estimate) for o in C.options(d, p, g)]
            get!(ranked, C.canonical(p, g), os) == os
        end
    end
    if CUDA_FUNCTIONAL
        # Proposals alike in the generator's plan are priced once and taken
        # together: the cover's trials do not grow with the layers. Each trial
        # plans the whole program, and the planner's dependency check
        # (`ancestors`, `depends`) still grows faster than the layers.
        function per_layer_prepare(L)
            layers, x = stack(L, CuArray)
            p = differentiate(trace(stack_loss, x, layers))
            seconds(() -> prepare(p, x, layers; target = T.CUDATarget(), measure = false)) / L
        end
        @test per_layer_prepare(16) < 2 * per_layer_prepare(4)
    end
end

# The coverage suite's compositions, in a module of their own: their names
# clash with this file's.
module Coverage include(joinpath(@__DIR__, "fixtures", "compositions.jl")) end

const GENERATOR_COST = "the generator's estimate counts what its schedule costs beyond bytes and products"
# Compositions and precisions where an estimate that decides is outside the close-call factor.
const ESTIMATE_GAPS = [("the verb, a training step", :tf32),
    ("a slice written, a training step", :full), ("a slice written, a training step", :tf32)]

within(e, t) = max(e, t) <= C.ESTIMATE_ERROR * min(e, t)

# Every option of a group chosen by estimate among several, and each side of
# a comparison made by estimate, is within the close-call factor of its
# measurement, and each such comparison stands when measured.
@testset "estimates that decide are within the close-call factor, or measured" begin
    @test !hasfield(Tempest.Implementations.Work, :rate)
    CUDA_FUNCTIONAL && @testset "$(c.name), $precision" for precision in (:full, :tf32), c in Coverage.COMPOSITIONS
        d = C.Decider(T.CUDATarget(; precision), C.rewrite(C.lower(c.program)); measure = :close)
        p, considered = C.cover(d)
        timed(q, g, o) = (C.measure!(d, q, g, o); d.record[o.key])
        held = Bool[]
        for g in filter(g -> C.chosen(p, g), p.groups)
            x = C.decide(d, p, g)
            x.how === :estimated && append!(held, [within(o.estimate, timed(p, g, o)) for o in x.options])
        end
        for k in filter(k -> k.comparison.how === :estimated, considered)
            measured = map(side -> sum(s -> timed(s[1], s[2], s[3].choice), side; init = 0.0), k.comparison.sides)
            # Times within the device's noise are a tie, which the estimate decides.
            @test abs(measured[2] - measured[1]) <= d.rates.noise * minimum(measured) || (measured[2] < measured[1]) == C.cheaper(k.comparison)
            append!(held, within.((k.comparison.old, k.comparison.new), measured))
        end
        if (c.name, precision) in ESTIMATE_GAPS
            @testset "gap: $GENERATOR_COST" begin @test_broken all(held) end
        else
            @test all(held)
        end
    end
end
