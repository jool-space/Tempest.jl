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

const RESULT_FORMULAS = "a result written through its formula, so a permute of a stored value costs no kernel"
const CANONICAL_DOMAIN = "one canonical axis order for a group's domain"
const SCALING_FORMS = "sum / n and sum · (1 / n) one form under the numerics policy"
const PRICED_GROUPING = "the cover prices every grouping (DESIGN.md, The decision)"

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
    # Reshapes around size-1 axes are checked in plan.jl, "axes of extent one".
    @testset "gap: $RESULT_FORMULAS" begin
        # The transpose after the product plans as a kernel of its own.
        @test_broken same_plan((a, b) -> (a' * b)', (a, b) -> b' * a, a, b)
        @test_broken C.structure(chosen((a, b) -> (a' * b)', a, b)).kernels == 1
    end
    @testset "gap: $CANONICAL_DOMAIN" begin
        @test_broken same_plan(a -> sum(permutedims(a, (2, 1)); dims = 2), a -> permutedims(sum(a; dims = 1), (2, 1)), a)
        @test_broken same_plan(a -> tanh.(reshape(a, 48)), a -> reshape(tanh.(a), 48), a)
    end
    @testset "gap: $SCALING_FORMS" begin
        @test_broken same_plan(a -> sum(a; dims = 1) ./ 8, a -> sum(a; dims = 1) .* (1 / 8), a)
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
    @testset "gap: $PRICED_GROUPING" begin
        # Wide enough for the fused kernel to be the cheaper by estimate.
        xw, ww = data(64, 512), data(512, 16)
        pw = chosen(map_fold_product, xw, ww)
        yw = findfirst(v -> v.definition isa C.Pointwise && v.dims == (64, 512), pw.graph.values)
        sw = findfirst(v -> v.definition isa C.Reduction && v.dims == (512,), pw.graph.values)
        @test estimated(C.plan(pw.graph; fixed = [[yw, sw]]), T.CPUTarget()) < estimated(pw, T.CPUTarget())
        @test_broken C.structure(pw).kernels == 2
        if CUDA_FUNCTIONAL
            xf, wf = CuArray(single(x)), CuArray(single(w))
            @test_broken C.structure(chosen(map_fold_product, xf, wf; target = T.CUDATarget())).kernels == 2
        end
    end
end

# Identical layers are identical groups, canonically, however many of them a
# program stacks. Preparing should then cost the same per layer, and a cold
# prepare should build one kernel per distinct group. Today the cover decides
# every group instance anew and re-plans the whole program for each proposal.
const LINEAR_PREPARE = "the cover per canonical group, a trial re-planning only its region"

stack_layer(x, (w, b)) = tanh.(w * x .+ b)
stack_loss(x, layers) = sum(abs2, foldl(stack_layer, layers; init = x))
stack(L, dev; width = 64) =
    (Tuple((dev(single(data(width, width; scale = 0.1k))), dev(single(data(width; scale = 0.2k)))) for k in 1:L), dev(single(data(width, 16))))
# The second run, once the Julia side has compiled.
seconds(f) = (f(); t = time(); f(); time() - t)

@testset "promise 6: identical layers cost the same each" begin
    distinct(p) = length(unique(C.canonical(p, g) for g in p.groups))
    function planned(L)
        layers, x = stack(L, identity)
        C.plan(C.lower(differentiate(trace(stack_loss, x, layers))); recompute_folds = true)
    end
    @test distinct(planned(32)) == distinct(planned(8))
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
    @testset "gap: $LINEAR_PREPARE" begin
        if CUDA_FUNCTIONAL
            function per_layer_prepare(L)
                layers, x = stack(L, CuArray)
                p = differentiate(trace(stack_loss, x, layers))
                seconds(() -> prepare(p, x, layers; target = T.CUDATarget(), measure = false)) / L
            end
            @test_broken per_layer_prepare(16) < 2 * per_layer_prepare(4)
        end
    end
end
