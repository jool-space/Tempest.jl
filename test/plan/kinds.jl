# What each kind of group answers (src/Compiler/plan/groups.jl), asked of the
# groups of small plans: where a group keeps a value, which of its folds a
# member may read, how a term of it reads a value, and that the loads found
# for a term before it is built are the term's. What each kind takes as it
# grows is fusion.jl's to test.
include("../fixtures/programs.jl")

planned(f, args...) = C.plan(C.lower(trace(f, args...)))
# A context that reads a plan's kernels as they are formed.
context(p) = C.Plain(C.Kernels(copy(p.stored), C.ancestors(p.graph)))
whole(dims) = C.identity_index(dims)

@testset "a kind is written as its name, which a group's view keeps" begin
    for (kind, name) in ((C.ReductionKind(), :reduction), (C.ScanKind(), :scan), (C.PointwiseKind(), :pointwise), (C.RecurrenceKind(), :recurrence))
        @test C.kind_name(kind) === name
        @test string(kind) == string(name)
    end
    p = planned(x -> cumsum(x; dims = 1), data(8, 6))
    @test only(p.groups).kind isa C.ScanKind
    @test C.group_view(p, only(p.groups)).kind === :scan
end

# A softmax: the sum over the first axis, and the map that divides by it,
# which computes the exponential in place.
@testset "a reduction's group" begin
    p = planned(x -> exp.(x) ./ sum(exp.(x); dims = 1), data(8, 6))
    g = only(p.groups)
    fold, quotient = only(g.reductions), only(g.outputs)
    @test g.kind isa C.ReductionKind && !C.stepwise(g.kind)
    @test C.placement(p.graph, g, fold) == (C.var(2),)
    @test C.placement(p.graph, g, quotient) == whole(g.domain)
    # A member reads the fold where the group keeps it, and nowhere else.
    @test C.reads_fold(g.kind, p.graph, g, fold, (C.var(2),))
    @test !C.reads_fold(g.kind, p.graph, g, fold, (C.var(1),))
    cx = context(p)
    input = findfirst(v -> v.definition isa C.Input, p.graph.values)
    exponential = only(v for v in eachindex(p.graph.values) if p.graph.values[v].definition isa C.Pointwise && !p.stored[v])
    @test C.reading(p.graph, cx, g, fold) === C.as_fold
    @test C.reading(p.graph, cx, g, quotient) === C.as_output
    @test C.reading(p.graph, cx, g, input) === C.as_load
    @test C.reading(p.graph, cx, g, exponential) === C.as_inlined
end

# Attention with its folds recomputed: the scores are read by the sum over
# keys as a fold recomputed where it is read.
@testset "a fold recomputed where it is read" begin
    q, k, v = data(4, 4, 6, 2; scale = 0.3), data(4, 2, 6, 2; scale = 0.2), data(4, 2, 6, 2; scale = 0.1)
    p = recomputing(C.lower(trace((q, k, v) -> Ops.attention(q, k, v; causal = true), q, k, v)))
    g = only(p.groups)
    cx = context(p)
    @test !isempty(C.recomputed_in(g))
    @test all(r -> C.reading(p.graph, cx, g, r) === C.as_recomputed, C.recomputed_in(g))
end

# A scan and a sum along its axis share the sweep.
@testset "a scan's group" begin
    p = planned(x -> (cumsum(x; dims = 1), sum(x; dims = 1)), data(8, 6))
    g = only(p.groups)
    scan = only(r for r in g.reductions if p.graph.values[r].definition isa C.Scan)
    fold = only(r for r in g.reductions if p.graph.values[r].definition isa C.Reduction)
    @test g.kind isa C.ScanKind && !C.stepwise(g.kind)
    @test C.placement(p.graph, g, scan) == whole(g.domain)
    @test C.placement(p.graph, g, fold) == (C.var(2),)
    @test C.reads_fold(g.kind, p.graph, g, scan, whole(g.domain))
    # A fold riding the sweep is complete only after it: no member reads it.
    @test !C.reads_fold(g.kind, p.graph, g, fold, (C.var(2),))
end

@testset "a group of maps" begin
    p = planned(x -> (y = sin.(x); (y, cos.(y))), data(8, 6))
    g = only(p.groups)
    @test g.kind isa C.PointwiseKind && isempty(g.reductions) && length(g.outputs) == 2
    @test all(m -> C.placement(p.graph, g, m) == whole(g.domain), g.outputs)
    @test !any(m -> C.restricted(p.graph, g, m), g.outputs)
end

# A loop's state, carried from step to step; and its backward with the
# weights' gradient riding the steps.
@testset "a recurrence's group" begin
    rnn(W, xs, a) = (h = a; for t in axes(xs, 2); h = tanh.(W * h .+ xs[:, t]); end; sum(abs2, h))
    args = (data(3, 3; scale = 0.5), data(3, 6; scale = 0.2), data(3; scale = 0.1))
    p = planned(rnn, args...)
    g = only(g for g in p.groups if g.kind isa C.RecurrenceKind)
    @test C.stepwise(g.kind)
    state = first(g.outputs)
    @test p.graph.values[state].definition isa C.Carried
    @test C.placement(p.graph, g, state) == whole(p.graph.values[state].dims)
    # A member reads what the group computes where this step stored it.
    cx = context(p)
    @test all(m -> C.reading(p.graph, cx, g, m) === C.as_load, g.outputs)
    lowered = C.lower(differentiate(trace(rnn, args...)))
    sums = C.riding_sums(C.plan(lowered))
    @test !isempty(sums)
    ridden = C.plan(lowered; ride = sums)
    g = only(g for g in ridden.groups if g.kind isa C.RecurrenceKind && !isempty(g.reductions))
    # A sum riding the steps is complete only after them: the group keeps it
    # nowhere a member could read it.
    @test all(r -> C.placement(ridden.graph, g, r) === nothing, g.reductions)
    @test !any(r -> C.reads_fold(g.kind, ridden.graph, g, r, ()), g.reductions)
end

# `try_add!` asks what a term would load before it builds the term, most
# candidates failing there: the loads it finds are those of the term built.
@testset "a term's loads are found before it is built" begin
    function differing(p)
        cx = context(p)
        count(((g, m) for g in p.groups if !C.stepwise(g.kind) for m in C.members(g))) do (g, m)
            found = Set{Int}()
            m in g.reductions ? C.operand_reads!(found, Set{Int}(), p.graph, cx, g, p.graph.values[m].definition.operand) :
                C.value_reads!(found, Set{Int}(), p.graph, cx, g, m)
            found != Set(IR.stored_reads(g.terms[m]))
        end
    end
    @testset "$name" for (name, f, args) in [PROGRAMS; REPEATED_WRITES]
        for program in (trace(f, args...), differentiate(trace(f, args...)))
            lowered = C.lower(program)
            @test differing(C.plan(lowered)) == 0
            @test differing(recomputing(lowered)) == 0
        end
    end
end
