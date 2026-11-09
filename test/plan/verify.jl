# The verifier: each invariant a plan and its storage keep holds for the
# plans the planner forms, and a plan or storage built by hand to break one is
# rejected, naming the invariant.
include("../fixtures/programs.jl")

# The plan of `f` on `args`, and with its kernels in order, its storage.
planned(f, args...) = C.plan(C.lower(trace(f, args...)))
stored_plan(f, args...; replaces = Pair{Int, Int}[]) = C.storage(planned(f, args...), replaces)

# The invariant `f` finds broken, or nothing.
function broken(f)
    try
        f()
        nothing
    catch err
        err isa C.InvalidPlan || rethrow()
        err.invariant
    end
end

# A plan with its groups or ownership changed.
with_groups(p, groups, owner = p.owner) = C.Plan(p.graph, p.stored, groups, owner)
copied(g) = C.Group(g.kind, g.domain, copy(g.reductions), copy(g.outputs), copy(g.terms), copy(g.phase))

chain(x, w1, w2, w3) = w3 * tanh.(w2 * tanh.(w1 * x))
chain_args = (data(5, 4), data(6, 5; scale = 0.3), data(7, 6; scale = 0.2), data(3, 7; scale = 0.1))

function two_states(W, x, h)
    c = h
    for _ in 1:6
        z = W * h .+ x
        c = tanh.(z) .* c .+ 0.1
        h = tanh.(z) .* tanh.(c)
    end
    sum(abs2, h)
end
two_args = (data(4, 4; scale = 0.3), data(4), data(4; scale = 0.5))

@testset "the planner's plans verify" begin
    for (f, args) in ((chain, chain_args), (softmax_columns, (data(8, 6),)), (layernorm, (data(8, 6), data(8))),
            (two_states, two_args), (x -> cumsum(exp.(x); dims = 1) .* 2, (data(8, 6),)))
        p = planned(f, args...)
        @test C.verify(p) === p
        q, _, s = C.storage(p)
        @test C.verify(q, s) === q
        @test agree(C.leaves(outputs(execute!(prepare(trace(f, args...), args...; verify = true)))), [f(args...)])
    end
    # A result written in place of the argument it replaces.
    p = planned(x -> x .* 2, data(8))
    input = findfirst(v -> v.definition isa C.Input, p.graph.values)
    q, _, s = C.storage(p, [p.graph.results[1].value => input])
    @test !isempty(s.donated)
    @test C.verify(q, s) === q
end

@testset "invariant 1: one owner, and reads written before" begin
    p = planned(chain, chain_args...)
    @test length(p.groups) == 3
    first_product = only(p.groups[1].reductions)
    # A value two kernels compute, as a recurrence's step value joined into a
    # second kernel was.
    groups = copied.(p.groups)
    push!(groups[2].outputs, first_product)
    @test broken(() -> C.verify(with_groups(p, groups))) == 1
    # A member whose owner is another kernel.
    owner = copy(p.owner)
    owner[first_product] = 2
    @test broken(() -> C.verify(with_groups(p, p.groups, owner))) == 1
    # A kernel that reads what a later kernel writes.
    owner = map(k -> k == 0 ? 0 : 4 - k, p.owner)
    @test broken(() -> C.verify(with_groups(p, reverse(p.groups), owner))) == 1
    # A kernel that reads what nothing stores.
    stored = copy(p.stored)
    stored[first_product] = false
    @test broken(() -> C.verify(C.Plan(p.graph, stored, p.groups, p.owner))) == 1
end

@testset "invariant 2: slots, lifetimes and pairs" begin
    p, _, s = stored_plan(chain, chain_args...)
    x, y = only(p.groups[1].reductions), only(p.groups[2].reductions)
    @test s.slots[x] > 0 && s.slots[y] > 0 && s.slots[x] != s.slots[y]
    with(s; kwargs...) = C.Storage((get(kwargs, f, getfield(s, f)) for f in fieldnames(C.Storage))...)
    # A slot given to a value while the one it holds has a reader yet to run.
    slots = copy(s.slots)
    slots[x] = slots[y]
    @test broken(() -> C.verify(p, with(s; slots))) == 2
    # Nor where a pair says so that its rule does not allow: the second
    # product does not read the first where it writes.
    @test broken(() -> C.verify(p, with(s; slots, reused = [y => x]))) == 2
    # A lifetime that ends before a reader.
    lifetimes = copy(s.lifetimes)
    lifetimes[x] = 1:1
    @test broken(() -> C.verify(p, with(s; lifetimes))) == 2
    # A slot too small for its value.
    @test broken(() -> C.verify(p, with(s; sizes = fill(1, length(s.sizes))))) == 2
    # A fold that does not start from what it is written over.
    @test broken(() -> C.verify(p, with(s; overwrites = [y => x]))) == 2
    # A result written over the argument it replaces, which its kernel reads
    # elsewhere: staged, and not donated.
    p = planned(x -> reverse(x) .* 2, data(8))
    input = findfirst(v -> v.definition isa C.Input, p.graph.values)
    q, _, s = C.storage(p, [p.graph.results[1].value => input])
    @test !isempty(s.staged) && isempty(s.donated)
    @test C.verify(q, s) === q
    @test broken(() -> C.verify(q, with(s; donated = s.staged, staged = Pair{Int, Int}[]))) == 2
end

@testset "invariant 3: a recurrence owns its cycle" begin
    p = planned(two_states, two_args...)
    k = findfirst(g -> g.kind isa C.RecurrenceKind, p.groups)
    g = p.groups[k]
    states = C.recurrence_states(p.graph, g.outputs)
    @test length(states) == 2
    # Two states carried from one value: one argument standing for two.
    graph = p.graph
    values = copy(graph.values)
    first_state, second_state = states
    d = values[second_state].definition
    values[second_state] = C.Value(C.Carried(d.init, values[first_state].definition.from, d.axis, d.reverse), values[second_state].dims,
        values[second_state].eltype, values[second_state].scalar)
    joined = C.IndexGraph(values, graph.arguments, graph.results, graph.result_dims, graph.result_scalar, graph.result_eltype)
    @test broken(() -> C.verify(C.Plan(joined, p.stored, p.groups, p.owner))) == 3
    # A value of the cycle, the one a state is carried from, written by a
    # kernel that runs before the recurrence's.
    from = C.carried_from(p.graph, second_state)
    apart = C.Group(C.PointwiseKind(), p.graph.values[from].dims)
    push!(apart.outputs, from)
    apart.terms[from] = IR.Lit(0.0)
    apart.phase[from] = 0
    stepping = copied(g)
    filter!(!=(from), stepping.outputs)
    delete!(stepping.terms, from)
    groups = [apart; [j == k ? stepping : h for (j, h) in enumerate(p.groups)]]
    owner = map(o -> o == 0 ? 0 : o + 1, p.owner)
    owner[from] = 1
    @test broken(() -> C.verify(with_groups(p, groups, owner))) == 3
end

@testset "invariant 4: terms, phases and places" begin
    p = planned(x -> exp.(x) ./ sum(exp.(x); dims = 1), data(8, 6))
    @test length(p.groups) == 1
    g = only(p.groups)
    fold, output = first(g.reductions), only(g.outputs)
    # An output computed before the fold it reads is complete.
    early = copied(g)
    early.phase[output] = 0
    @test broken(() -> C.verify(with_groups(p, [early]))) == 4
    # A term reading as a fold of the kernel what is not one.
    stray = copied(g)
    stray.terms[output] = IR.Stat(1)
    @test broken(() -> C.verify(with_groups(p, [stray]))) == 4
    # A term loading what its own kernel computes.
    loading = copied(g)
    loading.terms[output] = IR.Load(fold, C.placement(p.graph, g, fold))
    @test broken(() -> C.verify(with_groups(p, [loading]))) == 4
    # A member with no place in the domain.
    p = planned(x -> exp.(x) .* 2, data(8, 6))
    g = only(p.groups)
    wide = C.Group(g.kind, map(n -> n + 1, g.domain), g.reductions, g.outputs, g.terms, g.phase)
    @test broken(() -> C.verify(with_groups(p, [wide]))) == 4
    # A fold a term recomputes that is also a member of its kernel.
    p = recomputing(C.lower(trace((a, b) -> tanh.(a * b) * b', data(5, 6), data(6, 7))))
    k = findfirst(h -> !isempty(C.recomputed_in(h)), p.groups)
    @test k !== nothing
    h = copied(p.groups[k])
    v = first(C.recomputed_in(h))
    push!(h.reductions, v)
    h.terms[v] = IR.Lit(0.0)
    h.phase[v] = 1
    owner = copy(p.owner)
    owner[v] = k
    @test broken(() -> C.verify(with_groups(p, [j == k ? h : x for (j, x) in enumerate(p.groups)], owner))) == 4
end
