# Running a plan on the CPU, kernel by kernel, with a buffer per stored value.
# It checks the plan's decisions: what is stored, what is recomputed, and how
# reductions share a kernel.

function evaluate_term(t::Load, x, k, buffers, stats, graph)
    stored = buffers[t.value]
    graph.values[t.value].scalar && return stored
    stored[map(e -> evaluate_index(e, x, (v, i) -> buffers[v][(i .+ 1)...]) + 1, t.index)...]
end
evaluate_term(t::Lit, x, k, buffers, stats, graph) = t.value
evaluate_term(t::Ix{T}, x, k, buffers, stats, graph) where {T} =
    T(evaluate_index(t.expr, x, (v, i) -> buffers[v][(i .+ 1)...]))
# A fold's value at the domain index `x`: at the axes it keeps.
function evaluate_term(t::Stat, x, k, buffers, stats, graph)
    s = stats[t.value]
    s isa AbstractArray || return s
    d = graph.values[t.value].definition
    axes = d isa Reduction ? d.reduced : ()
    s[(x[i] + 1 for i in eachindex(x) if !(i in axes))...]
end
evaluate_term(t::Apply, x, k, buffers, stats, graph) =
    convert(t.eltype, t.f(map(a -> evaluate_term(a, x, k, buffers, stats, graph), t.args)...))
# A recomputed fold: its body over its reduced axes, after the domain's own.
function evaluate_term(t::Recomputed, x, k, buffers, stats, graph)
    val = graph.values[t.value]
    fold = val.definition.fold
    A = fold_types(fold, term_eltype(t.body, graph)).accumulator
    state = fold_init(fold, A)
    for J in CartesianIndices(t.extents)
        e = evaluate_term(t.body, (x..., (Tuple(J) .- 1)...), nothing, buffers, stats, graph)
        state = fold_combine(fold, state, fold_lift(fold, A(e)))
    end
    val.eltype(fold_observe(fold, state))
end

function run_group!(g::Group, buffers, graph)
    g.kind === :reduction && return run_reduction!(g, buffers, graph)
    g.kind === :scan && return run_scan!(g, buffers, graph)
    g.kind === :recurrence && return run_recurrence!(g, buffers, graph)
    run_pointwise!(g, buffers, graph)
end

# A step at a time: the state, where it starts at the first; then each other
# member over its axes, the value the state is carried from last, which is
# the state one step on.
function run_recurrence!(g::Group, buffers, graph)
    D = g.domain
    axis = recurrence_axes(graph, g).axis
    state = first(g.outputs)
    from = graph.values[state].definition.from.value
    places = recurrence_places(graph, g.outputs).places
    foreach(m -> buffers[m] = Array{graph.values[m].eltype}(undef, graph.values[m].dims), g.outputs)
    for n in 0:(D[axis] - 1), m in g.outputs
        m == state && n > 0 && continue
        spans = map(k -> k != axis && any(e -> mentions(e, k), places[m]) ? D[k] : 1, Base.OneTo(length(D)))
        for I in CartesianIndices(Tuple(spans))
            x = Base.setindex(Tuple(I) .- 1, n, axis)
            y = evaluate_term(g.terms[m], x, nothing, buffers, nothing, graph)
            at = map(e -> evaluate_index(e, x), places[m])
            buffers[m][(at .+ 1)...] = y
            m == from && n + 1 < D[axis] && (buffers[state][(Base.setindex(at, n + 1, axis) .+ 1)...] = y)
        end
    end
end

function run_pointwise!(g::Group, buffers, graph)
    for v in g.outputs
        val = graph.values[v]
        t = g.terms[v]
        if val.scalar
            buffers[v] = evaluate_term(t, (), nothing, buffers, nothing, graph)
            continue
        end
        out = Array{val.eltype}(undef, val.dims)
        for I in CartesianIndices(val.dims)
            out[I] = evaluate_term(t, Tuple(I) .- 1, I, buffers, nothing, graph)
        end
        buffers[v] = out
    end
end

# Every fold over its own axes, in phase order; then the outputs.
function run_reduction!(g::Group, buffers, graph)
    D = g.domain
    stats = Dict{Int, Any}()
    for r in sort(g.reductions; by = r -> (g.phase[r], r))
        val = graph.values[r]
        val.definition isa Keyed && (buffers[r] = run_keyed(g, r, buffers, stats, graph); continue)
        fold = val.definition.fold
        axes = val.definition.reduced
        A = fold_types(fold, term_eltype(g.terms[r], graph)).accumulator
        kept = kept_dims(D, axes)
        swept = ntuple(k -> k in axes ? D[k] : 1, length(D))
        out = Array{val.eltype}(undef, kept)
        for K in CartesianIndices(kept)
            state = fold_init(fold, A)
            for S in CartesianIndices(swept)
                x = ntuple(k -> k in axes ? S[k] - 1 : K[k] - 1, length(D))
                e = evaluate_term(g.terms[r], x, nothing, buffers, stats, graph)
                state = fold_combine(fold, state, fold_lift(fold, A(e)))
            end
            out[K] = val.eltype(fold_observe(fold, state))
        end
        stats[r] = val.scalar ? only(out) : reshape(out, val.dims)
        buffers[r] = stats[r]
    end
    # An output over the domain, at the positions of the primary folds, or at
    # those of a fold it is a slice of: its index over the domain, with the
    # axes it does not keep at zero.
    for m in g.outputs
        val = graph.values[m]
        if val.scalar
            buffers[m] = evaluate_term(g.terms[m], ntuple(_ -> 0, length(D)), nothing, buffers, stats, graph)
            continue
        end
        kept = [findfirst(k -> p == var(k), eachindex(D)) for p in placement(graph, g, m)]
        out = Array{val.eltype}(undef, val.dims)
        for I in CartesianIndices(val.dims)
            x = ntuple(k -> (j = findfirst(==(k), kept)) === nothing ? 0 : I[j] - 1, length(D))
            out[I] = evaluate_term(g.terms[m], x, nothing, buffers, stats, graph)
        end
        buffers[m] = out
    end
end

# A fold by key: the points of the domain in order, each element combined at
# its key, from the contents the fold starts from.
function run_keyed(g::Group, r, buffers, stats, graph)
    val = graph.values[r]
    d = val.definition
    states = map(CartesianIndices(val.dims)) do I
        d.into === nothing && return fold_init(d.fold, val.eltype)
        fold_lift(d.fold, convert(val.eltype, evaluate_term(stored_term(d.into), Tuple(I) .- 1, nothing, buffers, stats, graph)))
    end
    gather = (v, i) -> buffers[v][(i .+ 1)...]
    for I in CartesianIndices(g.domain)
        x = Tuple(I) .- 1
        e = evaluate_term(g.terms[r], x, nothing, buffers, stats, graph)
        at = map(k -> evaluate_index(k, x, gather) + 1, d.key)
        states[at...] = fold_combine(d.fold, states[at...], fold_lift(d.fold, convert(val.eltype, e)))
    end
    y = map(s -> fold_observe(d.fold, s), states)
    val.scalar ? only(y) : y
end

# Each line along the scanned axes, swept in the scans' direction: at each
# position every scan and fold takes its element, in phase order, and the
# outputs there read the scans where the sweep has them. The folds are
# complete after the line.
function run_scan!(g::Group, buffers, graph)
    D = g.domain
    first_scan = graph.values[first(g.reductions)].definition
    axes, reverse = first_scan.axes, first_scan.reverse
    folds = sort(g.reductions; by = r -> (g.phase[r], r))
    kept = ntuple(k -> k in axes ? 1 : D[k], length(D))
    # Each member's values, which a `Stat` reads at the position; a fold's at
    # the line.
    values = Dict{Int, Any}(m => Array{graph.values[m].eltype}(undef, m in folds && graph.values[m].definition isa Reduction ? kept : D)
        for m in members(g))
    along = CartesianIndices(ntuple(k -> k in axes ? D[k] : 1, length(D)))
    for K in CartesianIndices(kept)
        states = Dict{Int, Any}()
        observe(r) = graph.values[r].eltype(fold_observe(graph.values[r].definition.fold, states[r]))
        for S in (reverse ? Iterators.reverse(along) : along)
            I = K + S - oneunit(S)
            x = Tuple(I) .- 1
            for r in folds
                d = graph.values[r].definition
                scan = d isa Scan
                A = fold_types(d.fold, term_eltype(g.terms[r], graph)).accumulator
                haskey(states, r) || (states[r] = fold_init(d.fold, A))
                scan && d.exclusive && (values[r][I] = observe(r))
                e = evaluate_term(g.terms[r], x, nothing, buffers, values, graph)
                states[r] = fold_combine(d.fold, states[r], fold_lift(d.fold, A(e)))
                scan && !d.exclusive && (values[r][I] = observe(r))
            end
            for m in g.outputs
                values[m][I] = evaluate_term(g.terms[m], x, nothing, buffers, values, graph)
            end
        end
        foreach(r -> graph.values[r].definition isa Reduction && (values[r][K] = observe(r)), folds)
    end
    for m in members(g)
        val = graph.values[m]
        buffers[m] = !(val.definition isa Reduction) ? values[m] : val.scalar ? only(values[m]) : reshape(values[m], val.dims)
    end
end

"""
    execute(plan, inputs...) -> flat results

Run a plan on the CPU.
"""
function execute(p::Plan, inputs...)
    graph = p.graph
    arguments = filter(is_input, leaves(inputs))
    buffers = Vector{Any}(undef, length(graph.values))
    for (id, v) in enumerate(graph.values)
        d = v.definition
        d isa Input && (buffers[id] = shaped(arguments[d.argument], v))
        d isa ConstantArray && (buffers[id] = d.data)
    end
    foreach(g -> run_group!(g, buffers, graph), p.groups)
    map(eachindex(graph.results)) do k
        materialize(graph.results[k], graph.result_dims[k], buffers, graph; scalar = graph.result_scalar[k])
    end
end
