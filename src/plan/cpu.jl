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
# A fold's value at the domain index `x`, its reduced axes at zero.
function evaluate_term(t::Stat, x, k, buffers, stats, graph)
    s = stats[t.value]
    s isa Number && return s
    d = graph.values[t.value].definition
    axes = d isa Reduction ? d.reduced : ()
    s[ntuple(i -> i in axes ? 1 : x[i] + 1, length(x))...]
end
evaluate_term(t::Apply, x, k, buffers, stats, graph) =
    convert(t.eltype, t.f(map(a -> evaluate_term(a, x, k, buffers, stats, graph), t.args)...))
# A recomputed fold: its body over its reduced axes, after the domain's own.
function evaluate_term(t::Fold, x, k, buffers, stats, graph)
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
    g.kind === :pointwise && return run_pointwise!(g, buffers, graph)
    g.kind === :opaque && return run_opaque!(g, buffers, graph)
    v = only(g.outputs)
    buffers[v] = evaluate_value(graph.values[v].definition, graph.values[v], buffers, graph, ())
end

# An opaque group's results through their reference, then its epilogues: terms
# over a result's domain that read the result's tile as `Stat`.
function run_opaque!(g::Group, buffers, graph)
    results = Dict{Int, Any}()
    for v in g.outputs
        d = graph.values[v].definition
        d isa Opaque || continue
        results[v] = buffers[v] = evaluate_value(d, graph.values[v], buffers, graph, ())
    end
    for (o, e) in g.epilogues
        val = graph.values[e]
        if val.scalar
            buffers[e] = evaluate_term(g.terms[e], (), nothing, buffers, results, graph)
            continue
        end
        out = Array{val.eltype}(undef, val.dims)
        for I in CartesianIndices(val.dims)
            out[I] = evaluate_term(g.terms[e], Tuple(I) .- 1, I, buffers, results, graph)
        end
        buffers[e] = out
    end
end

function run_pointwise!(g::Group, buffers, graph)
    v = only(g.outputs)
    val = graph.values[v]
    t = g.terms[v]
    if val.scalar
        buffers[v] = evaluate_term(t, (), nothing, buffers, nothing, graph)
        return
    end
    out = Array{val.eltype}(undef, val.dims)
    for I in CartesianIndices(val.dims)
        out[I] = evaluate_term(t, Tuple(I) .- 1, I, buffers, nothing, graph)
    end
    buffers[v] = out
end

# Every fold over its own axes, in phase order; then the outputs.
function run_reduction!(g::Group, buffers, graph)
    D = g.domain
    stats = Dict{Int, Any}()
    for r in sort(g.reductions; by = r -> (g.phase[r], r))
        val = graph.values[r]
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
    for m in g.outputs
        val = graph.values[m]
        if val.scalar
            buffers[m] = evaluate_term(g.terms[m], ntuple(_ -> 0, length(D)), nothing, buffers, stats, graph)
            continue
        end
        out = Array{val.eltype}(undef, val.dims)
        for I in CartesianIndices(val.dims)
            out[I] = evaluate_term(g.terms[m], Tuple(I) .- 1, nothing, buffers, stats, graph)
        end
        buffers[m] = out
    end
end

"""
    execute(plan, inputs...) -> flat results

Run a plan on the CPU.
"""
function execute(p::Plan, inputs...)
    graph = p.graph
    arguments = leaves(inputs)
    buffers = Vector{Any}(undef, length(graph.values))
    for (id, v) in enumerate(graph.values)
        d = v.definition
        d isa Input && (buffers[id] = arguments[d.argument])
        d isa ConstantArray && (buffers[id] = d.data)
    end
    foreach(g -> run_group!(g, buffers, graph), p.groups)
    map(eachindex(graph.results)) do k
        materialize(graph.results[k], graph.result_dims[k], buffers, graph; scalar = graph.result_scalar[k])
    end
end
