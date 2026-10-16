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
evaluate_term(t::Stat, x, k, buffers, stats, graph) = stats[t.value] isa Number ? stats[t.value] : stats[t.value][k]
evaluate_term(t::Apply, x, k, buffers, stats, graph) =
    convert(t.eltype, t.f(map(a -> evaluate_term(a, x, k, buffers, stats, graph), t.args)...))

function run_group!(g::Group, buffers, graph)
    g.kind === :reduction && return run_reduction!(g, buffers, graph)
    g.kind === :pointwise && return run_pointwise!(g, buffers, graph)
    v = only(g.outputs)
    buffers[v] = evaluate_value(graph.values[v].definition, graph.values[v], buffers, graph, ())
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

function run_reduction!(g::Group, buffers, graph)
    D = g.domain
    kept = ntuple(k -> k in g.reduced ? 1 : D[k], length(D))
    swept = ntuple(k -> k in g.reduced ? D[k] : 1, length(D))
    stats = Dict{Int, Any}()
    for phase in sort!(unique(g.phase[r] for r in g.reductions))
        for r in g.reductions
            g.phase[r] == phase || continue
            val = graph.values[r]
            fold = val.definition.fold
            A = fold_types(fold, term_eltype(g.terms[r], graph)).accumulator
            out = Array{val.eltype}(undef, kept)
            for K in CartesianIndices(kept)
                state = fold_init(fold, A)
                for S in CartesianIndices(swept)
                    x = ntuple(k -> k in g.reduced ? S[k] - 1 : K[k] - 1, length(D))
                    e = evaluate_term(g.terms[r], x, K, buffers, stats, graph)
                    state = fold_combine(fold, state, fold_lift(fold, A(e)))
                end
                out[K] = val.eltype(fold_observe(fold, state))
            end
            stats[r] = val.scalar ? only(out) : reshape(out, val.dims)
            buffers[r] = stats[r]
        end
    end
    for m in g.outputs
        val = graph.values[m]
        full = val.dims == D
        if val.scalar
            buffers[m] = evaluate_term(g.terms[m], ntuple(_ -> 0, length(D)), CartesianIndex(), buffers, stats, graph)
            continue
        end
        out = Array{val.eltype}(undef, val.dims)
        for I in CartesianIndices(full ? D : kept)
            K = CartesianIndex(ntuple(k -> k in g.reduced ? 1 : I[k], length(D)))
            out[full ? I : CartesianIndex(Tuple(I)[1:length(val.dims)])] =
                evaluate_term(g.terms[m], Tuple(I) .- 1, K, buffers, stats, graph)
        end
        buffers[m] = out
    end
    for r in g.cross
        val = graph.values[r]
        d = val.definition
        fold = d.fold
        A = fold_types(fold, term_eltype(g.terms[r], graph)).accumulator
        kept_c = ntuple(k -> k in d.reduced ? 1 : D[k], length(D))
        swept_c = ntuple(k -> k in d.reduced ? D[k] : 1, length(D))
        out = Array{val.eltype}(undef, kept_c)
        for Kc in CartesianIndices(kept_c)
            state = fold_init(fold, A)
            for Sc in CartesianIndices(swept_c)
                x = ntuple(k -> k in d.reduced ? Sc[k] - 1 : Kc[k] - 1, length(D))
                K = CartesianIndex(ntuple(k -> k in g.reduced ? 1 : x[k] + 1, length(D)))
                e = evaluate_term(g.terms[r], x, K, buffers, stats, graph)
                state = fold_combine(fold, state, fold_lift(fold, A(e)))
            end
            out[Kc] = val.eltype(fold_observe(fold, state))
        end
        buffers[r] = val.scalar ? only(out) : reshape(out, val.dims)
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
