# Scalar code for a kernel: the floor that realizes any group, one item at a
# time. An item is an element of a pointwise group's domain, or a position
# along a reduction group's kept axes, whose reductions run sequentially. The
# same code runs on the CPU in a loop and on CUDA one thread per item.
#
# Buffers are dense and one-dimensional; the generated function receives them
# as a tuple, in the order of `slots`.

struct ScalarKernel
    slots::Vector{Int}
    items::Int
    body::Expr # computes item `lin` (zero-based)
end

index_vars(n) = [Symbol(:i, k) for k in 1:n]

function dense_strides(dims)
    s = 1
    map(dims) do n
        stride = s
        s *= n
        stride
    end
end

# The linear position of `index` in the dense storage of value `v`.
function address(graph, v, index, d::Domain)
    val = graph.values[v]
    (val.scalar || isempty(val.dims)) && return constant(0)
    total = constant(0)
    for (e, s) in zip(index, dense_strides(val.dims))
        total = add(total, scale(e, s), d)
    end
    total
end

struct EmitContext
    graph::IndexGraph
    slots::Vector{Int}
    vars::Vector{Symbol}
    domain::Domain
end

buffer(c::EmitContext, v) = :(buffers[$(findfirst(==(v), c.slots))])

function affine_expr(e::Affine, c)
    terms = [k == 1 ? atom_expr(a, c) : :($k * $(atom_expr(a, c))) for (a, k) in e.terms]
    isempty(terms) && return e.constant
    sum = foldl((acc, t) -> :($acc + $t), terms)
    e.constant == 0 ? sum : :($sum + $(e.constant))
end
atom_expr(a::Var, c) = c.vars[a.n]
atom_expr(a::Div, c) = :(fld($(affine_expr(a.e, c)), $(a.d)))
atom_expr(a::Mod, c) = :(mod($(affine_expr(a.e, c)), $(a.d)))
atom_expr(a::Indirect, c) =
    :(Int(@inbounds $(buffer(c, a.value))[$(affine_expr(address(c.graph, a.value, a.index, c.domain), c)) + 1]))

term_expr(t::Load, names, c) =
    :(@inbounds $(buffer(c, t.value))[$(affine_expr(address(c.graph, t.value, t.index, c.domain), c)) + 1])
term_expr(t::Lit, names, c) = t.value
term_expr(t::Ix{T}, names, c) where {T} = :($T($(affine_expr(t.expr, c))))
term_expr(t::Stat, names, c) = Symbol(:stat_, t.value)
term_expr(t::Apply, names, c) = :(convert($(t.eltype), $(t.f)($(map(a -> names[a], t.args)...))))

# Statements computing `roots`, each subterm once; returns the names.
function emit_terms!(statements, roots, c; names = Dict{Term, Any}())
    for t in subterms(roots)
        haskey(names, t) && continue
        if t isa Lit
            names[t] = t.value
            continue
        end
        name = gensym(:t)
        push!(statements, :($name = $(term_expr(t, names, c))))
        names[t] = name
    end
    names
end

function delinearize_expr(statements, lin, axes, extents, vars)
    rest = gensym(:rest)
    push!(statements, :($rest = $lin))
    for k in axes
        push!(statements, :($(vars[k]) = rem($rest, $(extents[k]))))
        push!(statements, :($rest = div($rest, $(extents[k]))))
    end
end

store_expr(c, v, index, value) =
    :(@inbounds $(buffer(c, v))[$(affine_expr(address(c.graph, v, index, c.domain), c)) + 1] = $value)

function scalar_kernel(graph, stored, g::Group, slots)
    D = g.domain
    vars = index_vars(length(D))
    c = EmitContext(graph, slots, vars, Domain(D))
    body = Expr(:block)
    s = body.args
    if g.kind === :pointwise
        delinearize_expr(s, :lin, 1:length(D), D, vars)
        names = emit_terms!(s, [g.terms[m] for m in g.outputs], c)
        for m in g.outputs
            push!(s, store_expr(c, m, identity_index(D), names[g.terms[m]]))
        end
        return ScalarKernel(slots, prod(D; init = 1), body)
    end
    # Items are the positions of the primary folds; other folds get launches of
    # their own (`fold_kernel`).
    reduced = collect(primary_axes(graph, g))
    kept = [k for k in 1:length(D) if !(k in reduced)]
    folds = [r for r in g.reductions if primary(graph, g, r)]
    delinearize_expr(s, :lin, kept, D, vars)
    count = prod(k -> D[k], reduced; init = 1)
    for phase in sort!(unique(g.phase[r] for r in folds))
        phase_reductions = [r for r in folds if g.phase[r] == phase]
        loop = Expr(:block)
        states = Dict{Int, Symbol}()
        for r in phase_reductions
            fold = graph.values[r].definition.fold
            A = fold_types(fold, term_eltype(g.terms[r], graph)).accumulator
            states[r] = Symbol(:state_, r)
            push!(s, :($(states[r]) = $fold_init($fold, $A)))
        end
        delinearize_expr(loop.args, :sweep, reduced, D, vars)
        names = emit_terms!(loop.args, [g.terms[r] for r in phase_reductions], c)
        for r in phase_reductions
            fold = graph.values[r].definition.fold
            A = fold_types(fold, term_eltype(g.terms[r], graph)).accumulator
            push!(loop.args, :($(states[r]) = $fold_combine($fold, $(states[r]), $fold_lift($fold, convert($A, $(names[g.terms[r]]))))))
        end
        push!(s, :(for sweep in 0:$(count - 1)
            $loop
        end))
        for r in phase_reductions
            val = graph.values[r]
            push!(s, :($(Symbol(:stat_, r)) = convert($(val.eltype), $fold_observe($(val.definition.fold), $(states[r])))))
        end
    end
    # Reductions other kernels read, and outputs shaped like a reduction, once.
    foreach(k -> push!(s, :($(vars[k]) = 0)), reduced)
    for r in folds
        stored[r] && push!(s, store_expr(c, r, placement(graph, g, r), Symbol(:stat_, r)))
    end
    statistic_outputs = [m for m in g.outputs if graph.values[m].dims != D && !(m in g.reductions)]
    names = emit_terms!(s, [g.terms[m] for m in statistic_outputs], c)
    for m in statistic_outputs
        push!(s, store_expr(c, m, placement(graph, g, m), names[g.terms[m]]))
    end
    full_outputs = [m for m in g.outputs if graph.values[m].dims == D && !(m in g.reductions)]
    if !isempty(full_outputs)
        loop = Expr(:block)
        delinearize_expr(loop.args, :sweep, reduced, D, vars)
        names = emit_terms!(loop.args, [g.terms[m] for m in full_outputs], c)
        for m in full_outputs
            push!(loop.args, store_expr(c, m, identity_index(D), names[g.terms[m]]))
        end
        push!(s, :(for sweep in 0:$(count - 1)
            $loop
        end))
    end
    ScalarKernel(slots, prod(k -> D[k], kept; init = 1), body)
end

# A term with the group's reductions read from storage, at their placements,
# for a launch that runs after the group's own.
stored_statistics(t::Stat, graph, g) = Load(t.value, placement(graph, g, t.value))
stored_statistics(t::Apply, graph, g) = Apply(t.f, map(a -> stored_statistics(a, graph, g), t.args), t.eltype)
stored_statistics(t::Term, graph, g) = t

# A fold the group's items do not cover, on the floor: one item per position of
# its kept axes, sweeping its reduced axes and recomputing its operand.
function fold_kernel(graph, g::Group, slots, r)
    D = g.domain
    vars = index_vars(length(D))
    c = EmitContext(graph, slots, vars, Domain(D))
    val = graph.values[r]
    d = val.definition
    fold = d.fold
    t = stored_statistics(g.terms[r], graph, g)
    A = fold_types(fold, term_eltype(t, graph)).accumulator
    kept = [k for k in 1:length(D) if !(k in d.reduced)]
    reduced = collect(d.reduced)
    body = Expr(:block)
    s = body.args
    delinearize_expr(s, :lin, kept, D, vars)
    loop = Expr(:block)
    delinearize_expr(loop.args, :sweep, reduced, D, vars)
    names = emit_terms!(loop.args, [t], c)
    push!(loop.args, :(state = $fold_combine($fold, state, $fold_lift($fold, convert($A, $(names[t]))))))
    push!(s, :(state = $fold_init($fold, $A)))
    push!(s, :(for sweep in 0:$(prod(k -> D[k], reduced; init = 1) - 1)
        $loop
    end))
    foreach(k -> push!(s, :($(vars[k]) = 0)), reduced)
    index = val.scalar ? () : identity_index(ntuple(k -> k in d.reduced ? 1 : D[k], length(D)))
    push!(s, store_expr(c, r, index, :(convert($(val.eltype), $fold_observe($fold, state)))))
    ScalarKernel(slots, prod(k -> D[k], kept; init = 1), body)
end

"The floor's launches for a group: its kernel, then one per fold over other axes."
scalar_kernels(graph, stored, g::Group, slots) =
    [scalar_kernel(graph, stored, g, slots); [fold_kernel(graph, g, slots, r) for r in g.reductions if !primary(graph, g, r)]]
