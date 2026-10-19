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
    :(Int(@inbounds $(buffer(c, a.value))[$(affine_expr(address(c.graph.values[a.value], a.index, c.domain), c)) + 1]))

term_expr(t::Load, names, c) =
    :(@inbounds $(buffer(c, t.value))[$(affine_expr(address(c.graph.values[t.value], t.index, c.domain), c)) + 1])
term_expr(t::Lit, names, c) = t.value
term_expr(t::Ix{T}, names, c) where {T} = :($T($(affine_expr(t.expr, c))))
term_expr(t::Stat, names, c) = Symbol(:stat_, t.value)
term_expr(t::Apply, names, c) = :(convert($(t.eltype), $(t.f)($(map(a -> names[a], t.args)...))))

# A recomputed fold: a loop over its reduced axes, the variables after the
# domain's own, computing its body at each.
function term_expr(t::Recomputed, names, c)
    val = c.graph.values[t.value]
    fold = val.definition.fold
    A = fold_types(fold, term_eltype(t.body, c.graph)).accumulator
    base = length(c.vars)
    inner = base .+ (1:length(t.extents))
    vars = [c.vars; [gensym(:i) for _ in inner]]
    extents = (c.domain.extents..., t.extents...)
    nested = EmitContext(c.graph, c.slots, vars, Domain(extents))
    state, j = gensym(:state), gensym(:j)
    loop = Expr(:block)
    delinearize_expr(loop.args, j, inner, extents, vars)
    body = emit_terms!(loop.args, [t.body], nested)
    push!(loop.args, :($state = $fold_combine($fold, $state, $fold_lift($fold, convert($A, $(body[t.body]))))))
    quote
        $state = $fold_init($fold, $A)
        for $j in 0:$(prod(t.extents; init = 1) - 1)
            $loop
        end
        convert($(val.eltype), $fold_observe($fold, $state))
    end
end

# Statements computing `roots`, each subterm once; returns the names. With
# `hoisted`, the subterms that do not depend on the variables `swept` go to
# `hoisted` instead, computed once before the loop of `statements`.
function emit_terms!(statements, roots, c; names = Dict{Term, Any}(), hoisted = nothing, swept = ())
    for t in subterms(roots)
        haskey(names, t) && continue
        if hoisted !== nothing && !any(k -> mentions(t, k), swept)
            emit_terms!(hoisted, [t], c; names)
            continue
        end
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
    :(@inbounds $(buffer(c, v))[$(affine_expr(address(c.graph.values[v], index, c.domain), c)) + 1] = $value)

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
        names = emit_terms!(loop.args, [g.terms[r] for r in phase_reductions], c; hoisted = s, swept = reduced)
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
    push!(s, store_expr(c, r, placement(graph, g, r), :(convert($(val.eltype), $fold_observe($fold, state)))))
    ScalarKernel(slots, prod(k -> D[k], kept; init = 1), body)
end

# An operand of the scatter at `index`, as a term.
scatter_term(o::Read, index, d) = Load(o.value, map(e -> substitute(e, index, d), o.index))
scatter_term(o::Literal, index, d) = Lit(o.value)
scatter_term(o::IndexOperand{T}, index, d) where {T} = Ix{T}(substitute(o.expr, index, d))

# A scatter on the floor, deterministic: an item per element of the result,
# summing the elements selected at its position in the order of their linear
# index, as the interpreter does. An integer or a range among the indices
# constrains the position; each index array is scanned for it, the last
# outermost. The work is the result's length times the index arrays'.
function scatter_kernel(graph, g::Group, slots)
    v = only(g.outputs)
    d = graph.values[v].definition
    n, op = d.node, d.node.op
    T = n.eltype
    # Variables: the result's axes, then each range's position along it and
    # each index array's axes, which are the selection's axes.
    extents = collect(n.dims)
    vars = index_vars(length(n.dims))
    selected = Affine[]
    ranges = Expr[]
    checks = Expr[]
    scans = Tuple{UnitRange{Int}, Int, Int}[]
    for (axis, i) in enumerate(op.indices)
        if i isa Integer
            push!(checks, :($(vars[axis]) == $(i - 1)))
        elseif i isa AbstractRange
            push!(extents, length(i))
            push!(vars, gensym(:j))
            offset = gensym(:offset)
            push!(ranges, step(i) > 0 ? :($offset = $(vars[axis]) - $(first(i) - 1)) : :($offset = $(first(i) - 1) - $(vars[axis])))
            push!(ranges, :($(last(vars)) = fld($offset, $(abs(step(i))))))
            push!(checks, :(mod($offset, $(abs(step(i)))) == 0), :(0 <= $(last(vars)) < $(length(i))))
            push!(selected, var(length(extents)))
        else
            first_axis = length(extents) + 1
            append!(extents, d.shapes[i.input][1])
            append!(vars, [gensym(:n) for _ in d.shapes[i.input][1]])
            push!(scans, (first_axis:length(extents), i.input, axis))
            append!(selected, map(var, first_axis:length(extents)))
        end
    end
    domain = Domain(Tuple(extents))
    c = EmitContext(graph, slots, vars, domain)
    acc = gensym(:acc)
    element = scatter_term(first(d.operands), selected, domain)
    inner = :($acc = convert($T, $acc + $(term_expr(element, nothing, c))))
    for (axes, input, axis) in scans
        lin = gensym(:lin)
        loop = Expr(:block)
        delinearize_expr(loop.args, lin, axes, extents, vars)
        index = term_expr(scatter_term(d.operands[input], Tuple(map(var, axes)), domain), nothing, c)
        push!(loop.args, :(if Int($index) - 1 == $(vars[axis])
            $inner
        end))
        inner = :(for $lin in 0:$(prod(extents[axes]; init = 1) - 1)
            $loop
        end)
    end
    body = Expr(:block)
    delinearize_expr(body.args, :lin, 1:length(n.dims), extents, vars)
    append!(body.args, ranges)
    push!(body.args, :($acc = zero($T)))
    push!(body.args, isempty(checks) ? inner : :(if $(foldl((a, b) -> :($a && $b), checks))
        $inner
    end))
    at = map(e -> substitute(e, Tuple(map(var, wide_axes(n.dims))), domain), d.index)
    push!(body.args, store_expr(c, v, at, acc))
    ScalarKernel(slots, prod(n.dims; init = 1), body)
end

"The floor's launches for a group: its kernel, then one per fold over other axes; a scatter's kernel."
scalar_kernels(graph, stored, g::Group, slots) = g.kind === :opaque ? [scatter_kernel(graph, g, slots)] :
    [scalar_kernel(graph, stored, g, slots); [fold_kernel(graph, g, slots, r) for r in g.reductions if !primary(graph, g, r)]]
