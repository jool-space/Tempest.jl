# Scalar code for a kernel: the floor that realizes any group, one item at a
# time. An item is an element of a pointwise group's domain, or a position
# along a reduction group's kept axes, whose reductions run sequentially. The
# same code runs on the CPU in a loop and on CUDA one thread per item.

"""
A kernel of the floor: `body` computes item `lin`, zero-based, of `items`,
over the buffers of the values `slots`, dense and one-dimensional, which the
function generated from the body receives as a tuple in that order (see
`floor_launch`).
"""
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
    stats::Dict{Int, Symbol} # the variable of each reduction's result, named when first used
end

EmitContext(graph, slots, vars, domain) = EmitContext(graph, slots, vars, domain, Dict{Int, Symbol}())

# Variables are generated names, numbered by position when the kernel is keyed,
# so that identical groups generate identical kernels whatever their value ids.
stat_name(c::EmitContext, r) = get!(() -> gensym(:stat), c.stats, r)

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
term_expr(t::Stat, names, c) = stat_name(c, t.value)
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
    nested = EmitContext(c.graph, c.slots, vars, Domain(extents), c.stats)
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
# `hoisted` instead, computed once before the loop of `statements`, unless
# `moves` says they change along the loop all the same, as a scan read where
# the sweep has it does.
function emit_terms!(statements, roots, c; names = Dict{Term, Any}(), hoisted = nothing, swept = (), moves = t -> false)
    for t in subterms(roots)
        haskey(names, t) && continue
        if hoisted !== nothing && !any(k -> mentions(t, k), swept) && !moves(t)
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
    g.kind === :scan && return scan_kernel(graph, stored, g, slots)
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
            states[r] = gensym(:state)
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
            push!(s, :($(stat_name(c, r)) = convert($(val.eltype), $fold_observe($(val.definition.fold), $(states[r])))))
        end
    end
    # Reductions other kernels read, and outputs shaped like a reduction, once.
    foreach(k -> push!(s, :($(vars[k]) = 0)), reduced)
    for r in folds
        stored[r] && push!(s, store_expr(c, r, placement(graph, g, r), stat_name(c, r)))
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

# A scan's group: an item per position of its other axes, which sweeps the
# scanned axes in the scans' direction. At each position every scan folds its
# element, in phase order, and the outputs there are computed and stored.
function scan_kernel(graph, stored, g::Group, slots)
    D = g.domain
    vars = index_vars(length(D))
    c = EmitContext(graph, slots, vars, Domain(D))
    first_scan = graph.values[first(g.reductions)].definition
    swept = collect(first_scan.axes)
    kept = [k for k in 1:length(D) if !(k in swept)]
    count = prod(k -> D[k], swept; init = 1)
    body = Expr(:block)
    s = body.args
    delinearize_expr(s, :lin, kept, D, vars)
    scans = sort(g.reductions; by = r -> (g.phase[r], r))
    states = Dict(r => gensym(:state) for r in scans)
    accumulators = Dict(r => fold_types(graph.values[r].definition.fold, term_eltype(g.terms[r], graph)).accumulator for r in scans)
    foreach(r -> push!(s, :($(states[r]) = $fold_init($(graph.values[r].definition.fold), $(accumulators[r])))), scans)
    loop = Expr(:block)
    delinearize_expr(loop.args, first_scan.reverse ? :($(count - 1) - sweep) : :sweep, swept, D, vars)
    names = Dict{Term, Any}()
    # What reads a scan changes along the sweep, though no index moves it.
    moves(t) = !isempty(stats(t))
    for r in scans
        val = graph.values[r]
        fold, A = val.definition.fold, accumulators[r]
        observed = :(convert($(val.eltype), $fold_observe($fold, $(states[r]))))
        val.definition.exclusive && push!(loop.args, :($(stat_name(c, r)) = $observed))
        emit_terms!(loop.args, [g.terms[r]], c; names, hoisted = s, swept, moves)
        push!(loop.args, :($(states[r]) = $fold_combine($fold, $(states[r]), $fold_lift($fold, convert($A, $(names[g.terms[r]]))))))
        val.definition.exclusive || push!(loop.args, :($(stat_name(c, r)) = $observed))
        stored[r] && push!(loop.args, store_expr(c, r, identity_index(D), stat_name(c, r)))
    end
    emit_terms!(loop.args, [g.terms[m] for m in g.outputs], c; names, hoisted = s, swept, moves)
    foreach(m -> push!(loop.args, store_expr(c, m, identity_index(D), names[g.terms[m]])), g.outputs)
    push!(s, :(for sweep in 0:$(count - 1)
        $loop
    end))
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

"""
    accumulate!(buffer, i, v)

Add `v` to element `i` of a buffer: in place on the CPU, where a floor
kernel's items run in order, and atomically on a device (see CUDAExt), where
they run at once and the sums come in no fixed order.
"""
accumulate!(buffer, i, v) = (@inbounds buffer[i] += v; nothing)

# A scatter on the floor: the result zeroed, then one item per element
# scattered, which adds the element at the position its indices name. An
# integer or a range among the indices names the position along its axis; an
# index array is read at the item's position among the array's own axes,
# which are axes of the item domain, as a gather. The work is the elements
# scattered, once each.
function scatter_kernels(graph, g::Group, slots)
    v = only(g.outputs)
    d = graph.values[v].definition
    n, op = d.node, d.node.op
    T = n.eltype
    slot(w) = :(buffers[$(findfirst(==(w), slots))])
    zeroing = ScalarKernel(slots, storage_length(graph.values[v]), :(@inbounds $(slot(v))[lin + 1] = $(zero(T))))
    # The item domain: an axis per range, and each index array's own axes.
    extents = Int[]
    placed = Any[]
    for i in op.indices
        if i isa Integer
            push!(placed, i)
        elseif i isa AbstractRange
            push!(extents, length(i))
            push!(placed, (i, length(extents)))
        else
            first_axis = length(extents) + 1
            append!(extents, d.shapes[i.input][1])
            push!(placed, (i, first_axis:length(extents)))
        end
    end
    domain = Domain(Tuple(extents))
    # Where each item's element comes from, as the scattered operand's axes,
    # and where it lands, zero-based along each axis of the result.
    selected = Affine[]
    target = Affine[]
    for (axis, entry) in enumerate(placed)
        if entry isa Integer
            push!(target, constant(entry - 1))
        elseif entry[1] isa AbstractRange
            r, j = entry
            push!(selected, var(j))
            push!(target, add(mul(var(j), step(r), domain), first(r) - 1, domain))
        else
            i, axes = entry
            append!(selected, map(var, axes))
            read = through(d.operands[i.input], Tuple(map(var, axes)), domain)
            push!(target, add(normalize(0, (Indirect(read.value, read.index, 1, n.dims[axis]) => 1,), domain), -1, domain))
        end
    end
    c = EmitContext(graph, slots, [gensym(:j) for _ in extents], domain)
    body = Expr(:block)
    delinearize_expr(body.args, :lin, 1:length(extents), extents, c.vars)
    element = scatter_term(first(d.operands), Tuple(selected), domain)
    names = emit_terms!(body.args, [element], c)
    at = Tuple(target[k] for k in wide_axes(n.dims))
    push!(body.args, :($accumulate!($(slot(v)), $(affine_expr(address(graph.values[v], at, domain), c)) + 1, $(names[element]))))
    [zeroing, ScalarKernel(slots, prod(extents; init = 1), body)]
end

"The floor's launches for a group: its kernel, then one per fold over other axes; a scatter's two."
scalar_kernels(graph, stored, g::Group, slots) = g.kind === :opaque ? scatter_kernels(graph, g, slots) :
    [scalar_kernel(graph, stored, g, slots); [fold_kernel(graph, g, slots, r) for r in g.reductions if !primary(graph, g, r)]]
