# A value masked by formulas over its positions: `ifelse(cond, x, literal)`
# with `cond` a conjunction of comparisons between index formulas, or a map
# of such a value that is a known number where the mask fails, as the
# probabilities exp(s - lse) of masked scores s are zero. An implementation
# that sweeps such a value skips the tiles the mask rules out entirely (the
# cuTile generator's `live_steps`), as FlexAttention's block mask does, which
# it derives from the formulas rather than by evaluating the mask on tiles.

# A mask: formulas over a value's axes, each at most zero where the mask
# holds. `a <= b` is `a - b <= 0`, `a < b` is `a - b + 1 <= 0`.
function comparison(f, a::Affine, b::Affine, d::Domain)
    f in (<=, <) || ((a, b) = (b, a))
    e = add(a, scale(b, -1), d)
    [f in (<, >) ? add(e, 1, d) : e]
end

"""
    masked(graph, v) -> (mask, value) or nothing

The formulas over `v`'s axes that mask it, each at most zero where the mask
holds, and the number `v` is where the mask fails; nothing when `v` is no
such value.
"""
function masked(graph::IndexGraph, v)
    d = Domain(graph.values[v].dims)
    index = Tuple(var(k) for k in 1:length(graph.values[v].dims))
    found = Vector{Affine}[]
    value_masks!(found, graph, v, index, d, 0)
    for m in unique(found)
        x = masked_value(graph, v, index, m, d, 0)
        x isa Number && return (m, x)
    end
    nothing
end

# The mask a value read at `index` is a condition for, when its definition
# conjoins comparisons of index formulas.
function mask_of(graph::IndexGraph, v, index, d::Domain, depth = 0)
    depth < 32 || return nothing
    def = graph.values[v].definition
    def isa Pointwise || return nothing
    through(e) = substitute(e, index, d)
    if def.f === (&)
        parts = map(o -> o isa Read ? mask_of(graph, o.value, map(through, o.index), d, depth + 1) : nothing, def.operands)
        return any(isnothing, parts) ? nothing : reduce(vcat, parts)
    end
    def.f in (<=, <, >=, >) && length(def.operands) == 2 && all(o -> o isa IndexOperand, def.operands) || return nothing
    comparison(def.f, through(def.operands[1].expr), through(def.operands[2].expr), d)
end

# The masks of the `ifelse` definitions `v`, read at `index`, is computed
# from, through the pointwise values it reads.
function value_masks!(found, graph::IndexGraph, v, index, d::Domain, depth)
    depth < 32 || return
    def = graph.values[v].definition
    def isa Pointwise || return
    through(e) = substitute(e, index, d)
    if def.f === ifelse && length(def.operands) == 3 && def.operands[1] isa Read && def.operands[3] isa Literal
        o = def.operands[1]
        m = mask_of(graph, o.value, map(through, o.index), d)
        m === nothing || push!(found, m)
    end
    for o in def.operands
        o isa Read && value_masks!(found, graph, o.value, map(through, o.index), d, depth + 1)
    end
end

# What `v`, read at `index`, is where `mask` fails, when the mask's literal
# decides it: a number; `:finite` for an input, a fold, an index, and for
# what sums, scales, exponentiates or converts those; nothing when unknown.
function masked_value(graph::IndexGraph, v, index, mask, d::Domain, depth)
    depth < 32 || return nothing
    val = graph.values[v]
    def = val.definition
    def isa Pointwise || return :finite
    through(e) = substitute(e, index, d)
    operand(o) = o isa Literal ? o.value : o isa IndexOperand ? :finite :
        o isa Read ? masked_value(graph, o.value, map(through, o.index), mask, d, depth + 1) : nothing
    if def.f === ifelse && length(def.operands) == 3 && def.operands[1] isa Read
        o = def.operands[1]
        mask_of(graph, o.value, map(through, o.index), d) == mask && return operand(def.operands[3])
    end
    combine(def.f, val.eltype, map(operand, def.operands))
end

# `f` of arguments each a number, `:finite` or nothing (unknown): a number
# when the numbers decide it, as a zero factor or an infinite summand does,
# in exact arithmetic; `:finite` for sums, products and the functions here of
# finite arguments, which an overflow could make infinite, so that a program
# whose unmasked positions overflow computes a different nothing here.
function combine(f, T, args)
    any(isnothing, args) && return nothing
    all(a -> a isa Number, args) && return f isa Cast ? convert(T, only(args)) : f === ifelse ? nothing : f(args...)
    known = [a for a in args if a isa Number]
    if f === (*)
        any(iszero, known) && return zero(T)
        return all(isfinite, known) ? :finite : nothing
    elseif f === (+)
        infinite = filter(isinf, known)
        return isempty(infinite) ? :finite : length(infinite) == 1 ? convert(T, only(infinite)) : nothing
    elseif f === (-) && length(args) == 2
        a, b = args
        a isa Number && isinf(a) && return b isa Number && isinf(b) ? nothing : convert(T, a)
        b isa Number && isinf(b) && return convert(T, -b)
        return :finite
    end
    f in (exp, tanh, abs, abs2, max, min) || f isa Cast ? (all(a -> a === :finite || isfinite(a), args) ? :finite : nothing) : nothing
end
