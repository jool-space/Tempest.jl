# A value masked by formulas over its positions: `ifelse(cond, x, literal)`
# with `cond` a conjunction of comparisons between index formulas, a
# position and an integer the program reads among them, or a map of such a
# value that is a known number where the mask fails, as the probabilities
# exp(s - lse) of masked scores s are zero. An implementation
# that sweeps such a value skips the tiles the mask rules out entirely (the
# cuTile generator's `live_steps`), as FlexAttention's block mask does, which
# it derives from the formulas rather than by evaluating the mask on tiles.

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
# conjoins comparisons of index formulas: those of its comparisons that are,
# when others are not, since the conjunction fails wherever one of them does.
function mask_of(graph::IndexGraph, v, index, d::Domain, depth = 0)
    depth < 32 || return nothing
    def = graph.values[v].definition
    def isa Pointwise || return nothing
    through(e) = substitute(e, index, d)
    if def.f === (&)
        parts = filter(!isnothing, map(o -> o isa Read ? mask_of(graph, o.value, map(through, o.index), d, depth + 1) : nothing, def.operands))
        return isempty(parts) ? nothing : reduce(vcat, parts)
    end
    def.f in (<=, <, >=, >) && length(def.operands) == 2 && any(o -> o isa IndexOperand, def.operands) || return nothing
    a, b = map(o -> compared(graph, o, through, d), def.operands)
    a === nothing || b === nothing ? nothing : comparison_mask(def.f, a, b, d)
end

# A side of a comparison with a position, as a formula: a position, an
# integer read from a value (see `held_formula`), or an integer, as a window's
# width is.
compared(graph, o::IndexOperand, through, d) = through(o.expr)
compared(graph, o::Literal{<:Integer}, through, d) = constant(o.value)
compared(graph, o::Read, through, d) = held_formula(o.value, map(through, o.index), graph.values[o.value].eltype, d)
compared(graph, o, through, d) = nothing

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
    masked_result(def.f, val.eltype, map(operand, def.operands))
end
