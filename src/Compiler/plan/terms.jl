# A group's members as terms over its domain: a value the group computes is
# its definition's term, one stored is loaded, a map between is computed in
# place, a fold of the group is read where the group keeps it, and a fold
# recomputed is its operand's term over the domain extended by its axes. A
# read the group cannot serve at the index asked is outside it
# (`OutsideGroup`), and the value does not join.

"The iteration index of fold `d` read at `index`, its value's index, with `inner[j]` along its `j`th reduced axis."
Base.@nospecializeinfer iteration_index(d::Reduction, @nospecialize(index), @nospecialize(inner)) = ntuple(length(d.iteration)) do k
    j = findfirst(==(k), d.reduced)
    j === nothing ? index[k - count(<(k), d.reduced)] : inner[j]
end

struct OutsideGroup <: Exception end

"""
The term computing value `v` at `index`, an index over `domain`: the group's
domain, or inside a recomputed fold that domain extended by the fold's axes.
Each value's term at an index is built once, in `memo`: terms are interned,
so a value read twice is one subterm, and built again at every path to it,
a chain of maps each reading the last twice would take time doubling with
each map.
"""
Base.@nospecializeinfer function value_term(graph, cx::Forming, g::Group, v, @nospecialize(index), domain = Domain(g.domain), memo = Dict{Any, Term}())
    key = (v, index, domain)
    haskey(memo, key) && return memo[key]
    memo[key] = built_term(graph, cx, g, v, index, domain, memo)
end

"""
How a term of a group reads a value: a fold of the group, where the group
keeps it (`as_fold`); another value the group computes, as that value's term
(`as_output`); from storage (`as_load`); a fold recomputed where it is read
(`as_recomputed`); or a map computed in place (`as_inlined`). See `reading`.
"""
@enum Reading as_fold as_output as_load as_recomputed as_inlined

"""
How a term of group `g` reads value `v`, formed through context `cx` (see
`Reading`): what a term is built by (`built_term`) and what it loads is
found by (`value_reads!`), so that the two cannot differ.
"""
function reading(graph, cx::Forming, g::Group, v)
    v in g.reductions && return as_fold
    v in g.outputs && return stepwise(g.kind) ? as_load : as_output
    isstored(cx, v) && return as_load
    d = graph.values[v].definition
    d isa Reduction && return as_recomputed
    d isa Pointwise ? as_inlined : as_load
end

function built_term(graph, cx::Forming, g::Group, v, index, domain, memo)
    tally!(:terms)
    how = reading(graph, cx, g, v)
    how === as_load && return Load(v, index)
    if how === as_fold
        reads_fold(g.kind, graph, g, v, index) || throw(OutsideGroup())
        return Stat(v)
    end
    how === as_recomputed && return fold_term(graph, cx, g, v, index, domain, memo)
    # A value the group computes is its term where the group keeps it.
    how === as_output && index != placement(graph, g, v) && throw(OutsideGroup())
    val = graph.values[v]
    definition_term(graph, cx, g, val.definition, val, index, domain, memo)
end

function definition_term(graph, cx::Forming, g, d::Pointwise, val, index, domain, memo = Dict{Any, Term}())
    args = map(d.operands) do o
        operand_term(graph, cx, g, o, index, domain, memo)
    end
    Apply(d.f, args, val.eltype)
end

# A fold recomputed where it is read: its operand over the domain extended by
# its reduced axes.
function fold_term(graph, cx::Forming, g, v, index, domain, memo = Dict{Any, Term}())
    d = graph.values[v].definition::Reduction
    base = length(domain)
    extents = Tuple(d.iteration[k] for k in d.reduced)
    iteration = iteration_index(d, index, ntuple(j -> var(base + j), length(d.reduced)))
    extended = Domain((domain.extents..., extents...))
    Recomputed(v, index, extents, operand_term(graph, cx, g, d.operand, iteration, extended, memo))
end

# The values `value_term` and `operand_term` would load for `v` or `o`, the
# term's `stored_reads`, added to `reads`: found by the same rules from the
# graph alone, each value visited once where the term repeats it at every
# reader. Its formulas' gathers are the same at any index.
function value_reads!(reads, seen, graph, cx::Forming, g::Group, v)
    v in seen && return reads
    push!(seen, v)
    how = reading(graph, cx, g, v)
    how === as_fold && return reads
    how === as_load && return push!(reads, v)
    d = graph.values[v].definition
    how === as_recomputed && return operand_reads!(reads, seen, graph, cx, g, d.operand)
    foreach(o -> operand_reads!(reads, seen, graph, cx, g, o), operands(d))
    reads
end

function operand_reads!(reads, seen, graph, cx::Forming, g, o)
    union!(reads, gathered_values(o))
    o isa Read && value_reads!(reads, seen, graph, cx, g, o.value)
    reads
end

operand_term(graph, cx::Forming, g, o::Read, index, domain, memo = Dict{Any, Term}()) =
    value_term(graph, cx, g, o.value, map(e -> substitute(e, index, domain), o.index), domain, memo)
operand_term(graph, cx::Forming, g, o::Literal, index, domain, memo = nothing) = Lit(o.value)
operand_term(graph, cx::Forming, g, o::IndexOperand{T}, index, domain, memo = nothing) where {T} =
    Ix{T}(substitute(o.expr, index, domain))

# An operand read from storage, as a term over its reader's domain: the
# contents a fold by key starts from.
stored_term(o::Read) = Load(o.value, o.index)
stored_term(o::Literal) = Lit(o.value)
stored_term(o::IndexOperand{T}) where {T} = Ix{T}(o.expr)
