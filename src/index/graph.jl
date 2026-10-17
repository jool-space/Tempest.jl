# The index level: every value is a definition over its own domain, and every
# read of another value goes through an index formula over the reader's domain.
# Layout operations do not exist here; they were absorbed into formulas.

abstract type Operand end

"Read `value` at `index`, one expression per axis of `value`, over the reader's domain."
struct Read <: Operand
    value::Int
    index::Tuple{Vararg{Affine}}
end

"A constant, broadcast over the reader's domain."
struct Literal{T} <: Operand
    value::T
end

"An integer computed from the reader's indices, converted to `T`: positions."
struct IndexOperand{T} <: Operand
    expr::Affine
end

IndexOperand(expr, T::Type) = IndexOperand{T}(expr)

abstract type Definition end

"The `argument`th input of the program."
struct Input <: Definition
    argument::Int
end

"A constant array the program needs at run time, such as the indices of a gather."
struct ConstantArray{A} <: Definition
    data::A
end

"`f` applied to the operands at each index of the value's domain."
struct Pointwise{F} <: Definition
    f::F
    operands::Tuple{Vararg{Operand}}
end

"""
A fold of `operand` over the `reduced` axes of `iteration`. The operand is read
over the iteration domain; the value's own domain is the iteration's other axes.
"""
struct Reduction{F} <: Definition
    fold::F
    operand::Operand
    iteration::Tuple{Vararg{Int}}
    reduced::Tuple{Vararg{Int}}
end

"""
A program-level operation realized on its operands materialized in their
program shapes: one the index level does not model (a scatter), or a call an
implementation claimed, which reads values whole. Its boundary is storage.
`epilogue` says whether the claiming kernel applies an epilogue over this
result's output tile.
"""
struct Opaque <: Definition
    node::Node
    operands::Tuple{Vararg{Operand}}
    shapes::Tuple{Vararg{Tuple{Tuple{Vararg{Int}}, Bool}}} # each operand's dims and scalar flag
    epilogue::Bool
end

Opaque(node, operands, shapes) = Opaque(node, operands, shapes, false)

"""
A value: its definition, the extents of its domain (none of them one; see
`wide_axes`), its element type, and whether it is a number rather than an array.
"""
struct Value
    definition::Definition
    dims::Tuple{Vararg{Int}}
    eltype::DataType
    scalar::Bool
end

domain(v::Value) = Domain(v.dims)
iteration_domain(v::Value) = v.definition isa Reduction ? Domain(v.definition.iteration) : domain(v)

function dense_address(dims, index, d::Domain)
    total = constant(0)
    stride = 1
    for (e, n) in zip(index, dims)
        total = add(total, scale(e, stride), d)
        stride *= n
    end
    total
end

"Whether value `v` read at `index` over a reader of shape `dims` is its storage as it is."
function reads_storage(v::Value, index, dims)
    prod(v.dims; init = 1) == prod(dims; init = 1) || return false
    d = Domain(dims)
    dense_address(v.dims, index, d) == dense_address(dims, identity_index(dims), d)
end

"""
    IndexGraph

A program at the index level: values in topological order, the values that are
arguments, and each result as a read over the result's own shape (results may be
layouts of values, or constants).
"""
struct IndexGraph
    values::Vector{Value}
    arguments::Vector{Int}
    results::Vector{Operand}
    result_dims::Vector{Tuple{Vararg{Int}}}
    result_scalar::Vector{Bool}
    result_eltype::Vector{DataType}
end

operands(d::Definition) = ()
operands(d::Pointwise) = d.operands
operands(d::Reduction) = (d.operand,)
operands(d::Opaque) = d.operands

reads(d::Definition) = (o for o in operands(d) if o isa Read)

function readers(graph::IndexGraph)
    users = [Int[] for _ in graph.values]
    for (id, v) in enumerate(graph.values)
        for r in reads(v.definition)
            push!(users[r.value], id)
        end
    end
    users
end

describe(d::Input) = "input $(d.argument)"
describe(d::ConstantArray) = "constant array $(size(d.data))"
describe(d::Pointwise) = "pointwise($(d.f))"
describe(d::Reduction) = "reduce($(fold_name(d.fold)), reduce $(d.reduced) of $(d.iteration))"
describe(d::Opaque) = "opaque $(op_name(d.node.op))$(op_detail(d.node.op))"

show_operand(io, o::Read) = print(io, "%", o.value, "[", join(map(string, o.index), ", "), "]")
show_operand(io, o::Literal) = print(io, repr(o.value))
show_operand(io, o::IndexOperand{T}) where {T} = print(io, T, "(", o.expr, ")")

function Base.show(io::IO, ::MIME"text/plain", g::IndexGraph)
    println(io, "IndexGraph with $(length(g.values)) values")
    for (id, v) in enumerate(g.values)
        print(io, "  %", id, " = ", describe(v.definition))
        ops = operands(v.definition)
        if !isempty(ops)
            print(io, " ")
            for (k, o) in enumerate(ops)
                k > 1 && print(io, ", ")
                show_operand(io, o)
            end
        end
        println(io, " :: ", v.eltype, v.scalar ? " scalar" : string(v.dims))
    end
    for (k, r) in enumerate(g.results)
        print(io, "  result ", k, " = ")
        show_operand(io, r)
        println(io)
    end
end
