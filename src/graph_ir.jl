# Operations carry mathematical meaning; nodes add operand ids and result metadata.
# This is host-side compiler data. A heterogeneous tape does not need to specialize
# on its entire graph, but each operation should have a meaningful payload type.
abstract type Operation end

abstract type LayoutOperation <: Operation end

struct Argument <: Operation end

struct Constant{T <: Real} <: Operation
    value::T
end

struct MatMul <: Operation end

struct Pointwise{F} <: Operation
    f::F
end

"""
    Fold

A reduction without a schedule. `fold_lift(f, x)` maps one element to a state
tuple, `fold_combine(f, a, b)` merges two states, `fold_observe(f, state)` maps
a final state to the result, and `fold_init(f, T)` is the identity state for
elements of type `T`. Backends group elements in any fixed order, so combine
must be associative and commutative up to rounding. See `src/folds.jl`.
"""
abstract type Fold end

struct FoldReduction{F <: Fold, A <: Union{Colon, Tuple{Vararg{Int}}}} <: Operation
    fold::F
    axes::A
end

struct SumFold <: Fold end

# A sum is the fold whose state is the running total. Backends that have a
# dedicated sum path dispatch on this alias.
const SumReduction{A <: Union{Colon, Tuple{Vararg{Int}}}} = FoldReduction{SumFold, A}
SumReduction(axes) = FoldReduction(SumFold(), axes)

struct TransposeOp <: LayoutOperation end
struct ReshapeOp <: LayoutOperation end

struct PermuteOp{P <: Tuple{Vararg{Int}}} <: LayoutOperation
    permutation::P
end

struct ViewOp{S <: Selection} <: LayoutOperation
    selection::S
end

struct SliceOp{S <: Selection} <: LayoutOperation
    selection::S
end

struct ScatterOp{S <: Selection} <: Operation
    selection::S
end

struct SemanticCall{F, K <: NamedTuple, S, R}
    f::F
    kwargs::K
    specs::S
    results::R
    origin::Base.RefValue{Nothing} # identity survives graph copies and preparation retries
end

struct CallResult{C <: SemanticCall} <: Operation
    call::C
end

# Diagnostic labels are not an operation dispatch protocol.
operation_name(op::Operation) = Symbol(nameof(typeof(op)))
operation_name(::Argument) = :argument
operation_name(::Constant) = :constant
operation_name(::MatMul) = :matmul
operation_name(::Pointwise) = :pointwise
operation_name(op::FoldReduction) = fold_name(op.fold)
operation_name(::TransposeOp) = :transpose
operation_name(::ReshapeOp) = :reshape
operation_name(::PermuteOp) = :permute
operation_name(::ViewOp) = :view
operation_name(::SliceOp) = :slice
operation_name(::ScatterOp) = :scatter
operation_name(::CallResult) = :call_result

operation_detail(::Operation) = ""
operation_detail(op::Pointwise) = "($(op.f))"
operation_detail(op::CallResult) = "($(op.call.f))"
operation_detail(op::FoldReduction{<:Fold, <:Tuple}) = " dims=$(op.axes)"

transform_layout(::ReshapeOp, dims, parentdims, mapping) = reshape_map(mapping, dims)
transform_layout(op::Union{ViewOp, SliceOp}, dims, parentdims, mapping) =
    select_map(mapping, op.selection)
transform_layout(op::PermuteOp, dims, parentdims, mapping) =
    permute_map(mapping, parentdims, op.permutation)

function transform_layout(::TransposeOp, dims, parentdims, mapping)
    # A vector's row presentation preserves linear order.
    length(parentdims) == 1 ? reshape_map(mapping, dims) :
    permute_map(mapping, parentdims, (2, 1))
end

# Logical values have no physical storage. Scalar/zero-dimensional-array identity
# is explicit even though both will eventually fit in a one-element device buffer.
struct Node
    op::Operation
    inputs::Tuple{Vararg{Int}}
    dims::Tuple{Vararg{Int}}
    dtype::DataType
    scalar::Bool
end

struct TensorProgram
    nodes::Vector{Node}
    arguments::Vector{Int}
    output::Any # leaf node id or nested tuple/named tuple of ids
    signatures::Vector{Tuple{DataType, Tuple}}
    input_structure::Tuple # positional arguments, containing leaf node ids
end

# Input and result structures contain tensor leaves, not arbitrary Julia objects.
Base.@nospecializeinfer function tree_map(f, @nospecialize(x::Tuple))
    length(x) > 32 ? map_wide_tuple(f, x) : map(v -> tree_map(f, v), x)
end

# Large parameter collections are host metadata. Keep their element count out
# of generated host code; scalar kernels still specialize on expression types.
Base.@nospecializeinfer function map_wide_tuple(f, @nospecialize(x::Tuple))
    results = Any[]
    sizehint!(results, length(x))
    for element in x
        push!(results, tree_map(f, element))
    end
    Tuple(results)
end

Base.@nospecializeinfer function tree_map(f, @nospecialize(x::NamedTuple))
    if length(x) > 32
        NamedTuple{keys(x)}(map_wide_tuple(f, values(x)))
    else
        map(v -> tree_map(f, v), x)
    end
end

tree_map(f, x) = f(x)

same_structure(a, b) = typeof(tree_map(_ -> nothing, a)) == typeof(tree_map(_ -> nothing, b))

Base.@nospecializeinfer function tree_leaves(@nospecialize(x))
    leaves = Any[]
    tree_map(x) do v
        push!(leaves, v)
        nothing
    end
    leaves
end

is_layout(n::Node) = n.op isa LayoutOperation
transform_layout(n::Node, parentdims, mapping) =
    transform_layout(n.op, n.dims, parentdims, mapping)

function live_nodes(nodes, results)
    live = falses(length(nodes))
    function visit(i)
        if live[i]
            return
        end
        live[i] = true
        foreach(visit, nodes[i].inputs)
        nodes[i].op isa CallResult && foreach(visit, call_ids(nodes[i].op.call))
    end
    foreach(visit, results)
    live
end

function show_nodes(io, nodes, results)
    for (id, node) in enumerate(nodes)
        detail = operation_detail(node.op)
        shape = node.scalar ? " scalar" : string(node.dims)
        operands = join((" %$input" for input in node.inputs), ",")
        output = id in results ? " [output]" : ""
        println(
            io,
            "  %$id = $(operation_name(node.op))$detail$operands :: $(node.dtype)$shape$output",
        )
    end
end

function Base.show(io::IO, ::MIME"text/plain", p::TensorProgram)
    println(io, "TensorProgram (concrete shapes, structured outputs)")
    show_nodes(io, p.nodes, tree_leaves(p.output))
end

# Executable graphs share the original input contract. Flat result IDs serve
# liveness; the separate structure reconstructs gradient trees and forward outputs.
abstract type ExecutionProgram end

struct TrainingProgram <: ExecutionProgram
    primal::TensorProgram
    nodes::Vector{Node}
    results::Vector{Int} # flattened live result ids
    structure::Tuple # loss, then one gradient tree per selected positional argument
end

struct ForwardProgram <: ExecutionProgram
    primal::TensorProgram
    nodes::Vector{Node}
    results::Vector{Int}
    structure::Any
end

function ForwardProgram(p::TensorProgram)
    ForwardProgram(p, p.nodes, Int[tree_leaves(p.output)...], p.output)
end

TrainingProgram(primal, nodes, results) =
    TrainingProgram(primal, nodes, results, Tuple(results))

result_structure(p::TrainingProgram, lookup) = tree_map(lookup, p.structure)
result_structure(p::ForwardProgram, lookup) = tree_map(lookup, p.structure)

function remap_program(program::TrainingProgram, nodes, lookup)
    TrainingProgram(
        program.primal,
        nodes,
        map(lookup, program.results),
        tree_map(lookup, program.structure),
    )
end

function remap_program(program::ForwardProgram, nodes, lookup)
    ForwardProgram(
        program.primal,
        nodes,
        map(lookup, program.results),
        tree_map(lookup, program.structure),
    )
end
