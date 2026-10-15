# Expression trees are backend-neutral scalar programs. Leaves refer to a tuple
# of flattened materialized arrays; transpose is indexing, not a copy.
abstract type ScalarExpression end

struct Literal{T} <: ScalarExpression
    value::T
end

struct Read{I, M, D, O} <: ScalarExpression
    mapping::M
    dims::D
    shape::O
end

Read{I}(m::M, dims::D, shape::O) where {I, M, D, O} = Read{I, M, D, O}(m, dims, shape)

# Julia broadcasts leading axes, with missing trailing axes treated as singleton.
@inline broadcast_index(i, dims, shape) = 1 + broadcast_offset(i - 1, dims, shape, 1)

@inline broadcast_offset(rest, ::Tuple{}, shape, stride) = 0

@inline function broadcast_offset(rest, dims::Tuple, shape, stride)
    extent = first(shape)
    coordinate = first(dims) == 1 ? 0 : rem(rest, extent)
    coordinate * stride + broadcast_offset(
        div(rest, extent),
        Base.tail(dims),
        Base.tail(shape),
        stride * first(dims),
    )
end

struct AxisReduction{D, R}
    dims::D
    axes::R
    count::Int
end

AxisReduction(dims, axes) = AxisReduction(dims, axes, prod(dims[d] for d in axes))

@inline reduction_index(layout::AxisReduction, output, reduced) =
    1 + reduction_offset(output - 1, reduced - 1, layout.dims, layout.axes, 1, 1)

@inline reduction_offset(o, r, ::Tuple{}, axes, d, stride) = 0

@inline function reduction_offset(o, r, dims::Tuple, axes, d, stride)
    extent = first(dims)
    next = stride * extent
    if d in axes
        rem(r, extent) * stride +
        reduction_offset(o, div(r, extent), Base.tail(dims), axes, d + 1, next)
    else
        rem(o, extent) * stride +
        reduction_offset(div(o, extent), r, Base.tail(dims), axes, d + 1, next)
    end
end

struct ScalarCall{F, A} <: ScalarExpression
    f::F
    args::A
end

# Regions describe executable work. Scheduling adds value ids and dependencies;
# reduction and scatter metadata belong to the work itself, not the request.
abstract type Region end

abstract type ScalarRegion <: Region end

struct Elementwise{E <: ScalarExpression} <: ScalarRegion
    expression::E
    count::Int
end

# Reductions carry their fold and its identity state in the fold's accumulator
# type (`fold_accumulator`), and their expression yields contributions in that
# type, so kernels need no element type to start a fold. The store rounds the
# observed value to the output's type. A provider without a general fold path
# checks for `SumFold`.
struct FullReduction{E <: ScalarExpression, F <: Fold, S <: Tuple} <: ScalarRegion
    expression::E
    count::Int
    fold::F
    init::S
end

struct AxisReductionRegion{E <: ScalarExpression, L <: AxisReduction, F <: Fold, S <: Tuple} <:
       ScalarRegion
    expression::E
    count::Int
    layout::L
    fold::F
    init::S
end

# The type a sum region accumulates in, the type of its state.
sum_accumulator(region::Union{FullReduction, AxisReductionRegion}) = typeof(only(region.init))

struct ScatterRegion{E <: ScalarExpression, S <: Selection} <: ScalarRegion
    expression::E
    count::Int
    selection::S
end

struct GemmRegion <: Region end

struct OpaqueCall{C <: SemanticCall} <: Region
    call::C
end

# Names are for printed plans only. Implementations dispatch on region types.
region_name(region::Region) = Symbol(nameof(typeof(region)))
region_name(::Elementwise) = :pointwise
region_name(r::FullReduction) = fold_name(r.fold)
region_name(r::AxisReductionRegion) = Symbol(:axis_, fold_name(r.fold))
region_name(::ScatterRegion) = :scatter
region_name(::GemmRegion) = :matmul
region_name(::OpaqueCall) = :call

struct Step{R <: Region}
    region::R
    output::Int
    dependencies::Vector{Int}
    outputs::Vector{Int}
end

Step(region::Region, output, dependencies) = Step(region, output, dependencies, [output])

struct StoragePlan
    program::ExecutionProgram
    steps::Vector{Step}
    slots::Dict{Int, Int}
    specifications::Vector{Tuple{DataType, Int}}
    lastuse::Dict{Int, Int}
    naive_bytes::Int
    reuse::Bool
    layouts::Dict{Int, Tuple{Int, IndexMap}} # logical value => (storage root, index map)
    donations::Dict{Int, Int} # output root => owned input id
end

# Whether an expression reads anything other than its own linear index, so a
# provider without general index mappings can decline it.
needs_indexing(::Literal) = false

function needs_indexing(x::Read)
    !linear_identity(x.mapping) || (prod(x.dims) != 1 && prod(x.dims) != prod(x.shape))
end

needs_indexing(x::ScalarCall) = any(needs_indexing, x.args)
