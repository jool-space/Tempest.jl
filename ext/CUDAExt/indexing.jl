# Generated kernels iterate a region's shape. Each read is resolved against that
# shape once, at build time: a dense read of the full shape uses the linear
# index, an affine read (broadcasts included) becomes strides over iteration
# coordinates, and a gather keeps its selection behind a strided logical index.
# Kernels compute coordinates once per element with unsigned 32-bit division;
# singleton extents are dropped, so they cost nothing.
struct LinearRead{I} end

struct StridedRead{I, N}
    strides::NTuple{N, Int}
    offset::Int
end

struct MappedRead{I, N, M}
    strides::NTuple{N, Int}
    mapping::M
end

# `kept` enumerates outputs and `reduced` the summed elements of each output.
# Pointwise, full and scatter regions iterate `kept` alone.
struct DeviceExpression{E, K, R}
    expression::E
    kept::NTuple{K, UInt32}
    reduced::NTuple{R, UInt32}
end

@inline coordinates(rest, ::Tuple{}) = ()
@inline coordinates(rest, ::Tuple{Any}) = (rest,)

@inline function coordinates(rest, shape::Tuple)
    extent = first(shape)
    quotient = Core.Intrinsics.udiv_int(rest, extent)
    (rest - quotient * extent, coordinates(quotient, Base.tail(shape))...)
end

@inline coordinates(e::DeviceExpression, i) = coordinates((i - 1) % UInt32, e.kept)

@inline coordinates(e::DeviceExpression, o, r) =
    (coordinates((o - 1) % UInt32, e.kept)..., coordinates((r - 1) % UInt32, e.reduced)...)

@inline strided_offset(::Tuple{}, ::Tuple{}) = 0
@inline strided_offset(c::Tuple, s::Tuple) =
    Int(first(c)) * first(s) + strided_offset(Base.tail(c), Base.tail(s))

@inline device_eval(x::Literal, arrays, i, c) = x.value
@inline device_eval(::LinearRead{I}, arrays, i, c) where {I} = @inbounds arrays[I][i]

@inline device_eval(x::StridedRead{I}, arrays, i, c) where {I} =
    @inbounds arrays[I][x.offset + 1 + strided_offset(c, x.strides)]

@inline device_eval(x::MappedRead{I}, arrays, i, c) where {I} =
    @inbounds arrays[I][device_index(x.mapping, strided_offset(c, x.strides) % UInt32)]

# Layout maps evaluated from a zero-based UInt32 logical index. Logical indices
# of a generated region fit in 32 bits (preparation checks the count), so each
# dimension costs an unsigned 32-bit division instead of a 64-bit one; storage
# offsets still accumulate in Int.
@inline device_index(m, rest::UInt32) = 1 + device_offset(m, rest)

@inline device_offset(m::AffineMap, rest::UInt32) =
    m.offset + device_affine(rest, m.dims, m.strides)

@inline device_affine(rest, ::Tuple{}, ::Tuple{}) = 0

@inline function device_affine(rest, dims::Tuple, strides::Tuple)
    extent = first(dims) % UInt32
    quotient = Core.Intrinsics.udiv_int(rest, extent)
    Int(rest - quotient * extent) * first(strides) +
    device_affine(quotient, Base.tail(dims), Base.tail(strides))
end

@inline device_offset(m::ComposedMap, rest::UInt32) =
    device_offset(m.parent, device_offset(m.localmap, rest) % UInt32)

@inline device_offset(m::Selection, rest::UInt32) =
    device_selection(m.indices, m.parentdims, rest, 1)

@inline device_selection(::Tuple{}, ::Tuple{}, rest, stride) = 0

@inline function device_selection(indices::Tuple, dims::Tuple, rest, stride)
    index = first(indices)
    if index isa Integer
        coordinate = index
    else
        extent = index_length(index) % UInt32
        quotient = Core.Intrinsics.udiv_int(rest, extent)
        coordinate = index_at(index, Int(rest - quotient * extent) + 1)
        rest = quotient
    end
    (coordinate - 1) * stride +
    device_selection(Base.tail(indices), Base.tail(dims), rest, stride * first(dims))
end

# Arguments recurse over the tuple instead of mapping a closure: an outlined
# closure would hold the captured array tuple in local memory.
@inline device_eval(x::ScalarCall, arrays, i, c) = x.f(device_args(x.args, arrays, i, c)...)

@inline device_args(::Tuple{}, arrays, i, c) = ()
@inline device_args(args::Tuple, arrays, i, c) =
    (device_eval(first(args), arrays, i, c), device_args(Base.tail(args), arrays, i, c)...)

@inline device_eval(e::DeviceExpression, arrays, i) =
    device_eval(e.expression, arrays, i, coordinates(e, i))

expression_shape(::Literal) = nothing
expression_shape(x::Read) = x.shape

function expression_shape(x::ScalarCall)
    for a in x.args
        shape = expression_shape(a)
        shape === nothing || return shape
    end
    nothing
end

# Strides of a read over each dimension of `shape`; broadcast dimensions and
# trailing dimensions missing from the read contribute nothing.
function broadcast_strides(dims, strides, shape)
    ntuple(length(shape)) do d
        d <= length(dims) && dims[d] != 1 ? strides[d] : 0
    end
end

function device_leaf(r, x::Read{I}, order, linear) where {I}
    if linear && x.dims == x.shape && linear_identity(x.mapping)
        return LinearRead{I}()
    end
    m = x.mapping
    if m isa AffineMap && m.dims == x.dims
        strides = broadcast_strides(x.dims, m.strides, x.shape)
        return StridedRead{I, length(order)}(map(d -> strides[d], order), m.offset)
    end
    strides = broadcast_strides(x.dims, dense_strides(x.dims), x.shape)
    mapping = gpu_mapping(r, m)
    MappedRead{I, length(order), typeof(mapping)}(map(d -> strides[d], order), mapping)
end

device_leaf(r, x::Literal, order, linear) = x

device_leaf(r, x::ScalarCall, order, linear) =
    ScalarCall(x.f, map(a -> device_leaf(r, a, order, linear), x.args))

uses_coordinates(x) = false
uses_coordinates(::Union{StridedRead, MappedRead}) = true
uses_coordinates(x::ScalarCall) = any(uses_coordinates, x.args)

# A pointwise, full-reduction or scatter expression iterating linearly over its
# shape. Without strided reads, it needs no coordinates at all.
function device_expression(r, e)
    shape = expression_shape(e)
    shape === nothing && return DeviceExpression(e, (), ())
    order = Tuple(d for d in eachindex(shape) if shape[d] != 1)
    leaves = device_leaf(r, e, order, true)
    kept = uses_coordinates(leaves) ? map(d -> UInt32(shape[d]), order) : ()
    DeviceExpression(leaves, kept, ())
end

# Coordinates are ordered kept dimensions first, then reduced ones, so outputs
# and reduction positions decompose independently.
function device_expression(r, e, layout::AxisReduction)
    shape = layout.dims
    kept = Tuple(d for d in eachindex(shape) if shape[d] != 1 && !(d in layout.axes))
    reduced = Tuple(d for d in eachindex(shape) if shape[d] != 1 && d in layout.axes)
    leaves = device_leaf(r, e, (kept..., reduced...), false)
    DeviceExpression(leaves, map(d -> UInt32(shape[d]), kept), map(d -> UInt32(shape[d]), reduced))
end

# Whether consecutive reduction positions are adjacent in iteration order, so
# the lanes of one block read contiguous storage for one output.
function contiguous_reduction(layout::AxisReduction)
    first_dim = findfirst(!=(1), layout.dims)
    first_dim !== nothing && first_dim in layout.axes
end
