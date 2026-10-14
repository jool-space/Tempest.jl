# Layouts map logical, one-based linear indices to a storage root. They carry no
# array data. Reshape changes the coordinate system, never the linear order.
abstract type IndexMap end

struct AffineMap{D, S} <: IndexMap
    dims::D
    strides::S
    offset::Int
end

struct ComposedMap{P <: IndexMap, L <: IndexMap} <: IndexMap
    parent::P
    localmap::L
end

# Gather indices are structural metadata, snapshotted at trace time. GPU lowering
# provisions stable device index buffers; metadata size stays independent of the
# gather length in generated kernel signatures.
struct IndexList{I}
    indices::I
end

struct Selection{D, I, O} <: IndexMap
    parentdims::D
    indices::I
    dims::O
    unique::Bool
end

dense_strides(dims) = ntuple(d -> prod(dims[1:(d - 1)]; init = 1), length(dims))
dense_map(dims) = AffineMap(dims, dense_strides(dims), 0)

# Per-dimension helpers recurse over tuples rather than loop with a runtime
# index, so GPU compilation unrolls them and keeps the metadata in registers.
@inline map_index(m::AffineMap, i) = m.offset + 1 + affine_offset(i - 1, m.dims, m.strides)

@inline affine_offset(rest, ::Tuple{}, ::Tuple{}) = 0

@inline function affine_offset(rest, dims::Tuple, strides::Tuple)
    extent = first(dims)
    rem(rest, extent) * first(strides) +
    affine_offset(div(rest, extent), Base.tail(dims), Base.tail(strides))
end

@inline map_index(m::ComposedMap, i) = map_index(m.parent, map_index(m.localmap, i))
index_length(i::Integer) = 0
index_length(i::IndexList) = length(i.indices)
index_length(i::AbstractRange) = length(i)
@inline index_at(i::IndexList, k) = i.indices[k]
@inline index_at(i::AbstractRange, k) = first(i) + (k - 1) * step(i)
# Recurse over the heterogeneous selector tuple so GPU compilation resolves
# integer/range/gather dispatch statically instead of boxing tuple fields.
@inline selection_offset(::Tuple{}, ::Tuple{}, rest, stride) = 0

@inline function selection_offset(indices::Tuple, dims::Tuple, rest, stride)
    idx = first(indices)
    if idx isa Integer
        coordinate = idx
    else
        extent = index_length(idx)
        coordinate = index_at(idx, rem(rest, extent) + 1)
        rest = div(rest, extent)
    end
    (coordinate - 1) * stride +
    selection_offset(Base.tail(indices), Base.tail(dims), rest, stride * first(dims))
end

@inline map_index(m::Selection, i) = 1 + selection_offset(m.indices, m.parentdims, i - 1, 1)

linear_identity(::IndexMap) = false

function linear_identity(m::AffineMap)
    if m.offset != 0
        return false
    end
    stride = 1
    for d in eachindex(m.dims)
        if m.dims[d] > 1 && m.strides[d] != stride
            return false
        end
        stride *= m.dims[d]
    end
    true
end

# A no-copy reshape may split or merge contiguous stride chunks, but must not
# cross a gap in the parent's storage. Singleton dimensions impose no stride.
function reshape_affine(m::AffineMap, dims)
    if prod(m.dims) != prod(dims)
        throw(DimensionMismatch("layout reshape changes length"))
    end
    if prod(dims) == 0
        return AffineMap(dims, dense_strides(dims), m.offset)
    end
    chunks = Tuple{Int, Int}[]
    for (n, s) in zip(m.dims, m.strides)
        if n == 1
            continue
        end
        if !isempty(chunks) && s == chunks[end][1] * chunks[end][2]
            count, stride = pop!(chunks)
            push!(chunks, (count * n, stride))
        else
            push!(chunks, (n, s))
        end
    end
    if isempty(chunks)
        return AffineMap(dims, dense_strides(dims), m.offset)
    end
    strides = Int[]
    c, consumed = 1, 1
    remaining, base = chunks[c]
    for n in dims
        if n == 1
            push!(strides, base * consumed)
            continue
        end
        if remaining % n != 0
            return nothing
        end
        push!(strides, base * consumed)
        remaining ÷= n
        consumed *= n
        if remaining == 1 && c < length(chunks)
            c += 1
            remaining, base = chunks[c]
            consumed = 1
        end
    end
    AffineMap(dims, Tuple(strides), m.offset)
end

reshape_affine(::IndexMap, dims) = nothing

function reshape_map(m::IndexMap, dims)
    simplified = reshape_affine(m, dims)
    simplified === nothing ? m : simplified
end

function permute_map(m::IndexMap, parentdims, perm)
    affine = reshape_affine(m, parentdims)
    dims = Tuple(parentdims[d] for d in perm)
    if affine !== nothing
        return AffineMap(dims, Tuple(affine.strides[d] for d in perm), affine.offset)
    end
    localmap = AffineMap(dims, Tuple(dense_strides(parentdims)[d] for d in perm), 0)
    ComposedMap(m, localmap)
end

function select_map(m::IndexMap, s::Selection)
    affine = reshape_affine(m, s.parentdims)
    if affine !== nothing && !any(i -> i isa IndexList, s.indices)
        offset = affine.offset
        strides = Int[]
        for (idx, stride) in zip(s.indices, affine.strides)
            if idx isa Integer
                offset += (idx - 1) * stride
            else
                offset += (first(idx) - 1) * stride
                push!(strides, step(idx) * stride)
            end
        end
        return AffineMap(s.dims, Tuple(strides), offset)
    end
    linear_identity(m) ? s : ComposedMap(m, s)
end

# Native representations accepted by the executor and BLAS. Other layouts can
# still feed generated kernels; only consumers needing these forms force a copy.
function native_layout(m::IndexMap, dims, rootdims)
    if linear_identity(m) && prod(dims) == prod(rootdims)
        return :dense
    end
    a = reshape_affine(m, dims)
    if length(dims) == 2 &&
       a !== nothing &&
       a.offset == 0 &&
       prod(dims) == prod(rootdims) &&
       a.strides == (dims[2], 1)
        return :transpose
    end
    :mapped
end

# Reference paths use Julia indexing, independently of map_index.
thaw_index(i::IndexList) = collect(i.indices)
thaw_index(i) = i
native_indices(s::Selection) = map(thaw_index, s.indices)

# Stable source-order buckets for non-injective pullbacks. O(parent + selected)
# storage/work at preparation; GPU execution needs no floating-point atomics.
function scatter_schedule(s::Selection, parentcount)
    offsets = zeros(Int, parentcount + 1)
    count = prod(s.dims)
    for i = 1:count
        offsets[map_index(s, i) + 1] += 1
    end
    cumsum!(offsets, offsets)
    cursor = copy(offsets)
    sources = Vector{Int}(undef, count)
    for i = 1:count
        target = map_index(s, i)
        cursor[target] += 1
        sources[cursor[target]] = i
    end
    offsets, sources
end
