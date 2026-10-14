
# Read logical scalar lanes from flattened storage. Mask both loads and the final
# reduction: f(0) can be nonzero, infinite, or NaN in padded lanes.
@inline tile_eval(x::Literal, arrays, indices, valid) = x.value

@inline function tile_eval(x::Read{I}, arrays, indices, valid) where {I}
    a = arrays[I]
    if prod(x.dims) == 1
        return a[1]
    end
    # Dense reads use tile loads. cuTile 1.0's dynamic gather/scatter path
    # exhibited an off-by-one on this toolchain; transpose regions use CUDA.
    ct.load(a, (ct.bid(1),), (TILE_SIZE,); padding_mode = ct.PaddingMode.Zero)
end

@inline function tile_eval(x::ScalarCall, arrays, indices, valid)
    operands = map(a -> tile_eval(a, arrays, indices, valid), x.args)
    x.f.(operands...)
end

@inline function lane_indices(block, tile)
    base = ct.Tile((block - Int32(1)) * Int32(tile))
    ct.broadcast_to(base, (tile,)) .+ ct.arange(tile)
end

function pointwise_kernel(out, expression, arrays, count::Int32, tile::Int)
    indices = lane_indices(ct.bid(1), tile)
    valid = indices .<= count
    y = tile_eval(expression, arrays, indices, valid)
    # broadcast_to handles expressions containing only scalar constants;
    # ct.store's bounds check discards the final tile's padded lanes.
    ct.store(out, (ct.bid(1),), ct.broadcast_to(ct.Tile(y), (tile,)))
    return
end

function reduction_kernel(work, expression, arrays, count::Int32, tile::Int)
    indices = lane_indices(ct.bid(1), tile)
    valid = indices .<= count
    y = tile_eval(expression, arrays, indices, valid)
    masked = ifelse.(valid, y, zero(eltype(work)))
    work[ct.bid(1)] = sum(masked)
    return
end

function finish_reduction_kernel(out, work, count::Int32, tile::Int)
    acc = zeros(eltype(out), tile)
    for block = Int32(1):cld(count, Int32(tile))
        indices = lane_indices(block, tile)
        loaded = ct.load(work, (block,), (tile,); padding_mode = ct.PaddingMode.Zero)
        acc = acc .+ ifelse.(indices .<= count, loaded, zero(eltype(out)))
    end
    out[1] = sum(acc)
    return
end

function tile_pointwise!(out, expression, arrays, count)
    if count == 0
        return nothing
    end
    CUDACore.@cuda backend=ct occupancy=TILE_OCCUPANCY blocks=cld(count, TILE_SIZE) pointwise_kernel(
        out,
        ct.Constant(expression),
        arrays,
        Int32(count),
        ct.Constant(TILE_SIZE),
    )
    nothing
end

function tile_sum!(out, expression, arrays, count, work)
    if count == 0
        fill!(out, zero(eltype(out)))
        return nothing
    end
    blocks = cld(count, TILE_SIZE)
    CUDACore.@cuda backend=ct occupancy=TILE_OCCUPANCY blocks=blocks reduction_kernel(
        work,
        ct.Constant(expression),
        arrays,
        Int32(count),
        ct.Constant(TILE_SIZE),
    )
    CUDACore.@cuda backend=ct occupancy=TILE_OCCUPANCY blocks=1 finish_reduction_kernel(
        out,
        work,
        Int32(blocks),
        ct.Constant(TILE_SIZE),
    )
    nothing
end
