# First axis-specific cuTile schedule: reduce the leading axis of a dense matrix.
# Inputs to the scalar region are dense (same element count) or scalar. More
# complicated broadcast/transpose indexing stays with the general CUDA schedule.
function tiled_columns(layout::AxisReduction, e)
    length(layout.dims) == 2 && layout.axes == (1,) && !needs_indexing(e)
end

@inline column_eval(x::Literal, arrays, block, tile) = x.value

@inline function column_eval(x::Read{I}, arrays, block, tile) where {I}
    a = arrays[I]
    if prod(x.dims) == 1
        return a[1, 1]
    end
    ct.load(a, (block, ct.bid(1)), (tile, 1); padding_mode = ct.PaddingMode.Zero)
end

@inline column_eval(x::ScalarCall, arrays, block, tile) =
    x.f.(map(a -> column_eval(a, arrays, block, tile), x.args)...)

function column_reduction_kernel(out, expression, arrays, rows::Int32, tile::Int)
    acc = zeros(eltype(out), (tile, 1))
    for block = Int32(1):cld(rows, Int32(tile))
        indices = reshape(lane_indices(block, tile), (tile, 1))
        y = column_eval(expression, arrays, block, tile)
        acc = acc .+ ifelse.(indices .<= rows, y, zero(eltype(out)))
    end
    out[ct.bid(1)] = sum(acc)
    return
end

function tile_columns!(out, expression, arrays, layout)
    if isempty(out)
        return nothing
    end
    CUDACore.@cuda backend=ct occupancy=TILE_OCCUPANCY blocks=length(out) column_reduction_kernel(
        out,
        ct.Constant(expression),
        arrays,
        Int32(layout.dims[1]),
        ct.Constant(TILE_SIZE),
    )
    nothing
end
