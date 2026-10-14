
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

# cuTile's scalar/tile promotion may narrow Float64 literals to a Float32
# operand tile, unlike Julia scalar promotion. Keep these regions on CUDA;
# even exactly representable literals can change intermediate precision.
function tile_expression_support(x::Literal)
    x.value isa Float64 ? Unsupported("Float64 literals require CUDA scalar promotion") :
    Supported()
end

tile_expression_support(::Read) = Supported()

function tile_expression_support(x::ScalarCall)
    # cuTile has no overlays for these Base implementations. Their scalar
    # branches do not lower correctly to tile-shaped values. Keep Julia's CUDA
    # implementations rather than replacing log1p/expm1 with less accurate formulas.
    # Still so on cuTile main (86b1f83): log1p, expm1, asin and acos fail to
    # compile, comparing a 0-D tile with a 256-lane one; one-argument atan needs
    # a copysign without a Tile IR equivalent; sign compiles over one-axis tiles
    # but not over the two-axis tiles of column reductions (see graph/codegen.jl).
    if x.f in (log1p, expm1, asin, acos, sign) || (x.f === atan && length(x.args) == 1)
        return Unsupported(
            "cuTile has no verified lowering for $(x.f) with $(length(x.args)) argument(s)",
        )
    end
    for argument in x.args
        verdict = tile_expression_support(argument)
        verdict isa Unsupported && return verdict
    end
    Supported()
end

function support(::CuTileKernels, request::RegionRequest)
    if !(request.target isa CUDATarget)
        return Unsupported("requires a CUDA target")
    end
    step = request.step
    if !(step.region isa Union{Elementwise, FullReduction, AxisReductionRegion})
        return Unsupported("cuTile does not handle $(region_name(step.region))")
    end
    if step.region.count > typemax(Int32)
        return Unsupported("GPU scalar region exceeds Int32 indexing")
    end
    if step.region isa Union{FullReduction, AxisReductionRegion} &&
       !(step.region.fold isa Tempest.SumFold)
        return Unsupported("cuTile scalar regions implement only sum folds")
    end
    if needs_indexing(step.region.expression)
        return Unsupported("mapped or expanded reads require CUDA indexing kernels")
    end
    verdict = tile_expression_support(step.region.expression)
    verdict isa Unsupported && return verdict
    if step.region isa AxisReductionRegion &&
       !tiled_columns(step.region.layout, step.region.expression)
        return Unsupported("axis reduction is not a supported column reduction")
    end
    Supported()
end

function build_implementation(
    ::CuTileKernels,
    request::RegionRequest{<:ScalarRegion},
    runtime,
    out,
    arrays,
)
    region = request.step.region
    isbits(region.expression) ||
        throw(ArgumentError("GPU scalar expressions must have isbits captures"))
    build_tile(region, out, arrays)
end

function build_tile(region::AxisReductionRegion, out, arrays)
    expression, layout = region.expression, region.layout
    count = prod(layout.dims)
    arrays = map(arrays) do array
        # Dense reads may cover a prefix of their root; smaller roots are scalar reads.
        if length(array) < count
            return reshape(array, (length(array), 1))
        end
        prefix = length(array) == count ? array : view(array, 1:count)
        reshape(prefix, layout.dims)
    end
    launch = () -> tile_columns!(out, expression, arrays, layout)
    BuiltImplementation(launch, (out, arrays), 0)
end

function build_tile(region::FullReduction, out, arrays)
    expression, count = region.expression, region.count
    if count == 0
        launch = () -> tile_sum!(out, expression, arrays, count, nothing)
        return BuiltImplementation(launch, (out, arrays), 0)
    end
    workspace = WorkspaceSpec(eltype(out), cld(count, TILE_SIZE))
    ImplementationPlan((workspace,)) do buffers
        work = only(buffers)
        launch = () -> tile_sum!(out, expression, arrays, count, work)
        BuiltImplementation(launch, (out, arrays, work), 0)
    end
end

function build_tile(region::Elementwise, out, arrays)
    launch = () -> tile_pointwise!(out, region.expression, arrays, region.count)
    BuiltImplementation(launch, (out, arrays), 0)
end
