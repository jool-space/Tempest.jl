module cuTileExt
using Tempest
using Tempest:
    RegionRequest, Supported, Unsupported, BuiltImplementation, ImplementationPlan, WorkspaceSpec
using Tempest: Elementwise, FullReduction, AxisReductionRegion, ScalarRegion, region_name
using CUDACore
import cuTile as ct
using Tempest: Literal, Read, ScalarCall, AxisReduction, needs_indexing
import Tempest: support, build_implementation
const TILE_SIZE = 256
# Scalar-region kernels compile for 8 blocks per SM. Without a hint, tileiras
# gave them 255 registers once cuTile placed each array in its own alias class;
# on GB10 the residual RMSNorm fixture's unfused schedule then ran at 128 x 4096
# in 132/160 us warm/cold and at 1024 x 1024 in 67/115 us. With occupancy 8 it
# ran in 27/57 and 28/86 us; 16 matched 8, and 32 was slower at 1024 x 1024.
const TILE_OCCUPANCY = 8
include("tile.jl")
include("tile_columns.jl")
include("attention.jl")

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
    # but not over the two-axis tiles of column reductions (see graph_codegen.jl).
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

include("graph_regions.jl")

end
