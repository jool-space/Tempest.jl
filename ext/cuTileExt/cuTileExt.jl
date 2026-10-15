module cuTileExt
using Tempest
using Tempest:
    RegionRequest, Supported, Unsupported, BuiltImplementation, ImplementationPlan, WorkspaceSpec
using Tempest: Elementwise, FullReduction, AxisReductionRegion, ScalarRegion, region_name
using Tempest: Literal, Read, ScalarCall, AxisReduction, needs_indexing
using Tempest:
    GraphRegion,
    GraphRequest,
    Argument,
    Constant,
    Pointwise,
    FoldReduction,
    SumFold,
    fold_init,
    fold_width,
    has_resident_lowering,
    recomputable,
    standard_numbering
import Tempest: support, build_implementation
using CUDACore
import cuTile as ct

const TILE_SIZE = 256
# Scalar-region kernels compile for 8 blocks per SM. Without a hint, tileiras
# gives the reduction kernels 255 registers (the pointwise kernel uses 16
# either way). Measured on GB10 with cuTile main d2fafa3, 15 October 2026, warm
# us with no hint / 4 / 8: column sums over 256 x 32768 Float32 1164 / 326 / 155
# and Float16 951 / 305 / 131, a sum of 2^24 values 1300 / 418 / 298, and the
# residual RMSNorm fixture's unfused schedule at 128 x 4096 132 / 48 / 26.
# 16 matched 8.
const TILE_OCCUPANCY = 8

# Scalar regions on tiles: pointwise, full and leading-axis reductions.
include("scalar.jl")
include("attention.jl")
# Graph regions: schedule analysis, generated tile code, kernel entry points,
# and construction of the provider's implementations.
include("graph/schedule.jl")
include("graph/codegen.jl")
include("graph/kernels.jl")
include("graph/build.jl")
# Launches for region estimates.
include("costs.jl")

# Active once loaded; Tempest.deactivate(cuTile) removes it.
__init__() = Tempest.register!(ct, CuTileKernels; target = Tempest.CUDATarget, rank = 10)

end
