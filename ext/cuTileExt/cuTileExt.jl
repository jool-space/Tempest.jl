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
    recomputable
import Tempest: support, build_implementation
using CUDACore
import cuTile as ct

const TILE_SIZE = 256
# Scalar-region kernels compile for 8 blocks per SM. Without a hint, tileiras
# gave them 255 registers once cuTile placed each array in its own alias class;
# on GB10 the residual RMSNorm fixture's unfused schedule then ran at 128 x 4096
# in 132/160 us warm/cold and at 1024 x 1024 in 67/115 us. With occupancy 8 it
# ran in 27/57 and 28/86 us; 16 matched 8, and 32 was slower at 1024 x 1024.
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

end
