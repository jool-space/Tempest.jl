module CUDAExt
using Tempest
using Tempest:
    RegionRequest,
    CallRequest,
    InPlaceRequest,
    GroupedRequest,
    Supported,
    Unsupported,
    BuiltImplementation,
    ImplementationPlan,
    WorkspaceSpec
using Tempest:
    Elementwise,
    FullReduction,
    AxisReductionRegion,
    ScatterRegion,
    ScalarRegion,
    GemmRegion,
    region_name
using LinearAlgebra
using CUDACore
import cuBLAS
using Tempest:
    AbstractRuntime,
    CUDATarget,
    Read,
    AxisReduction,
    linear_identity,
    dense_strides,
    dense_map,
    map_index,
    scatter_schedule,
    AffineMap,
    ComposedMap,
    Selection,
    IndexList,
    index_length,
    index_at,
    Literal,
    ScalarCall
import Tempest:
    array_target,
    make_runtime,
    allocate,
    check_runtime,
    default_implementations,
    support,
    build_implementation,
    capture_runtime!,
    replay_runtime!,
    explain_runtime,
    time_calls

const TILE_SIZE = 256

include("blas.jl")
include("runtime.jl")
include("indexing.jl")
include("scalar.jl")
include("axis_reduction.jl")
include("scatter.jl")
include("grouping.jl")
include("kernels.jl")
include("verbs.jl")

end
