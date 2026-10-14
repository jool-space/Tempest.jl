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

function support(impl::CUDAKernels, request::InPlaceRequest)
    request.region.step.region isa Elementwise ||
        return Unsupported("in-place execution requires elementwise work")
    support(impl, request.region)
end

const TILE_SIZE = 256
include("blas.jl")
include("indexing.jl")
include("cuda_scalar.jl")
include("axis_reduction.jl")
include("scatter.jl")
include("grouping.jl")
include("primitives.jl")

mutable struct CUDARuntime <: AbstractRuntime
    target::CUDATarget
    device::CUDACore.CuDevice
    context::CUDACore.CuContext
    blas::Union{Nothing, BlasResources}
    graph::Any
    graph_exec::Any
    layout_buffers::Vector{Any}
    layout_cache::IdDict{Any, Any}
end

array_target(a::CuArray) = CUDATarget(device = CUDACore.device(a))

function make_runtime(target::CUDATarget, storage, args)
    if !CUDACore.functional()
        throw(ArgumentError("CUDA is not functional"))
    end
    device = CUDACore.device()
    if !(target.device === nothing || target.device == device)
        throw(ArgumentError("activate the target CUDA device before preparing"))
    end
    CUDARuntime(
        target,
        device,
        CUDACore.context(),
        nothing,
        nothing,
        nothing,
        Any[],
        IdDict{Any, Any}(),
    )
end

allocate(::CUDARuntime, T, dims) = CuArray{T}(undef, dims)

function check_runtime(r::CUDARuntime)
    if !(CUDACore.device() == r.device && CUDACore.context() == r.context)
        throw(ArgumentError("prepared executable belongs to another CUDA device/context"))
    end
    nothing
end

function default_implementations(target::CUDATarget)
    tile_loaded = Base.get_extension(Tempest, :cuTileExt) !== nothing
    if target.kernels === :cutile && !tile_loaded
        throw(ArgumentError("kernels=:cutile requires loading cuTile"))
    end
    if target.kernels !== :cuda && tile_loaded
        (CuTileKernels(), CUDAKernels(), CuBLASGemm())
    else
        (CUDAKernels(), CuBLASGemm())
    end
end

function support(::CUDAKernels, request::RegionRequest)
    if !(request.target isa CUDATarget)
        return Unsupported("requires a CUDA target")
    end
    if !(
        request.step.region isa
        Union{Elementwise, FullReduction, AxisReductionRegion, ScatterRegion}
    )
        return Unsupported(
            "CUDA scalar kernels do not handle $(region_name(request.step.region))",
        )
    end
    if request.step.region.count > typemax(Int32)
        return Unsupported("GPU scalar region exceeds Int32 indexing")
    end
    if !isbitstype(request.dtype) || !all(isbitstype, request.input_types)
        return Unsupported("CUDA scalar kernels require bits element types")
    end
    Supported()
end

function support(::CuBLASGemm, request::RegionRequest)
    if !(request.target isa CUDATarget)
        return Unsupported("requires a CUDA target")
    end
    if !(request.step.region isa GemmRegion)
        return Unsupported("requires a matrix product")
    end
    if !(request.dtype in (Float32, Float64)) || !all(==(request.dtype), request.input_types)
        return Unsupported("CUDA GEMM operands must have matching Float32/Float64 dtypes")
    end
    if max(request.dims..., request.input_dims[1][2]) > typemax(Int32)
        return Unsupported("GEMM dimensions exceed Int32")
    end
    Supported()
end

function build_implementation(::CuBLASGemm, request::RegionRequest, runtime, out, arrays)
    workspace = WorkspaceSpec(UInt8, runtime.target.blas_workspace_bytes)
    ImplementationPlan((workspace,)) do buffers
        if runtime.blas === nothing
            runtime.blas = BlasResources(
                runtime.target.blas_workspace_bytes,
                runtime.target.matmul_precision,
            )
        end
        work = only(buffers)
        launch = gemm_call(runtime.blas, out, arrays..., work)
        BuiltImplementation(launch, (runtime.blas, out, arrays, work), 0)
    end
end

function build_implementation(
    ::CUDAKernels,
    request::RegionRequest{<:ScalarRegion},
    runtime,
    out,
    arrays,
)
    region = request.step.region
    expression = device_expression(runtime, region)
    isbits(expression) ||
        throw(ArgumentError("GPU scalar expressions must have isbits captures"))
    build_scalar(region, runtime, out, expression, arrays)
end

device_expression(r, region::ScalarRegion) = device_expression(r, region.expression)
device_expression(r, region::AxisReductionRegion) =
    device_expression(r, region.expression, region.layout)

function build_scalar(
    region::FullReduction{<:Tempest.ScalarExpression, Tempest.SumFold},
    runtime,
    out,
    expression,
    arrays,
)
    count = region.count
    if count == 0
        launch = () -> cuda_sum!(out, expression, arrays, count, nothing)
        return BuiltImplementation(launch, (runtime, out, arrays), 0)
    end
    workspace = WorkspaceSpec(eltype(out), reduction_blocks(count))
    ImplementationPlan((workspace,)) do buffers
        work = only(buffers)
        launch = () -> cuda_sum!(out, expression, arrays, count, work)
        BuiltImplementation(launch, (runtime, out, arrays, work), 0)
    end
end

function build_scalar(region::Elementwise, runtime, out, expression, arrays)
    launch = () -> cuda_pointwise!(out, expression, arrays, region.count)
    BuiltImplementation(launch, (runtime, out, arrays), 0)
end

function build_scalar(region::AxisReductionRegion, runtime, out, expression, arrays)
    launch = fold_call(out, region, expression, arrays)
    BuiltImplementation(launch, (runtime, out, arrays), 0)
end

# Partial states of a general fold are tuples; the workspace holds them as such.
function build_scalar(region::FullReduction, runtime, out, expression, arrays)
    workspace = WorkspaceSpec(typeof(region.init), reduction_blocks(max(region.count, 1)))
    ImplementationPlan((workspace,)) do buffers
        work = only(buffers)
        launch = fold_call(out, region, expression, arrays, work)
        BuiltImplementation(launch, (runtime, out, arrays, work), 0)
    end
end
function build_scalar(
    region::AxisReductionRegion{<:Tempest.ScalarExpression, <:Tempest.AxisReduction, Tempest.SumFold},
    runtime,
    out,
    expression,
    arrays,
)
    multiprocessors =
        CUDACore.attribute(runtime.device, CUDACore.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    launch = axis_sum_call(out, expression, arrays, region.layout, multiprocessors)
    BuiltImplementation(launch, (runtime, out, arrays), 0)
end

function build_scalar(region::ScatterRegion, runtime, out, expression, arrays)
    launch = scatter_call(runtime, out, expression, arrays, region.count, region.selection)
    BuiltImplementation(launch, (runtime, out, arrays), 0)
end

# Device pointers in mapped expressions are backed by arrays owned by the
# runtime. Reuse each snapshotted gather's upload across its consumers.
gpu_index(r, i) = i

function gpu_index(r, i::IndexList)
    get!(r.layout_cache, i.indices) do
        owner = CuArray(i.indices)
        push!(r.layout_buffers, owner)
        IndexList(CUDACore.cudaconvert(owner))
    end
end

gpu_mapping(r, m::AffineMap) = m

function gpu_mapping(r, m::ComposedMap)
    ComposedMap(gpu_mapping(r, m.parent), gpu_mapping(r, m.localmap))
end

function gpu_mapping(r, m::Selection)
    Selection(m.parentdims, map(i -> gpu_index(r, i), m.indices), m.dims, m.unique)
end


function capture_runtime!(r::CUDARuntime, ex)
    execute!(ex)
    CUDACore.synchronize()
    graph = CUDACore.capture() do
        execute!(ex)
    end
    types = graph_node_types(graph)
    if any(
        t -> t in (CUDACore.CU_GRAPH_NODE_TYPE_MEM_ALLOC, CUDACore.CU_GRAPH_NODE_TYPE_MEM_FREE),
        types,
    )
        throw(
            ArgumentError(
                "capture recorded memory allocation/free nodes; provision all workspace before capture",
            ),
        )
    end
    executable = CUDACore.instantiate(graph)
    r.graph, r.graph_exec = graph, executable
    nothing
end

function replay_runtime!(r::CUDARuntime)
    if r.graph_exec === nothing
        throw(ArgumentError("call capture! first"))
    end
    CUDACore.launch(r.graph_exec)
    nothing
end

# Device time of each call, from events recorded between consecutive calls.
function time_calls(::CUDARuntime, calls)
    events = [CUDACore.CuEvent() for _ = 0:length(calls)]
    CUDACore.record(events[1])
    for (index, call) in enumerate(calls)
        call()
        CUDACore.record(events[index + 1])
    end
    CUDACore.synchronize(events[end])
    [CUDACore.elapsed(events[i], events[i + 1]) for i in eachindex(calls)]
end

function graph_node_types(graph)
    n = Ref{Csize_t}(0)
    CUDACore.cuGraphGetNodes(graph, C_NULL, n)
    nodes = Vector{CUDACore.CUgraphNode}(undef, n[])
    CUDACore.cuGraphGetNodes(graph, nodes, n)
    map(nodes) do node
        t = Ref{CUDACore.CUgraphNodeType}()
        CUDACore.cuGraphNodeGetType(node, t)
        t[]
    end
end

function explain_runtime(io, r::CUDARuntime)
    println(
        io,
        "Target: CUDA (",
        r.device,
        ")",
        r.graph_exec === nothing ? " (uncaptured)" : " (captured)",
    )
    if r.graph !== nothing
        types = graph_node_types(r.graph)
        println(
            io,
            "Captured nodes: ",
            length(types),
            "; kernels: ",
            count(==(CUDACore.CU_GRAPH_NODE_TYPE_KERNEL), types),
        )
    end
    println(io, "Owned layout indices: ", sum(sizeof, r.layout_buffers; init = 0), " bytes")
    if r.blas !== nothing
        scalarbytes = sum(T -> 2sizeof(T), keys(r.blas.coefficients); init = 0)
        math = r.target.matmul_precision === :tf32 ? "CUBLAS_TF32_TENSOR_OP_MATH" : "CUBLAS_DEFAULT_MATH"
        println(io, "Owned cuBLAS coefficients: $scalarbytes bytes; math: $math")
    end
end

end
