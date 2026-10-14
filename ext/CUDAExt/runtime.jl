# The CUDA runtime owns a prepared executable's device, context, cuBLAS
# resources, captured graph and uploaded index arrays. Everything is created
# during preparation; capture rejects graphs that would allocate.
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
