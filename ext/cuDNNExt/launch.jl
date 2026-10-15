# Running a built cuDNN graph inside a Tempest schedule. Everything a launch
# touches is fixed outside capture: the execution plan, an owned handle, the
# by-value scalars, and a variant pack holding the final device pointers,
# including the engine workspace from Tempest's shared pool. A launch only sets
# the stream and calls cudnnBackendExecute, which CUDA graph capture records.

# From Tempest's former Stiletto bridge (git show ccdac5a:ext/StilettoExt/StilettoExt.jl).
mutable struct HandleOwner
    handle::cuDNN.cudnnHandle_t
    context::CUDACore.CuContext
end

function HandleOwner()
    owner = HandleOwner(cuDNN.cudnnCreate(), CUDACore.context())
    finalizer(owner) do r
        if CUDACore.isvalid(r.context)
            CUDACore.context!(r.context) do
                cuDNN.cudnnDestroy(r.handle)
            end
        end
    end
    owner
end

# Execution plans do not belong to a handle; the handle only carries the
# stream. One handle per executable runtime therefore serves all of its
# launches, which run in schedule order on one stream.
const HANDLES = WeakKeyDict{Any, HandleOwner}()

runtime_handle(runtime) = get!(HandleOwner, HANDLES, runtime)

"""
A built cuDNN graph bound to final buffers. Each variant pack in `packs` is one
execution of the plan, in order: a graph split into column chunks binds one
pack per chunk. `before` and `after` are dtype conversions around the graph;
`summary` is the line `explain` prints.
"""
struct GraphLaunch{B, A}
    handle::HandleOwner
    graph::Graph
    packs::Vector{cuDNN.BackendDescriptor}
    scalars::Vector{Any}
    before::B
    after::A
    summary::String
end

function (launch::GraphLaunch)()
    foreach(convert! -> convert!(), launch.before)
    handle = launch.handle.handle
    cuDNN.cudnnSetStream(handle, CUDACore.stream())
    for pack in launch.packs
        cuDNN.cudnnBackendExecute(handle, launch.graph.plan.ptr, pack.ptr)
    end
    foreach(convert! -> convert!(), launch.after)
    nothing
end

Tempest.implementation_summary(launch::GraphLaunch) = launch.summary

"""
    bind_graph(graph, values, workspace; offsets = nothing) -> (pack, scalars)

Build the variant pack for a built `graph`. `values` maps the name of every
non-virtual tensor to a dense CUDA array or, for by-value tensors, a number.
The arrays must outlive the pack; `scalars` holds the numbers' storage. With
`offsets`, a map from names to byte offsets, the graph covers part of each
array (a column chunk) at that offset: its tensors are laid out with the whole
array's strides, which the caller guarantees, so only the element type is
checked.
"""
function bind_graph(graph::Graph, values::AbstractDict{String}, workspace; offsets = nothing)
    scalars = Any[]
    pointers = map(graph.variant_tensors) do t
        value = values[t.name]
        if t.by_value
            ref = Ref{cuDNN.juliaDataType(t.dtype)}(value)
            push!(scalars, ref)
            return Base.unsafe_convert(Ptr{eltype(ref)}, ref)
        end
        pointer = if offsets === nothing
            cuDNN.checked_array_pointer(t, value)
        else
            cuDNN.cudnnDataType(eltype(value)) == t.dtype ||
                throw(ArgumentError("binding for $(t.name) has eltype $(eltype(value))"))
            Base.pointer(value) + offsets[t.name]
        end
        if UInt(pointer) % t.alignment != 0
            throw(ArgumentError("buffer for cuDNN tensor $(t.name) is less aligned than declared"))
        end
        pointer
    end
    if sizeof(workspace) < graph.workspace_size
        throw(ArgumentError("cuDNN engine workspace is smaller than its plan requires"))
    end
    pack = cuDNN.variant_pack(;
        uids = Int64[t.uid for t in graph.variant_tensors],
        pointers,
        workspace = graph.workspace_size == 0 ? C_NULL : pointer(workspace),
    )
    pack, scalars
end

# Build `graph` with engines that keep the declared operand types: tensor
# cores, but no engine that down-converts inputs. Unavailable engines become a
# reason; invalid graphs and other errors propagate.
function build_graph(graph::Graph, description)
    try
        cuDNN.build!(graph; math_mode = CUDACore.DEFAULT_MATH, deterministic = true)
    catch err
        cuDNN.graph_unsupported(err) || rethrow()
        return Unsupported("cuDNN has no engine for $description")
    end
    graph
end

# Elementwise dtype conversion between dense arrays of one layout, compiled
# outside capture.
function convert_kernel(destination, source)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    if i <= length(destination)
        @inbounds destination[i] = convert(eltype(destination), source[i])
    end
    return
end

function conversion(destination, source)
    length(destination) == length(source) ||
        throw(DimensionMismatch("conversion buffers have different lengths"))
    isempty(destination) && return () -> nothing
    kernel = CUDACore.@cuda launch = false convert_kernel(destination, source)
    threads = 256
    blocks = cld(length(destination), threads)
    () -> kernel(destination, source; threads, blocks)
end

# A buffer of `dims` at the start of a (possibly larger) workspace slot.
workspace_array(buffer, dims) = reshape(view(buffer, 1:prod(dims)), dims)
