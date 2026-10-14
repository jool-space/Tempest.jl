# Generated scalar kernels: support and construction of CUDAKernels for
# elementwise, reduction and scatter regions.
function support(impl::CUDAKernels, request::InPlaceRequest)
    request.region.step.region isa Elementwise ||
        return Unsupported("in-place execution requires elementwise work")
    support(impl, request.region)
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
