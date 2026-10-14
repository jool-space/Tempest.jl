function support(impl::CUDAKernels, request::GroupedRequest)
    first_region = Tempest.region_request(first(request.requests)).step.region
    first_region isa Elementwise || return Unsupported("groups require elementwise regions")
    for member in request.requests
        verdict = Tempest.checked_support(impl, member)
        verdict isa Unsupported && return verdict
        region = Tempest.region_request(member).step.region
        typeof(region) === typeof(first_region) ||
            return Unsupported("group expression types differ")
        isbits(region.expression) ||
            return Unsupported("group expressions require bits metadata")
    end
    Supported()
end

# Descriptor tables have a fixed element type, independent of the number of
# arrays. Each block chooses one array; array shapes are data, not specialization.
function grouped_pointwise_kernel(outputs, expressions, inputs)
    group = CUDACore.blockIdx().y
    i = (CUDACore.blockIdx().x - 1) * CUDACore.blockDim().x + CUDACore.threadIdx().x
    @inbounds out = outputs[group]
    if i <= length(out)
        @inbounds out[i] = device_eval(expressions[group], inputs[group], i)
    end
    return
end

function descriptor_table(values)
    T = typeof(first(values))
    all(value -> typeof(value) === T, values) || return nothing
    CuArray(T[values...])
end

function build_implementation(::CUDAKernels, request::GroupedRequest, runtime, outputs, inputs)
    regions = [Tempest.region_request(member).step.region for member in request.requests]
    device_outputs = descriptor_table(map(CUDACore.cudaconvert, outputs))
    device_inputs = descriptor_table([map(CUDACore.cudaconvert, arrays) for arrays in inputs])
    expressions =
        descriptor_table([device_expression(runtime, region.expression) for region in regions])
    if device_outputs === nothing || device_inputs === nothing || expressions === nothing
        return Unsupported("group members need matching descriptor types")
    end
    blocks = (cld(maximum(region -> region.count, regions), TILE_SIZE), length(regions))
    kernel = CUDACore.@cuda launch=false grouped_pointwise_kernel(
        device_outputs,
        expressions,
        device_inputs,
    )
    launch =
        () -> begin
            kernel(device_outputs, expressions, device_inputs; threads = TILE_SIZE, blocks)
            nothing
        end
    bytes = sizeof(device_outputs) + sizeof(device_inputs) + sizeof(expressions)
    BuiltImplementation(
        launch,
        (outputs, inputs, device_outputs, device_inputs, expressions),
        bytes,
    )
end
