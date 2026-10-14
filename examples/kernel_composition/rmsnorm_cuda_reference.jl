# A small, deterministic CUDA kernel library for the integration tests. It has
# no Tempest dependency. These kernels establish the execution contract; they
# are not intended to compete with tuned normalization implementations.
module ReferenceNormKernels

using CUDACore

function block_sum(value)
    shared = CuStaticSharedArray(Float32, 256)
    thread = threadIdx().x
    shared[thread] = value
    sync_threads()
    stride = 128
    while stride > 0
        if thread <= stride
            shared[thread] += shared[thread + stride]
        end
        sync_threads()
        stride ÷= 2
    end
    shared[1]
end

function forward_kernel(
    y::CuDeviceArray,
    rstd::CuDeviceArray,
    x::CuDeviceArray,
    weight::CuDeviceArray,
    eps,
)
    column = blockIdx().x
    total = 0.0f0
    for row = threadIdx().x:256:size(x, 1)
        total += x[row, column] * x[row, column]
    end
    inverse = inv(sqrt(block_sum(total) / size(x, 1) + eps))
    threadIdx().x == 1 && (rstd[column] = inverse)
    for row = threadIdx().x:256:size(x, 1)
        y[row, column] = x[row, column] * inverse * weight[row]
    end
    nothing
end

function input_gradient_kernel(
    dx::CuDeviceArray,
    dy::CuDeviceArray,
    x::CuDeviceArray,
    weight::CuDeviceArray,
    rstd::CuDeviceArray,
)
    column = blockIdx().x
    total = 0.0f0
    for row = threadIdx().x:256:size(x, 1)
        total += dy[row, column] * weight[row] * x[row, column]
    end
    correction = block_sum(total) / size(x, 1)
    inverse = rstd[column]
    for row = threadIdx().x:256:size(x, 1)
        dx[row, column] =
            inverse *
            (dy[row, column] * weight[row] - x[row, column] * inverse * inverse * correction)
    end
    nothing
end

function weight_gradient_kernel(
    dw::CuDeviceArray,
    dy::CuDeviceArray,
    x::CuDeviceArray,
    rstd::CuDeviceArray,
)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if row <= length(dw)
        total = 0.0f0
        for column in axes(x, 2)
            total += dy[row, column] * x[row, column] * rstd[column]
        end
        dw[row] = total
    end
    nothing
end

function forward_call(y::CuArray, rstd::CuArray, x::CuArray, weight::CuArray, eps)
    eps = eltype(x)(eps)
    kernel = CUDACore.@cuda launch=false forward_kernel(y, rstd, x, weight, eps)
    () -> kernel(y, rstd, x, weight, eps; threads = 256, blocks = size(x, 2))
end

function backward_call(
    dx::CuArray,
    dw::CuArray,
    dy::CuArray,
    x::CuArray,
    weight::CuArray,
    rstd::CuArray,
)
    input_kernel = CUDACore.@cuda launch=false input_gradient_kernel(dx, dy, x, weight, rstd)
    weight_kernel = CUDACore.@cuda launch=false weight_gradient_kernel(dw, dy, x, rstd)
    () -> begin
        input_kernel(dx, dy, x, weight, rstd; threads = 256, blocks = size(x, 2))
        weight_kernel(dw, dy, x, rstd; threads = 256, blocks = cld(length(dw), 256))
        nothing
    end
end

end

# The adapter alone knows Tempest. Both allocating reference mathematics and
# concrete destination-writing kernels keep their natural interfaces.
using Tempest:
    Implementation, CallRequest, Supported, Unsupported, BuiltImplementation, ReverseRule

struct ReferenceRMSNorm <: Implementation end
Tempest.implementation_name(::ReferenceRMSNorm) = :reference_rmsnorm

function reference_rms_capability(request)
    request.target isa CUDATarget || return Unsupported("CUDA only")
    all(s -> s.dtype === Float32, request.inputs) || return Unsupported("Float32 only")
    Supported()
end

function Tempest.reverse_rule(::ReferenceRMSNorm, request::CallRequest)
    request.call.f === rmsnorm || return Unsupported("not RMSNorm")
    verdict = reference_rms_capability(request)
    verdict isa Unsupported && return verdict
    ReverseRule(rms_rule)
end

function Tempest.support(::ReferenceRMSNorm, request::CallRequest)
    request.call.f in (rms_forward, rms_backward) ||
        return Unsupported("private RMSNorm calls only")
    reference_rms_capability(request)
end

function Tempest.build_implementation(
    ::ReferenceRMSNorm,
    request::CallRequest,
    runtime,
    out,
    inputs,
)
    launch = if request.call.f === rms_forward
        ReferenceNormKernels.forward_call(out.y, out.rstd, inputs..., request.call.kwargs.eps)
    else
        ReferenceNormKernels.backward_call(out..., inputs...)
    end
    BuiltImplementation(launch, (out, inputs), 0)
end
