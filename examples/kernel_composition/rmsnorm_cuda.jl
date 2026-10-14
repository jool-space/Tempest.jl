include("archived_library.jl")

using CUDACore
import cuTile
using Tempest:
    Implementation,
    CallRequest,
    Supported,
    Unsupported,
    BuiltImplementation,
    ImplementationPlan,
    WorkspaceSpec,
    ReverseRule

struct ArchivedRMSNorm <: Implementation end
Tempest.implementation_name(::ArchivedRMSNorm) = :archived_rmsnorm

function rms_capability(request)
    request.target isa CUDATarget || return Unsupported("RMSNorm wrapper requires CUDA")
    all(s -> s.dtype === Float32, request.inputs) || return Unsupported("Float32 only")
    Supported()
end

function Tempest.reverse_rule(::ArchivedRMSNorm, request::CallRequest)
    request.call.f === rmsnorm || return Unsupported("not RMSNorm")
    verdict = rms_capability(request)
    verdict isa Unsupported && return verdict
    ReverseRule(rms_rule)
end

function Tempest.support(::ArchivedRMSNorm, request::CallRequest)
    request.call.f in (rms_forward, rms_backward) ||
        return Unsupported("private RMSNorm calls only")
    rms_capability(request)
end

# The adapter's launch wrappers deliberately require concrete CUDA arrays. They
# never see a tracer, and they call the original library's mutating wrappers.
function rms_forward_call(out, x::CuArray, weight::CuArray, eps)
    kernel = CUDACore.@cuda backend=cuTile launch=false ArchivedRMSKernels.rms_norm_fwd(
        x,
        weight,
        out.y,
        out.rstd,
        0.0f0,
        Float32(eps),
        cuTile.Constant(256),
    )
    checkpoints = (; Rstd = out.rstd)
    launch =
        () -> ArchivedRMSKernels.rms_norm!(out.y, x, weight; eps, TILE_M = 256, checkpoints)
    BuiltImplementation(launch, (out, x, weight, kernel, checkpoints), 0)
end

function rms_backward_call(
    out,
    dy::CuArray,
    x::CuArray,
    weight::CuArray,
    rstd::CuArray,
    buffers,
)
    rows, columns = size(x)
    # One group per column removes lock contention and fixes the accumulation
    # order in this experiment. This intentionally spends rows*columns scratch;
    # it is not a claim that the archived backward is the best implementation.
    partials = reshape(view(buffers[1], 1:(rows * columns)), rows, columns)
    locks = view(buffers[2], 1:columns)
    fill!(partials, 0)
    fill!(locks, 0)
    first_kernel =
        CUDACore.@cuda backend=cuTile launch=false ArchivedRMSKernels.rms_norm_bwd_dx_partial_dw(
            out[1],
            dy,
            partials,
            x,
            weight,
            rstd,
            locks,
            cuTile.Constant(0.0f0),
            cuTile.Constant(columns),
            cuTile.Constant(256),
        )
    second_kernel =
        CUDACore.@cuda backend=cuTile launch=false ArchivedRMSKernels.rms_norm_bwd_dw(
            partials,
            out[2],
            cuTile.Constant(64),
            cuTile.Constant(64),
        )
    checkpoints = (; Rstd = rstd)
    scratch = (; W̄_partial = partials, Locks = locks)
    Duplicated = ArchivedRMSKernels.Duplicated
    launch =
        () -> ArchivedRMSKernels.∇rms_norm!(
            Duplicated(dy, dy),
            Duplicated(x, out[1]),
            Duplicated(weight, out[2]);
            checkpoints,
            scratch,
            N_GROUPS = columns,
        )
    BuiltImplementation(
        launch,
        (out, dy, x, weight, rstd, buffers, first_kernel, second_kernel),
        0,
    )
end

function Tempest.build_implementation(
    ::ArchivedRMSNorm,
    request::CallRequest,
    runtime,
    out,
    inputs,
)
    if request.call.f === rms_forward
        return rms_forward_call(out, inputs..., request.call.kwargs.eps)
    end
    rows, columns = request.inputs[2].dims
    ImplementationPlan((
        WorkspaceSpec(Float32, rows * columns),
        WorkspaceSpec(Int, columns),
    )) do buffers
        rms_backward_call(out, inputs..., buffers)
    end
end
