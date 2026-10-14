# A training integration fixture. Only this module knows RMSNorm; the TensorIR
# provider compiles outlined array graphs without recognizing their names.
if !isdefined(@__MODULE__, :TensorIRBackend)
    include("tensor_ir_backend.jl")
end

module TensorIRNormalization

using Tempest
using Tempest:
    Implementation,
    CallRequest,
    Unsupported,
    support,
    build_implementation,
    TensorSpec,
    output_spec,
    record_call,
    ReverseRule
using ..TensorIRBackend

export residual_rmsnorm, NormalizationImplementation

function inverse_rms(h, eps)
    inv.(sqrt.(sum(h .* h; dims = 1) ./ size(h, 1) .+ eltype(h)(eps)))
end

function residual_rmsnorm(x, residual, weight; eps = 1f-5)
    h = x .+ residual
    h .* inverse_rms(h, eps) .* weight
end
residual_rmsnorm(x::Tempest.TracedArray, residual, weight; eps = 1f-5) =
    record_call(residual_rmsnorm, x, residual, weight; eps)

function Tempest.output_spec(::typeof(residual_rmsnorm), x, residual, weight; eps)
    ndims(x) == 2 && size(residual) == size(x) && size(weight) == (size(x, 1),) ||
        throw(DimensionMismatch("residual RMSNorm needs two matrices and one weight per row"))
    eltype(x) === eltype(residual) === eltype(weight) ||
        throw(ArgumentError("residual RMSNorm input types must match"))
    isfinite(eps) && eps > 0 || throw(ArgumentError("epsilon must be finite and positive"))
    TensorSpec(x)
end

function Tempest.reference(::typeof(residual_rmsnorm), x, residual, weight; eps)
    h = x .+ residual
    h .* inverse_rms(h, eps) .* weight
end

function input_backward(dy, h, weight, rstd)
    scaled = dy .* weight
    correction = sum(scaled .* h; dims = 1) ./ size(h, 1)
    rstd .* (scaled .- h .* rstd .* rstd .* correction)
end

function weight_backward(dy, h, rstd)
    reshape(sum(dy .* h .* rstd; dims = 2), (size(h, 1),))
end

# The reference derivative is independent of the selected compiler. It computes
# its own statistic; the specialized rule below reuses the forward's statistic.
function Tempest.verb_pullback(::typeof(residual_rmsnorm), inputs, y, dy; eps)
    x, residual, weight = inputs
    h = x .+ residual
    rstd = inverse_rms(h, eps)
    dh = input_backward(dy, h, weight, rstd)
    (dh, dh, weight_backward(dy, h, rstd))
end

function forward_saved(x, residual, weight; eps)
    h = x .+ residual
    rstd = inverse_rms(h, eps)
    (y = h .* rstd .* weight, residual = h, rstd = rstd)
end

function forward_specs(x, residual, weight; eps)
    y = output_spec(residual_rmsnorm, x, residual, weight; eps)
    (y = y, residual = TensorSpec(x), rstd = TensorSpec(eltype(x), (1, size(x, 2))))
end

const forward_region = GraphCall(:residual_rms_forward, forward_saved, forward_specs)
const input_region = GraphCall(:rms_input_backward, input_backward,
                              (dy, h, weight, rstd) -> TensorSpec(h))
const weight_region = GraphCall(:rms_weight_backward, weight_backward,
                               (dy, h, rstd) -> TensorSpec(eltype(h), (size(h, 1),)))

function saved_rule(x, residual, weight; eps)
    saved = forward_region(x, residual, weight; eps)
    function pullback(dy)
        dh = input_region(dy, saved.residual, weight, saved.rstd)
        dw = weight_region(dy, saved.residual, saved.rstd)
        (dh, dh, dw)
    end
    saved.y, pullback
end

struct NormalizationImplementation <: Implementation
    regions::TensorIRImplementation
end
NormalizationImplementation(compiler::Compiler) =
    NormalizationImplementation(TensorIRImplementation(compiler))
Tempest.implementation_name(::NormalizationImplementation) = :tensor_ir

function Tempest.reverse_rule(::NormalizationImplementation, request::CallRequest)
    request.call.f === residual_rmsnorm || return Unsupported("not the residual RMSNorm fixture")
    request.target isa CUDATarget || return Unsupported("requires CUDA")
    all(s -> s.dtype === Float32, request.inputs) || return Unsupported("Float32 only")
    ReverseRule(saved_rule)
end

Tempest.support(impl::NormalizationImplementation, request::CallRequest) =
    support(impl.regions, request)
Tempest.build_implementation(impl::NormalizationImplementation, request::CallRequest,
                             runtime, out, inputs) =
    build_implementation(impl.regions, request, runtime, out, inputs)

end # module
