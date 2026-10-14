module RMSNormExample

using Tempest
using Tempest: TensorSpec, output_spec, reference, record_call

export rmsnorm, rms_forward, rms_backward, rms_rule

"""Reference RMSNorm, with one feature weight per row and a fixed epsilon."""
rmsnorm(x, weight; eps = 1.0f-5) = reference(rmsnorm, x, weight; eps)
rmsnorm(x::Tempest.TracedArray, weight; eps = 1.0f-5) = record_call(rmsnorm, x, weight; eps)

function Tempest.output_spec(::typeof(rmsnorm), x, weight; eps)
    ndims(x) == 2 && size(weight) == (size(x, 1),) ||
        throw(DimensionMismatch("RMSNorm expects a matrix and one weight per feature"))
    eltype(x) === eltype(weight) || throw(ArgumentError("RMSNorm dtypes must match"))
    TensorSpec(x)
end

function inverse_rms(x, eps)
    eps = eltype(x)(eps)
    one(eltype(x)) ./ sqrt.(sum(abs2, x; dims = 1) ./ size(x, 1) .+ eps)
end

Tempest.reference(::typeof(rmsnorm), x, weight; eps) = x .* inverse_rms(x, eps) .* weight

function rms_derivative(x, weight, rstd, dy)
    scaled = dy .* weight
    correction = sum(scaled .* x; dims = 1) ./ size(x, 1)
    dx = rstd .* (scaled .- x .* rstd .* rstd .* correction)
    dw = reshape(sum(dy .* x .* rstd; dims = 2), size(weight))
    (dx, dw)
end

function Tempest.verb_pullback(::typeof(rmsnorm), xs, y, dy; eps)
    x, weight = xs
    rms_derivative(x, weight, inverse_rms(x, eps), dy)
end

# These functions are the adapter's private vocabulary. The model only sees
# rmsnorm's single output; the selected rule introduces the extra statistic.
rms_forward(x::Tempest.TracedArray, weight; eps) = record_call(rms_forward, x, weight; eps)

function Tempest.output_spec(::typeof(rms_forward), x, weight; eps)
    (; y = output_spec(rmsnorm, x, weight; eps), rstd = TensorSpec(eltype(x), (size(x, 2),)))
end

function Tempest.reference(::typeof(rms_forward), x, weight; eps)
    rstd = inverse_rms(x, eps)
    (; y = x .* rstd .* weight, rstd = reshape(rstd, (size(x, 2),)))
end

rms_backward(dy::Tempest.TracedArray, x, weight, rstd) =
    record_call(rms_backward, dy, x, weight, rstd)
Tempest.output_spec(::typeof(rms_backward), dy, x, weight, rstd) =
    (TensorSpec(x), TensorSpec(weight))
Tempest.reference(::typeof(rms_backward), dy, x, weight, rstd) =
    rms_derivative(x, weight, reshape(rstd, (1, size(x, 2))), dy)

function rms_rule(x, weight; eps)
    saved = rms_forward(x, weight; eps)
    saved.y, dy -> rms_backward(dy, x, weight, saved.rstd)
end

end
