# NNlib's functions whose bodies do not trace, as methods on Tempest's traced
# arrays and numbers: `softmax` and `logsoftmax` branch on array values,
# `batched_mul`, pooling, convolution and linear upsampling loop over
# scalars, and `softplus` branches on its argument.
# Each is written in Tempest's primitives, so it fuses with its neighbours.
# Functions whose bodies trace as they are (the other activations,
# `logsumexp`) need nothing here. `dropout` is left out: what NNlib's global
# random number generator means under tracing is an open question.
module NNlibExt

using Tempest, NNlib
import Tempest: logsumexp
import Tempest.Compiler: TracedArray, Traced, contract

# Written with the log-sum-exp, a softmax streams: a sum weighted by it
# accumulates online.
NNlib.softmax(x::TracedArray; dims = 1) = exp.(x .- logsumexp(x; dims))
NNlib.logsoftmax(x::TracedArray; dims = 1) = x .- logsumexp(x; dims)

# NNlib writes `softplus` without branches for Float32 and Float64 only, and a
# traced number takes the same form: both terms switch below zero, so the
# derivative at zero is one half. `logσ` and `mish` are written with it.
function NNlib.softplus(x::Traced)
    negative = x < 0
    ifelse(negative, zero(x), x) + log1p(exp(ifelse(negative, x, -x)))
end

# Inside `Tempest.gradient`, as inside Zygote's: Flux's normalisations and
# dropout train there.
NNlib.within_gradient(x::TracedArray) = Tempest.Compiler.differentiating(x)

# A batch of one is shared by every batch of the other factor.
function batched_product(a, b)
    size(a, 3) == size(b, 3) && return contract(a, b, (1, 2, 4), (2, 3, 4), (1, 3, 4))
    size(a, 3) == 1 && return contract(reshape(a, size(a, 1), size(a, 2)), b, (1, 2), (2, 3, 4), (1, 3, 4))
    size(b, 3) == 1 && return contract(a, reshape(b, size(b, 1), size(b, 2)), (1, 2, 4), (2, 3), (1, 3, 4))
    throw(DimensionMismatch("batch sizes $(size(a, 3)) and $(size(b, 3)) differ"))
end

# Each product with a traced factor, the other traced or not.
for (A, B) in ((TracedArray, TracedArray), (TracedArray, AbstractArray), (AbstractArray, TracedArray))
    @eval NNlib.batched_mul(a::$A{<:Any, 3}, b::$B{<:Any, 3}) = batched_product(a, b)
end

# A batched transpose is a layout that shares its array's memory, as NNlib's
# lazy one does.
NNlib.batched_transpose(a::TracedArray{<:Real, 3}) = PermutedDimsArray(a, (2, 1, 3))
NNlib.batched_adjoint(a::TracedArray{<:Real, 3}) = PermutedDimsArray(a, (2, 1, 3))

# Pooling folds each window of `x`. Windows that tile their axes are blocks of a
# reshape, folded over the blocks' axes; others are the slices that an offset
# within the window selects of every window, strided as the windows are, and
# combined elementwise. Padding holds the fold's neutral value, and reaches as
# far as the last window does, past the input where ceil mode lets it run over.
function pooled(fold, combine, neutral, x, pdims)
    n = ndims(x) - 2
    k, s, d = NNlib.kernel_size(pdims), NNlib.stride(pdims), NNlib.dilation(pdims)
    out = NNlib.output_size(pdims)
    x = padded(x, NNlib.padding(pdims), ntuple(i -> (out[i] - 1) * s[i] + (k[i] - 1) * d[i] + 1, n), neutral)
    rest = size(x)[(n + 1):end]
    if k == s && all(==(1), d)
        tiled = x[ntuple(i -> 1:(out[i] * k[i]), n)..., :, :]
        blocks = reshape(tiled, (Iterators.flatten(zip(k, out))..., rest...))
        return reshape(fold(blocks; dims = ntuple(i -> 2i - 1, n)), (out..., rest...))
    end
    slices = map(CartesianIndices(k)) do o
        x[ntuple(i -> range((o[i] - 1) * d[i] + 1; step = s[i], length = out[i]), n)..., :, :]
    end
    reduce((a, b) -> combine.(a, b), vec(slices))
end

# `x` with its spatial axes padded by `value`: low as `padding` says, high as far
# as `reach`, or as `padding` says if it says more.
function padded(x, padding, reach, value)
    n = length(reach)
    low = ntuple(i -> padding[2i - 1], n)
    high = ntuple(i -> max(padding[2i], reach[i] - low[i] - size(x, i)), n)
    all(==(0), low) && all(==(0), high) && return x
    NNlib.pad_constant(x, (map(tuple, low, high)..., (0, 0), (0, 0)), value)
end

NNlib.maxpool(x::TracedArray{<:Real}, pdims::NNlib.PoolDims) = pooled(maximum, max, typemin(eltype(x)), x, pdims)

function NNlib.meanpool(x::TracedArray{<:Real}, pdims::NNlib.PoolDims; count_include_pad = true)
    total = pooled(sum, +, zero(eltype(x)), x, pdims)
    count_include_pad && return total ./ prod(NNlib.kernel_size(pdims))
    # How many of each window's elements are the input's: the same pooling of
    # ones, untraced, a constant of the program.
    total ./ pooled(sum, +, 0, ones(Int32, size(x)), pdims)
end

# A convolution sums over the kernel's offsets: at each, the slice of `x` its
# windows select at that offset, padded with zeros, times the kernel's slice
# there, summed over a group's input channels. Without `flipkernel` the kernel
# is flipped, as a convolution's is.
function convolved(x, w, cdims)
    n = ndims(x) - 2
    k, s, d = NNlib.kernel_size(cdims), NNlib.stride(cdims), NNlib.dilation(cdims)
    out = NNlib.output_size(cdims)
    groups = cdims isa NNlib.DepthwiseConvDims ? NNlib.channels_in(cdims) : NNlib.groupcount(cdims)
    cin, cout, batch = NNlib.channels_in(cdims) ÷ groups, NNlib.channels_out(cdims) ÷ groups, size(x, n + 2)
    x = padded(x, NNlib.padding(cdims), ntuple(i -> (out[i] - 1) * s[i] + (k[i] - 1) * d[i] + 1, n), zero(eltype(x)))
    x = reshape(x, (size(x)[1:n]..., cin, groups, batch))
    w = reshape(w, (k..., cin, cout, groups))
    spatial = ntuple(identity, n)
    terms = map(CartesianIndices(k)) do o
        slice = x[ntuple(i -> range((o[i] - 1) * d[i] + 1; step = s[i], length = out[i]), n)..., :, :, :]
        at = NNlib.flipkernel(cdims) ? Tuple(o) : k .- Tuple(o) .+ 1
        contract(slice, w[at..., :, :, :], (spatial..., n + 1, n + 2, n + 3), (n + 1, n + 4, n + 2), (spatial..., n + 4, n + 2, n + 3))
    end
    reshape(reduce(+, vec(terms)), (out..., cout * groups, batch))
end

# The data's gradient of a convolution, as a transposed convolution is: what
# Tempest's derivative of `convolved` gives, at any input, for `dy`.
function convolved_data(dy, w, cdims)
    x = zeros(eltype(dy), NNlib.input_size(cdims)..., NNlib.channels_in(cdims), size(dy, ndims(dy)))
    Tempest.gradient(x -> sum(convolved(x, w, cdims) .* dy), x)[2]
end

# Each with a traced array, the other traced or not.
for (A, B) in ((TracedArray, TracedArray), (TracedArray, AbstractArray), (AbstractArray, TracedArray))
    @eval begin
        NNlib.conv(x::$A{<:Any, N}, w::$B{<:Any, N}, cdims::NNlib.DenseConvDims; kwargs...) where {N} = convolved(x, w, cdims)
        NNlib.depthwiseconv(x::$A{<:Any, N}, w::$B{<:Any, N}, cdims::NNlib.DepthwiseConvDims; kwargs...) where {N} =
            convolved(x, w, cdims)
        NNlib.∇conv_data(dy::$A{<:Any, N}, w::$B{<:Any, N}, cdims::NNlib.DenseConvDims; kwargs...) where {N} =
            convolved_data(dy, w, cdims)
    end
end


# Linear upsampling, bilinear and trilinear too, is separable: along each
# spatial axis, a product with the matrix that NNlib's own upsampling makes of
# the identity, untraced.
function NNlib.upsample_linear(x::TracedArray{<:AbstractFloat, N}; size, align_corners::Bool = true) where {N}
    out = Tuple(size)
    Base.size(x)[1:(N - 2)] == out && return x
    labels = ntuple(identity, N)
    for i in 1:(N - 2)
        n = Base.size(x, i)
        unit = reshape(eltype(x).((1:n) .== (1:n)'), n, n, 1)
        weights = reshape(NNlib.upsample_linear(unit; size = out[i], align_corners), out[i], n)
        x = contract(weights, x, (N + 1, i), labels, ntuple(j -> j == i ? N + 1 : j, N))
    end
    x
end

end
