# NNlib's functions whose bodies do not trace, as methods on Tempest's traced
# arrays: `softmax` and `logsoftmax` branch on array values, and `batched_mul`
# loops over scalars. Each is written in Tempest's primitives, so it fuses with
# its neighbours. Functions whose bodies trace as they are (the activations,
# `logsumexp`) need nothing here. `dropout` is left out: what NNlib's global
# random number generator means under tracing is an open question.
module NNlibExt

using Tempest, NNlib
import Tempest: logsumexp
import Tempest.Compiler: TracedArray, contract

# Written with the log-sum-exp, a softmax streams: a sum weighted by it
# accumulates online.
NNlib.softmax(x::TracedArray; dims = 1) = exp.(x .- logsumexp(x; dims))
NNlib.logsoftmax(x::TracedArray; dims = 1) = x .- logsumexp(x; dims)

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

end
