# NNlib's functions whose bodies do not trace, as operations: `softmax` and
# `logsoftmax` branch on array values, and `batched_mul` loops over scalars.
# Each keeps its name in the program, with a reference in Tempest's
# primitives, so it fuses with its neighbours. Functions whose bodies trace as
# they are (the activations, `logsumexp`) need nothing here. `dropout` is left
# out: what NNlib's global random number generator means under tracing is an
# open question.
module NNlibExt

using Tempest, NNlib
import Tempest: @operation, contract, logsumexp
import Tempest.Compiler: TracedArray

@operation NNlib.softmax(x::AbstractArray; dims = 1)
@operation NNlib.logsoftmax(x::AbstractArray; dims = 1)
@operation NNlib.batched_mul(a::AbstractArray{<:Any, 3}, b::AbstractArray{<:Any, 3})

# Written with the log-sum-exp, a softmax streams: a sum weighted by it
# accumulates online.
Tempest.reference(::typeof(NNlib.softmax), x; dims = 1) = exp.(x .- logsumexp(x; dims))
Tempest.reference(::typeof(NNlib.logsoftmax), x; dims = 1) = x .- logsumexp(x; dims)

# A batch of one is shared by every batch of the other factor.
function Tempest.reference(::typeof(NNlib.batched_mul), a, b)
    size(a, 3) == size(b, 3) && return contract(a, b, (1, 2, 4), (2, 3, 4), (1, 3, 4))
    size(a, 3) == 1 && return contract(reshape(a, size(a, 1), size(a, 2)), b, (1, 2), (2, 3, 4), (1, 3, 4))
    size(b, 3) == 1 && return contract(a, reshape(b, size(b, 1), size(b, 2)), (1, 2, 4), (2, 3), (1, 3, 4))
    throw(DimensionMismatch("batch sizes $(size(a, 3)) and $(size(b, 3)) differ"))
end

# A batched transpose is a layout that shares its array's memory, as NNlib's
# lazy one does.
NNlib.batched_transpose(a::TracedArray{<:Real, 3}) = Tempest.Compiler.permuted(a, (2, 1, 3))
NNlib.batched_adjoint(a::TracedArray{<:Real, 3}) = Tempest.Compiler.permuted(a, (2, 1, 3))

end
