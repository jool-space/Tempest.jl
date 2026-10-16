# Dropout as a verb with no realization of its own: it always expands to its
# reference, a pointwise function of the element, the seed and the position,
# so it fuses with the work around it and its derivative redraws the mask.

"""
The keep-scale of one element: `1 / (1 - rate)` or zero, from a counter hash of
the seed, the stream and the position, the same on every target.
"""
struct DropoutScale{T}
    rate::Float32
    stream::UInt32
end

function (d::DropoutScale{T})(key::Integer, i::Integer) where {T}
    h = (UInt32(key % UInt32) * 0x9e3779b9) ⊻ (d.stream * 0x85ebca6b) ⊻ (UInt32(i % UInt32) * 0xc2b2ae35)
    h ⊻= h >> 16
    h *= 0x7feb352d
    h ⊻= h >> 15
    h *= 0x846ca68b
    h ⊻= h >> 16
    u = Float32(h >> 8) * Float32(2.0^-24)
    u >= d.rate ? T(1 / (1 - d.rate)) : zero(T)
end

# The scale depends only on integers.
scalar_partials(::DropoutScale, g, y, key, i) = (nothing, nothing)

"""
    Tempest.dropout(x, seed; rate, stream = 0)

Zero each element of `x` with probability `rate` and scale the rest by
`1 / (1 - rate)`. `seed` is an integer array whose first element keys the draw;
distinct dropout sites should use distinct `stream` numbers.
"""
function dropout(x, seed; rate::Real, stream::Integer = 0)
    0 <= rate < 1 || throw(ArgumentError("dropout rate must lie in [0, 1)"))
    eltype(seed) <: Integer || throw(ArgumentError("dropout seeds are integer arrays"))
    call(dropout, x, seed; rate = Float32(rate), stream = UInt32(stream))
end

reference(::typeof(dropout), x, seed; rate, stream) =
    x .* DropoutScale{eltype(x)}(rate, stream).(seed[1], positions(x))
