# Attention and dropout as plain array code on Tempest's primitives: what a
# model author writes. Tests and benchmarks include this file into a module
# `Ops`. Nothing here is special to the compiler; it plans the bodies.

# Scaled dot-product attention. Arrays are (features, heads, positions,
# batch); query head h reads key/value head cld(h, G), where G is the number
# of query heads per key/value head. The forward pass is two contractions
# around a softmax. Heads group as (G, kv): a query array reshaped to
# (d, G, kv, M, B) shares the kv label with the keys, so the contractions need
# no gather.

"""
    attention(q, k, v; causal = false, scale = 1 / sqrt(size(q, 1)))

Scaled dot-product attention over `(features, heads, positions, batch)` arrays,
with grouped-query attention when `k` and `v` have fewer heads than `q`. Three
dimensional arrays are one head.
"""
function attention(q, k, v; causal::Bool = false, scale::Real = inv(sqrt(size(q, 1))))
    if ndims(q) == ndims(k) == ndims(v) == 3
        head(x) = reshape(x, size(x, 1), 1, size(x, 2), size(x, 3))
        o = attention(head(q), head(k), head(v); causal, scale)
        return reshape(o, size(o, 1), size(o, 3), size(o, 4))
    end
    attention_checks(q, k, v, causal)
    attention_forward(q, k, v; causal, scale = Float32(scale)).o
end

function attention_checks(q, k, v, causal)
    ndims(q) == ndims(k) == ndims(v) == 4 ||
        throw(ArgumentError("attention takes (features, heads, positions, batch) arrays"))
    size(q, 1) == size(k, 1) || throw(DimensionMismatch("query and key features differ"))
    size(k)[2:4] == size(v)[2:4] || throw(DimensionMismatch("keys and values differ"))
    size(q, 4) == size(k, 4) || throw(DimensionMismatch("attention batches differ"))
    size(q, 2) % size(k, 2) == 0 ||
        throw(DimensionMismatch("query heads must be a multiple of key/value heads"))
    causal && size(q, 3) != size(k, 3) &&
        throw(DimensionMismatch("causal attention needs as many queries as keys"))
    nothing
end

# Labels: d features, g query heads per kv head, h kv heads, m queries,
# b batch, n keys.
const D_, G_, H_, M_, B_, N_ = 1, 2, 3, 4, 5, 6

grouped(x, kv) = reshape(x, size(x, 1), size(x, 2) ÷ kv, kv, size(x, 3), size(x, 4))

# Scores (n, g, h, m, b), scaled, keys after a query masked out when causal.
function attention_scores(q, k; causal, scale)
    s = Tempest.contract(k, grouped(q, size(k, 2)), (D_, H_, N_, B_), (D_, G_, H_, M_, B_), (N_, G_, H_, M_, B_)) .* scale
    causal || return s
    ifelse.(Tempest.axis_positions(s, 1) .<= Tempest.axis_positions(s, 4), s, typemin(eltype(s)))
end

"""
    attention_forward(q, k, v; causal, scale) -> (o, lse)

Attention's output and the log-sum-exp of each query's scores, of size
`(1, heads, queries, batch)`.
"""
function attention_forward(q, k, v; causal, scale)
    s = attention_scores(q, k; causal, scale)
    lse = Tempest.logsumexp(s; dims = 1)
    o = Tempest.contract(v, exp.(s .- lse), (D_, H_, N_, B_), (N_, G_, H_, M_, B_), (D_, G_, H_, M_, B_))
    (o = reshape(o, size(q, 1), size(q, 2), size(q, 3), size(q, 4)), lse = reshape(lse, 1, size(q, 2), size(q, 3), size(q, 4)))
end

# Dropout as plain array code: a pointwise function of the element, the seed
# and the position, so it fuses with the work around it and its derivative
# redraws the mask.

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
Tempest.scalar_partials(::DropoutScale, g, y, key, i) = (nothing, nothing)

"""
    dropout(x, seed; rate, stream = 0)

Zero each element of `x` with probability `rate` and scale the rest by
`1 / (1 - rate)`. `seed` is an integer array whose first element keys the draw;
distinct dropout sites should use distinct `stream` numbers.
"""
function dropout(x, seed; rate::Real, stream::Integer = 0)
    0 <= rate < 1 || throw(ArgumentError("dropout rate must lie in [0, 1)"))
    eltype(seed) <: Integer || throw(ArgumentError("dropout seeds are integer arrays"))
    x .* DropoutScale{eltype(x)}(rate, stream).(reshape(seed[1:1], ntuple(_ -> 1, ndims(x))), Tempest.positions(x))
end
