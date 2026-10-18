# Scaled dot-product attention as an operation. Arrays are (features, heads,
# positions, batch); query head h reads key/value head cld(h, G), where G is
# the number of query heads per key/value head.
#
# The forward and backward operations are written as two contractions around
# a softmax, which is also their reference. Heads group as (G, kv): a query
# array reshaped to (d, G, kv, M, B) shares the kv label with the keys, so the
# contractions need no gather.

"""
    Tempest.Operations.attention(q, k, v; causal = false, scale = 1 / sqrt(size(q, 1)))

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
    s = contract(k, grouped(q, size(k, 2)), (D_, H_, N_, B_), (D_, G_, H_, M_, B_), (N_, G_, H_, M_, B_)) .* scale
    causal || return s
    ifelse.(axis_positions(s, 1) .<= axis_positions(s, 4), s, typemin(eltype(s)))
end

"""
    Tempest.Operations.attention_forward(q, k, v; causal, scale) -> (o, lse)

Attention's output and the log-sum-exp of each query's scores, of size
`(1, heads, queries, batch)`.
"""
function attention_forward(q, k, v; causal, scale)
    s = attention_scores(q, k; causal, scale)
    lse = logsumexp(s; dims = 1)
    o = contract(v, exp.(s .- lse), (D_, H_, N_, B_), (N_, G_, H_, M_, B_), (D_, G_, H_, M_, B_))
    (o = reshape(o, size(q, 1), size(q, 2), size(q, 3), size(q, 4)), lse = reshape(lse, 1, size(q, 2), size(q, 3), size(q, 4)))
end

@operation attention_forward(q, k, v; causal, scale)

has_pullback(::typeof(attention_forward)) = true

verb_pullback(::typeof(attention_forward), (q, k, v), y, dy; causal, scale) =
    Tuple(attention_backward(q, k, v, y.o, y.lse, dy.o, dy.lse; causal, scale))

"""
    Tempest.Operations.attention_backward(q, k, v, o, lse, dout, dlse; causal, scale) -> (dq, dk, dv)

The pullback of attention, as an operation of its own so a kernel can claim
it. With p the probabilities and delta = Σ_d dout·o − dlse, the scores'
cotangent is ds = p (dp − delta) scale.
"""
function attention_backward(q, k, v, o, lse, dout, dlse; causal, scale)
    kv = size(k, 2)
    s = attention_scores(q, k; causal, scale)
    p = exp.(s .- grouped(lse, kv))
    g = grouped(dout, kv)
    delta = sum(g .* grouped(o, kv); dims = 1) .- grouped(dlse, kv)
    dv = contract(g, p, (D_, G_, H_, M_, B_), (N_, G_, H_, M_, B_), (D_, H_, N_, B_))
    dp = contract(v, g, (D_, H_, N_, B_), (D_, G_, H_, M_, B_), (N_, G_, H_, M_, B_))
    ds = p .* (dp .- delta) .* scale
    dq = contract(k, ds, (D_, H_, N_, B_), (N_, G_, H_, M_, B_), (D_, G_, H_, M_, B_))
    dk = contract(grouped(q, kv), ds, (D_, G_, H_, M_, B_), (N_, G_, H_, M_, B_), (D_, H_, N_, B_))
    (dq = reshape(dq, size(q)...), dk = dk, dv = dv)
end

@operation attention_backward(q, k, v, o, lse, dout, dlse; causal, scale)
