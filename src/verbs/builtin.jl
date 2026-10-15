# Built-in verbs. `batched_mul` is a primitive product and `logsumexp` a fold;
# the others are semantic calls, which keep their identity through the graph
# so providers can realize them directly. Their pullbacks are ordinary array
# programs or further calls.

"""
    Tempest.batched_mul(a, b; transpose_a = false, transpose_b = false)

For three-dimensional `a` and `b`, compute `op(a)[:, :, k] * op(b)[:, :, k]`
for each `k` along the third axis, where `op` transposes each matrix when
requested. Both operands need the same batch size. On traced arrays it is a
primitive product, like `*`: one `BatchedMatMul` node, realized by a batched
GEMM, that a proposal can claim with its neighbours.
"""
function batched_mul(a, b; transpose_a::Bool = false, transpose_b::Bool = false)
    dims = batched_dims(a, b, transpose_a, transpose_b)
    out = Array{promote_type(eltype(a), eltype(b))}(undef, dims)
    batched_mul!(out, a, b, transpose_a, transpose_b)
end

function batched_mul(
    a::TracedArray,
    b::TracedArray;
    transpose_a::Bool = false,
    transpose_b::Bool = false,
)
    nodes = same_trace(a, b)
    dims = batched_dims(a, b, transpose_a, transpose_b)
    T = promote_type(eltype(a), eltype(b))
    emit(nodes, BatchedMatMul(transpose_a, transpose_b), (a.id, b.id), dims, T; scalar = false)
end

function batched_dims(a, b, transpose_a, transpose_b)
    if !(ndims(a) == ndims(b) == 3)
        throw(ArgumentError("batched_mul requires three-dimensional operands"))
    end
    if size(a, 3) != size(b, 3)
        throw(DimensionMismatch("batched_mul batch sizes differ"))
    end
    m, k = transpose_a ? (size(a, 2), size(a, 1)) : (size(a, 1), size(a, 2))
    k2, n = transpose_b ? (size(b, 2), size(b, 1)) : (size(b, 1), size(b, 2))
    if k != k2
        throw(DimensionMismatch("batched_mul inner dimensions differ"))
    end
    (m, n, size(a, 3))
end

batch_operand(x, transposed) = transposed ? transpose(x) : x

function batched_mul!(out, a, b, transpose_a, transpose_b)
    for k in axes(out, 3)
        mul!(
            view(out, :, :, k),
            batch_operand(view(a, :, :, k), transpose_a),
            batch_operand(view(b, :, :, k), transpose_b),
        )
    end
    out
end

evaluate(op::BatchedMatMul, node, xs) =
    batched_mul(xs...; transpose_a = op.transpose_a, transpose_b = op.transpose_b)

"""
    Tempest.logsumexp(x; dims = 1)

`log.(sum(exp.(x); dims))`, computed without overflow by shifting each slice by
its maximum. `dims = :` returns a number, like `sum(x)`. It is the fold
`Tempest.fold(LogSumExpFold(), x; dims)`: one reduction node that fuses with the
work around it, so softmax `exp.(x .- logsumexp(x))` and log-softmax
`x .- logsumexp(x)` each become one region with their neighbours.
"""
logsumexp(x; dims = 1) =
    fold(LogSumExpFold(), x; dims = dims isa Colon ? dims : reduction_axes(x, dims))

function reduction_axes(x, dims)
    axes = dims isa Integer ? (Int(dims),) : Tuple(Int.(dims))
    if !all(d -> 1 <= d <= ndims(x), axes) || !allunique(axes)
        throw(ArgumentError("logsumexp dimensions must be distinct axes of the input"))
    end
    Tuple(sort!(collect(axes)))
end

# Runtime indices. Integer index arrays are traced inputs; these verbs read
# through them. Their pullbacks give the indices no gradient, and accumulate
# into the table in a fixed order, so repeated indices sum deterministically.

"""
    table[:, indices]

With a traced integer array `indices`, gather whole columns of the matrix
`table` at runtime: the result has size `(size(table, 1), size(indices)...)`.
Indices must lie in `1:size(table, 2)`. The gradient accumulates each
output column into its table column in index order.
"""
Base.getindex(table::TracedArray, ::Colon, indices::TracedArray{<:Integer}) =
    gather_columns(table, indices)

gather_columns(table, indices) = reference(gather_columns, table, indices)
gather_columns(table::TracedArray, indices::TracedArray) = record_call(gather_columns, table, indices)

function output_spec(::typeof(gather_columns), table, indices)
    ndims(table) == 2 || throw(ArgumentError("column gathers need a matrix table"))
    TensorSpec(eltype(table), (size(table, 1), size(indices)...))
end

function reference(::typeof(gather_columns), table::AbstractArray, indices::AbstractArray)
    reshape(table[:, vec(indices)], size(table, 1), size(indices)...)
end

verb_pullback(::typeof(gather_columns), (table, indices), y, dy) =
    (scatter_columns(dy, indices; columns = size(table, 2)), nothing)

"""
    Tempest.scatter_columns(values, indices; columns)

Sum each column of `values` (the columns of `values` flattened past its first
axis) into column `indices[n]` of a `size(values, 1) × columns` matrix of zeros,
in increasing `n`. The pullback of a column gather.
"""
scatter_columns(values, indices; columns::Integer) =
    reference(scatter_columns, values, indices; columns = Int(columns))
scatter_columns(values::TracedArray, indices::TracedArray; columns::Integer) =
    record_call(scatter_columns, values, indices; columns = Int(columns))

function output_spec(::typeof(scatter_columns), values, indices; columns)
    size(values)[2:end] == size(indices) ||
        throw(DimensionMismatch("scatter_columns needs one value column per index"))
    TensorSpec(eltype(values), (size(values, 1), columns))
end

function reference(::typeof(scatter_columns), values::AbstractArray, indices::AbstractArray; columns)
    out = zeros(eltype(values), size(values, 1), columns)
    flat = reshape(values, size(values, 1), :)
    for (n, index) in enumerate(indices)
        out[:, index] .+= view(flat, :, n)
    end
    out
end

verb_pullback(::typeof(scatter_columns), (values, indices), y, dy; columns) =
    (gather_columns(dy, indices), nothing)

"""
    Tempest.take_rows(x, rows)

For each column of `x` (flattened past its first axis) pick the element in
row `rows[n]`: the result has size `(1, size(rows)...)`. With logits `x` and
integer targets, `logsumexp(x) .- take_rows(x, targets)` is the cross-entropy
per position.
"""
take_rows(x, rows) = reference(take_rows, x, rows)
take_rows(x::TracedArray, rows::TracedArray) = record_call(take_rows, x, rows)

function output_spec(::typeof(take_rows), x, rows)
    size(x)[2:end] == size(rows) ||
        throw(DimensionMismatch("take_rows needs one row index per column"))
    TensorSpec(eltype(x), (1, size(rows)...))
end

function reference(::typeof(take_rows), x::AbstractArray, rows::AbstractArray)
    flat = reshape(x, size(x, 1), :)
    reshape([flat[r, n] for (n, r) in enumerate(rows)], 1, size(rows)...)
end

verb_pullback(::typeof(take_rows), (x, rows), y, dy) =
    (put_rows(dy, rows; rows_count = size(x, 1)), nothing)

"""
    Tempest.put_rows(values, rows; rows_count)

The pullback of `take_rows`: a `rows_count × ...` array of zeros holding
`values[1, n]` in row `rows[n]` of column `n`.
"""
put_rows(values, rows; rows_count::Integer) =
    reference(put_rows, values, rows; rows_count = Int(rows_count))
put_rows(values::TracedArray, rows::TracedArray; rows_count::Integer) =
    record_call(put_rows, values, rows; rows_count = Int(rows_count))

function output_spec(::typeof(put_rows), values, rows; rows_count)
    size(values) == (1, size(rows)...) ||
        throw(DimensionMismatch("put_rows needs one value per row index"))
    TensorSpec(eltype(values), (rows_count, size(rows)...))
end

function reference(::typeof(put_rows), values::AbstractArray, rows::AbstractArray; rows_count)
    out = zeros(eltype(values), rows_count, length(rows))
    for (n, r) in enumerate(rows)
        out[r, n] = values[n]
    end
    reshape(out, rows_count, size(rows)...)
end

verb_pullback(::typeof(put_rows), (values, rows), y, dy; rows_count) =
    (take_rows(dy, rows), nothing)

const INDEX_VERBS = (gather_columns, scatter_columns, take_rows, put_rows)

# Counter-based randomness: every element's draw is a hash of the seed, a
# per-site stream number, and its linear index, so a mask depends on nothing
# but those three and is the same on every target. The backward pass redraws
# it instead of storing it; advancing the seed is ordinary integer arithmetic
# in a training step's update.

"""
    Tempest.dropout(x, seed; rate, stream = 0)

Zero each element of `x` with probability `rate` and scale the rest by
`1 / (1 - rate)`. `seed` is a traced integer array whose first element keys
the draw; distinct dropout sites in one program should use distinct `stream`
numbers. The mask is a counter hash of `(seed, stream, index)`, identical on
the CPU and CUDA, and recomputed for the gradient.
"""
dropout(x, seed; rate::Real, stream::Integer = 0) =
    reference(dropout, x, seed; rate = Float64(rate), stream = Int(stream))
dropout(x::TracedArray, seed::TracedArray; rate::Real, stream::Integer = 0) =
    record_call(dropout, x, seed; rate = Float64(rate), stream = Int(stream))

function output_spec(::typeof(dropout), x, seed; rate, stream)
    0 <= rate < 1 || throw(ArgumentError("dropout rate must lie in [0, 1)"))
    eltype(seed) <: Integer || throw(ArgumentError("dropout seeds are integer arrays"))
    TensorSpec(eltype(x), size(x))
end

# SplitMix64's finalizer over the seed, stream and index; the top 24 bits give
# a uniform Float32 in [0, 1).
@inline function hash_uniform(seed, stream, index)
    z = UInt64(seed % UInt32) * 0x9e3779b97f4a7c15 ⊻ (UInt64(stream % UInt32) << 32) ⊻ UInt64(index % UInt32)
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    z ⊻= z >> 31
    Float32(z >> 40) * Float32(0x1p-24)
end

@inline function dropout_value(x, u, rate)
    T = typeof(x)
    u >= rate ? x / (one(T) - T(rate)) : zero(T)
end

function reference(::typeof(dropout), x::AbstractArray, seed::AbstractArray; rate, stream)
    key = first(seed)
    [dropout_value(x[i], hash_uniform(key, stream, i), rate) for i in eachindex(IndexLinear(), x)] |>
        values -> reshape(values, size(x))
end

verb_pullback(::typeof(dropout), (x, seed), y, dy; rate, stream) =
    (dropout(dy, seed; rate, stream), nothing)

# Scaled dot-product attention. Queries have size (features, heads, positions,
# batch) and keys and values (features, key/value heads, positions, batch);
# query head j reads key/value head cld(j, group) with group = heads ÷ kv heads.
# s[n, m] = scale * k[:, n] ⋅ q[:, m] per head and batch, p = softmax over n,
# o[:, m] = v * p[:, m]. With `causal`, key n is visible from query m only when
# n <= m. The forward call also returns each query's log-sum-exp, which the
# backward call uses to recompute p.

"""
    Tempest.attention(q, k, v; causal = false, scale = 1 / sqrt(size(q, 1)))

Scaled dot-product attention. `q` has size `(d, heads, queries, batch)`, `k`
`(d, kv_heads, keys, batch)` and `v` `(dv, kv_heads, keys, batch)`; the result has
size `(dv, heads, queries, batch)`. Query head `j` attends with key/value head
`cld(j, heads ÷ kv_heads)` (grouped-query attention). Three-dimensional
`(d, positions, batch)` arrays are a single head. Providers may compute it
without materializing the scores; the reference is the composed
`batched_mul`/`logsumexp` form.
"""
function attention(q, k, v; causal::Bool = false, scale::Real = inv(sqrt(size(q, 1))))
    if ndims(q) == ndims(k) == ndims(v) == 3
        head(x) = reshape(x, size(x, 1), 1, size(x, 2), size(x, 3))
        o = attention(head(q), head(k), head(v); causal, scale)
        return reshape(o, size(o, 1), size(o, 3), size(o, 4))
    end
    attention_forward(q, k, v; causal, scale = Float32(scale)).o
end

attention_forward(q, k, v; causal, scale) =
    reference(attention_forward, q, k, v; causal, scale)
attention_forward(q::TracedArray, k::TracedArray, v::TracedArray; causal, scale) =
    record_call(attention_forward, q, k, v; causal, scale)

function attention_dims(q, k, v, causal)
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

function output_spec(::typeof(attention_forward), q, k, v; causal, scale)
    attention_dims(q, k, v, causal)
    T = eltype(q)
    H, Tq, B = size(q, 2), size(q, 3), size(q, 4)
    (o = TensorSpec(T, (size(v, 1), H, Tq, B)), lse = TensorSpec(T, (1, H, Tq, B)))
end

causal_scores(s, causal) = causal ? causal_fill(s; value = -Inf32) : s

# Heads to the batch axis for the composed form: (d, h, T, B) → (d, T, h * B),
# repeating each key/value head for its group of query heads.
function batched_heads(x, heads)
    group = heads ÷ size(x, 2)
    x = group == 1 ? x : x[:, [cld(j, group) for j = 1:heads], :, :]
    reshape(permutedims(x, (1, 3, 2, 4)), size(x, 1), size(x, 3), :)
end

# (d, T, h * B) → (d, h, T, B).
unbatched_heads(x, heads) =
    permutedims(reshape(x, size(x, 1), size(x, 2), heads, :), (1, 3, 2, 4))

# Sum the query-head copies of each key/value head: (d, T, h * B) → (d, kv, T, B).
function grouped_heads(x, heads, kv_heads)
    group = heads ÷ kv_heads
    x = reshape(x, size(x, 1), size(x, 2), group, kv_heads, :)
    x = group == 1 ? x : sum(x; dims = 3)
    permutedims(reshape(x, size(x, 1), size(x, 2), kv_heads, :), (1, 3, 2, 4))
end

function attention_scores(q, k, causal, scale)
    H = size(q, 2)
    causal_scores(
        batched_mul(batched_heads(k, H), batched_heads(q, H); transpose_a = true) .* scale,
        causal,
    )
end

function reference(::typeof(attention_forward), q, k, v; causal, scale)
    H = size(q, 2)
    s = attention_scores(q, k, causal, scale)
    lse = logsumexp(s; dims = 1)
    o = batched_mul(batched_heads(v, H), exp.(s .- lse))
    (o = unbatched_heads(o, H), lse = unbatched_heads(lse, H))
end

function verb_pullback(::typeof(attention_forward), (q, k, v), y, dy; causal, scale)
    g = attention_backward(q, k, v, y.o, y.lse, dy.o, dy.lse; causal, scale)
    (g.dq, g.dk, g.dv)
end

"""
    Tempest.attention_backward(q, k, v, o, lse, dout, dlse; causal, scale)

The pullback of attention: `(dq, dk, dv)` from the forward outputs and their
cotangents, recomputing the probabilities from `lse`.
"""
attention_backward(q, k, v, o, lse, dout, dlse; causal, scale) =
    reference(attention_backward, q, k, v, o, lse, dout, dlse; causal, scale)
attention_backward(
    q::TracedArray, k::TracedArray, v::TracedArray, o::TracedArray, lse::TracedArray,
    dout::TracedArray, dlse::TracedArray; causal, scale,
) = record_call(attention_backward, q, k, v, o, lse, dout, dlse; causal, scale)

function output_spec(::typeof(attention_backward), q, k, v, o, lse, dout, dlse; causal, scale)
    (dq = TensorSpec(q), dk = TensorSpec(k), dv = TensorSpec(v))
end

function reference(::typeof(attention_backward), q, k, v, o, lse, dout, dlse; causal, scale)
    H, KV = size(q, 2), size(k, 2)
    s = attention_scores(q, k, causal, scale)
    p = exp.(s .- batched_heads(lse, H))
    d = batched_heads(dout, H)
    delta = sum(d .* batched_heads(o, H); dims = 1) .- batched_heads(dlse, H)
    dp = batched_mul(batched_heads(v, H), d; transpose_a = true)
    ds = p .* (dp .- delta) .* scale
    (
        dq = unbatched_heads(batched_mul(batched_heads(k, H), ds), H),
        dk = grouped_heads(batched_mul(batched_heads(q, H), ds; transpose_b = true), H, KV),
        dv = grouped_heads(batched_mul(d, p; transpose_b = true), H, KV),
    )
end

"""
    Tempest.causal_fill(x; value)

Replace `x[n, m, ...]` with `value` wherever `n > m`: the entries a causal
query `m` may not see. Its pullback zeroes the same entries.
"""
causal_fill(x; value::Real) = reference(causal_fill, x; value = Float32(value))
causal_fill(x::TracedArray; value::Real) = record_call(causal_fill, x; value = Float32(value))

output_spec(::typeof(causal_fill), x; value) = TensorSpec(x)

function reference(::typeof(causal_fill), x::AbstractArray; value)
    y = copy(x)
    for index in CartesianIndices(y)
        index[1] > index[2] && (y[index] = value)
    end
    y
end

verb_pullback(::typeof(causal_fill), (x,), y, dy; value) = (causal_fill(dy; value = 0.0f0),)
