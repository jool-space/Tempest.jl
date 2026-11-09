# Attention as cuDNN's SDPA engines compute it, which cuda/cudnn.jl and
# fixtures/at_once.jl prepare: the scores and softmax in Float32 and the
# output in the operands' type.
const LABELS = (d = 1, g = 2, h = 3, m = 4, b = 5, n = 6)
# As a model writes it: (features, heads, positions, batch). With
# `statistics`, the log-sum-exp of each query's scores too, (heads,
# positions, batch), as a training forward keeps it.
function sdpa(q, k, v; causal, round = true, statistics = false)
    (; d, g, h, m, b, n) = LABELS
    q5 = reshape(q, size(q, 1), size(q, 2) ÷ size(k, 2), size(k, 2), size(q, 3), size(q, 4))
    s = C.contract(Float32.(k), Float32.(q5), (d, h, n, b), (d, g, h, m, b), (n, g, h, m, b)) .* Float32(inv(sqrt(size(q, 1))))
    causal && (s = ifelse.(reshape(1:size(s, 1), :, 1, 1, 1) .<= reshape(1:size(s, 4), 1, 1, 1, :), s, -Inf32))
    lse = T.logsumexp(s; dims = 1)
    o = C.contract(v, exp.(s .- lse), (d, h, n, b), (n, g, h, m, b), (d, g, h, m, b))
    o = reshape(round ? eltype(q).(o) : o, size(q))
    statistics ? (o, reshape(lse, size(q, 2), size(q, 3), size(q, 4))) : o
end
