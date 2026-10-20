# The coverage suite's cases: compositions nobody wrote code for, each with the
# structure recorded for its streaming plan. test/coverage.jl checks them and
# test/bench/coverage.jl times them.
#
# A structure is the number of kernels, the bytes stored (inputs and constants
# apart) and the largest single value stored, a bound: attention stores no
# value as large as its scores. A `Gap` is an expectation the plan does not
# meet yet, with what closes it.
using Tempest, Tempest.Compiler
import Tempest as T
import Tempest.Compiler as C

# Attention as a user writes it, when this file is not included beside setup.jl.
@isdefined(Ops) || @eval module Ops
    using Tempest
    include($(joinpath(@__DIR__, "operations.jl")))
end

"An expectation not met yet, and the work that closes the gap."
struct Gap
    value::Int
    closer::String
end

struct Composition
    name::String
    program::C.Program
    args::Tuple
    expected::NamedTuple # with every implementation loaded
    generated::NamedTuple # with the generator and the floor alone
end

Composition(name, program, args, expected; generated = expected) = Composition(name, program, args, expected, generated)

sample(dims...; scale = 1.0f0) = 2.0f0 .* sin.(scale .* reshape(Float32.(1:prod(dims)), dims...))

## Attention: (features, heads, positions, batch), causal, one key head per
## query head. Small enough to compile quickly, long enough to stream keys.

const D, H, S, B = 64, 4, 256, 2
const O = 4 * D * H * S * B          # bytes of the output, or of q, k or v
const LSE = 4 * H * S * B            # bytes of one value per query

q, k, v = sample(D, H, S, B; scale = 0.1f0), sample(D, H, S, B; scale = 0.2f0), sample(D, H, S, B; scale = 0.3f0)
x = sample(D, H, S, B; scale = 0.5f0)

attend(q, k, v) = Ops.attention(q, k, v; causal = true)

# Written out: scores (keys, heads, queries, batch), a mask, and the weighted
# sum of values.
scores(q, k) = T.contract(k, q, (1, 2, 3, 4), (1, 2, 5, 4), (3, 2, 5, 4)) .* inv(sqrt(Float32(size(q, 1))))
causal(s, masked) = ifelse.(reshape(1:size(s, 1), :, 1, 1) .<= reshape(1:size(s, 3), 1, 1, :), s, masked)
combine(v, p) = T.contract(v, p, (1, 2, 3, 4), (3, 2, 5, 4), (1, 2, 5, 4))

written(q, k, v) = (s = causal(scores(q, k), -Inf32); combine(v, exp.(s .- T.logsumexp(s; dims = 1))))
softcap(q, k, v) = (s = causal(30.0f0 .* tanh.(scores(q, k) ./ 30.0f0), -Inf32); combine(v, exp.(s .- T.logsumexp(s; dims = 1))))
silu(x) = x / (1 + exp(-x))
hstu(q, k, v) = combine(v, causal(silu.(scores(q, k)), 0.0f0) ./ Float32(size(k, 3)))
function shifted(q, k, v)
    s = causal(scores(q, k), -Inf32)
    e = exp.(s .- maximum(s; dims = 1))
    combine(v, e ./ sum(e; dims = 1))
end

training(loss, args...) = differentiate(trace(loss, args...))

const ATTENTION = [
    Composition("attention as a function", trace(attend, q, k, v), (q, k, v), (kernels = 1, bytes = O, largest = O)),
    Composition("written out with contract and logsumexp", trace(written, q, k, v), (q, k, v), (kernels = 1, bytes = O, largest = O)),
    Composition("softcap 30·tanh(s/30)", trace(softcap, q, k, v), (q, k, v), (kernels = 1, bytes = O, largest = O)),
    Composition("SiLU scores, no softmax (HSTU)", trace(hstu, q, k, v), (q, k, v), (kernels = 1, bytes = O, largest = O)),
    # The output, the loss's gradient at it, the loss and three gradients.
    Composition("SiLU scores, a training step (HSTU)", training((q, k, v) -> sum(abs2, hstu(q, k, v)), q, k, v), (q, k, v),
        (kernels = 3, bytes = 5O + 4, largest = O)),
    Composition("softmax spelled exp(s - max) / sum", trace(shifted, q, k, v), (q, k, v), (kernels = 1, bytes = O, largest = O)),
    Composition("the function, then a residual", trace((q, k, v, x) -> attend(q, k, v) .+ x, q, k, v, x), (q, k, v, x),
        (kernels = 1, bytes = O, largest = O)),
    Composition("written out, then a residual", trace((q, k, v, x) -> written(q, k, v) .+ x, q, k, v, x), (q, k, v, x),
        (kernels = 1, bytes = O, largest = O)),
    Composition("the function, and a reduction of its output", trace((q, k, v) -> (o = attend(q, k, v); (o, sum(abs2, o; dims = 1))), q, k, v),
        (q, k, v), (kernels = 2, bytes = O + LSE, largest = O)),
    # The streaming plan recomputes the scores in the backward pass too, so no
    # value as large as them is stored; the cover stores them at this size,
    # measured faster.
    Composition("the function, a training step", training((q, k, v) -> sum(abs2, attend(q, k, v)), q, k, v), (q, k, v),
        (bytes = 5O + 2LSE + 4, largest = O)),
    Composition("written out, a training step", training((q, k, v) -> sum(abs2, written(q, k, v)), q, k, v), (q, k, v),
        (largest = O,)),
]

## Spellings outside attention, over the columns of a matrix.

const N, M = 512, 1024
const Y = 4 * N * M

X, w = sample(N, M; scale = 0.3f0), sample(N; scale = 0.1f0)

rmsnorm(x, w) = x ./ sqrt.(sum(abs2, x; dims = 1) ./ size(x, 1) .+ 1.0f-5) .* w
rmsnorm_inverse(x, w) = w .* x .* inv.(sqrt.(sum(x .* x; dims = 1) .* (1.0f0 / size(x, 1)) .+ 1.0f-5))
function layernorm(x, w)
    m = sum(x; dims = 1) ./ size(x, 1)
    (x .- m) ./ sqrt.(sum(abs2, x .- m; dims = 1) ./ size(x, 1) .+ 1.0f-5) .* w
end
function layernorm_squares(x, w)
    m = sum(x; dims = 1) ./ size(x, 1)
    v = sum(x .^ 2; dims = 1) ./ size(x, 1) .- m .^ 2
    w .* (x .- m) ./ sqrt.(v .+ 1.0f-5)
end
softmax_logsumexp(x) = exp.(x .- T.logsumexp(x; dims = 1))
softmax_shifted(x) = (e = exp.(x .- maximum(x; dims = 1)); e ./ sum(e; dims = 1))
softmax_reused(x) = (e = exp.(x); e ./ sum(e; dims = 1))
softmax_twice(x) = exp.(x) ./ sum(exp.(x); dims = 1)

gelu(x) = x / 2 * (1 + tanh(0.7978846f0 * (x + 0.044715f0 * x * x * x)))
W, Z, b = sample(256, 384; scale = 0.01f0), sample(384, 512; scale = 0.2f0), sample(256; scale = 0.3f0)
const P = 4 * 256 * 512

const SPELLINGS = [
    Composition("RMSNorm, divided by the root", trace(rmsnorm, X, w), (X, w), (kernels = 1, bytes = Y)),
    Composition("RMSNorm, times the inverse root", trace(rmsnorm_inverse, X, w), (X, w), (kernels = 1, bytes = Y)),
    # The loss, the gradients and a value per column, between the folds.
    Composition("RMSNorm, a training step", training((x, w) -> sum(abs2, rmsnorm(x, w)), X, w), (X, w),
        (kernels = 1, bytes = Y + 4N + 4M + 4)),
    Composition("layer norm, two passes", trace(layernorm, X, w), (X, w), (kernels = 1, bytes = Y)),
    Composition("layer norm, from the mean of squares", trace(layernorm_squares, X, w), (X, w), (kernels = 1, bytes = Y)),
    Composition("softmax through logsumexp", trace(softmax_logsumexp, X), (X,), (kernels = 1, bytes = Y)),
    Composition("softmax shifted by the maximum", trace(softmax_shifted, X), (X,), (kernels = 1, bytes = Y)),
    Composition("softmax, exp written twice", trace(softmax_twice, X), (X,), (kernels = 1, bytes = Y)),
    Composition("softmax, exp computed once", trace(softmax_reused, X), (X,), (kernels = 1, bytes = Y)),
    # cuBLAS wins the product by estimate, and the epilogue is a kernel of its own.
    Composition("linear, bias, GELU", trace((W, Z, b) -> gelu.(W * Z .+ b), W, Z, b), (W, Z, b),
        (kernels = 2, bytes = 2P); generated = (kernels = 1, bytes = P)),
]

## Writes into arrays the program allocates, which are functional updates:
## each planned with its readers, storing no version of the array written.

function padded(x, p)
    y = fill!(similar(x, size(x) .+ 2p), 0)
    y[(p + 1):(end - p), (p + 1):(end - p)] = x
    y
end
overwritten(x, v) = (y = copy(x); y[1:size(v, 1), :] = v; y)

const RECOMPUTE = "a map recomputed by the kernels that read it over other domains (planner)"
Xh, V = sample(N ÷ 2, M; scale = 0.2f0), sample(64, M; scale = 0.4f0)

const UPDATES = [
    Composition("padding, then a reduction", trace(x -> sum(abs2, padded(x, 2); dims = 1), X), (X,), (kernels = 1, bytes = 4 * (M + 4))),
    Composition("a concatenation, then a map", trace((x, h) -> tanh.(vcat(x, h)) .* 2, X, Xh), (X, Xh), (kernels = 1, bytes = 6 * N * M)),
    Composition("a slice written, then a reduction", trace((x, v) -> sum(overwritten(x, v); dims = 1), X, V), (X, V), (kernels = 1, bytes = 4M)),
    # The loss and the two gradients: the gradient of the values written
    # selects, and that of the array written is masked. The map between the
    # update and the loss is stored for the gradients' kernels.
    Composition("a slice written, a training step", training((x, v) -> sum(abs2, tanh.(overwritten(x, v))), X, V), (X, V),
        (kernels = 2, bytes = Gap(Y + 4 * 64 * M + 4, RECOMPUTE), largest = Y)),
]

const COMPOSITIONS = [ATTENTION; SPELLINGS; UPDATES]
