# Qwen3.5-4B's gradient, the session's program by path: the loss of Vallmo's
# model (branch `train`, `Vallmo.loss`, the mean cross-entropy of every
# token's logits against the token that follows) on the real checkpoint, its
# weights on the device in BFloat16, and its gradient with respect to every
# weight, `Tempest.withgradient`, on T tokens of B sequences drawn at random:
# (16, 1) at the small shape, (32, 1) at the doubled one. The checkpoint is
# TEMPEST_CORPUS_QWEN's directory, else ~/models/Qwen/Qwen3.5-4B; the seconds
# reading it are the case's `checkpoint_s`.
using Random
using BFloat16s, CUDACore, Vallmo
import Tempest

const CKPT = get(ENV, "TEMPEST_CORPUS_QWEN", joinpath(homedir(), "models", "Qwen", "Qwen3.5-4B"))

tokens_of(shape) = shape === :doubled ? (32, 1) : (16, 1)

# The tokens, drawn on the host from the vocabulary.
draw(vocabulary, (T, B)) = Int32.(rand(MersenneTwister(1), 1:vocabulary, T, B))

function instance(shape)
    checkpoint_s = @elapsed m = Vallmo.load(CKPT; Ctx = first(tokens_of(shape)), backend = Vallmo.TempestBackend(CuArray))
    w, cfg = m.weights, m.config
    tokens = Vallmo.on_device(m, draw(size(w.embed_tokens.weight, 2), tokens_of(shape)))
    step = (w, tokens) -> Tempest.withgradient(w -> Vallmo.loss(w, cfg, tokens), w)
    (; step, args = (w, tokens), target = Tempest.CUDATarget(), precision = "bf16", checkpoint_s,
        note = "$(tokens_of(shape)) tokens")
end

# The same function untraced: on the host, the weights widened to Float32,
# the gradient Tempest's interpreter's.
function untraced(shape)
    m = Vallmo.load(CKPT; Ctx = first(tokens_of(shape)))
    w = Vallmo.mapleaves(x -> eltype(x) === BFloat16s.BFloat16 ? Float32.(x) : x, m.weights)
    tokens = draw(size(w.embed_tokens.weight, 2), tokens_of(shape))
    Tempest.withgradient(w -> Vallmo.loss(w, m.config, tokens), w)
end
