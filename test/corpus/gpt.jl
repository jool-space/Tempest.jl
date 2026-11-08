# A GPT training step, test/bench/gpt.jl's (module GPT): token embedding with
# a tied head, RMSNorm, grouped-query attention with rotary positions and a
# causal mask, SwiGLU, cross-entropy and Adam, the parameters and Adam's state
# replaced in place, TF32. The small shape is the bench's, a character model of
# Tempest's docs/src: 4 layers, 128 hidden, 4 heads over 2 key/value heads, FFN
# 384, T 64, batch 16. The full shape is GPT-2 124M's geometry: 12 layers, 768
# hidden, 12 heads each its own key/value head, FFN 3072, T 1024, vocabulary
# 50257, batch TEMPEST_CORPUS_GPT_BATCH (4), on tokens drawn at random.
include(joinpath(@__DIR__, "..", "bench", "gpt.jl"))
using Random
import Tempest as T

function configuration(shape, rng)
    if shape === :small
        corpus = GPT.load_corpus(joinpath(@__DIR__, "..", "..", "docs", "src"))
        cfg = (; vocabulary = length(corpus.chars), hidden = 128, heads = 4, kv_heads = 2, head_dim = 32, ffn = 384, layers = 4,
            T = 64, B = 16)
        return cfg, corpus.data
    end
    cfg = (; vocabulary = 50257, hidden = 768, heads = 12, kv_heads = 12, head_dim = 64, ffn = 3072, layers = 12, T = 1024,
        B = parse(Int, get(ENV, "TEMPEST_CORPUS_GPT_BATCH", "4")))
    cfg, rand(rng, Int32(1):Int32(cfg.vocabulary), cfg.T * cfg.B + 1024)
end

function instance(shape)
    rng = MersenneTwister(0)
    cfg, data = configuration(shape, rng)
    params = GPT.parameters(rng, cfg)
    c = GPT.rotary(cfg)
    batch = GPT.get_batch(rng, data, cfg.T, cfg.B)
    step = (p, s, b) -> GPT.train_step(p, s, b, c, cfg)
    # The Float64 reference interprets the gradient on the CPU holding every
    # value it computes, which at the full shape is past the memory the host
    # shares with the GPU.
    full = shape === :full
    (; step, args = (params, GPT.state_of(params), batch), replaces = (2 => 1, 3 => 2),
        target = T.CUDATarget(; precision = :tf32), precision = "tf32", check = !full,
        note = full ? "batch $(cfg.B); unchecked, the Float64 reference past the host's memory" : "")
end
