# A character-level GPT training step: embedding, RMSNorm, grouped-query
# attention with rotary positions and a causal mask, a SwiGLU feedforward, a tied
# output head, cross-entropy, and Adam, prepared as one captured CUDA graph.
#
# The model follows nanoGPT's `model_full.jl` (jool-space) layer for layer, and
# initializes and samples batches with the same random stream, so the loss
# curve can be compared step for step with that implementation. Differences in
# formulation, not in mathematics: rotary embedding uses the rotate-half form
# with a fixed row permutation instead of concatenation. Tokens and targets are
# runtime integer indices: `wte[:, tokens]` gathers embeddings and
# `Tempest.take_rows` picks the target logits. `Tempest.attention` computes
# causal attention; under TF32 products it runs fused cuTile kernels, otherwise
# the composed batched products and logsumexp.
#
# Run: julia --project=examples examples/gpt.jl
# Environment: TEMPEST_GPT_TEXT (a .md directory or a file; default: this
# repository's docs), TEMPEST_GPT_STEPS (default 200), and the shape variables
# TEMPEST_GPT_{HIDDEN,HEADS,KV_HEADS,FFN,LAYERS,T,B}; TEMPEST_GPT_PRECISION=tf32
# lets matrix products use TensorFloat-32, TEMPEST_GPT_DROPOUT=0.1 adds
# residual dropout keyed by a seed the step advances, TEMPEST_GPT_KERNELS=cuda
# excludes cuTile, and TEMPEST_GPT_SEED changes the random stream (0 matches
# nanoGPT).
module GPTExample

using Tempest
using CUDACore
using Random
import cuBLAS
# Run as a script, the example loads cuTile for its fused kernels. Code that
# includes it, such as the test suite, decides which providers are loaded.
if abspath(PROGRAM_FILE) == @__FILE__
    import cuTile
end

const logsumexp = Tempest.logsumexp

# ---- data (as nanoGPT's data.jl) ---------------------------------------------------

function load_corpus(path)
    files = isdir(path) ?
        filter(f -> endswith(f, ".md"), [joinpath(r, f) for (r, _, fs) in walkdir(path) for f in fs]) :
        [path]
    text = join(read.(files, String), "\n")
    chars = sort!(unique(text))
    lookup = Dict(c => i for (i, c) in enumerate(chars))
    (; data = [lookup[c] for c in text], chars)
end

function get_batch(rng, data, T, B)
    starts = rand(rng, 1:(length(data) - T), B)
    tokens = [data[s + t - 1] for t = 1:T, s in starts]
    targets = [data[s + t] for t = 1:T, s in starts]
    tokens, targets
end

batch_arrays(tokens, targets, V) = (tokens = Int32.(tokens), targets = Int32.(targets))

# ---- parameters, in nanoGPT's initialization order --------------------------------

init(rng, dims...) = 0.02f0 .* randn(rng, Float32, dims...)

function block_parameters(rng, cfg)
    (; hidden, head_dim, heads, kv_heads, ffn) = cfg
    norm1 = ones(Float32, hidden)
    wq = init(rng, hidden, head_dim, heads)
    wk = init(rng, hidden, head_dim, kv_heads)
    wv = init(rng, hidden, head_dim, kv_heads)
    wo = init(rng, head_dim, heads, hidden)
    norm2 = ones(Float32, hidden)
    wg = init(rng, hidden, ffn)
    wu = init(rng, hidden, ffn)
    wd = init(rng, ffn, hidden)
    (; norm1, wq, wk, wv, wo, norm2, wg, wu, wd)
end

function parameters(rng, cfg)
    wte = init(rng, cfg.hidden, cfg.vocabulary)
    blocks = Tuple(block_parameters(rng, cfg) for _ = 1:cfg.layers)
    (; wte, blocks, norm = ones(Float32, cfg.hidden))
end

# Rotary tables in rotate-half form: rope(x) = x .* C .+ x[swap, ...] .* S.
function constants(cfg; theta = 10000.0f0)
    d, T = cfg.head_dim, cfg.T
    inv_freq = theta .^ (-Float32.(0:2:(d - 1)) ./ d)
    angles = inv_freq .* Float32.(0:(T - 1))'
    C = reshape(vcat(cos.(angles), cos.(angles)), d, 1, T)
    S = reshape(vcat(-sin.(angles), sin.(angles)), d, 1, T)
    (; C, S)
end

# ---- the model ---------------------------------------------------------------------

rms_norm(x, weight; eps = 1.0f-5) = weight .* x ./ sqrt.(sum(abs2, x; dims = 1) ./ size(x, 1) .+ eps)

rope(x, c, swap) = x .* c.C .+ x[swap, :, :, :] .* c.S

function attention(x, p, c, cfg)
    (; head_dim, T) = cfg
    D, N = size(x)
    B = N ÷ T
    project(w) = reshape(reshape(w, D, :)' * x, head_dim, size(w, 3), T, B)
    swap = [(head_dim ÷ 2 + 1):head_dim; 1:(head_dim ÷ 2)]
    q = rope(project(p.wq), c, swap)
    k = rope(project(p.wk), c, swap)
    # Query head j reads key/value head cld(j, group): nanoGPT's (g h) axis, g fastest.
    o = Tempest.attention(q, k, project(p.wv); causal = true, scale = 1 / sqrt(Float32(head_dim)))
    reshape(p.wo, :, D)' * reshape(o, :, N)
end

swish(x) = x / (1 + exp(-x))

function Tempest.scalar_partials(::typeof(swish))
    function dswish(x)
        s = 1 / (1 + exp(-x))
        s + x * s * (1 - s)
    end
    (dswish,)
end

feedforward(x, p) = p.wd' * (swish.(p.wg' * x) .* (p.wu' * x))

# Residual-branch dropout, as in nanoGPT's original model, when cfg.dropout > 0.
# Each site draws from its own stream of the step's seed.
residual_dropout(x, seed, cfg, stream) =
    cfg.dropout > 0 ? Tempest.dropout(x, seed; rate = cfg.dropout, stream) : x

function loss(p, batch, c, cfg; seed = nothing)
    x = reshape(p.wte[:, batch.tokens], size(p.wte, 1), :)
    for (layer, block) in enumerate(p.blocks)
        a = attention(rms_norm(x, block.norm1), block, c, cfg)
        x = x + residual_dropout(a, seed, cfg, 2layer)
        f = feedforward(rms_norm(x, block.norm2), block)
        x = x + residual_dropout(f, seed, cfg, 2layer + 1)
    end
    logits = p.wte' * rms_norm(x, p.norm)
    picked = Tempest.take_rows(logits, reshape(batch.targets, :))
    sum(logsumexp(logits; dims = 1) .- picked) / size(logits, 2)
end

# Adam as Optimisers.AdamW(rate) with its default zero weight decay.
function adam(p, state, gradients; rate = 3.0f-3, beta1 = 0.9f0, beta2 = 0.999f0, epsilon = 1.0f-8)
    step = state.step + 1.0f0
    first = map_leaves((m, g) -> beta1 .* m .+ (1 - beta1) .* g, state.first, gradients)
    second = map_leaves((v, g) -> beta2 .* v .+ (1 - beta2) .* abs2.(g), state.second, gradients)
    correction1, correction2 = 1 - beta1^step, 1 - beta2^step
    updated = map_leaves(p, first, second) do x, m, v
        x .- rate .* (m ./ correction1) ./ (sqrt.(v ./ correction2) .+ epsilon)
    end
    updated, (; first, second, step, seed = state.seed .+ Int32(1))
end

map_leaves(f, xs::Union{Tuple, NamedTuple}...) = map((ys...) -> map_leaves(f, ys...), xs...)
map_leaves(f, xs...) = f(xs...)

function configuration(vocabulary)
    int(name, default) = parse(Int, get(ENV, "TEMPEST_GPT_$name", string(default)))
    hidden, heads = int("HIDDEN", 128), int("HEADS", 4)
    (;
        vocabulary,
        hidden,
        heads,
        kv_heads = int("KV_HEADS", 2),
        head_dim = hidden ÷ heads,
        ffn = int("FFN", 384),
        layers = int("LAYERS", 4),
        T = int("T", 64),
        B = int("B", 16),
        dropout = parse(Float64, get(ENV, "TEMPEST_GPT_DROPOUT", "0")),
    )
end

parameter_count(p) = sum(length, leaves(p))
leaves(x::Union{Tuple, NamedTuple}) = mapreduce(leaves, vcat, x; init = Any[])
leaves(x) = Any[x]

# The prepared, captured training step and the random stream positioned after
# the batch used for preparation.
function build(; text = get(ENV, "TEMPEST_GPT_TEXT", joinpath(@__DIR__, "..", "docs", "src")))
    corpus = load_corpus(text)
    cfg = configuration(length(corpus.chars))
    rng = MersenneTwister(parse(Int, get(ENV, "TEMPEST_GPT_SEED", "0")))
    params = parameters(rng, cfg)
    state = (
        first = map_leaves(zero, params),
        second = map_leaves(zero, params),
        step = 0.0f0,
        seed = Int32[0],
    )
    c = constants(cfg)
    batch = batch_arrays(get_batch(rng, corpus.data, cfg.T, cfg.B)..., cfg.vocabulary)
    println("corpus: ", length(corpus.data), " chars, vocabulary ", cfg.vocabulary)
    println("shape: T=", cfg.T, " B=", cfg.B, ", ", parameter_count(params), " parameters, ",
            cfg.layers, " layers")
    step_loss(p, s, b, c) = loss(p, b, c, cfg; s.seed)
    seconds = @elapsed begin
        gradients = differentiate(trace(step_loss, params, state, batch, c); wrt = (1,))
        step = with_update(gradients) do value, grads, p, s, b, c
            next_params, next_state = adam(p, s, only(grads))
            (next_params, next_state, value)
        end
        precision = Symbol(get(ENV, "TEMPEST_GPT_PRECISION", "full"))
        kernels = Symbol(get(ENV, "TEMPEST_GPT_KERNELS", "auto"))
        target = CUDATarget(; matmul_precision = precision, kernels)
        ex = prepare(step, params, state, batch, c; target, donate = (1, 2))
        capture!(ex)
    end
    println("trace, differentiate, prepare and capture: ", round(seconds; digits = 1), " s")
    (; ex, rng, corpus, cfg)
end

function main(; steps = parse(Int, get(ENV, "TEMPEST_GPT_STEPS", "200")))
    CUDACore.functional() || error("this example requires CUDA")
    (; ex, rng, corpus, cfg) = build()
    # Batches are built on the host while the previous step runs; binding is
    # ordered after it on the stream. Reading a loss synchronizes, as does
    # Reactant's loop, every ten steps. Times exclude the first step, and the
    # steady rate covers the second half of the run.
    losses = Float32[]
    started = 0.0
    halfway = 0.0
    for i = 1:steps
        batch = batch_arrays(get_batch(rng, corpus.data, cfg.T, cfg.B)..., cfg.vocabulary)
        bind_argument!(ex, 3, batch)
        value = replay!(ex)[3]
        if i == 1
            CUDACore.synchronize()
            started = time()
        end
        if i == steps ÷ 2
            CUDACore.synchronize()
            halfway = time()
        end
        if i % 10 == 0
            push!(losses, Array(value)[])
            println("step ", i, "  loss ", round(losses[end]; digits = 3), "  ",
                    round((time() - started) / (i - 1) * 1000; digits = 2), " ms/step")
        end
    end
    CUDACore.synchronize()
    steady = (time() - halfway) / (steps - steps ÷ 2) * 1000
    println("steady: ", round(steady; digits = 2), " ms/step over the last ", steps - steps ÷ 2, " steps")
    (; ex, losses, cfg)
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    GPTExample.main()
end
