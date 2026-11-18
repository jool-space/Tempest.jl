# Qwen3.5-4B's decode, the session's program by path: one step of Vallmo's
# model (branch `train`, `forward!` of its model.jl, the program its Tempest
# backend compiles for each call shape) on the real checkpoint, the weights
# and the caches on the device in BFloat16, a token of each of B sequences
# at position 1 into empty states of 512 positions, the states written in
# place and the next token of each returned: B is 1 at the small shape, 2 at
# the doubled one. The checkpoint is TEMPEST_CORPUS_QWEN's directory, else
# ~/models/Qwen/Qwen3.5-4B; the seconds reading it are the case's
# `checkpoint_s`. What is compared is the tokens returned and the states.
using Random
using BFloat16s, CUDACore, Vallmo
import Tempest

const CKPT = get(ENV, "TEMPEST_CORPUS_QWEN", joinpath(homedir(), "models", "Qwen", "Qwen3.5-4B"))
const CTX = 512

batch_of(shape) = shape === :doubled ? 2 : 1

# The tokens, drawn on the host from the vocabulary, and their positions.
draw(vocabulary, B) = (Int32.(rand(MersenneTwister(1), 1:vocabulary, 1, B)), ones(Int32, 1, B))

compared(result, args) = (result, args[3])

function instance(shape)
    B = batch_of(shape)
    checkpoint_s = @elapsed m = Vallmo.load(CKPT; Ctx = CTX, backend = Vallmo.TempestBackend(CuArray))
    tokens, positions = map(x -> Vallmo.on_device(m, x), draw(size(m.weights.embed_tokens.weight, 2), B))
    args = (m.weights, m.config, Vallmo.states(m, B), tokens, positions)
    (; step = Vallmo.forward!, args, target = Tempest.CUDATarget(), precision = "bf16", checkpoint_s, compared,
        note = "B = $B, Ctx $CTX")
end

# The same function untraced: on the host, the weights widened to Float32 and
# the states in Float32, called twice as a run calls it, since each call
# steps the states it writes; the tokens the second returned and the states.
function untraced(shape)
    B = batch_of(shape)
    m = Vallmo.load(CKPT; Ctx = CTX)
    w = Vallmo.mapleaves(x -> eltype(x) === BFloat16s.BFloat16 ? Float32.(x) : x, m.weights)
    tokens, positions = draw(size(w.embed_tokens.weight, 2), B)
    args = (w, m.config, Vallmo.states(m.config, B, w.embed_tokens.weight), tokens, positions)
    Vallmo.forward!(args...)
    compared(Vallmo.forward!(args...), args)
end
