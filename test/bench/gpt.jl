# A character-level GPT training step, the model of today's Tempest's
# examples/gpt.jl (nanoGPT's model_full.jl, jool-space) layer for layer, with
# its initialization and random stream, so losses compare step for step:
# token embedding with a tied head, RMSNorm, grouped-query attention with
# rotary positions and a causal mask, SwiGLU, cross-entropy and Adam.
#
# The step is one traced program, the gradient of the loss and Adam, prepared
# with the parameters and Adam's state replaced by its results in place, and
# replayed as one captured graph.
#
#   TEMPEST_GPT_TEXT=~/dev/jool/Tempest/docs/src julia --project=test test/bench/gpt.jl
#
# TEMPEST_GPT_TEXT is a directory of .md files or a file (today's Tempest
# trains on its docs/src; nanoGPT on Reactant's). TEMPEST_GPT_PRECISION is full
# or tf32, TEMPEST_GPT_STEPS the steps timed (default 200; 0 times nothing),
# TEMPEST_GPT_SEED the random stream (0 matches nanoGPT), TEMPEST_GPT_CHECK=0
# skips the interpreter, TEMPEST_GPT_PROFILE=0 the profile,
# TEMPEST_GPT_EXPLAIN a file for the plans, and
# TEMPEST_GPT_{HIDDEN,HEADS,KV_HEADS,FFN,LAYERS,T,B} give the shape.
module GPT

using Tempest, Tempest.Compiler, CUDACore, Random, Printf
import Tempest as T
import Tempest.Compiler as C
module Ops
using Tempest
include(joinpath(@__DIR__, "..", "fixtures", "operations.jl"))
end

## Data, as nanoGPT's data.jl.

function load_corpus(path)
    files = isdir(path) ?
        filter(f -> endswith(f, ".md"), [joinpath(r, f) for (r, _, fs) in walkdir(path) for f in fs]) : [path]
    text = join(read.(files, String), "\n")
    chars = sort!(unique(text))
    lookup = Dict(c => i for (i, c) in enumerate(chars))
    (; data = [lookup[c] for c in text], chars)
end

function get_batch(rng, data, T, B)
    starts = rand(rng, 1:(length(data) - T), B)
    tokens = Int32[data[s + t - 1] for t in 1:T, s in starts]
    targets = Int32[data[s + t] for t in 1:T, s in starts]
    (; tokens, targets)
end

## Parameters, in nanoGPT's initialization order.

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
    blocks = Tuple(block_parameters(rng, cfg) for _ in 1:cfg.layers)
    (; wte, blocks, norm = ones(Float32, cfg.hidden))
end

# Rotary tables in rotate-half form: rope(x) = x .* C .+ x[swap, ...] .* S.
function rotary(cfg; theta = 10000.0f0)
    d, T = cfg.head_dim, cfg.T
    inv_freq = theta .^ (-Float32.(0:2:(d - 1)) ./ d)
    angles = inv_freq .* Float32.(0:(T - 1))'
    C = reshape(vcat(cos.(angles), cos.(angles)), d, 1, T)
    S = reshape(vcat(-sin.(angles), sin.(angles)), d, 1, T)
    (; C, S)
end

## The model: plain array code, which runs eagerly as it is.

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
    # Query head j reads key/value head cld(j, group): nanoGPT's (g h) axis.
    o = Ops.attention(q, k, project(p.wv); causal = true, scale = 1 / sqrt(Float32(head_dim)))
    reshape(p.wo, :, D)' * reshape(o, :, N)
end

swish(x) = x / (1 + exp(-x))

feedforward(x, p) = p.wd' * (swish.(p.wg' * x) .* (p.wu' * x))

# The logit of each column's target: a fold over the vocabulary of the logits
# where the row is the target, so it fuses with the logsumexp beside it and
# its gradient is pointwise.
take_rows(logits, targets) =
    sum(ifelse.((1:size(logits, 1)) .== reshape(targets, 1, :), logits, zero(eltype(logits))); dims = 1)

function loss(p, batch, c, cfg)
    x = reshape(p.wte[:, batch.tokens], size(p.wte, 1), :)
    for block in p.blocks
        x = x + attention(rms_norm(x, block.norm1), block, c, cfg)
        x = x + feedforward(rms_norm(x, block.norm2), block)
    end
    logits = p.wte' * rms_norm(x, p.norm)
    sum(Tempest.logsumexp(logits; dims = 1) .- take_rows(logits, batch.targets)) / size(logits, 2)
end

# Adam as Optimisers.AdamW(rate) with its default zero weight decay. The
# state holds the moments and the step count.
function adam(p, state, gradients; rate = 3.0f-3, beta1 = 0.9f0, beta2 = 0.999f0, epsilon = 1.0f-8)
    step = state.step .+ 1.0f0
    first = map_leaves((m, g) -> beta1 .* m .+ (1 - beta1) .* g, state.first, gradients)
    second = map_leaves((v, g) -> beta2 .* v .+ (1 - beta2) .* abs2.(g), state.second, gradients)
    correction1, correction2 = 1 .- beta1 .^ step, 1 .- beta2 .^ step
    updated = map_leaves(p, first, second) do x, m, v
        x .- rate .* (m ./ correction1) ./ (sqrt.(v ./ correction2) .+ epsilon)
    end
    updated, (; first, second, step)
end

map_leaves(f, xs::Union{Tuple, NamedTuple}...) = map((ys...) -> map_leaves(f, ys...), xs...)
map_leaves(f, xs...) = f(xs...)

function configuration(vocabulary)
    int(name, default) = parse(Int, get(ENV, "TEMPEST_GPT_$name", string(default)))
    hidden, heads = int("HIDDEN", 128), int("HEADS", 4)
    (; vocabulary, hidden, heads, kv_heads = int("KV_HEADS", 2), head_dim = hidden ÷ heads, ffn = int("FFN", 384),
        layers = int("LAYERS", 4), T = int("T", 64), B = int("B", 16))
end

## The step, prepared.

state_of(params) = (first = map_leaves(zero, params), second = map_leaves(zero, params), step = zeros(Float32, 1))

# A training step as a function: the loss and its gradient, then Adam.
function train_step(p, state, batch, c, cfg)
    l, g = Tempest.gradient(q -> loss(q, batch, c, cfg), p)
    p, state = adam(p, state, g)
    (l, p, state)
end

step_program(params, state, batch, c, cfg) = trace((p, s, b) -> train_step(p, s, b, c, cfg), params, state, batch)

"""
The training step on a target, prepared from the initial parameters, Adam's
state and the first batch: its parameters and state are replaced by its
results, so each execution takes the next step, and only the batch is bound
between steps.
"""
struct Step
    program::C.Program
    ex::C.Executable
end

function Step(params, batch, c, cfg; target, device)
    state = state_of(params)
    program = step_program(params, state, batch, c, cfg)
    # TEMPEST_GPT_PARTITION=0 saves every map the backward reads (TRIALS: store or recompute).
    ex = prepare(program, map_leaves(device, params), map_leaves(device, state), map_leaves(device, batch); target,
        replaces = (2 => 1, 3 => 2), partition = get(ENV, "TEMPEST_GPT_PARTITION", "1") != "0")
    Step(program, ex)
end

# One step: the captured graph once there is one, else kernel by kernel.
launch!(s::Step) = s.ex.captured === nothing ? execute!(s.ex) : replay!(s.ex)

bind_batch!(s::Step, batch) = bind!(s.ex, 3 => batch)

"Start again from parameters, a fresh state and a batch."
restart!(s::Step, params, batch) = bind!(s.ex, params, state_of(params), batch)

loss_value(s::Step) = Array(first(outputs(s.ex)))[]

## Measuring.

"Device time of one step, the minimum over samples of a captured graph of several."
function device_milliseconds(s::Step; samples = 10, milliseconds = 200.0)
    launch!(s)
    capture!(s.ex)
    launch!(s)
    CUDACore.synchronize()
    once = minimum(_ -> CUDACore.@elapsed(launch!(s)), 1:3)
    repetitions = clamp(round(Int, 1.0e-3 * milliseconds / once), 1, 100)
    capture!(s.ex; repetitions)
    launch!(s)
    t = minimum(_ -> 1.0e3 * CUDACore.@elapsed(launch!(s)) / repetitions, 1:samples)
    capture!(s.ex)
    t
end

"""
The steady time per step as today's Tempest's example measures it: a batch is
drawn on the host and bound before each replay of the captured step, the loss
is read every ten steps, and the time covers the second half of the run.
"""
function steady_milliseconds(s::Step, rng, corpus, cfg; steps)
    s.ex.captured === nothing && capture!(s.ex)
    halfway = 0.0
    for i in 1:steps
        bind_batch!(s, get_batch(rng, corpus.data, cfg.T, cfg.B))
        launch!(s)
        i % 10 == 0 && loss_value(s)
        i == steps ÷ 2 && (CUDACore.synchronize(); halfway = time())
    end
    CUDACore.synchronize()
    1.0e3 * (time() - halfway) / (steps - steps ÷ 2)
end

# What a kernel is, for a profile: its implementation and sketch.
kind(k) = k isa C.FloorKernels ? "floor" : (x = string(k); startswith(x, "cuTile ") ? join(split(x)[1:2], " ") :
    startswith(x, "cuBLAS") ? "cuBLAS product" : first(split(x, ",")))

"Device time of every kernel of a prepared program, timed alone: (kernel, seconds)."
profile(ex) = [(k, C.time_launch(ex.target, l)) for (k, l) in zip(ex.kernels, ex.launches)]

function report_profile(io, s::Step)
    rows = profile(s.ex)
    totals = Dict{String, Tuple{Float64, Int}}()
    for (k, t) in rows
        a, n = get(totals, kind(k), (0.0, 0))
        totals[kind(k)] = (a + t, n + 1)
    end
    copies = isempty(s.ex.copies) ? 0.0 : C.time_launch(s.ex.target, () -> foreach(copy -> copy(), s.ex.copies))
    @printf(io, "%-44s %10s %8s\n", "kernels timed alone", "ms", "count")
    for (name, (t, n)) in sort!(collect(totals); by = x -> -x[2][1])
        @printf(io, "%-44s %10.3f %8d\n", name, 1.0e3 * t, n)
    end
    @printf(io, "%-44s %10.3f %8d\n", "copies of staged results", 1.0e3 * copies, length(s.ex.copies))
    @printf(io, "%-44s %10.3f\n", "sum", 1.0e3 * (sum(last, rows) + copies))
    println(io, "slowest kernels:")
    for (k, t) in sort(rows; by = r -> -r[2])[1:min(8, length(rows))]
        @printf(io, "  %8.3f ms  %s\n", 1.0e3 * t, k)
    end
    rows
end

## The run.

function main()
    text = get(ENV, "TEMPEST_GPT_TEXT", "")
    isempty(text) && error("set TEMPEST_GPT_TEXT to a directory of .md files; today's Tempest trains on its docs/src")
    corpus = load_corpus(text)
    cfg = configuration(length(corpus.chars))
    rng = MersenneTwister(parse(Int, get(ENV, "TEMPEST_GPT_SEED", "0")))
    params = parameters(rng, cfg)
    c = rotary(cfg)
    batch = get_batch(rng, corpus.data, cfg.T, cfg.B)
    precision = Symbol(get(ENV, "TEMPEST_GPT_PRECISION", "full"))
    target = T.CUDATarget(; precision)
    println("corpus: ", length(corpus.data), " chars, vocabulary ", cfg.vocabulary, "; T=", cfg.T, " B=", cfg.B, ", ",
        sum(length, C.leaves(params)), " parameters, ", cfg.layers, " layers; ", target)
    seconds = @elapsed s = Step(params, batch, c, cfg; target, device = CuArray)
    storage = s.ex.storage
    println("trace, differentiate and prepare: ", round(seconds; digits = 1), " s; ", length(s.ex.kernels), " kernels; ",
        length(storage.donated), " results in place of their arguments, ", length(storage.staged), " staged")
    for (k, g) in zip(s.ex.kernels, s.ex.plan.groups)
        k isa C.FloorKernels && println("  on the floor: ", g.kind === :opaque ? "a scatter" : "a $(g.kind) group", " over ", g.domain, " (", k, ")")
    end
    path = get(ENV, "TEMPEST_GPT_EXPLAIN", "")
    isempty(path) || open(io -> explain(io, s.ex), path, "w")
    # The first step from the initial parameters, twice, bit for bit; its
    # first moments are a tenth of the gradients, against the interpreter's.
    first_moments() = map(Array, C.leaves(outputs(s.ex)[3].first))
    launch!(s)
    first_run = first_moments()
    restart!(s, params, batch)
    launch!(s)
    println("first step bitwise identical in a second run: ", first_run == first_moments())
    if get(ENV, "TEMPEST_GPT_CHECK", "1") != "0"
        reference = interpret(s.program, params, state_of(params), batch)
        errors = map((a, b) -> maximum(abs.(a .- b)) / maximum(abs.(b)), first_run, C.leaves(reference[3].first))
        @printf("loss %.7f, the interpreter's %.7f; gradients' largest relative difference %.1e (median %.1e)\n",
            loss_value(s), reference[1], maximum(errors), sort(errors)[(length(errors) + 1) ÷ 2])
    end
    # Losses step by step, the first on the batch the step was prepared with.
    restart!(s, params, batch)
    losses = Float32[]
    for i in 1:3
        i > 1 && bind_batch!(s, get_batch(rng, corpus.data, cfg.T, cfg.B))
        launch!(s)
        push!(losses, loss_value(s))
    end
    println("losses at steps 1 to 3: ", join(map(repr, losses), ", "))
    steps = parse(Int, get(ENV, "TEMPEST_GPT_STEPS", "200"))
    steps > 0 || return s
    @printf("device time per step: %.3f ms\n", device_milliseconds(s))
    @printf("steady: %.3f ms/step over the last %d of %d steps\n", steady_milliseconds(s, rng, corpus, cfg; steps), steps - steps ÷ 2, steps)
    get(ENV, "TEMPEST_GPT_PROFILE", "1") == "0" || report_profile(stdout, s)
    s
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    import cuBLAS, cuTile
    GPT.main()
end
