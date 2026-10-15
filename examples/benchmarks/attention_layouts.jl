# Layout copies around attention: how many steps does Tempest add to present
# q, k, v and the output densely to the attention call, and what share of the
# time do they take?
#
# Attention takes dense (features, heads, positions, batch) arrays. A layout
# operation between a producer and the call, or between the call and a matrix
# product, is materialized as its own copy step unless a pointwise region
# reads it anyway (rotary embedding, a cast in training). These cases measure
# the copies, with cuDNN's and with cuTile's attention:
#
# - `gpt`: examples/gpt.jl, training step and forward loss. Float32 data;
#   cuTile computes products and attention in TF32, cuDNN rounds attention
#   operands to BFloat16 (`float32_cast`), on a TF32 target.
# - `mha reshape`: a multi-head attention block whose head split and merge are
#   reshapes of the projections, the order attention takes.
# - `mha rows`: the same block written over (tokens, features) rows, as a
#   literal port of `x @ W` code: the head split is reshape + permutedims of
#   `x * W`, the merge permutedims + reshape before `* wo`.
# - `mha packed`: one projection to `[q; k; v]` features, sliced per operand
#   (GPT-2's `c_attn`), merged by a reshape.
#
# The MHA cases run Float16 and BFloat16 data on cuDNN; cuTile attention is
# Float32 only, so its MHA rows run Float32 data under TF32.
#
# For each executable, `explain` names the layout steps (a step whose output is
# a layout operation), and `Tempest.profile` times every call with CUDA events:
# layout steps, attention calls (including cuDNN's casts inside a forward-only
# call), and the sum of all calls. "graph" is the median device time per step of
# a CUDA graph of `repetitions` back-to-back executions, without event gaps.
# Run: julia --project=examples examples/benchmarks/attention_layouts.jl
# Environment: CASES (comma-separated case names), D, HEADS, T, B.
using Tempest, CUDACore, Statistics, Random
import cuBLAS, cuTile, cuDNN
using Tempest: attention

include(joinpath(@__DIR__, "..", "gpt.jl"))

int(name, default) = parse(Int, get(ENV, name, string(default)))
const D, HEADS, T, B = int("D", 768), int("HEADS", 12), int("T", 1024), int("B", 8)

## Blocks

function mha_reshape(x, wq, wk, wv, wo)
    D, T, B = size(x)
    X = reshape(x, D, :)
    split(w) = reshape(w * X, D ÷ HEADS, HEADS, T, B)
    o = attention(split(wq), split(wk), split(wv); causal = true)
    wo * reshape(o, D, :)
end

function mha_rows(x, wq, wk, wv, wo)
    N, D = size(x)
    split(w) = permutedims(reshape(x * w, T, N ÷ T, D ÷ HEADS, HEADS), (3, 4, 1, 2))
    o = attention(split(wq), split(wk), split(wv); causal = true)
    reshape(permutedims(o, (3, 4, 1, 2)), N, D) * wo
end

function mha_packed(x, wqkv, wo)
    D, T, B = size(x)
    y = reshape(wqkv * reshape(x, D, :), D ÷ HEADS, HEADS, 3, T, B)
    o = attention(y[:, :, 1, :, :], y[:, :, 2, :, :], y[:, :, 3, :, :]; causal = true)
    wo * reshape(o, D, :)
end

weight(rng, E, dims...) = CuArray(E.(0.02f0 .* randn(rng, Float32, dims...)))

function mha_arguments(kind, E)
    rng = MersenneTwister(1)
    if kind === :rows
        x = weight(rng, E, T * B, D) .* E(50)
        (mha_rows, (x, (weight(rng, E, D, D) for _ = 1:4)...))
    elseif kind === :packed
        x = weight(rng, E, D, T, B) .* E(50)
        (mha_packed, (x, weight(rng, E, 3D, D), weight(rng, E, D, D)))
    else
        x = weight(rng, E, D, T, B) .* E(50)
        (mha_reshape, (x, (weight(rng, E, D, D) for _ = 1:4)...))
    end
end

## Providers

const PROVIDERS = (
    cuDNN = (; implementations = (CuDNNKernels(),), target = CUDATarget()),
    cuDNN_cast = (;
        implementations = (CuDNNKernels(float32_cast = Core.BFloat16),),
        target = CUDATarget(matmul_precision = :tf32),
    ),
    cuTile = (; target = CUDATarget(matmul_precision = :tf32)),
)

function prepare_mha(kind, E, provider, training)
    f, args = mha_arguments(kind, E)
    flags = PROVIDERS[provider]
    training ? prepare((args...) -> sum(abs2, f(args...)), args...; flags...) :
        prepare_forward(f, args...; flags...)
end

function prepare_gpt(provider, training)
    G = GPTExample
    cfg = (;
        vocabulary = 96, hidden = D, heads = HEADS, kv_heads = HEADS ÷ 3, head_dim = D ÷ HEADS,
        ffn = 4D, layers = 2, T, B = max(1, B ÷ 2), dropout = 0.0,
    )
    rng = MersenneTwister(0)
    data = rand(rng, 1:(cfg.vocabulary), 100_000)
    params = G.map_leaves(CuArray, G.parameters(rng, cfg))
    c = G.map_leaves(CuArray, G.constants(cfg))
    batch = G.map_leaves(CuArray, G.batch_arrays(G.get_batch(rng, data, cfg.T, cfg.B)..., cfg.vocabulary))
    loss(p, b, c) = G.loss(p, b, c, cfg)
    flags = PROVIDERS[provider]
    training ? prepare(loss, params, batch, c; wrt = (1,), flags...) :
        prepare_forward(loss, params, batch, c; flags...)
end

## Classification and timing

const ATTENTION_CALLS = (:attention_forward, :attention_backward, :sdpa_forward, :sdpa_backward)

layout_step(ex, step) = Tempest.is_layout(ex.storage.program.nodes[step.output])
attention_step(step) =
    step.region isa Tempest.OpaqueCall && nameof(step.region.call.f) in ATTENTION_CALLS
attention_node(node) = node.op isa Tempest.CallResult && nameof(node.op.call.f) in ATTENTION_CALLS

# A layout value next to attention: connected to an attention call's operand or
# result through layout operations alone.
function next_to_attention(nodes, users, id)
    seen, stack = Set{Int}(), [id]
    while !isempty(stack)
        i = pop!(stack)
        i in seen && continue
        push!(seen, i)
        attention_node(nodes[i]) && return true
        Tempest.is_layout(nodes[i]) || continue
        push!(stack, only(nodes[i].inputs))
        for j in users[i]
            attention_node(nodes[j]) && return true
            Tempest.is_layout(nodes[j]) && push!(stack, j)
        end
    end
    false
end

function users_of(nodes)
    users = [Int[] for _ in nodes]
    for (i, node) in enumerate(nodes), j in node.inputs
        push!(users[j], i)
    end
    users
end

function graph_time(ex; repetitions = 10, samples = 7)
    execute!(ex)
    CUDACore.synchronize()
    graph = CUDACore.capture() do
        for _ = 1:repetitions
            execute!(ex)
        end
    end
    exec = CUDACore.instantiate(graph)
    CUDACore.launch(exec)
    median(1e6 * CUDACore.@elapsed(CUDACore.launch(exec)) / repetitions for _ = 1:samples)
end

function measure(ex)
    steps, nodes = ex.storage.steps, ex.storage.program.nodes
    users = users_of(nodes)
    near(step) = next_to_attention(nodes, users, step.output)
    layouts = [i for (i, step) in enumerate(steps) if layout_step(ex, step)]
    adjacent = filter(i -> near(steps[i]), layouts)
    rows = Tempest.profile(devnull, ex; samples = 21)
    layout_time = adjacent_time = attention_time = 0.0
    for row in rows
        group = steps[ex.groups[row.call]]
        if all(step -> layout_step(ex, step), group)
            layout_time += row.seconds
            all(near, group) && (adjacent_time += row.seconds)
        elseif any(attention_step, group)
            attention_time += row.seconds
        end
    end
    calls = sum(row -> row.seconds, rows)
    (; layouts, adjacent, layout_time = 1e6 * layout_time, adjacent_time = 1e6 * adjacent_time,
       attention_time = 1e6 * attention_time, calls = 1e6 * calls, graph = graph_time(ex))
end

function describe_layouts(ex, r)
    nodes = ex.storage.program.nodes
    map(r.layouts) do i
        step = ex.storage.steps[i]
        node = nodes[step.output]
        near = i in r.adjacent ? "next to attention" : "elsewhere"
        "$(nameof(typeof(node.op))) $(join(node.dims, "×")) $(node.dtype) ($near)"
    end
end

const CASES = Dict(
    "gpt" => (training -> [("gpt", "Float32", p, () -> prepare_gpt(p, training)) for p in (:cuDNN_cast, :cuTile)]),
    "mha reshape" => (training -> mha_cases(:reshape, training)),
    "mha rows" => (training -> mha_cases(:rows, training)),
    "mha packed" => (training -> mha_cases(:packed, training)),
)

mha_cases(kind, training) = [
    [("mha $kind", string(E), :cuDNN, () -> prepare_mha(kind, E, :cuDNN, training)) for E in (Float16, Core.BFloat16)];
    [("mha $kind", "Float32", :cuTile, () -> prepare_mha(kind, Float32, :cuTile, training))]
]

function run(names)
    CUDACore.functional() || error("requires CUDA")
    println(CUDACore.name(CUDACore.device()), "; cuDNN ", cuDNN.version(), "; D=$D heads=$HEADS T=$T B=$B")
    println("| case | pass | data | attention | layout steps: next to attention / all | their μs | attention μs | calls μs | graph μs | share next to attention |")
    println("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    details = String[]
    for name in names, training in (false, true), (label, data, provider, build) in CASES[name](training)
        ex = build()
        expected = provider === :cuTile ? :cutile : :cuDNN
        expected in ex.providers || error("$label $provider did not select $expected")
        r = measure(ex)
        pass = training ? "fwd+bwd" : "fwd"
        share = round(100 * r.adjacent_time / r.calls; digits = 1)
        println("| $label | $pass | $data | $provider | $(length(r.adjacent)) / $(length(r.layouts)) | ",
                round(r.adjacent_time; digits = 1), " / ", round(r.layout_time; digits = 1), " | ",
                round(r.attention_time; digits = 1), " | ",
                round(r.calls; digits = 1), " | ", round(r.graph; digits = 1), " | $share% |")
        isempty(r.layouts) ||
            push!(details, "$label $pass $data $provider: " * join(describe_layouts(ex, r), "; "))
        ex = nothing
        GC.gc()
        CUDACore.reclaim()
    end
    println()
    foreach(println, details)
end

run(split(get(ENV, "CASES", "gpt,mha reshape,mha rows,mha packed"), ","))
