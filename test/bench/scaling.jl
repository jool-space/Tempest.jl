# How preparation grows with a program's size. Families of programs, each
# scaled by one parameter, are traced and prepared twice in one process and
# the second timed, so that Julia's compilation is out of the numbers: for
# each phase its seconds and bytes allocated, the counts of the planner's and
# the cover's work (`Tempest.Compiler.tallied`), and between successive sizes
# the exponent of each in the parameter, which a phase linear in the
# program's size keeps near 1. Preparations are dry (`build = false`) on the
# CUDA target, its claims priced on nominal rates: no device is touched, and
# the numbers are the compiler's, not the device's.
#
#   julia --project=test test/bench/scaling.jl [family ...] [--sizes 1,2,4] [--out file.tsv] [--build]
#
# Families: gpt_layers, gpt_length, gpt_heads (the corpus's GPT step at its
# small width), node_steps and node_unrolled (the neural ODE's RK4, rolled,
# and unrolled with steps of their own lengths), lstm_loop and
# lstm_accumulate (the LSTM's step, as a plain loop and as `accumulate`),
# heat_loop and heat_accumulate (a stencil's steps, as a plain loop under the
# gradient and as `accumulate`), wide (independent branches), maps and norms
# (a deep chain of maps, and of maps and folds), leaves (a step over a nested
# structure of many arrays) and decode (a transformer's decoding step over a
# cache, forward only). Without any, all. `--build` prepares on the device,
# timing the build and bind: run it under the GPU lock. Each size's time is
# printed to stderr as it ends.
using Tempest, CUDACore, cuTile, Random, Printf
import Tempest as T
import Tempest.Compiler as C
include(joinpath(@__DIR__, "gpt.jl"))
include(joinpath(@__DIR__, "node.jl"))
include(joinpath(@__DIR__, "..", "corpus", "lstm.jl"))

module Heat
import Tempest as T
include(joinpath(@__DIR__, "..", "corpus", "heat.jl"))
end

module Ops
using Tempest
include(joinpath(@__DIR__, "..", "fixtures", "operations.jl"))
end

## The families: each a function of its parameter returning the step, its
## arguments, the results that replace arguments, and the target.

const TF32 = T.CUDATarget(precision = :tf32)

function gpt(; layers = 4, T = 64, heads = 4)
    rng = MersenneTwister(0)
    cfg = (; vocabulary = 96, hidden = 128, heads, kv_heads = max(1, heads ÷ 2), head_dim = 128 ÷ heads, ffn = 384, layers,
        T, B = 16)
    params = GPT.parameters(rng, cfg)
    c = GPT.rotary(cfg)
    batch = GPT.get_batch(rng, rand(rng, 1:96, 100_000), cfg.T, cfg.B)
    step = (p, s, b) -> GPT.train_step(p, s, b, c, cfg)
    (step, (params, GPT.state_of(params), batch), (2 => 1, 3 => 2), TF32)
end

# The neural ODE's RK4, which rolls into one loop; unrolled, each step's
# length its own, so that no two steps are alike.
function node(steps; unrolled = false)
    rng = Xoshiro(0)
    cfg = (; dims = 32, hidden = 256, steps, batch = 128)
    params = NODE.parameters(rng, cfg)
    function rk4(f, u)
        for k in 1:steps
            dt = (1.0f0 + Float32(k) / 1024) / Float32(steps)
            k1 = f(u)
            k2 = f(u .+ (dt / 2) .* k1)
            k3 = f(u .+ (dt / 2) .* k2)
            k4 = f(u .+ dt .* k3)
            u = u .+ (dt / 6) .* (k1 .+ 2.0f0 .* k2 .+ 2.0f0 .* k3 .+ k4)
        end
        u
    end
    loss(p, b) = (d = rk4(u -> NODE.field(p, u), b.u0) .- b.y; sum(abs2.(d)) / length(d))
    function unrolled_step(p, s, b)
        l, (g,) = T.withgradient(q -> loss(q, b), p)
        p, s = NODE.radam(p, s, g)
        (l, p, s)
    end
    step = unrolled ? unrolled_step : (p, s, b) -> NODE.train_step(p, s, b, steps)
    (step, (params, NODE.state_of(params), NODE.batch(rng, cfg)), (2 => 1, 3 => 2), TF32)
end

lstm_parameters(rng, D, H, B) = (w(o, i) = 0.1f0 .* randn(rng, Float32, o, i);
    (; Wi = w(H, D), Wf = w(H, D), Wg = w(H, D), Wo = w(H, D), Ui = w(H, H), Uf = w(H, H), Ug = w(H, H), Uo = w(H, H),
        bi = w(H, 1), bf = w(H, 1), bg = w(H, 1), bo = w(H, 1), h0 = zeros(Float32, H, B)))

function lstm(steps; accumulated = false)
    D, H, B = 32, 256, 64
    rng = Xoshiro(0)
    p = lstm_parameters(rng, D, H, B)
    xs, y = randn(rng, Float32, D, B, steps), randn(rng, Float32, H, B)
    run(q, xs) = first(last(accumulate((s, x) -> cell(q, s[1], s[2], x), eachslice(xs; dims = 3); init = (q.h0, q.h0))))
    accumulated_loss(q, xs, y) = sum(abs2, run(q, xs) .- y) / length(y)
    function accumulated_step(q, xs, y)
        l, (g,) = T.withgradient(r -> accumulated_loss(r, xs, y), q)
        (l, map((x, d) -> x .- 1.0f-2 .* d, q, g))
    end
    (accumulated ? accumulated_step : train_step, (p, xs, y), (2 => 1,), TF32)
end

# A stencil's steps as `accumulate` over the source's slices, or as a plain
# loop under the gradient.
function heat(steps; accumulated = true)
    rng = Xoshiro(0)
    u, q = randn(rng, Float32, 64, 64), 0.01f0 .* randn(rng, Float32, 64, 64, steps)
    function looped(u, q)
        inside = Heat.interior(u)
        for t in 1:size(q, 3)
            u = Heat.diffuse(u, inside) .+ q[:, :, t]
        end
        sum(abs2, u)
    end
    (accumulated ? Heat.step : (u, q) -> T.withgradient(v -> looped(v, q), u), (u, q), (), T.CUDATarget())
end

# `n` branches, each a layer of its own weights on the shared input, summed:
# a program as wide as it is long.
function wide(n)
    rng = Xoshiro(0)
    ws = Tuple(0.1f0 .* randn(rng, Float32, 64, 64) for _ in 1:n)
    x = randn(rng, Float32, 64, 32)
    loss(ws, x) = sum(w -> sum(abs2, tanh.(w * x)), ws)
    step(ws, x) = ((l, (g,)) = T.withgradient(v -> loss(v, x), ws); (l, map((w, d) -> w .- 1.0f-2 .* d, ws, g)))
    (step, (ws, x), (2 => 1,), TF32)
end

# A chain of `n` maps, each reading the last: one term `n` deep.
function maps(n)
    x = randn(Xoshiro(0), Float32, 256, 256)
    function chain(x)
        for k in 1:n
            x = tanh.(x) .* 0.9f0 .+ Float32(k)
        end
        x
    end
    (chain, (x,), (), TF32)
end

# A chain of `n` normalizations, each a fold and the maps that read it.
function norms(n)
    x = randn(Xoshiro(0), Float32, 256, 256)
    function chain(x)
        for _ in 1:n
            x = tanh.(x .- sum(x; dims = 1) ./ size(x, 1))
        end
        sum(abs2, x)
    end
    (x -> T.withgradient(chain, x), (x,), (), TF32)
end

# A step over parameters held as a nested structure of `n` arrays: each
# leaf's gradient and an update, as an optimizer's step over a model's.
function leaves(n)
    rng = Xoshiro(0)
    params = Tuple((; w = 0.1f0 .* randn(rng, Float32, 32, 32), b = zeros(Float32, 32)) for _ in 1:(n ÷ 2))
    x = randn(rng, Float32, 32, 16)
    loss(ps, x) = sum(p -> sum(abs2, p.w * x .+ p.b), ps)
    function step(ps, x)
        l, (g,) = T.withgradient(q -> loss(q, x), ps)
        (l, map((p, d) -> (; w = p.w .- 1.0f-2 .* d.w, b = p.b .- 1.0f-2 .* d.b), ps, g))
    end
    (step, (params, x), (2 => 1,), TF32)
end

# A transformer's decoding step: one token through `layers` layers, each
# attending over a cache read up to a length the device holds, then the
# logits. Forward only.
function decode(layers)
    rng = Xoshiro(0)
    D, H, KV, d, S, V = 256, 8, 2, 32, 512, 1024
    w(dims...) = 0.02f0 .* randn(rng, Float32, dims...)
    blocks = Tuple((; norm1 = ones(Float32, D), wq = w(D, d * H), wk = w(D, d * KV), wv = w(D, d * KV), wo = w(d * H, D),
        kc = w(d, S, KV), vc = w(d, S, KV), norm2 = ones(Float32, D), wg = w(D, 4D), wu = w(D, 4D), wd = w(4D, D)) for _ in 1:layers)
    params = (; blocks, wte = w(D, V), norm = ones(Float32, D))
    x, len = w(D, 1), Int32[S - 7]
    norm(x, g) = g .* x ./ sqrt.(sum(abs2, x; dims = 1) ./ size(x, 1) .+ 1.0f-5)
    function attend(x, b, len)
        q = reshape(b.wq' * x, d, H)
        s = reshape(sum(reshape(b.kc, d, S, 1, KV) .* reshape(q, d, 1, H ÷ KV, KV); dims = 1), S, H ÷ KV, KV) .* 0.25f0
        s = ifelse.((Int32(1):Int32(S)) .<= reshape(len, 1, 1, 1), s, -Inf32)
        a = exp.(s .- T.logsumexp(s; dims = 1))
        o = sum(reshape(b.vc, d, S, 1, KV) .* reshape(a, 1, S, H ÷ KV, KV); dims = 2)
        b.wo' * reshape(o, d * H, 1)
    end
    function step(p, x, len)
        for b in p.blocks
            x = x .+ attend(norm(x, b.norm1), b, len)
            h = norm(x, b.norm2)
            x = x .+ b.wd' * (GPT.swish.(b.wg' * h) .* (b.wu' * h))
        end
        p.wte' * norm(x, p.norm)
    end
    (step, (params, x, len), (), TF32)
end

const FAMILIES = [
    "gpt_layers" => (n -> gpt(layers = n), [1, 2, 4, 8, 16, 32]),
    "gpt_length" => (n -> gpt(layers = 2, T = n), [64, 128, 256, 512, 1024]),
    "gpt_heads" => (n -> gpt(layers = 2, heads = n), [2, 4, 8, 16]),
    "node_steps" => (node, [5, 10, 20, 40, 80]),
    "node_unrolled" => (n -> node(n; unrolled = true), [2, 4, 8, 16, 32]),
    "lstm_loop" => (lstm, [16, 32, 64, 128, 256]),
    "lstm_accumulate" => (n -> lstm(n; accumulated = true), [16, 32, 64, 128, 256]),
    "heat_loop" => (n -> heat(n; accumulated = false), [5, 10, 20, 40]),
    "heat_accumulate" => (heat, [25, 50, 100, 200, 400]),
    "wide" => (wide, [8, 16, 32, 64, 128]),
    "maps" => (maps, [16, 32, 64, 128, 256]),
    "norms" => (norms, [8, 16, 32, 64, 128]),
    "leaves" => (leaves, [16, 32, 64, 128, 256, 512]),
    "decode" => (decode, [2, 4, 8, 16, 32]),
]

## Measuring.

"""
One preparation of the family's program at size `n`, its work tallied: the
program's size, the counts, and each phase's seconds, bytes allocated and
seconds collecting garbage.
"""
const OUTERMOST = (:trace, :lower, :rewrite, :partition, :decider, :planned, :bind, :explain)

function measured(make, n; build)
    f, args, replaces, target = make(n)
    program, traced = spent(() -> C.trace(f, args...))
    inputs = build ? map(on_device, args) : args
    ex, counts, phases = C.tallied(() -> C.prepare(program, inputs...; target, replaces, build, measure = :none))
    text, explained = spent(() -> sprint(T.explain, ex))
    phases = [:trace => traced; phases; :explain => explained]
    # The whole is the sum of the outermost phases: the others are within them.
    total = ntuple(k -> sum(p -> first(p) in OUTERMOST ? p[2][k] : 0, phases), 3)
    size = (; nodes = length(program.nodes), values = length(ex.plan.graph.values), kernels = length(ex.plan.groups),
        retained = Base.summarysize(ex), explained = length(text))
    (; size, counts, phases = [phases; :total => total])
end

# What `f()` returns, and its seconds, bytes allocated and seconds collecting garbage.
function spent(f)
    t, b, g = time_ns(), Base.gc_bytes(), Base.gc_num().total_time
    x = f()
    x, (1.0e-9 * (time_ns() - t), Base.gc_bytes() - b, 1.0e-9 * (Base.gc_num().total_time - g))
end

on_device(x::AbstractArray) = CuArray(x)
on_device(x::Union{Tuple, NamedTuple}) = map(on_device, x)
on_device(x) = x

exponent(a, b, n, m) = a > 0 && b > 0 ? log(b / a) / log(m / n) : NaN

function report(io, family, sizes, runs)
    @printf(io, "\n%s\n", family)
    row(name, xs, fmt) = (@printf(io, "  %-22s", name); foreach(x -> print(io, fmt(x)), xs))
    row("size", sizes, x -> @sprintf("%11d", x)); @printf(io, "%9s\n", "exponent")
    for key in (:nodes, :values, :kernels)
        xs = [getfield(r.size, key) for r in runs]
        row(string(key), xs, x -> @sprintf("%11d", x)); @printf(io, "%9.2f\n", exponent(xs[end - 1], xs[end], sizes[end - 1], sizes[end]))
    end
    xs = [r.size.retained / 2^20 for r in runs]
    row("retained MB", xs, x -> @sprintf("%11.2f", x)); @printf(io, "%9.2f\n", exponent(xs[end - 1], xs[end], sizes[end - 1], sizes[end]))
    names = unique(first(p) for r in runs for p in r.phases)
    for (title, k, scale, fmt) in (("seconds", 1, 1.0, "%11.3f"), ("seconds collecting garbage", 3, 1.0, "%11.3f"),
            ("MB allocated", 2, 2.0^-20, "%11.1f"))
        @printf(io, "  %s\n", title)
        for name in names
            xs = [scale * something(get(Dict(r.phases), name, nothing), (0.0, 0, 0.0))[k] for r in runs]
            row(string(name), xs, x -> Printf.format(Printf.Format(fmt), x))
            @printf(io, "%9.2f\n", exponent(xs[end - 1], xs[end], sizes[end - 1], sizes[end]))
        end
    end
    @printf(io, "  counts\n")
    for name in C.COUNTED
        xs = [r.counts[name] for r in runs]
        row(string(name), xs, x -> @sprintf("%11d", x)); @printf(io, "%9.2f\n", exponent(xs[end - 1], xs[end], sizes[end - 1], sizes[end]))
    end
end

function record(path, family, sizes, runs)
    open(path, "a") do io
        for (n, r) in zip(sizes, runs)
            for (k, v) in pairs(r.size)
                println(io, join((family, n, "size", k, v), '\t'))
            end
            for (name, (s, b, g)) in r.phases
                println(io, join((family, n, "seconds", name, s), '\t'))
                println(io, join((family, n, "bytes", name, b), '\t'))
                println(io, join((family, n, "garbage", name, g), '\t'))
            end
            for (k, v) in r.counts
                println(io, join((family, n, "count", k, v), '\t'))
            end
        end
    end
end

function main(args)
    option(name) = (k = findfirst(==(name), args); k === nothing ? nothing : args[k + 1])
    sizes = option("--sizes")
    out = option("--out")
    build = "--build" in args
    flags = Set(filter(!isnothing, [option("--sizes"), option("--out")]))
    chosen = filter(a -> !startswith(a, "--") && !(a in flags), args)
    for (family, (make, default)) in FAMILIES
        isempty(chosen) || family in chosen || continue
        ns = sizes === nothing ? default : parse.(Int, split(sizes, ','))
        # Each size twice, the second kept: Julia compiles the preparation's
        # code, and the tracer's for the arguments' types, on the first.
        runs = map(ns) do n
            t = time()
            measured(make, n; build)
            r = measured(make, n; build)
            @printf(stderr, "%s %d: %.1f s\n", family, n, time() - t)
            r
        end
        report(stdout, family, ns, runs)
        out === nothing || record(out, family, ns, runs)
        flush(stdout)
    end
end

# Run as a script; included, it defines the families.
abspath(PROGRAM_FILE) == abspath(@__FILE__()) && main(ARGS)
