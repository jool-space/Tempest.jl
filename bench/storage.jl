# Bytes of prepared programs' buffers, with slots shared by values whose
# lifetimes do not overlap and with each value stored apart: training steps of
# attention, of an attention block (projections, attention, an output
# projection and a residual) and of an MLP, and the coverage suite's
# compositions. Kernels' own workspaces are not counted.
#   julia --project=test bench/storage.jl
using Tempest, CUDACore, cuBLAS, cuTile, Printf
import Tempest as T

# Plans by estimate, as the tests make them.
ENV["TEMPEST_CACHE"] = mktempdir()
include(joinpath(@__DIR__, "..", "test", "compositions.jl"))

function report(name, p, args; target = T.CUDATarget())
    ex = prepare(p, map(CuArray, args)...; target, measure = false)
    got = filter(!isnothing, T.leaves(execute!(ex)))
    ref = filter(!isnothing, T.leaves(interpret(p, args...)))
    err = maximum(map((a, b) -> maximum(abs.(Array(a) .- b)) / max(maximum(abs.(b)), eps(Float32)), got, ref))
    bytes, apart = T.footprint(ex)
    @printf("%-46s %9.2f MiB apart %9.2f MiB in slots  %5.1f%% less  err %.1e\n", name, apart / 2^20, bytes / 2^20,
        100 * (1 - bytes / apart), err)
end

let (D, H, S, B) = (64, 12, 256, 16), E = 64 * 12
    q, k, v, x = sample(D, H, S, B; scale = 0.001f0), sample(D, H, S, B; scale = 0.002f0), sample(D, H, S, B; scale = 0.003f0),
        sample(D, H, S, B; scale = 0.5f0)
    report("attention, a training step", training((q, k, v, x) -> sum(abs2, attend(q, k, v) .+ x), q, k, v, x), (q, k, v, x))
    X = sample(E, S * B; scale = 0.01f0)
    Wq, Wk, Wv, Wo = (sample(E, E; scale = 0.001f0 * i) ./ 32 for i in 1:4)
    function block(Wq, Wk, Wv, Wo, X)
        heads(y) = reshape(y, D, H, S, B)
        o = attend(heads(Wq * X), heads(Wk * X), heads(Wv * X))
        sum(abs2, Wo * reshape(o, E, S * B) .+ X) / (S * B)
    end
    report("attention block, a training step", training(block, Wq, Wk, Wv, Wo, X), (Wq, Wk, Wv, Wo, X))
end

let n = 1024
    x, y = sample(n, n; scale = 0.01f0), sample(n, n; scale = 0.02f0)
    W = Tuple(sample(n, n; scale = 0.001f0 * l) ./ 32 for l in 1:4)
    b = Tuple(sample(n; scale = 0.1f0 * l) for l in 1:4)
    function mlp(W, b, x, y)
        h = x
        for l in 1:4
            h = W[l] * h .+ b[l]
            l < 4 && (h = tanh.(h))
        end
        sum(abs2, h .- y) / size(y, 2)
    end
    report("MLP of 4 layers of 1024, a training step", training(mlp, W, b, x, y), (W..., b..., x, y))
end

foreach(c -> report(c.name, c.program, c.args), COMPOSITIONS)
