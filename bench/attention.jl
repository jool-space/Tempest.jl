# Attention on GPT-like shapes: the kernel generated from the reference (one
# fold over keys, online softmax or two passes), the hand-written kernel, and
# the kernel with a residual epilogue against the kernel followed by a
# separate add. Device time is the minimum over samples of a captured graph of
# a few milliseconds of executions.
#   julia --project=dev bench/attention.jl
using Tempest, CUDACore, cuBLAS, cuTile, Printf
import Tempest as T

function replay_graph(ex; repetitions)
    execute!(ex)
    CUDACore.synchronize()
    graph = CUDACore.capture() do
        for _ = 1:repetitions
            foreach(launch -> launch(), ex.launches)
        end
    end
    CUDACore.instantiate(graph)
end

function device_microseconds(ex; samples = 15, milliseconds = 4.0)
    once = let g = replay_graph(ex; repetitions = 1)
        CUDACore.launch(g)
        minimum(_ -> CUDACore.@elapsed(CUDACore.launch(g)), 1:3)
    end
    repetitions = clamp(round(Int, 1.0e-3 * milliseconds / once), 1, 200)
    g = replay_graph(ex; repetitions)
    CUDACore.launch(g)
    minimum(_ -> 1.0e6 * CUDACore.@elapsed(CUDACore.launch(g)) / repetitions, 1:samples)
end

data(dims...; scale = 1.0f0) = 2.0f0 .* sin.(scale .* reshape(Float32.(1:prod(dims)), dims...))
D, H, KV, S, B = 64, 12, 12, 256, 16
q, k, v = data(D, H, S, B; scale = 0.001f0), data(D, KV, S, B; scale = 0.002f0), data(D, KV, S, B; scale = 0.003f0)
x = data(D, H, S, B; scale = 0.5f0)
attend(q, k, v) = T.attention(q, k, v; causal = true)
residual(q, k, v, x) = T.attention(q, k, v; causal = true) .+ x
loss(q, k, v, x) = sum(abs2, T.attention(q, k, v; causal = true) .+ x)

function measure(name, p, args; target, check = true)
    ex = prepare(p, map(CuArray, args)...; target)
    if check
        got = map(r -> Array(r), filter(!isnothing, T.leaves(execute!(ex))))
        ref = filter(!isnothing, T.leaves(interpret(p, args...)))
        err = maximum(map((a, b) -> maximum(abs.(a .- b)) / max(maximum(abs.(b)), eps(Float32)), got, ref))
    else
        err = NaN
    end
    t = device_microseconds(ex)
    kinds = join(unique(d isa T.FloorKernels ? "floor" : first(split(string(d), ":")) for d in ex.kernels), " + ")
    @printf("%-58s %10.1f us  err %.1e  %s\n", name, t, err, kinds)
    t
end

ext = Base.get_extension(Tempest, :cuTileExt)
# Full precision: the hand-written kernels do not claim; the reference is
# planned, and generated as one kernel.
println("== full")
measure("generated, forward", trace(attend, q, k, v), (q, k, v); target = T.CUDATarget())
measure("generated, forward, two passes", trace(attend, q, k, v), (q, k, v); target = T.CUDATarget(online = false))
measure("training step (loss of attention + residual)", differentiate(trace(loss, q, k, v, x)), (q, k, v, x); target = T.CUDATarget())
for precision in (:tf32,)
    target = T.CUDATarget(precision = precision)
    println("== ", precision)
    measure("hand-written kernel, forward", trace(attend, q, k, v), (q, k, v); target)
    with_epilogue = measure("kernel + residual as an epilogue", trace(residual, q, k, v, x), (q, k, v, x); target)
    @eval Tempest.epilogue_results(::$(ext.cuTileAttention), ::typeof(Tempest.attention_forward), kwargs) = ()
    separate = measure("kernel, then a separate residual add", trace(residual, q, k, v, x), (q, k, v, x); target)
    @eval Tempest.epilogue_results(::$(ext.cuTileAttention), ::typeof(Tempest.attention_forward), kwargs) = (1,)
    @printf("  epilogue saves %.1f us (%.0f%%)\n", separate - with_epilogue, 100 * (separate - with_epilogue) / separate)
    measure("training step (loss of attention + residual)", differentiate(trace(loss, q, k, v, x)), (q, k, v, x); target)
end

# The generated kernel in TF32, the hand-written forward kernel stepping aside.
@eval Tempest.claims_call(::$(ext.cuTileAttention), ::T.CUDATarget, ::typeof(T.attention_forward), kwargs, inputs) = false
println("== tf32, generated")
measure("generated, forward", trace(attend, q, k, v), (q, k, v); target = T.CUDATarget(precision = :tf32))
measure("generated, forward, two passes", trace(attend, q, k, v), (q, k, v); target = T.CUDATarget(precision = :tf32, online = false))
