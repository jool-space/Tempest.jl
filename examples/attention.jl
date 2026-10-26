# Attention's training step, compiled
#
# Causal attention written in plain Julia with NNlib, and the gradient of a loss
# through it. `compile` traces the step once, plans it into kernels and runs
# them on the GPU. The reference is the same step in Float64 on the CPU, which
# Tempest's interpreter computes; the loss alone is also computed eagerly.
#
#   julia --project=test examples/attention.jl

using Tempest, CUDACore, cuTile, cuBLAS, NNlib, Printf
import Tempest.Compiler as C

# q, k and v are (features, positions, heads × batch). Each query scores every
# key, the keys after it are masked out, and a softmax over the keys weights
# the values.
function attention(q, k, v)
    s = batched_mul(batched_transpose(k), q) .* inv(sqrt(eltype(q)(size(q, 1))))
    n = size(s, 1)
    s = ifelse.(reshape(1:n, :, 1) .<= reshape(1:n, 1, :), s, typemin(eltype(s)))
    batched_mul(v, softmax(s; dims = 1))
end

loss(q, k, v, x) = sum(abs2, attention(q, k, v) .+ x)

# The loss and its gradient with respect to each argument.
step(q, k, v, x) = Tempest.gradient(loss, q, k, v, x)

# 64 features, 256 positions, 12 heads × a batch of 16.
d, n, hb = 64, 256, 12 * 16
sample(scale) = 2 .* sin.(scale .* reshape(1:d * n * hb, d, n, hb))
q, k, v, x = sample(0.001), sample(0.002), sample(0.003), sample(0.5)
args = map(a -> CuArray(Float32.(a)), (q, k, v, x))

c = compile(step, args...)
got = c(args...)
reference = step(q, k, v, x)

relative(a::Number, b) = abs(a - b) / abs(b)
relative(a, b) = maximum(abs, Array(a) .- b) / maximum(abs, b)
for (name, g, r) in zip(("loss", "dq", "dk", "dv", "dx"), got, reference)
    @printf("%-4s relative difference from Float64 %.1e\n", name, relative(g, r))
end
@printf("the interpreter's loss against eager Julia's: %.1e\n", relative(reference[1], loss(q, k, v, x)))

# What runs: each kernel, what it computes, stores and recomputes.
println()
explain(c)

# What the step keeps on the device: the intermediates share slots, and for the
# backward pass FlashAttention-2 would keep only the output and each query's
# log-sum-exp.
ex = c.executable
intermediates = sum(ex.storage.sizes; init = 0)
flash = sizeof(Float32) * (d * n * hb + n * hb)
c(args...)
ms = minimum(1:20) do _
    t = time_ns()
    c(args...)
    CUDACore.synchronize()
    (time_ns() - t) / 1.0e6
end
println()
@printf("attention step: %d kernels, intermediates %.1f MiB (FlashAttention-2 keeps %.1f MiB), %.2f ms\n",
    length(ex.plan.groups), intermediates / 2^20, flash / 2^20, ms)
