# Run: julia --project=examples examples/gated_mlp.jl
# A real composed forward/backward, not one specially named fused kernel.
using Tempest
using CUDACore
import cuBLAS
import cuTile

linear(x, w) = w' * x
silu(x) = x / (one(x) + exp(-x))

function dsilu(x)
    s = inv(one(x) + exp(-x))
    # Algebraically equivalent to s*(1+x*(1-s)); this spelling avoids a
    # mixed scalar/tile FMA verifier issue in the current cuTile toolchain.
    s + (x * s) * (one(x) - s)
end

Tempest.scalar_partials(::typeof(silu)) = (dsilu,)

function loss(x, gate, up, down, target)
    h = silu.(linear(x, gate)) .* linear(x, up)
    y = x + linear(h, down)
    sum(abs2, y .- target) / length(y)
end

# Deterministic inputs; no random-state or optimizer effects in this prototype.
array(m, n, scale) = reshape(scale .* sin.(Float32.(1:(m * n))), m, n)
D, I, N = 128, 256, 192
args = (
    array(D, N, 0.1f0),
    array(D, I, 0.03f0),
    array(D, I, 0.04f0),
    array(I, D, 0.03f0),
    array(D, N, 0.2f0),
)

program = differentiate(trace(loss, args...))
reference = value_and_gradient(program, args...)
ex = prepare(program, args...; target = CUDATarget())
execute!(ex)
@assert isapprox(Array(outputs(ex)[1])[], reference[1]; rtol = 2e-4, atol = 1e-6)
for (g, r) in zip(outputs(ex)[2:end], reference[2])
    @assert isapprox(Array(g), r; rtol = 2e-4, atol = 1e-6)
end

capture!(ex)
for _ = 1:5
    replay!(ex)
end

CUDACore.synchronize()
explain(ex)

# Measures this implementation against itself, not PyTorch or a SOTA claim.
# Event intervals include gaps caused by host submission. Preparation, binding,
# compilation, and transfers are excluded in both cases.
function interval(f; repetitions = 100)
    f()
    CUDACore.synchronize()
    best = Inf
    for _ = 1:5
        t = CUDACore.@elapsed begin
            for _ = 1:repetitions
                f()
            end
        end
        best = min(best, t / repetitions)
    end
    best
end

println(
    "Uncaptured prepared forward/backward: ",
    round(interval(() -> execute!(ex)) * 1e6; digits = 1),
    " μs",
)
println("Captured same schedule: ", round(interval(() -> replay!(ex)) * 1e6; digits = 1), " μs")
println("All input gradients are computed, including x and target; no optimizer update yet.")
