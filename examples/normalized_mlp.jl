# Run: julia --project=examples examples/normalized_mlp.jl
# These functions use the ordinary array interface. RMSNorm is a composition,
# not an opaque primitive or a dedicated normalization engine in this experiment.
using Tempest
using CUDACore
import cuBLAS
import cuTile

function rmsnorm(x::AbstractMatrix{T}, gain::AbstractVector{T}; eps = T(1e-4)) where {T}
    invrms = inv.(sqrt.(sum(abs2, x; dims = 1) / size(x, 1) .+ eps))
    x .* invrms .* gain
end

linear(x::AbstractMatrix, w::AbstractMatrix) = w' * x

function loss(
    x::AbstractMatrix,
    gain::AbstractVector,
    gate::AbstractMatrix,
    up::AbstractMatrix,
    down::AbstractMatrix,
    bias::AbstractVector,
    target::AbstractMatrix,
    eps::Real,
)
    h = rmsnorm(x, gain; eps)
    h = tanh.(linear(h, gate)) .* linear(h, up)
    y = x + linear(h, down)
    sum(abs2, y .+ bias .- target) / length(y)
end

array(dims, s) = reshape(s .* sin.(Float32.(1:prod(dims))), dims)
D, I, N = 64, 96, 97
args = (
    array((D, N), 0.2f0),
    fill(1.1f0, D),
    array((D, I), 0.01f0),
    array((D, I), 0.015f0),
    array((I, D), 0.01f0),
    array((D,), 0.05f0),
    array((D, N), 0.3f0),
    1.0f-3,
)
p = trace(loss, args...)
ex = prepare(p, args...; target = CUDATarget())
execute!(ex)
capture!(ex)
pointers = map(pointer, ex.buffers)

# eps is a graph input, not a baked host scalar. Replay sees its new value via
# stable device storage; no graph update or recapture is necessary.
for eps in (1.0f-3, 2.0f-3, 5.0f-4)
    updated = Base.setindex(args, eps, length(args))
    bind!(ex, updated...)
    got = replay!(ex)
    ref = value_and_gradient(p, updated...)
    @assert isapprox(Array(got[1])[], ref[1]; atol = 2e-5, rtol = 2e-4)
    for (g, r) in zip(got[2:end], ref[2])
        a = r isa Number ? only(Array(g)) : Array(g)
        @assert isapprox(a, r; atol = 2e-4, rtol = 2e-4)
    end
    @assert map(pointer, ex.buffers) == pointers
    println("eps=", eps, " loss=", Array(got[1])[], " ∂loss/∂eps=", Array(got[end])[])
end

CUDACore.synchronize()
explain(ex)
println(
    "CPU and GPU gradients agree. No optimizer, mixed precision, or tuned RMSNorm engine is claimed.",
)
