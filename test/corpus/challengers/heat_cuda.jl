# The corpus's heat equation (test/corpus/heat.jl) as one would write it in
# CUDA.jl by hand: a kernel a time step forward, then the adjoint a kernel a
# step back, the gradient of the loss with respect to the initial state. The adjoint of a step is its transpose: it needs no
# state of the forward but the last, the step being linear. The steps are
# captured as one CUDA graph and replayed; a step's time is the minimum over
# samples of a replay. The gradient is checked against the same derivation
# run on the CPU in Float64 from the same inputs.
#
#   julia --project=test test/corpus/challengers/heat_cuda.jl small|full
#
# small is 100 steps, full 1000, both on a 256×256 grid. TEMPEST_CORPUS_HEAT_STEPS
# sets the steps. Prints one JSON line.
using CUDACore, Random, Printf

const N = 256
const α = 0.2f0

steps_of(shape) = parse(Int, get(ENV, "TEMPEST_CORPUS_HEAT_STEPS", shape == "full" ? "1000" : "100"))

inside(i, j, n, m) = i > 1 && i < n && j > 1 && j < m

# One step forward: the interior diffuses, the border stays, and the step's
# source is added everywhere.
function forward!(v, u, q, t)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    n, m = size(u)
    if i <= n && j <= m
        @inbounds x = u[i, j]
        @inbounds if inside(i, j, n, m)
            x += α * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4.0f0 * x)
        end
        @inbounds v[i, j] = x + q[i, j, t]
    end
    nothing
end

# The step's transpose: what each position contributes to the next step's
# positions, weighted by the cotangent there.
function backward!(μ, λ)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    n, m = size(λ)
    if i <= n && j <= m
        @inbounds begin
            x = inside(i, j, n, m) ? (1.0f0 - 4.0f0 * α) * λ[i, j] : λ[i, j]
            i > 1 && inside(i - 1, j, n, m) && (x += α * λ[i - 1, j])
            i < n && inside(i + 1, j, n, m) && (x += α * λ[i + 1, j])
            j > 1 && inside(i, j - 1, n, m) && (x += α * λ[i, j - 1])
            j < m && inside(i, j + 1, n, m) && (x += α * λ[i, j + 1])
            μ[i, j] = x
        end
    end
    nothing
end

function square!(λ, u)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    i <= length(u) && @inbounds λ[i] = 2.0f0 * u[i]
    nothing
end

const THREADS = (32, 8)
blocks(n, m) = (cld(n, THREADS[1]), cld(m, THREADS[2]))

"The loss's gradient with respect to the initial state, into `du`."
function gradient!(du, u0, q, a, b)
    n, m, steps = size(q)
    copyto!(a, u0)
    for t in 1:steps
        @cuda threads = THREADS blocks = blocks(n, m) forward!(b, a, q, t)
        a, b = b, a
    end
    @cuda threads = 256 blocks = cld(length(a), 256) square!(b, a)
    a, b = b, a
    for _ in 1:steps
        @cuda threads = THREADS blocks = blocks(n, m) backward!(b, a)
        a, b = b, a
    end
    copyto!(du, a)
end

# The same derivation on the CPU, in Float64.
function reference(u0, q)
    n, m, steps = size(q)
    u = Float64.(u0)
    for t in 1:steps
        v = copy(u)
        for j in 2:(m - 1), i in 2:(n - 1)
            v[i, j] += α * (u[i - 1, j] + u[i + 1, j] + u[i, j - 1] + u[i, j + 1] - 4 * u[i, j])
        end
        u = v .+ q[:, :, t]
    end
    λ = 2 .* u
    for _ in 1:steps
        μ = copy(λ)
        for j in 2:(m - 1), i in 2:(n - 1)
            μ[i, j] -= 4α * λ[i, j]
            μ[i - 1, j] += α * λ[i, j]
            μ[i + 1, j] += α * λ[i, j]
            μ[i, j - 1] += α * λ[i, j]
            μ[i, j + 1] += α * λ[i, j]
        end
        λ = μ
    end
    λ
end

relative(a, b) = maximum(abs.(a .- b)) / maximum(abs.(b))

function main(shape)
    steps = steps_of(shape)
    rng = Xoshiro(0)
    u0 = randn(rng, Float32, N, N)
    q = 0.01f0 .* randn(rng, Float32, N, N, steps)
    du, a, b = CUDACore.zeros(Float32, N, N), CUDACore.zeros(Float32, N, N), CUDACore.zeros(Float32, N, N)
    device_u0, device_q = CuArray(u0), CuArray(q)
    started = time()
    gradient!(du, device_u0, device_q, a, b)
    CUDACore.synchronize()
    first_step_s = time() - started
    agreement = relative(Array(du), reference(u0, q))
    graph = CUDACore.capture() do
        gradient!(du, device_u0, device_q, a, b)
    end
    exec = CUDACore.instantiate(graph)
    CUDACore.launch(exec)
    CUDACore.synchronize()
    step_ms = minimum(_ -> 1.0e3 * CUDACore.@elapsed(CUDACore.launch(exec)), 1:10)
    @printf("{\"side\": \"cuda by hand\", \"precision\": \"fp32\", \"first_step_s\": %.2f, \"step_ms\": %.3f, \"kernels\": %d, \"agreement\": %.3e, \"note\": \"%d steps, a CUDA graph's replay, best of 10\"}\n",
        first_step_s, step_ms, 2 * steps + 1, agreement, steps)
end

main(get(ARGS, 1, "small"))
