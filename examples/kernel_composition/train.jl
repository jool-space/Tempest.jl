# Run: julia --project=examples examples/kernel_composition/train.jl
using Tempest
using CUDACore
import cuBLAS

include("rmsnorm.jl")
using .RMSNormExample
include("rmsnorm_cuda_reference.jl")

function loss(params, unused, batch)
    h = rmsnorm(params.w1 * batch.x .+ batch.x, params.gain1; eps = 0.03f0)
    y = rmsnorm(params.w2 * h .+ h, params.gain2; eps = 0.03f0)
    sum(abs2, y .- batch.y) / length(batch.y)
end

function adam(params, opt, gradients)
    step = opt.step + 1.0f0
    first = map((m, g) -> 0.9f0 .* m .+ 0.1f0 .* g, opt.first, gradients)
    second = map((v, g) -> 0.999f0 .* v .+ 0.001f0 .* abs2.(g), opt.second, gradients)
    updated = map(params, first, second) do p, m, v
        p .-
        0.001f0 .* (m ./ (1.0f0 - 0.9f0 ^ step)) ./
        (sqrt.(v ./ (1.0f0 - 0.999f0 ^ step)) .+ 1.0f-8)
    end
    updated, (; first, second, step)
end

function main()
    CUDACore.functional() || error("this example requires CUDA")
    params = (
        w1 = reshape(0.08f0 .* sin.(Float32.(1:49)), 7, 7),
        w2 = reshape(0.05f0 .* cos.(Float32.(1:49)), 7, 7),
        gain1 = fill(0.7f0, 7),
        gain2 = fill(0.8f0, 7),
    )
    opt = (first = map(zero, params), second = map(zero, params), step = 0.0f0)
    batch = (x = reshape(sin.(Float32.(1:35)), 7, 5), y = fill(0.1f0, 7, 5))
    derivative = differentiate(trace(loss, params, opt, batch); wrt = (1,))
    step = with_update(derivative) do value, gradients, p, o, b
        next_params, next_opt = adam(p, o, only(gradients))
        (next_params, next_opt, value)
    end
    ex = prepare(
        step,
        params,
        opt,
        batch;
        target = CUDATarget(),
        donate = (1, 2),
        implementations = (ReferenceRMSNorm(),),
    )
    @assert count(==(:reference_rmsnorm), ex.providers) == 4
    capture!(ex)
    expected_params, expected_opt = params, opt
    for iteration = 1:5
        next_batch = (x = batch.x .* (0.8f0 + 0.1f0 * iteration), y = batch.y)
        value, (gradients,) =
            value_and_gradient(derivative, expected_params, expected_opt, next_batch)
        expected_params, expected_opt = adam(expected_params, expected_opt, gradients)
        bind_argument!(ex, 3, next_batch)
        actual = replay!(ex)
        expected = (expected_params, expected_opt, value)
        for (a, b) in zip(Tempest.tree_leaves(actual), Tempest.tree_leaves(expected))
            observed = b isa Number ? only(Array(a)) : Array(a)
            @assert isapprox(observed, b; rtol = 1e-3, atol = 3e-5)
        end
    end
    explain(ex)
    println("Five captured Adam updates agree with the reference, including optimizer state.")
end

main()
