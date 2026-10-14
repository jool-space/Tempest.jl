# Run: julia --project=examples examples/adam.jl
using Tempest
using CUDACore
import cuBLAS
import cuTile

function model_loss(params, opt, batch)
    hidden = tanh.(params.w1 * batch.x .+ params.b1)
    prediction = params.w2 * hidden .+ params.b2
    sum(abs2, prediction .- batch.y) / length(batch.y)
end

function adam(params, opt, gradients; rate = 0.001f0)
    beta1, beta2, epsilon = 0.9f0, 0.999f0, 1.0f-8
    step = opt.step + 1.0f0
    first = map((m, g) -> beta1 .* m .+ (1.0f0 - beta1) .* g, opt.first, gradients)
    second = map((v, g) -> beta2 .* v .+ (1.0f0 - beta2) .* abs2.(g), opt.second, gradients)
    updated = map(params, first, second) do p, m, v
        p .-
        rate .* (m ./ (1.0f0 - beta1 ^ step)) ./ (sqrt.(v ./ (1.0f0 - beta2 ^ step)) .+ epsilon)
    end
    updated, (first = first, second = second, step = step)
end

function main()
    CUDACore.functional() || error("this example requires a functional CUDA device")
    array(dims, scale) = reshape(scale .* sin.(Float32.(1:prod(dims))), dims)
    params = (
        w1 = array((64, 32), 0.03f0),
        b1 = zeros(Float32, 64, 1),
        w2 = array((16, 64), 0.02f0),
        b2 = zeros(Float32, 16, 1),
    )
    opt = (first = map(zero, params), second = map(zero, params), step = 0.0f0)
    batch = (x = array((32, 48), 0.2f0), y = array((16, 48), 0.1f0))
    differentiated = differentiate(trace(model_loss, params, opt, batch); wrt = (1,))
    step = with_update(differentiated) do loss, gradients, p, o, b
        next_params, next_opt = adam(p, o, only(gradients))
        (next_params, next_opt, loss)
    end

    # Explicit placement copies all nested leaves to owned device storage.
    preparation_seconds =
        @elapsed ex = prepare(step, params, opt, batch; target = CUDATarget(), donate = (1, 2))
    capture!(ex)
    @assert Array(outputs(ex)[2].step)[] == 0.0f0
    reference_params, reference_opt = params, opt
    for iteration = 1:5
        next_batch = (x = batch.x .* Float32(iteration), y = batch.y)
        loss, (gradients,) =
            value_and_gradient(differentiated, reference_params, reference_opt, next_batch)
        reference_params, reference_opt = adam(reference_params, reference_opt, gradients)
        bind_argument!(ex, 3, next_batch)
        next_params, next_opt, actual_loss = replay!(ex)
        @assert isapprox(Array(actual_loss)[], loss; rtol = 2.0f-4, atol = 1.0f-6)
        for field in keys(params)
            @assert isapprox(
                Array(next_params[field]),
                reference_params[field];
                rtol = 2.0f-4,
                atol = 2.0f-6,
            )
            @assert isapprox(
                Array(next_opt.first[field]),
                reference_opt.first[field];
                rtol = 2.0f-4,
                atol = 2.0f-6,
            )
            @assert isapprox(
                Array(next_opt.second[field]),
                reference_opt.second[field];
                rtol = 2.0f-4,
                atol = 2.0f-6,
            )
        end
        @assert Array(next_opt.step)[] == Float32(iteration)
    end

    # Each measurement starts with the same state. Timing includes the staged
    # commits and synchronization, but excludes binding and capture.
    function seconds_per_step(run; iterations = 100)
        bind!(ex, params, opt, batch)
        CUDACore.synchronize()
        elapsed = @elapsed begin
            for _ = 1:iterations
                run(ex)
            end
            CUDACore.synchronize()
        end
        elapsed / iterations
    end
    seconds_per_step(execute!)
    seconds_per_step(replay!)
    timings = [(seconds_per_step(execute!), seconds_per_step(replay!)) for _ = 1:5]
    uncaptured = minimum(first, timings)
    captured = minimum(last, timings)
    explain(ex)
    println("Preparation (including first compilation): ", preparation_seconds, " s")
    println(
        "Best of five warmed batches, uncaptured step: ",
        1e6 * uncaptured,
        " μs; captured step: ",
        1e6 * captured,
        " μs",
    )
    println(
        "Five changing-batch steps agree with the allocating reference. State donation and remaining commits are reported above.",
    )
end

main()
