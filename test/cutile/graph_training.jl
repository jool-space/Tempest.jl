include("../fixtures/state.jl")

# Residual normalization fixture for the training test.
fusion_activation(x) = x * x * x
Tempest.scalar_partials(::typeof(fusion_activation)) = (x -> 3 * x * x,)

function residual_normalization(x, residual, weight, bias, epsilon)
    h = x .+ residual
    rstd = inv.(sqrt.(sum(abs2, h; dims = 1) ./ eltype(x)(size(x, 1)) .+ epsilon))
    y = fusion_activation.(h .* rstd .* weight .+ bias)
    (; y, h, rstd)
end

function fusion_inputs(T, rows = 5, columns = 4)
    x = reshape(T.(sin.(1:(rows * columns))), rows, columns)
    residual = T(0.2) .* reverse(x; dims = 1)
    weight = T.(range(0.3, 0.8; length = rows))
    bias = reshape(T.(range(-0.2, 0.1; length = columns)), 1, columns)
    (x, residual, weight, bias, T(0.03))
end

# Two normalization layers, Adam, and donation through the graph-region provider.
@testset "Captured Adam with two fused normalization layers" begin
    rows, columns = 7, 5
    matrix(scale) = reshape(scale .* sin.(Float32.(1:(rows ^ 2))), rows, rows)
    params = (
        w1 = matrix(0.07f0),
        w2 = matrix(0.05f0),
        gain1 = fill(0.7f0, rows),
        gain2 = fill(0.8f0, rows),
        bias1 = zeros(Float32, rows, 1),
        bias2 = zeros(Float32, rows, 1),
    )
    opt = (first = map(zero, params), second = map(zero, params), step = 0.0f0)
    batch = (x = fusion_inputs(Float32, rows, columns)[1], y = fill(0.1f0, rows, columns))
    loss(p, unused, b) = begin
        first = residual_normalization(p.w1 * b.x, b.x, p.gain1, p.bias1, 0.1f0).y
        second = residual_normalization(p.w2 * first, first, p.gain2, p.bias2, 0.1f0).y
        sum(abs2, second .- b.y) / length(b.y)
    end
    differentiated = differentiate(trace(loss, params, opt, batch); wrt = (1,))
    step = with_update(differentiated) do value, gradients, p, o, b
        next_params, next_opt = adam_update(p, o, only(gradients); rate = 0.001f0)
        (next_params, next_opt, value)
    end
    ex = prepare(step, params, opt, batch; target = CUDATarget(), donate = (1, 2))
    @test count(s -> s.region isa Tempest.GraphRegion, ex.storage.steps) >= 2
    @test :cutile in ex.providers
    original = map(a -> copy(Array(a)), ex.inputs)
    capture!(ex)
    @test all(Array(a) == b for (a, b) in zip(ex.inputs, original))
    expected_params, expected_opt = params, opt
    for iteration = 1:3
        next_batch = (x = batch.x .* (0.8f0 + 0.1f0 * iteration), y = batch.y)
        value, (gradients,) =
            value_and_gradient(differentiated, expected_params, expected_opt, next_batch)
        expected_params, expected_opt =
            adam_update(expected_params, expected_opt, gradients; rate = 0.001f0)
        bind_argument!(ex, 3, next_batch)
        actual_params, actual_opt, actual_loss = replay!(ex)
        @test Array(actual_loss)[] ≈ value rtol=5e-4 atol=2e-5
        test_state_close(actual_params, expected_params; rtol = 5e-4, atol = 3e-5)
        test_state_close(actual_opt, expected_opt; rtol = 1e-3, atol = 3e-5)
    end

    # Padded lanes must be masked after the map: exp(0) is not zero.
    x = fusion_inputs(Float32, 259, 7)[1]
    soft_denominator(x) = x ./ sum(exp.(x); dims = 1)
    ex = prepare_forward(soft_denominator, CuArray(x))
    @test any(s -> s.region isa Tempest.GraphRegion, ex.storage.steps)
    @test Array(execute!(ex)) ≈ soft_denominator(x) rtol=2e-5

    # Disabling the cuTile provider retains the decomposed CUDA schedule.
    args = fusion_inputs(Float32, 11, 7)
    device_args = map(a -> a isa AbstractArray ? CuArray(a) : a, args)
    ex = Tempest.deactivate(cuTile) do
        prepare_forward(residual_normalization, device_args...)
    end
    @test all(s -> !(s.region isa Tempest.GraphRegion), ex.storage.steps)
    @test Array(execute!(ex).y) ≈ residual_normalization(args...).y rtol=3e-5
end
