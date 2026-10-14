# Ordinary Julia update code serves both the allocating baseline and the trace.
function adam_update(
    params,
    opt,
    gradients;
    rate = 0.01f0,
    beta1 = 0.9f0,
    beta2 = 0.999f0,
    epsilon = 1.0f-8,
)
    step = opt.step + 1.0f0
    first_moment = map((m, g) -> beta1 .* m .+ (1.0f0 - beta1) .* g, opt.first, gradients)
    second_moment =
        map((v, g) -> beta2 .* v .+ (1.0f0 - beta2) .* abs2.(g), opt.second, gradients)
    updated = map(params, first_moment, second_moment) do p, m, v
        p .-
        rate .* (m ./ (1.0f0 - beta1 ^ step)) ./ (sqrt.(v ./ (1.0f0 - beta2 ^ step)) .+ epsilon)
    end
    updated, (first = first_moment, second = second_moment, step = step)
end

function adam_problem()
    params = (
        w1 = reshape(Float32.(range(-0.3, 0.4; length = 12)), 4, 3),
        b1 = zeros(Float32, 4, 1),
        w2 = reshape(Float32.(range(-0.2, 0.3; length = 8)), 2, 4),
        b2 = zeros(Float32, 2, 1),
    )
    opt = (first = map(zero, params), second = map(zero, params), step = 0.0f0)
    batch = (x = reshape(Float32.(range(-0.6, 0.7; length = 15)), 3, 5), y = fill(0.2f0, 2, 5))
    loss(p, unused, b) =
        sum(abs2, p.w2 * tanh.(p.w1 * b.x .+ p.b1) .+ p.b2 .- b.y) / length(b.y)
    differentiated = differentiate(trace(loss, params, opt, batch); wrt = (1,))
    step = with_update(differentiated) do loss, gradients, p, o, b
        next_params, next_opt = adam_update(p, o, only(gradients))
        (next_params, next_opt, loss)
    end
    (; params, opt, batch, differentiated, step)
end

# Independent eager derivatives for the two-layer model, outside Tempest's AD.
function adam_reference_gradient(params, batch)
    hidden = tanh.(params.w1 * batch.x .+ params.b1)
    residual = params.w2 * hidden .+ params.b2 .- batch.y
    output_gradient = (2.0f0 / length(batch.y)) .* residual
    hidden_gradient = (params.w2' * output_gradient) .* (1.0f0 .- hidden .^ 2)
    gradients = (
        w1 = hidden_gradient * batch.x',
        b1 = sum(hidden_gradient; dims = 2),
        w2 = output_gradient * hidden',
        b2 = sum(output_gradient; dims = 2),
    )
    sum(abs2, residual) / length(batch.y), gradients
end

function test_state_close(actual, expected; rtol = 2.0f-5, atol = 2.0f-6)
    @test Tempest.same_structure(actual, expected)
    for (a, b) in zip(Tempest.tree_leaves(actual), Tempest.tree_leaves(expected))
        observed = a isa AbstractArray ? Array(a) : a
        @test (b isa Number ? only(observed) : observed) ≈ b rtol=rtol atol=atol
    end
end
