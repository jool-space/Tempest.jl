state_unruled(x) = x * x + x

@testset "Structured arguments and selected gradients" begin
    params = (weight = [0.2 -0.3; 0.4 0.1], bias = reshape([0.1, -0.2], 2, 1))
    batch = (x = reshape([0.3, -0.5], 2, 1), scale = 0.7)
    loss(p, b, unused) = sum(abs2, p.weight * b.x .+ p.bias) * b.scale
    unused = (nested = (ones(2),),)
    primal = trace(loss, params, batch, unused)
    differentiated = differentiate(primal; wrt = (2, 1, 3))
    y, (db, dp, du) = value_and_gradient(differentiated, params, batch, unused)
    residual = params.weight * batch.x + params.bias
    @test y ≈ loss(params, batch, unused)
    @test dp.weight ≈ 2batch.scale .* residual * batch.x'
    @test dp.bias ≈ 2batch.scale .* residual
    @test db.x ≈ 2batch.scale .* params.weight' * residual
    @test db.scale ≈ sum(abs2, residual)
    @test du == (nested = (zeros(2),),)
    ex = prepare(differentiated, params, batch, unused)
    actual = execute!(ex)
    @test actual[2].x ≈ db.x
    @test actual[3].weight ≈ dp.weight
    @test keys(actual[4]) == keys(unused)
    @test length(execute!(prepare(primal, params, batch, unused; wrt = (1,)))) == 2
    @test value_and_gradient(differentiate(primal; wrt = ()), params, batch, unused) == (y, ())
    inactive = trace((x, state) -> sum(abs2, x) + sum(state_unruled.(state)), ones(2), ones(2))
    selected = differentiate(inactive; wrt = (1,))
    @test value_and_gradient(selected, ones(2), ones(2)) == (6.0, ([2.0, 2.0],))
    @test only(execute!(prepare(differentiate(inactive; wrt = ()), ones(2), ones(2)))[1]) == 6.0
    for wrt in ((0,), (4,), (true,), (1, 1), [1])
        @test_throws ArgumentError differentiate(primal; wrt)
    end
    @test_throws ArgumentError bind!(
        ex,
        (bias = params.bias, weight = params.weight),
        batch,
        unused,
    )
    @test_throws ArgumentError bind_argument!(ex, 2, (x = ones(3), scale = 1.0))
    @test_throws ArgumentError bind_argument!(ex, 0, batch)
    before = copy(ex.inputs[3])
    @test_throws ArgumentError bind_argument!(ex, 2, (x = ones(2, 1), scale = 1.0f0))
    @test ex.inputs[3] == before # validation precedes all writes
    bind_argument!(ex, 2, (x = 2batch.x, scale = batch.scale))
    @test only(first(execute!(ex))) ≈ loss(params, (x = 2batch.x, scale = batch.scale), unused)
end

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

@testset "Functional Adam with owned state" begin
    for reuse in (false, true)
        problem = adam_problem()
        (; params, opt, batch, differentiated, step) = problem
        ex = prepare(step, params, opt, batch; donate = (1, 2), reuse)
        pointers = map(pointer, Tempest.tree_leaves(outputs(ex)))
        for iteration = 1:5
            next_batch = (x = batch.x .* Float32(iteration), y = batch.y)
            loss, gradients = adam_reference_gradient(params, next_batch)
            params, opt = adam_update(params, opt, gradients)
            bind_argument!(ex, 3, next_batch)
            actual = execute!(ex)
            test_state_close(actual, (params, opt, loss))
            @test map(pointer, Tempest.tree_leaves(actual)) == pointers
        end
        @test problem.opt.step == 0.0f0
        @test all(iszero, problem.opt.first.w1)
        @test occursin("staged commits", sprint(explain, ex))
    end
end

@testset "Simultaneous donation and alias observations" begin
    a, b = [1.0, 2.0], [3.0, 4.0]
    for reuse in (false, true)
        ex = prepare_forward(
            (a, b) -> (b, a, (old = a, reversed = a[[2, 1]])),
            a,
            b;
            donate = (1, 2),
            reuse,
        )
        first_result = execute!(ex)
        @test first_result == (b, a, (old = a, reversed = reverse(a)))
        @test execute!(ex) == (a, b, (old = b, reversed = reverse(b)))
        @test a == [1, 2] && b == [3, 4]
        mapped = prepare_forward(a -> (a[[2, 1]] .+ 1,), a; donate = (1,), reuse)
        @test only(execute!(mapped)) == [3, 2]
        @test only(execute!(mapped)) == [3, 4]
    end
    repeated = prepare_forward(a -> (a .+ 1, a .+ 2), a; donate = (1,))
    @test execute!(repeated) == (a .+ 1, a .+ 2)
    named = prepare_forward(a -> (state = a .+ 1, old = a), a; donate = (1,))
    @test execute!(named) == (state = a .+ 1, old = a)
    for donate in ((0,), (true,), (1, 1), (3,), [1])
        @test_throws ArgumentError prepare_forward((a, b) -> (b, a), a, b; donate)
    end
    @test_throws ArgumentError prepare_forward((a, b) -> (b, a), a, a; donate = (1,))
    @test_throws ArgumentError prepare_forward((a, b) -> (b, a), a, view(a, :); donate = (1,))
    @test_throws ArgumentError prepare_forward(a -> (sum(a),), a; donate = (1,))
    @test_throws ArgumentError prepare_forward(a -> ((a,),), a; donate = (1,))
    @test_throws ArgumentError prepare_forward(identity, a; donate = (1,))
    @test_throws ArgumentError prepare(sum, a; donate = (1,))
    @test_throws ArgumentError with_update((args...) -> 1, differentiate(trace(sum, a)))
end
