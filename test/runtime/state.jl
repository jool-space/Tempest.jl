include("../fixtures/state.jl")

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
