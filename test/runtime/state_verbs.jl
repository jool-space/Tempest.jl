include("../fixtures/verbs.jl")

@testset "State updates through opaque and expanded verbs" begin
    params = (x = [0.2, -0.3, 0.4],)
    batch = [0.7, -0.5, 0.1]
    loss(p, b) = sum(pair_wave(p.x).s .* b)
    differentiated = differentiate(trace(loss, params, batch); wrt = (1,))
    step = with_update(differentiated) do loss, gradients, p, b
        ((x = p.x .- 0.1 .* only(gradients).x,), loss)
    end
    original_count = length(differentiated.nodes)
    for mode in (:accept, :reject, :unavailable)
        ex = prepare(
            step,
            params,
            batch;
            donate = (1,),
            implementations = (WaveImplementation(mode),),
        )
        @test (:wave_kernel in ex.providers) == (mode === :accept)
        expected = copy(params.x)
        for _ = 1:3
            expected_loss = sum(sin.(expected) .* batch)
            expected = expected .- 0.1 .* cos.(expected) .* batch
            actual, actual_loss = execute!(ex)
            @test actual.x ≈ expected
            @test actual_loss[] ≈ expected_loss
        end
        @test length(differentiated.nodes) == original_count
    end
end
