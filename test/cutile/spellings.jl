# Plan shapes of documented recipes and of equivalent spellings (see
# docs/src/spelling-invariance.md). Equivalent spellings must reach the same
# plan; a recipe falling off its fused path fails here rather than only in a
# benchmark.

# One label per scheduled step: the region kind and the implementation.
function plan_shape(ex)
    map(eachindex(ex.storage.steps)) do i
        region = ex.storage.steps[i].region
        kind = if region isa Tempest.ProposedRegion
            :proposed
        elseif region isa Tempest.GraphRegion
            :graph
        elseif region isa Tempest.OpaqueCall
            Symbol(:call_, nameof(region.call.f))
        else
            Tempest.region_name(region)
        end
        (kind, ex.providers[i])
    end
end

softmax_naive(x) = (e = exp.(x .- maximum(x; dims = 1)); e ./ sum(e; dims = 1))
softmax_fold(x) = exp.(x .- Tempest.fold(Tempest.LogSumExpFold(), x; dims = 1))
softmax_verb(x) = exp.(x .- Tempest.logsumexp(x; dims = 1))
log_softmax_naive(x) = (m = maximum(x; dims = 1); x .- m .- log.(sum(exp.(x .- m); dims = 1)))
log_softmax_verb(x) = x .- Tempest.logsumexp(x; dims = 1)

@testset "Softmax spellings reach one graph region" begin
    # Resident (short columns) and streamed (long columns) schedules.
    for (rows, columns) in ((96, 40), (5000, 3))
        x = CuArray(4.0f0 .* sin.(reshape(Float32.(1:(rows * columns)), rows, columns)))
        expected = softmax_naive(Array(x))
        for f in (softmax_naive, softmax_fold, softmax_verb)
            ex = prepare_forward(f, x)
            @test Array(execute!(ex)) ≈ expected rtol = 1e-5
            @test plan_shape(ex) == [(:graph, :cutile)]
        end
        for f in (log_softmax_naive, log_softmax_verb)
            ex = prepare_forward(f, x)
            @test Array(execute!(ex)) ≈ log.(expected) rtol = 1e-4 atol = 1e-5
            @test plan_shape(ex) == [(:graph, :cutile)]
        end
        # Rows: the same verb along the second axis.
        ex = prepare_forward(x -> exp.(x .- Tempest.logsumexp(x; dims = 2)), x)
        @test plan_shape(ex) == [(:graph, :cutile)]
    end
end

@testset "logsumexp is its fold" begin
    x = CuArray(30.0f0 .* sin.(reshape(Float32.(1:(512 * 24)), 512, 24)))
    verb = prepare_forward(x -> Tempest.logsumexp(x; dims = 1), x)
    folded = prepare_forward(x -> Tempest.fold(Tempest.LogSumExpFold(), x; dims = 1), x)
    @test plan_shape(verb) == plan_shape(folded) == [(:graph, :cutile)]
    @test Array(execute!(verb)) ≈ Array(execute!(folded))
    # The softmax gradient: the verb and the fold give the same training plan.
    t = CuArray(cos.(reshape(Float32.(1:(512 * 24)), 512, 24)))
    loss(f) = (x, t) -> sum(f(x) .* t)
    a = prepare(loss(softmax_verb), x, t)
    b = prepare(loss(softmax_fold), x, t)
    @test plan_shape(a) == plan_shape(b)
    @test all(step -> !(step.region isa Tempest.OpaqueCall), a.storage.steps)
    @test Array(execute!(a)[2]) ≈ Array(execute!(b)[2]) rtol = 1e-4 atol = 1e-6
end
