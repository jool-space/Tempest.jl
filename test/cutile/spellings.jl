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

# Base's functions are pure, so a graph region may recompute any of them.
@testset "Spellings with other Base functions plan alike" begin
    x = CuArray(sin.(reshape(Float32.(1:(256 * 40)), 256, 40)))
    w = CuArray(Float32.(range(0.5, 1.5; length = 256)))
    rms_sqrt(x, w) = x .* w ./ sqrt.(sum(abs2, x; dims = 1) ./ size(x, 1) .+ 1.0f-5)
    rms_power(x, w) = w .* x .* (sum(abs2, x; dims = 1) ./ size(x, 1) .+ 1.0f-5) .^ -0.5f0
    a, b = prepare_forward(rms_sqrt, x, w), prepare_forward(rms_power, x, w)
    @test plan_shape(a) == plan_shape(b) == [(:graph, :cutile)]
    @test Array(execute!(a)) ≈ Array(execute!(b)) rtol = 1e-5
    a = prepare_forward(x -> maximum(x; dims = 1), x)
    b = prepare_forward(x -> reduce(max, x; dims = 1, init = -Inf32), x)
    @test plan_shape(a) == plan_shape(b) == [(:graph, :cutile)]
    # A sum's pullback broadcasts its cotangent with `identity`: the softmax
    # spelled with maximum and sum trains in as many regions as the fold.
    t = CuArray(cos.(reshape(Float32.(1:(256 * 40)), 256, 40)))
    loss(f) = (x, t) -> sum(f(x) .* t)
    a, b = prepare(loss(softmax_naive), x, t), prepare(loss(softmax_fold), x, t)
    @test plan_shape(a) == plan_shape(b)
    @test Array(execute!(a)[2]) ≈ Array(execute!(b)[2]) rtol = 1e-4 atol = 1e-6
end

# batched_mul is a primitive product like `*`, for every GEMM element type.
@testset "Batched and plain products plan alike" begin
    a = reshape(Float32.(sin.(1:(64 * 32))), 64, 32, 1)
    b = reshape(Float32.(cos.(1:(32 * 48))), 32, 48, 1)
    for T in (Float32, Float16)
        A, B = CuArray(T.(a)), CuArray(T.(b))
        batched(a, b) = tanh.(Tempest.batched_mul(a, b) .+ one(T))
        plain(a, b) = tanh.(reshape(reshape(a, 64, 32) * reshape(b, 32, 48), 64, 48, 1) .+ one(T))
        x, y = prepare_forward(batched, A, B), prepare_forward(plain, A, B)
        @test first.(plan_shape(x)) == [:batched_matmul, :pointwise]
        @test first.(plan_shape(y)) == [:matmul, :pointwise]
        @test Array(execute!(x)) ≈ Array(execute!(y)) rtol = 1e-2
    end
    # Transposed operands, in Float16 and in training.
    q = CuArray(Float16.(reshape(sin.(1:(16 * 24 * 3)), 16, 24, 3)))
    k = CuArray(Float16.(reshape(cos.(1:(16 * 20 * 3)), 16, 20, 3)))
    ex = prepare_forward((q, k) -> Tempest.batched_mul(k, q; transpose_a = true), q, k)
    @test plan_shape(ex) == [(:batched_matmul, :cuBLAS)]
    @test Array(execute!(ex)) ≈ Tempest.batched_mul(Array(k), Array(q); transpose_a = true) rtol = 1e-2
    loss(q, k) = sum(abs2, Tempest.batched_mul(k, q; transpose_a = true))
    ex = prepare(loss, Float32.(Array(q)), Float32.(Array(k)); target = CUDATarget())
    @test count(s -> s.region isa Tempest.BatchedGemmRegion, ex.storage.steps) == 3
end

# Dropout is pointwise: it fuses with the work on either side.
@testset "Dropout fuses with its neighbours" begin
    x = CuArray(sin.(reshape(Float32.(1:(512 * 64)), 512, 64)))
    seed = CuArray(Int32[5])
    f(x, s) = tanh.(Tempest.dropout(x .* 2.0f0 .+ 1.0f0, s; rate = 0.2) .* 3.0f0)
    ex = prepare_forward(f, x, seed)
    @test first.(plan_shape(ex)) == [:pointwise]
    @test Array(execute!(ex)) ≈ f(Array(x), Int32[5]) rtol = 1e-6
end

# Without a fused kernel (full-precision products), attention expands to its
# reference: products, a logsumexp fold and pointwise work, with no calls left.
@testset "The attention reference has no calls" begin
    q, k, v = (CuArray(Float32.(reshape(sin.((1:(16 * 2 * 32 * 2)) .* s), 16, 2, 32, 2))) for s in (0.3, 0.7, 1.1))
    for causal in (false, true)
        ex = prepare_forward((q, k, v) -> Tempest.attention(q, k, v; causal), q, k, v)
        @test !any(step -> step.region isa Tempest.OpaqueCall, ex.storage.steps)
        @test count(step -> step.region isa Tempest.BatchedGemmRegion, ex.storage.steps) == 2
        @test Array(execute!(ex)) ≈ Tempest.attention(Array(q), Array(k), Array(v); causal) rtol = 1e-4
    end
end

# Statistics with a dropped axis, and reductions of permuted arrays, reach the
# plan of their direct spellings: exact aliases collapse before planning and
# permuted folds fold their source.
@testset "Reshaped statistics and permuted reductions plan alike" begin
    x = CuArray(sin.(reshape(Float32.(1:(256 * 40)), 256, 40)))
    w = CuArray(Float32.(range(0.5, 1.5; length = 256)))
    rms(x, w) = x .* w .* inv.(sqrt.(sum(abs2, x; dims = 1) ./ size(x, 1) .+ 1.0f-5))
    rms_dropdims(x, w) = x .* w .* inv.(sqrt.(dropdims(sum(abs2, x; dims = 1); dims = 1) ./ size(x, 1) .+ 1.0f-5))'
    rms_vec(x, w) = x .* w .* inv.(sqrt.(vec(sum(abs2, x; dims = 1)) ./ size(x, 1) .+ 1.0f-5))'
    expected = Array(execute!(prepare_forward(rms, x, w)))
    for f in (rms_dropdims, rms_vec)
        ex = prepare_forward(f, x, w)
        @test plan_shape(ex) == [(:graph, :cutile)]
        @test Array(execute!(ex)) ≈ expected rtol = 1e-5
    end
    # In training the forward statistics chain is also read by backward, so
    # the reshape cannot sink through it and the plan keeps two more steps;
    # that residual needs region growth through reshapes, not a rewrite.
    loss(f) = (x, w) -> sum(abs2, f(x, w))
    a, b = prepare(loss(rms_dropdims), x, w), prepare(loss(rms), x, w)
    @test first(plan_shape(a)) == first(plan_shape(b)) == (:graph, :cutile)
    @test Array(execute!(a)[2]) ≈ Array(execute!(b)[2]) rtol = 1e-4 atol = 1e-6

    row_softmax(x) = exp.(x .- Tempest.logsumexp(x; dims = 2))
    permuted(x) = exp.(x .- permutedims(Tempest.logsumexp(permutedims(x); dims = 1)))
    transposed(x) = exp.(x .- Tempest.logsumexp(x')')
    expected = Array(execute!(prepare_forward(row_softmax, x)))
    for f in (permuted, transposed)
        ex = prepare_forward(f, x)
        @test plan_shape(ex) == [(:graph, :cutile)]
        @test Array(execute!(ex)) ≈ expected rtol = 1e-5
    end
    # Full reductions read through reshapes, and reshapes sink through
    # pointwise work to the value they present.
    ex = prepare_forward(x -> sum(abs2, vec(x)) + sum(reshape(x, 40, :)), x)
    @test plan_shape(ex) == plan_shape(prepare_forward(x -> sum(abs2, x) + sum(x), x))
    for f in (x -> sum(abs2, vec(tanh.(x))), x -> sum(abs2, reshape(tanh.(x), 40, :)))
        ex = prepare_forward(f, x)
        @test plan_shape(ex) == [(:graph, :cutile)]
        @test Array(execute!(ex))[] ≈ sum(abs2, tanh.(Array(x))) rtol = 1e-5
    end
end

# Reductions over leading or trailing axes of any rank collapse to a matrix
# domain: rank-3 softmax plans as its reshaped rank-2 spelling and shares its
# kernel. Interior axes and partial broadcasts within a collapsed group are
# declined, and the CUDA kernels run them.
@testset "Rank-N reductions collapse to matrix domains" begin
    x = CuArray(sin.(reshape(Float32.(1:(96 * 24 * 5)), 96, 24, 5)))
    h = Array(x)
    softmax(x; dims) = exp.(x .- Tempest.logsumexp(x; dims))
    rank3 = prepare_forward(x -> softmax(x; dims = 1), x)
    rank2 = prepare_forward(x -> softmax(reshape(x, 96, :); dims = 1), x)
    @test plan_shape(rank3) == plan_shape(rank2) == [(:graph, :cutile)]
    @test Array(execute!(rank3)) ≈ reshape(Array(execute!(rank2)), 96, 24, 5) rtol = 1e-5
    @test occursin("of 96×24×5 as 96×120", Tempest.implementation_summary(only(rank3.calls)))
    naive(x) = (e = exp.(x .- maximum(x; dims = 1)); e ./ sum(e; dims = 1))
    ex = prepare_forward(naive, x)
    @test plan_shape(ex) == [(:graph, :cutile)]
    @test Array(execute!(ex)) ≈ naive(h) rtol = 1e-5
    # A weight over the leading axis and statistics over the rest collapse too.
    w = CuArray(Float32.(range(0.5, 1.5; length = 96)))
    layernorm(x, w) = (m = sum(x; dims = 1) ./ 96; c = x .- m; c .* inv.(sqrt.(sum(abs2, c; dims = 1) ./ 96 .+ 1.0f-5)) .* w)
    ex = prepare_forward(layernorm, x, w)
    @test plan_shape(ex) == [(:graph, :cutile)]
    @test Array(execute!(ex)) ≈ layernorm(h, Array(w)) rtol = 1e-4
    # Trailing axes, all axes, rank four, and the gradient of a rank-3 softmax.
    ex = prepare_forward(x -> sum(x; dims = (2, 3)) .+ sum(abs2, x; dims = (2, 3)), x)
    @test plan_shape(ex) == [(:graph, :cutile)]
    @test Array(execute!(ex)) ≈ sum(h; dims = (2, 3)) .+ sum(abs2, h; dims = (2, 3)) rtol = 1e-5
    ex = prepare_forward(x -> softmax(x; dims = 3), x)
    @test plan_shape(ex) == [(:graph, :cutile)]
    @test Array(execute!(ex)) ≈ softmax(h; dims = 3) rtol = 1e-5
    ex = prepare_forward(x -> sum(abs2, x) + sum(x), x)
    @test plan_shape(ex) == [(:graph, :cutile)]
    @test Array(execute!(ex))[] ≈ sum(abs2, h) + sum(h) rtol = 1e-5
    x4 = CuArray(sin.(reshape(Float32.(1:(32 * 4 * 24 * 3)), 32, 4, 24, 3)))
    ex = prepare_forward(x -> softmax(x; dims = 1), x4)
    @test plan_shape(ex) == [(:graph, :cutile)]
    @test Array(execute!(ex)) ≈ softmax(Array(x4); dims = 1) rtol = 1e-5
    t = CuArray(cos.(reshape(Float32.(1:(96 * 24 * 5)), 96, 24, 5)))
    a = prepare((x, t) -> sum(softmax(x; dims = 1) .* t), x, t)
    b = prepare((x, t) -> sum(softmax(reshape(x, 96, :); dims = 1) .* reshape(t, 96, :)), x, t)
    @test plan_shape(a) == [(:graph, :cutile), (:graph, :cutile)]
    @test length(plan_shape(a)) <= length(plan_shape(b))
    @test Array(execute!(a)[2]) ≈ reshape(Array(execute!(b)[2]), 96, 24, 5) rtol = 1e-4 atol = 1e-6
    # Declined: an interior axis, and a bias over one axis of a collapsed group.
    ex = prepare_forward(x -> x .- sum(x; dims = 2) ./ 24, x)
    @test all(step -> !(step.region isa Tempest.GraphRegion), ex.storage.steps)
    @test Array(execute!(ex)) ≈ h .- sum(h; dims = 2) ./ 24 rtol = 1e-5
    bias = CuArray(Float32.(reshape(range(-0.2, 0.3; length = 24), 1, 24, 1)))
    ex = prepare_forward((x, b) -> softmax(x .+ b; dims = 1), x, bias)
    @test Array(execute!(ex)) ≈ softmax(h .+ Array(bias); dims = 1) rtol = 1e-5
    @test any(note -> occursin("does not collapse", note), ex.semantic_notes)
    # The attention reference's softmax over (keys, queries, heads × batch) is
    # one graph region.
    q, k, v = (CuArray(Float32.(reshape(sin.((1:(16 * 2 * 64 * 2)) .* s), 16, 2, 64, 2))) for s in (0.3, 0.7, 1.1))
    ex = prepare_forward((q, k, v) -> Tempest.attention(q, k, v), q, k, v)
    @test count(step -> step.region isa Tempest.GraphRegion, ex.storage.steps) == 1
    @test Array(execute!(ex)) ≈ Tempest.attention(Array(q), Array(k), Array(v)) rtol = 1e-4
end

# Statistics' mean, var and std are their sums (ext/StatisticsExt.jl), so a
# layer norm written with them plans as the spelling with sums, forward and in
# training. Without the `mean` keyword, var computes its own mean: a second
# sum over the same values, in the same region.
@testset "Statistics spellings plan as their sums" begin
    x = CuArray(sin.(reshape(Float32.(1:(256 * 40)), 256, 40)))
    w = CuArray(Float32.(range(0.5, 1.5; length = 256)))
    function by_statistics(x, w)
        m = Statistics.mean(x; dims = 1)
        v = Statistics.var(x; dims = 1, mean = m, corrected = false)
        (x .- m) ./ sqrt.(v .+ 1.0f-5) .* w
    end
    function by_sums(x, w)
        m = sum(x; dims = 1) ./ size(x, 1)
        v = sum(abs2, x .- m; dims = 1) ./ size(x, 1)
        (x .- m) ./ sqrt.(v .+ 1.0f-5) .* w
    end
    by_std(x, w) = (x .- Statistics.mean(x; dims = 1)) ./ Statistics.std(x; dims = 1, corrected = false) .* w
    a, b, c = prepare_forward(by_statistics, x, w), prepare_forward(by_sums, x, w), prepare_forward(by_std, x, w)
    @test plan_shape(a) == plan_shape(b) == plan_shape(c) == [(:graph, :cutile)]
    expected = Array(execute!(b))
    @test Array(execute!(a)) ≈ expected rtol = 1e-5
    @test Array(execute!(c)) ≈ expected rtol = 1e-3
    loss(f) = (x, w) -> sum(abs2, f(x, w))
    a, b = prepare(loss(by_statistics), x, w), prepare(loss(by_sums), x, w)
    @test plan_shape(a) == plan_shape(b)
    @test Array(execute!(a)[2]) ≈ Array(execute!(b)[2]) rtol = 1e-5
    # A full mean, and the mean of a rank-3 array over its last axis.
    @test plan_shape(prepare_forward(x -> Statistics.mean(abs2, x), x)) ==
          plan_shape(prepare_forward(x -> sum(abs2, x) / length(x), x))
    x3 = CuArray(sin.(reshape(Float32.(1:(32 * 24 * 5)), 32, 24, 5)))
    ex = prepare_forward(x -> x .- Statistics.mean(x; dims = 3), x3)
    @test plan_shape(ex) == [(:graph, :cutile)]
    @test Array(execute!(ex)) ≈ Array(x3) .- Statistics.mean(Array(x3); dims = 3) rtol = 1e-5
end
