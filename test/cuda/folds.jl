# Folds on CUDA: cuTile graph regions for every schedule, and the general CUDA
# kernels for folds outside a region. Fixtures come from test/folds.jl.
function fold_mix(x, dims)
    l = Tempest.fold(LogSumExpFold(), x; dims)
    p = exp.(x .- l)
    (
        lse = l,
        max = maximum(x; dims),
        softmax = p,
        mean = sum(p .* x; dims),
        variance = Tempest.fold(WelfordVariance(), x; dims),
        range = Tempest.fold(range_fold, x; dims),
    )
end

function check_fold_outputs(got, expected; rtol = 2e-5, atol = 2e-5)
    for (g, r) in zip(values(got), values(expected))
        observed = r isa Number ? only(Array(g)) : Array(g)
        @test all(isfinite, observed)
        @test isapprox(observed, r; rtol, atol)
    end
end

@testset "cuTile graph regions with folds" begin
    for (shape, dims) in (
        ((300, 37), 1),   # resident column
        ((50, 300), 1),   # packed resident tile
        ((5000, 70), 1),  # streamed
        ((5000, 8), 1),   # split
        ((2100, 40), 2),  # streamed second axis
        ((64, 300), 2),   # split second axis
        ((300, 37), :),   # full
        ((7, 5), :),      # tiny full
    )
        indices = reshape(Float32.(1:prod(shape)), shape)
        x = 40.0f0 .* sin.(0.37f0 .* indices) .+ cos.(0.011f0 .* indices)
        ex = prepare_forward(x -> fold_mix(x, dims), CuArray(x))
        check_fold_outputs(execute!(ex), fold_mix(x, dims))
        # Every operation is inside a cuTile graph region.
        steps = filter(step -> step.region isa Tempest.GraphRegion, ex.storage.steps)
        covered = sort!(reduce(vcat, [step.region.members for step in steps]))
        @test covered == findall(node -> !(node.op isa Tempest.Argument), ex.storage.program.nodes)
        @test all(==(:cutile), ex.providers)
        y = reverse(x; dims = 1)
        bind!(ex, CuArray(y))
        capture!(ex)
        replay!(ex)
        check_fold_outputs(outputs(ex), fold_mix(y, dims))
    end
    # A logsumexp held in registers lowers as a maximum and a shifted sum.
    x = randn(Float32, 300, 37)
    ex = prepare_forward(x -> Tempest.fold(LogSumExpFold(), x; dims = 1), CuArray(x))
    @test Array(execute!(ex)) ≈ lse_reference(x; dims = 1) rtol = 1e-5
end

@testset "General CUDA fold kernels" begin
    x = 40.0f0 .* sin.(0.37f0 .* reshape(Float32.(1:(64 * 300)), 64, 300))
    for dims in (1, 2, :)
        expected = fold_mix(x, dims)
        ex = prepare_forward(x -> fold_mix(x, dims), CuArray(x); fuse_regions = false)
        check_fold_outputs(execute!(ex), expected)
        @test !any(step -> step.region isa Tempest.GraphRegion, ex.storage.steps)
        capture!(ex)
        replay!(ex)
        check_fold_outputs(outputs(ex), expected)
        # Float64 stays off cuTile.
        x64 = Float64.(x)
        got = execute!(prepare_forward(x -> fold_mix(x, dims), CuArray(x64)))
        check_fold_outputs(got, fold_mix(x64, dims); rtol = 1e-10, atol = 1e-10)
    end
    # A fold carrying data is declined by cuTile and runs on the CUDA kernels.
    f(x) = Tempest.fold(TemperedLogSumExp(2.0f0), x; dims = 1)
    ex = prepare_forward(f, CuArray(x))
    @test Array(execute!(ex)) ≈ tempered_reference(x, 2; dims = 1) rtol = 1e-5
    @test ex.providers == [:cuda]
    # An empty full fold observes its identity.
    @test Tempest.fold(LogSumExpFold(), Float32[]) == -Inf32
end

@testset "logsumexp off the CUDA kernel's axis stays finite" begin
    x = 100.0f0 .* sin.(reshape(Float32.(1:(64 * 300)), 64, 300))
    for kwargs in ((;), (; fuse_regions = false))
        ex = prepare_forward(x -> Tempest.logsumexp(x; dims = 2), CuArray(x); kwargs...)
        y = Array(execute!(ex))
        @test all(isfinite, y)
        @test y ≈ Tempest.logsumexp(x; dims = 2) rtol = 1e-5
    end
end

@testset "Fold gradients on CUDA" begin
    cross_entropy(x, t) = sum(Tempest.fold(LogSumExpFold(), x; dims = 1) .- sum(x .* t; dims = 1))
    centered(x, t) = sum(abs2, x .- maximum(x; dims = 2)) + sum(x .* t)
    spread(x, t) = sum(Tempest.fold(WelfordVariance(), x; dims = 1) .* sum(t; dims = 1))
    for (f, shape) in ((cross_entropy, (300, 37)), (cross_entropy, (5000, 8)), (centered, (64, 300)), (spread, (300, 37)))
        indices = reshape(Float32.(1:prod(shape)), shape)
        x = 20.0f0 .* sin.(0.37f0 .* indices)
        t = zeros(Float32, shape)
        t[1, :] .= 1
        reference = differentiate(trace(f, Float64.(x), Float64.(t)); wrt = (1,))
        loss, (dx,) = value_and_gradient(reference, Float64.(x), Float64.(t))
        ex = prepare(differentiate(trace(f, x, t); wrt = (1,)), CuArray(x), CuArray(t))
        got = execute!(ex)
        @test only(Array(got[1])) ≈ loss rtol = 1e-4
        @test Array(got[2]) ≈ dx rtol = 1e-3 atol = 1e-3
        @test any(step -> step.region isa Tempest.GraphRegion, ex.storage.steps)
    end
    # The rule table is target-independent: no rule, no derivative.
    @test_throws ArgumentError prepare(x -> sum(Tempest.fold(range_fold, x; dims = 1)), CuArray(randn(Float32, 8, 3)))
end

@testset "Infinite slices and ties in cuTile folds" begin
    # An all -Inf column gives -Inf and a +Inf element +Inf, on the resident,
    # packed, streamed, split and full schedules.
    for (shape, dims) in (((300, 4), 1), ((50, 300), 1), ((5000, 40), 1), ((5000, 4), 1), ((300, 37), :))
        x = 3.0f0 .* sin.(reshape(Float32.(1:prod(shape)), shape))
        x[:, 2] .= -Inf32
        x[7, 3] = Inf32
        f(x) = (lse = Tempest.fold(LogSumExpFold(), x; dims), max = maximum(x; dims), min = minimum(x; dims))
        ex = prepare_forward(f, CuArray(x))
        @test all(==(:cutile), ex.providers)
        got = map(Array, execute!(ex))
        expected = (lse = Tempest.logsumexp(x; dims), max = maximum(x; dims), min = minimum(x; dims))
        for key in keys(expected)
            g, r = got[key], expected[key]
            g, r = r isa Number ? (vec(g), [r]) : (g, r)
            @test isequal(isfinite.(g), isfinite.(r))
            @test isequal(g[.!isfinite.(g)], r[.!isfinite.(r)])
            @test g[isfinite.(g)] ≈ r[isfinite.(r)] rtol = 1e-5
        end
    end
    # Tied extrema share the cotangent equally on the GPU as well.
    x = Float32[1 2 5; 3 2 5; 3 0 5]
    for f in (x -> sum(maximum(x; dims = 1)), x -> sum(minimum(x; dims = 2)), maximum)
        p = differentiate(trace(f, x))
        _, (expected,) = value_and_gradient(p, x)
        @test Array(execute!(prepare(p, CuArray(x)))[2]) ≈ expected
    end
end

@testset "Products and Base reductions on CUDA" begin
    shape = (300, 37)
    x = 1.0f0 .+ 0.01f0 .* sin.(reshape(Float32.(1:prod(shape)), shape))
    x[5, 2] = 0                  # one zero in a column
    x[7, 3] = x[9, 3] = 0        # two zeros in a column
    for dims in (:, (1,), (2,))
        result(got) = dims isa Colon ? only(Array(got)) : Array(got)
        f(x) = (
            prod = prod(x; dims),
            max = reduce(max, x; dims),
            sum = mapreduce(abs2, +, x; dims, init = 1.0f0),
            norm = mapreduce(abs, norm_combine, x; dims, init = 0.0f0),
        )
        ex = prepare_forward(f, CuArray(x))
        @test all(==(:cutile), ex.providers)
        got = execute!(ex)
        @test result(got.prod) ≈ prod(x; dims) rtol = 1e-4
        @test result(got.max) == maximum(x; dims)
        @test result(got.sum) ≈ mapreduce(abs2, +, x; dims, init = 1.0f0) rtol = 1e-5
        @test result(got.norm) ≈ mapreduce(abs, norm_combine, x; dims, init = 0.0f0) rtol = 1e-5
        # An identity the element type cannot rebuild is captured: CUDA kernels.
        g(x) = reduce(smaller, x; dims, init = 2.0f0)
        ex = prepare_forward(g, CuArray(x))
        @test ex.providers == [:cuda]
        @test result(execute!(ex)) == minimum(x; dims)
        # Gradients through the product and maximum rules match the CPU.
        loss(x) = sum(sin, prod(x; dims)) + sum(abs2, reduce(max, x; dims))
        _, (expected,) = value_and_gradient(differentiate(trace(loss, Float64.(x))), Float64.(x))
        got = execute!(prepare(differentiate(trace(loss, x)), CuArray(x)))
        @test Array(got[2]) ≈ expected rtol = 1e-3 atol = 1e-4
    end
end
