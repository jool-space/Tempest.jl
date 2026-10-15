# Folds on the general CUDA kernels, outside graph regions.
include("../fixtures/folds.jl")

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

@testset "logsumexp along either axis stays finite" begin
    x = 100.0f0 .* sin.(reshape(Float32.(1:(64 * 300)), 64, 300))
    for kwargs in ((;), (; fuse_regions = false))
        ex = prepare_forward(x -> Tempest.logsumexp(x; dims = 2), CuArray(x); kwargs...)
        y = Array(execute!(ex))
        @test all(isfinite, y)
        @test y ≈ Tempest.logsumexp(x; dims = 2) rtol = 1e-5
    end
end

@testset "Half-precision folds on the CUDA kernels accumulate in Float32" begin
    for T in (Float16, Core.BFloat16)
        x = half_fold_input(T)
        for case in HALF_FOLD_CASES
            elements, reduce = case
            f(x) = reduce(elements(x))
            ex = without_cutile(() -> prepare_forward(f, CuArray(x); fuse_regions = false))
            @test all(==(:cuda), ex.providers)
            check_half_fold(execute!(ex), half_fold_reference(case, x))
        end
    end
end
