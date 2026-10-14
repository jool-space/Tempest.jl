@testset "CUDA placement and provider isolation" begin
    x = reshape(Float32.(1:12) ./ 12, 3, 4)
    dx = CuArray(x)
    f(x, s) = sum(abs2, x * s)
    ex = prepare(f, dx, 2.0f0)
    @test all(a -> a isa CuArray, ex.inputs)
    @test Array(execute!(ex)[1])[] ≈ f(x, 2.0f0)
    capture!(ex)
    bind!(ex, dx, 3.0f0)
    @test Array(replay!(ex)[1])[] ≈ f(x, 3.0f0)
    @test occursin("Target: CUDA", sprint(explain, ex))
    @test_throws ArgumentError prepare((x, y) -> sum(x .* y), dx, x)
    copied = prepare((x, y) -> sum(x .* y), dx, x; target = CUDATarget())
    @test Array(execute!(copied)[1])[] ≈ sum(abs2, x)
    host = prepare(x -> sum(abs2, x), dx; target = CPUTarget())
    @test execute!(host)[1][] ≈ sum(abs2, x)
    for input in (transpose(dx), view(dx, :, 1:2), reshape(dx, 2, 6))
        wrapped = prepare(x -> sum(abs2, x), input)
        @test Array(execute!(wrapped)[1])[] ≈ sum(abs2, Array(input))
    end
    scalar = prepare(x -> x * x, 2.0f0; target = CUDATarget())
    capture!(scalar)
    bind!(scalar, 3.0f0)
    @test Array(replay!(scalar)[1])[] ≈ 9.0f0
    # The plain CUDA finalizer must cover tails, multiple scratch tiles, and reuse.
    for n in (1, 133, 257, 65537)
        a = Float32.(range(-0.3; step = 1 / max(n - 1, 1), length = n))
        plain = prepare(
            x -> sum(exp.(0 * x)) + sum(abs2, x),
            a;
            target = CUDATarget(kernels = :cuda),
        )
        @test all(==(:cuda), plain.providers)
        capture!(plain)
        bind!(plain, 2a)
        @test Array(replay!(plain)[1])[] ≈ n + sum(abs2, 2a) rtol=2e-5
        @test Array(outputs(plain)[2]) ≈ 4a rtol=2e-5
    end
    tile_loaded = Base.get_extension(Tempest, :cuTileExt) !== nothing
    if tile_loaded
        @test :cutile in ex.providers
        tiled = prepare(x -> sum(abs2, x), x; target = CUDATarget(kernels = :cutile))
        @test :cutile in tiled.providers
        @test Array(execute!(tiled)[1])[] ≈ sum(abs2, x)
    else
        @test all(==(:cuda), ex.providers)
        @test !any(id -> id.name == "cuTile", keys(Base.loaded_modules))
        @test_throws ArgumentError prepare(f, x, 2.0f0; target = CUDATarget(kernels = :cutile))
    end
end

@testset "CUDA defaults expose their support decisions" begin
    x = reshape(Float32.(1:64), 8, 8) ./ 64
    tiled = Base.get_extension(Tempest, :cuTileExt) !== nothing
    ex = prepare_forward(x -> sin.(view(x, 1:2:8, :)), CuArray(x))
    @test ex.providers == [:cuda]
    @test Array(execute!(ex)) ≈ sin.(view(x, 1:2:8, :))
    @test :cuda in ex.candidates
    @test :cuBLAS in ex.candidates
    if tiled
        @test first(ex.candidates) === :cutile
        @test occursin("mapped or expanded reads", sprint(explain, ex))
    else
        @test :cutile ∉ ex.candidates
    end
    plain = prepare_forward(x -> sin.(x), CuArray(x); target = CUDATarget(kernels = :cuda))
    @test :cutile ∉ plain.candidates
    @test plain.providers == [:cuda]
end
