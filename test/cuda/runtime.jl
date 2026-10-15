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
        plain = without_cutile(() -> prepare(x -> sum(exp.(0 * x)) + sum(abs2, x), a; target = CUDATarget()))
        @test all(==(:cuda), plain.providers)
        capture!(plain)
        bind!(plain, 2a)
        @test Array(replay!(plain)[1])[] ≈ n + sum(abs2, 2a) rtol=2e-5
        @test Array(outputs(plain)[2]) ≈ 4a rtol=2e-5
    end
    if CUTILE_ENABLED
        @test :cutile in ex.providers
    else
        @test all(==(:cuda), ex.providers)
        @test !any(id -> id.name == "cuTile", keys(Base.loaded_modules))
    end
end

@testset "CUDA defaults expose their support decisions" begin
    x = reshape(Float32.(1:64), 8, 8) ./ 64
    ex = prepare_forward(x -> sin.(view(x, 1:2:8, :)), CuArray(x))
    @test ex.providers == [:cuda]
    @test Array(execute!(ex)) ≈ sin.(view(x, 1:2:8, :))
    @test :cuda in ex.candidates
    @test :cuBLAS in ex.candidates
    if CUTILE_ENABLED
        @test first(ex.candidates) === :cutile
        @test occursin("mapped or expanded reads", sprint(explain, ex))
    else
        @test :cutile ∉ ex.candidates
    end
    plain = without_cutile(() -> prepare_forward(x -> sin.(x), CuArray(x)))
    @test :cutile ∉ plain.candidates
    @test plain.providers == [:cuda]
end

@testset "cuBLAS half-precision products accumulate in Float32" begin
    a = reshape(Float32.(sin.(1:(24 * 40))), 24, 40) ./ 8
    b = reshape(Float32.(cos.(1:(40 * 16))), 40, 16) ./ 8
    for T in (Float16, Core.BFloat16)
        f(a, b) = a * b
        g(a, b) = transpose(a) * b
        ex = prepare_forward(f, CuArray(T.(a)), CuArray(T.(b)))
        @test ex.providers == [:cuBLAS]
        y = execute!(ex)
        @test eltype(y) === T
        # One rounding of a Float32 accumulation.
        expected = T.(Float32.(T.(a)) * Float32.(T.(b)))
        @test maximum(abs.(Float32.(Array(y)) .- Float32.(expected))) <= 2 * eps(T(1))
        at = Matrix(transpose(a))
        ex = prepare_forward(g, CuArray(T.(at)), CuArray(T.(b)))
        @test maximum(abs.(Float32.(Array(execute!(ex))) .- Float32.(expected))) <= 2 * eps(T(1))
    end
end
