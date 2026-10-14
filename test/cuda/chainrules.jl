# ChainRules supplies the log1p, expm1 and atan rules; this file runs on its own
# worker (see runtests.jl).
using ChainRules

@testset "Shared scalar rules in captured CUDA training" begin
    x = reshape(Float32.(range(0.2, 0.7; length = 6)), 2, 3)
    y = reshape(Float32[0.4, 0.6], 2, 1)
    loss(x, y) = sum(log1p.(x) .* expm1.(y) .+ atan.(x, y))
    executable = prepare(loss, CuArray(x), CuArray(y))
    if Base.get_extension(Tempest, :cuTileExt) !== nothing
        @test occursin("cuTile has no verified lowering for log1p", sprint(explain, executable))
    end
    capture!(executable)
    for scale in (1.0f0, 1.3f0)
        a, b = scale .* x, scale .* y
        bind!(executable, CuArray(a), CuArray(b))
        actual = replay!(executable)
        @test only(Array(actual[1])) ≈ loss(a, b) rtol = 2e-5
        for i = 1:2
            expected = finite_difference(loss, (Float64.(a), Float64.(b)), i)
            @test Array(actual[i + 1]) ≈ expected rtol = 2e-5 atol = 2e-6
        end
    end
end

@testset "Shared unary rules on CUDA and cuTile" begin
    for f in (expm1, log1p, sinh, cosh, tan, asin, acos, atan, abs)
        x = f === abs ? Float32[-0.3, 0.0, 0.4] : Float32[0.2, 0.4, 0.6]
        loss(x) = sum(f.(x))
        executable = prepare(loss, CuArray(x))
        capture!(executable)
        for scale in (1.0f0, 0.7f0)
            input = scale .* x
            bind!(executable, CuArray(input))
            value, gradient = replay!(executable)
            expected = finite_difference(loss, (Float64.(input),), 1)
            @test only(Array(value)) ≈ loss(input) rtol = 2e-5 atol = 2e-6
            @test Array(gradient) ≈ expected rtol = 2e-5 atol = 2e-6
        end
    end
end
