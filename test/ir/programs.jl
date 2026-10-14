# End-to-end: trace, differentiate and interpret a small program.
@testset "First tensor program" begin
    f(x, w) = sum(abs2, tanh.(w' * x))
    x = reshape(collect(range(-0.4, 0.5; length = 12)), 3, 4)
    w = reshape(collect(range(-0.3, 0.6; length = 6)), 3, 2)
    p = trace(f, x, w)
    loss, (dx, dw) = value_and_gradient(p, x, w)
    @test loss ≈ f(x, w)
    @test dx ≈ finite_difference(f, (x, w), 1) atol=1e-8
    @test dw ≈ finite_difference(f, (x, w), 2) atol=1e-8
    @test occursin("matmul", sprint(show, MIME"text/plain"(), p))
    @test value_and_gradient(p, 2x, w)[1] ≈ f(2x, w)
    @test_throws ArgumentError value_and_gradient(p, Float32.(x), w)
    @test_throws ArgumentError value_and_gradient(p, x)
    @test_throws ArgumentError differentiate(trace(identity, x))
    @test_throws DimensionMismatch trace(f, x, zeros(5, 2))
    @test_throws ArgumentError trace(f, complex.(x), w)
end

@testset "Repeated and unused inputs" begin
    f(x, unused) = sum(abs2, x' * x)
    x = reshape([0.1, 0.4, -0.2, 0.3], 2, 2)
    p = trace(f, x, x)
    loss, (dx, unused) = value_and_gradient(p, x, x)
    @test loss ≈ f(x, x)
    @test dx ≈ finite_difference(f, (x, copy(x)), 1) atol=1e-8
    @test all(iszero, unused)
end
