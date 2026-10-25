# Extremes and where they are as Base finds them (`EXTREMES` and `FOUND` in
# fixtures/programs.jl), and the derivative: the cotangent at the element
# found.
include("../fixtures/programs.jl")

@testset "extremes" begin
    @testset "$(eltype(x)) $i" for (i, x) in enumerate(EXTREMES), f in FOUND
        @test isequal(found(interpret(trace(f, x), x)), found(f(x)))
    end
    x = Float64[1 5 2; 3 -5 7; 0 4 -1; 2 2 9]
    _, g = interpret(differentiate(trace(x -> sum(findmax(x; dims = 1)[1] .* [1.0 2.0 3.0]) + sum(abs2, x[argmin(x)]), x)), x)
    @test g == [0 2 0; 1 -10 0; 0 0 0; 0 0 3]
end
