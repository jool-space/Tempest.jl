# SpecialFunctions' error functions are primitives of a traced program: one
# map each, which runs SpecialFunctions' method on the elements, with the
# Gaussian for its derivative. Functions written with them trace through.
using SpecialFunctions: erf, erfc

@testset "error functions" begin
    x = data(8, 6; scale = 2.0)
    p = trace(x -> erf.(x) .+ erfc.(x ./ 2), x)
    @test count(n -> n.op isa C.Map, p.nodes) == 4
    @test interpret(p, x) ≈ erf.(x) .+ erfc.(x ./ 2)
    @test compile(x -> erf.(x) .+ erfc.(x ./ 2), x)(x) ≈ erf.(x) .+ erfc.(x ./ 2)
    _, g = T.gradient(x -> sum(erf.(x) .+ erfc.(x ./ 2)), x)
    @test g ≈ 2 / sqrt(π) .* (exp.(.-x .^ 2) .- exp.(.-(x ./ 2) .^ 2) ./ 2)
    # NNlib's exact gelu, x Φ(x), traces through to one error function, and its
    # derivative is NNlib's.
    q = trace(x -> sum(NNlib.gelu_erf.(x)), x)
    @test count(n -> n.op isa C.Map{typeof(erf)}, q.nodes) == 1
    @test T.gradient(x -> sum(NNlib.gelu_erf.(x)), x)[2] ≈ NNlib.deriv_gelu_erf.(x)
    if CUDA_FUNCTIONAL
        y = CuArray(single(x))
        @test isapprox(Array(compile(y -> NNlib.gelu_erf.(y), y)(y)), NNlib.gelu_erf.(single(x)); rtol = 1.0e-5)
        @test isapprox(Array(compile(y -> T.gradient(y -> sum(NNlib.gelu_erf.(y)), y)[2], y)(y)), NNlib.deriv_gelu_erf.(single(x)); rtol = 1.0e-5)
    end
end
