# A derivative a package states for its scalar function with ChainRulesCore's
# `@scalar_rule` is the function's: where its code's derivative is another, at
# a branch, and where its code has none, recorded whole.
using ChainRulesCore: ChainRulesCore

# `x log y`, zero where `x` is, as Flux's `xlogy` is: the branch guards the
# product against `0 log 0`, and its derivative in `x` there is none, where the
# function's is `log y`.
guarded(x, y) = ifelse(iszero(x), zero(x * log(y)), x * log(y))
ChainRulesCore.@scalar_rule guarded(x, y) (log(y), x / y)
unstated(x, y) = ifelse(iszero(x), zero(x * log(y)), x * log(y))

# A function that branches on its argument, which Tempest records whole.
branching(x) = x > 0 ? x * x : -x
ChainRulesCore.@scalar_rule branching(x) ifelse(x > 0, 2x, -one(x))

# A statement that branches on its argument does not run on handles.
tripled(x) = 3x
ChainRulesCore.@scalar_rule tripled(x) (x > 0 ? 3.0 : 3.0)

@testset "stated derivatives" begin
    x, y = [0.0, 0.5, 2.0], [0.3, 1.5, 2.0]
    _, gx, gy = T.gradient((x, y) -> sum(guarded.(x, y)), x, y)
    @test gx ≈ log.(y) && gy ≈ x ./ y
    @test T.gradient((x, y) -> sum(unstated.(x, y)), x, y)[2] ≈ [0.0, log(1.5), log(2.0)]
    # The value is the code's.
    p = trace((x, y) -> guarded.(x, y), x, y)
    @test count(n -> n.op isa C.Map{<:C.Stated}, p.nodes) == 1
    @test interpret(p, x, y) == unstated.(x, y)
    c = compile((x, y) -> T.gradient((x, y) -> sum(guarded.(x, y)), x, y), x, y)
    @test all(c(x, y) .≈ (sum(unstated.(x, y)), log.(y), x ./ y))
    @test T.gradient(x -> sum(branching.(x)), [-1.0, 2.0])[2] == [-1.0, 4.0]
    @test T.gradient(x -> sum(tripled.(x)), [-1.0, 2.0])[2] == [3.0, 3.0]
    # NNlib states its activations' derivatives: relu's is zero at zero, where
    # its code's is one.
    @test T.gradient(x -> sum(relu.(x)), [-1.0, 0.0, 2.0])[2] == [0.0, 0.0, 1.0]
    if CUDA_FUNCTIONAL
        a, b = CuArray(Float32[0, 0.5, 2, 0]), CuArray(Float32[0.3, 1.5, 2, 1])
        _, ga, gb = compile((a, b) -> T.gradient((a, b) -> sum(guarded.(a, b) .+ relu.(a .- 1)), a, b), a, b)(a, b)
        @test Array(ga) ≈ log.(Array(b)) .+ (Array(a) .> 1) && Array(gb) ≈ Array(a) ./ Array(b)
    end
end
