# Programs that differ by axes of extent one get the same plan: reshapes that
# insert them around pointwise work, reductions, products and their gradients,
# and attention, whose grouped heads have such an axis with one query head per
# key head, against attention written out without it.
include("../fixtures/programs.jl")

@testset "axes of extent one" begin
    explained(p) = [sprint(explain, C.plan(C.lower(p))), sprint(explain, recomputing(C.lower(p)))]
    same(f, g, args...) = explained(trace(f, args...)) == explained(trace(g, args...))
    same_gradient(f, g, args...) = explained(differentiate(trace(f, args...))) == explained(differentiate(trace(g, args...)))
    x, w, c = data(8, 6), data(8; scale = 0.1), data(5)
    a, b = data(5, 7; scale = 0.3), data(7, 6)
    pointwise(x) = tanh.(x) .* 2
    @test same(pointwise, x -> reshape(pointwise(reshape(x, 1, 8, 1, 6)), 8, 6), x)
    @test same(x -> sum(x; dims = 1), x -> dropdims(sum(x; dims = 1); dims = 1), x)
    @test same(x -> T.logsumexp(x; dims = 2), x -> reshape(T.logsumexp(reshape(x, 8, 1, 6); dims = 3), 8, 1), x)
    unit_norm(x, w) = reshape(rmsnorm(reshape(x, 8, 1, 6), reshape(w, 8, 1)), 8, 6)
    @test same(rmsnorm, unit_norm, x, w)
    @test same_gradient((x, w) -> sum(abs2, rmsnorm(x, w)), (x, w) -> sum(abs2, unit_norm(x, w)), x, w)
    unit_product(a, b) = reshape(C.contract(reshape(a, 5, 7, 1), reshape(b, 7, 6, 1), (1, 2, 4), (2, 3, 4), (1, 3, 4)), 5, 6)
    @test same((a, b, c) -> tanh.(a * b .+ c), (a, b, c) -> tanh.(unit_product(a, b) .+ reshape(c, 5, 1, 1)), a, b, c)
    @test same_gradient((a, b) -> sum(abs2, a * b), (a, b) -> sum(abs2, unit_product(a, b)), a, b)
    q, k, v, r = data(8, 3, 16, 2; scale = 0.3), data(8, 3, 16, 2; scale = 0.2), data(8, 3, 16, 2; scale = 0.1), data(8, 3, 16, 2)
    scale = Float32(1 / sqrt(8))
    scores(q, k) = C.contract(k, q, (1, 2, 3, 4), (1, 2, 5, 4), (3, 2, 5, 4)) .* scale
    causal(s) = ifelse.(reshape(1:size(s, 1), :, 1, 1) .<= reshape(1:size(s, 3), 1, 1, :), s, -Inf)
    written(q, k, v) = (s = causal(scores(q, k)); C.contract(v, exp.(s .- T.logsumexp(s; dims = 1)), (1, 2, 3, 4), (3, 2, 5, 4), (1, 2, 5, 4)))
    @test same((q, k, v, r) -> Ops.attention(q, k, v; causal = true) .+ r, (q, k, v, r) -> written(q, k, v) .+ r, q, k, v, r)
end
