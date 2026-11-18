# An einsum spelled with Einops traces as one product: the labels are
# numbered, grouped axes are reshapes, and the ellipsis is resolved before
# Tempest sees it. The interpreter and the compiled program agree with the
# eager einsum.
using Einops

@testset "an einsum is one contract" begin
    x, w = randn(Float32, 8, 5, 3), randn(Float32, 8, 8)
    heads(w, x) = einsum(w, x, einops"D (d h), D ... -> d h ..."; h = 2)
    scores(k, q) = einsum(k, q, einops"d h k ..., d h q ... -> k q h ...")
    f(w, x) = sum(abs2, scores(heads(w, x), heads(w, x)))
    p = trace(f, w, x)
    @test count(n -> n.op isa C.Contract, p.nodes) == 3
    @test only(C.leaves(interpret(p, w, x))) ≈ f(w, x)
    @test compile(f, w, x)(w, x) ≈ f(w, x)
    # A host array beside a traced one is a constant of the program.
    g(x) = einsum(w, x, einops"D (d h), D ... -> d h ..."; h = 2)
    @test only(C.leaves(interpret(trace(g, x), x))) ≈ g(x)
end

@testset "three operands are two contracts, in the order Einops finds" begin
    a, b, c = randn(Float32, 3, 4), randn(Float32, 4, 5), randn(Float32, 5, 2)
    h(a, b, c) = einsum(a, b, c, einops"i j, j k, k l -> i l")
    p = trace(h, a, b, c)
    @test count(n -> n.op isa C.Contract, p.nodes) == 2
    @test only(C.leaves(interpret(p, a, b, c))) ≈ h(a, b, c)
end

# Einops drops the axis a fold reduces and puts back the one its pattern
# keeps: two reshapes that are the fold's own result. So a log-sum-exp
# spelled with `@reduce` is the one spelled with `dims`, whose maximum the
# loss does not depend on and no rule differentiates, where a reshape
# between the sum and its logarithm hid that from the laws that find it.
@testset "a fold whose pattern keeps its axis is the fold along `dims`" begin
    x = randn(Float32, 7, 3, 2)
    mask = axes(x, 1) .== reshape(rand(1:7, 3, 2), 1, 3, 2)
    pick(logp) = -sum(ifelse.(mask, logp, 0.0f0))
    kept(x) = (z = x .- @reduce(maximum, x, "v t b -> 1 t b"); pick(z .- log.(@reduce(sum, exp.(z), "v t b -> 1 t b"))))
    plain(x) = (z = x .- maximum(x; dims = 1); pick(z .- log.(sum(exp.(z); dims = 1))))
    folds(f) = count(n -> n.op isa C.Reduce, differentiate(trace(f, x)).nodes)
    @test folds(kept) == folds(plain)
    @test T.gradient(kept, x)[1] ≈ T.gradient(plain, x)[1]
end
