# Positions on cuTile's tiles, read by masks, pointwise work and folds, and
# hashed as dropout draws.

# Positions on tiles, and formulas of the axes the grid binds computed per
# block: a causal mask after a product, positions in pointwise work and in a
# fold, and key heads shared among query heads, spelled as a broadcast that
# reads key head h ÷ G.
causal_scores(a, b) = (s = a' * b; ifelse.((1:size(s, 1)) .<= (1:size(s, 2))', s, -Inf32))
function grouped_scores(q, k)
    D, H, S, B = size(q)
    G = H ÷ size(k, 2)
    shared = reshape(reshape(k, D, 1, size(k, 2), S, B) .* ones(Float32, 1, G, 1, 1, 1), D, H, S, B)
    C.contract(shared, q, (1, 2, 3, 4), (1, 2, 5, 4), (3, 5, 2, 4))
end

const POSITIONAL = [
    ("causal mask after a product", (a, b) -> exp.(causal_scores(a, b) .- T.logsumexp(causal_scores(a, b); dims = 1)),
     (data(24, 70; scale = 0.1), data(24, 70; scale = 0.2))),
    ("positions in pointwise work", x -> x .* reshape(1:size(x, 2), 1, :) .+ reshape(1:length(x), size(x)), (data(50, 30, 3),)),
    ("positions with arithmetic", x -> x .* fld.(reshape(1:size(x, 2), 1, :) .- 1, 4) .+ (2 .* reshape(Int32(1):Int32(size(x, 1)), :, 1)), (data(50, 30, 3),)),
    ("positions in a fold", x -> sum(ifelse.((1:size(x, 1)) .<= (1:size(x, 2))', x, 0.0f0); dims = 1), (data(300, 200),)),
    ("grouped heads by division", grouped_scores, (data(16, 6, 40, 2; scale = 0.1), data(16, 2, 40, 2; scale = 0.2))),
    ("shared columns by division in a fold",
     x -> sum(reshape(reshape(x, 64, 1, 3, 10) .* ones(Float32, 1, 2, 1, 1), 64, 6, 10); dims = 1), (data(64, 3, 10),)),
]

@testset "positions and per-block formulas on cuTile" begin
    @testset "$name" for (name, f, args) in POSITIONAL
        args = map(single, args)
        p = trace(f, args...)
        ex = prepare(p, map(CuArray, args)...; target = T.CUDATarget(cuTile))
        @test all(d -> !(d isa C.FloorKernels), ex.kernels)
        @test agree(C.leaves(outputs(execute!(ex))), C.leaves(interpret(p, args...)); rtol = 2.0e-4)
    end
end

# The dropout fixture's hash, wrapping Int32 multiplies and logical shifts on
# tiles, is claimed by the generator and then fails to build: the assembler
# rejects the bytecode cuTile emits for it. The floor runs it, correct, and
# the failed build warns once.
const INTEGER_HASH = "an integer hash (wrapping multiplies, logical shifts) on Int32 tiles (cuTile's bytecode)"
@testset "gap: $INTEGER_HASH" begin
    x, s = single(data(64, 48)), Int32[7]
    p = trace((x, s) -> Ops.dropout(x, s; rate = 0.25), x, s)
    ex = prepare(p, CuArray(x), CuArray(s); target = T.CUDATarget(cuTile))
    @test_broken !(only(ex.kernels) isa C.FloorKernels)
    @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, x, s)])
end
