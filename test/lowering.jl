# The index level computes what the program level computes: every program,
# lowered, interpreted value by value, agrees with Tempest's interpreter.
using Test, Tempest, TempestSeed, Statistics
import TempestSeed as S

softmax_columns(x) = exp.(x .- Tempest.logsumexp(x; dims = 1))
softmax_naive(x) = exp.(x) ./ sum(exp.(x); dims = 1)
softmax_rows(x) = exp.(x .- Tempest.logsumexp(x; dims = 2))
rmsnorm(x, w) = x ./ sqrt.(mean(abs2, x; dims = 1) .+ 1.0f-5) .* w
function layernorm(x, w)
    m = mean(x; dims = 1)
    (x .- m) ./ sqrt.(var(x; dims = 1, mean = m, corrected = false) .+ 1.0f-5) .* w
end
mlp(x, w1, b1, w2, b2) = w2 * tanh.(w1 * x .+ b1) .+ b2

data(dims...; scale = 1.0f0) = 2.0f0 .* sin.(scale .* reshape(Float32.(1:prod(dims)), dims...))

const FORWARD = [
    ("dot product", (a, b) -> sum(a .* b), (data(64), data(64; scale = 0.3f0))),
    ("sum abs2 tanh", x -> sum(abs2, tanh.(x)), (data(16, 8),)),
    ("full logsumexp", x -> Tempest.logsumexp(x), (data(16, 8),)),
    ("maximum and shifted sum", x -> sum(exp.(x .- maximum(x))), (data(16, 8),)),
    ("x .- mean(x)", x -> x .- mean(x), (data(16, 8),)),
    ("column sums", x -> sum(abs2, x; dims = 1), (data(16, 8),)),
    ("row sums", x -> sum(abs2, x; dims = 2), (data(16, 8),)),
    ("row maxima", x -> maximum(x; dims = 2), (data(16, 8),)),
    ("bias gradient", (g, h) -> sum(g .* (1 .- h .^ 2); dims = 2), (data(16, 8), data(16, 8; scale = 0.3f0))),
    ("softmax columns", softmax_columns, (data(16, 8),)),
    ("naive softmax", softmax_naive, (data(16, 8),)),
    ("softmax rows", softmax_rows, (data(16, 8),)),
    ("RMSNorm", rmsnorm, (data(16, 8), data(16; scale = 0.1f0))),
    ("layer norm", layernorm, (data(16, 8), data(16; scale = 0.1f0))),
    ("softmax axis 1, rank 3", softmax_columns, (data(8, 4, 3),)),
    ("softmax axis 3, rank 3", x -> exp.(x .- Tempest.logsumexp(x; dims = 3)), (data(8, 4, 3),)),
    ("layer norm, rank 3", layernorm, (data(8, 4, 3), data(8; scale = 0.1f0))),
    # Layout spellings.
    ("transpose then column sums", x -> sum(permutedims(x) .* 2.0f0; dims = 1), (data(16, 8),)),
    ("adjoint", x -> sum(x' .* 2.0f0; dims = 2), (data(16, 8),)),
    ("reshape round trip", x -> reshape(reshape(x, 128) .+ 1.0f0, 16, 8) .* x, (data(16, 8),)),
    ("reshape of a permutation", x -> sum(reshape(permutedims(x, (2, 1, 3)), 4, 24); dims = 1), (data(8, 4, 3),)),
    ("strided view", x -> sum(view(x, 2:2:16, :); dims = 1), (data(16, 8),)),
    ("slice with a fixed index", x -> x[3, :] .* 2.0f0, (data(16, 8),)),
    ("gather by a constant list", x -> x[:, [3, 1, 3, 8]] .+ 1.0f0, (data(16, 8),)),
    ("dropout positions", (x, s) -> Tempest.dropout(x, s; rate = 0.25f0), (data(16, 8), Int32[7])),
    ("MLP", mlp, (data(16, 32), data(24, 16; scale = 0.1f0), data(24), data(16, 24; scale = 0.2f0), data(16))),
    ("causal attention", (q, k, v) -> Tempest.attention(q, k, v; causal = true),
     (data(8, 2, 6, 1), data(8, 2, 6, 1; scale = 0.5f0), data(8, 2, 6, 1; scale = 0.25f0))),
]

const TRAINING = [
    ("RMSNorm", (x, w) -> sum(abs2, rmsnorm(x, w)), (data(16, 8), data(16; scale = 0.1f0))),
    ("softmax", (x, t) -> sum(softmax_columns(x) .* t), (data(16, 8), data(16, 8; scale = 0.7f0))),
    ("MLP", (x, w1, b1, w2, b2) -> sum(abs2, mlp(x, w1, b1, w2, b2)),
     (data(16, 32), data(24, 16; scale = 0.1f0), data(24), data(16, 24; scale = 0.2f0), data(16))),
    ("layout spellings", x -> sum(abs2, reshape(permutedims(x, (2, 1, 3)), 4, 24) .* 0.5f0), (data(8, 4, 3),)),
]

agree(a, b) = all(map((x, y) -> isapprox(x, y; rtol = 1.0e-4, atol = 1.0e-5), a, b))

@testset "lowering agrees with the program interpreter" begin
    @testset "forward: $name" for (name, f, args) in FORWARD
        program = Tempest.ForwardProgram(trace(f, args...))
        graph = S.lower(program)
        @test agree(S.interpret(graph, args...), Tempest.interpret(program, args...))
        @test !any(v -> v.definition isa S.Opaque && v.definition.node.op isa Tempest.LayoutOperation, graph.values)
    end
    @testset "training: $name" for (name, f, args) in TRAINING
        program = differentiate(trace(f, args...))
        graph = S.lower(program)
        @test agree(S.interpret(graph, args...), Tempest.interpret(program, args...))
    end
end
