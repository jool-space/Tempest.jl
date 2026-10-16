# The program level: traced programs agree with plain Julia, and gradients of
# differentiated programs agree with finite differences.
using Test, Tempest, LinearAlgebra
import Tempest as T

swish(x) = x / (1 + exp(-x))
leaky(x) = x > 0 ? x : 0.01 * x
T.scalar_partials(::typeof(leaky), g, y, x) = (ifelse(x > 0, g, 0.01 * g),)
rmsnorm(x, w) = x ./ sqrt.(sum(abs2, x; dims = 1) ./ size(x, 1) .+ 1.0e-5) .* w
softmax_columns(x) = exp.(x .- T.logsumexp(x; dims = 1))
softmax_rows(x) = exp.(x .- T.logsumexp(x; dims = 2))
function layernorm(x, w)
    m = sum(x; dims = 1) ./ size(x, 1)
    v = sum(abs2, x .- m; dims = 1) ./ size(x, 1)
    (x .- m) ./ sqrt.(v .+ 1.0e-5) .* w
end
mlp(x, w1, b1, w2, b2) = w2 * tanh.(w1 * x .+ b1) .+ b2

data(dims...; scale = 1.0) = 2.0 .* sin.(scale .* reshape(Float64.(1:prod(dims)), dims...))

const PROGRAMS = [
    ("dot product", (a, b) -> sum(a .* b), (data(32), data(32; scale = 0.3))),
    ("sum abs2 tanh", x -> sum(abs2, tanh.(x)), (data(8, 6),)),
    ("full logsumexp", x -> T.logsumexp(x), (data(8, 6),)),
    ("maximum and shifted sum", x -> sum(exp.(x .- maximum(x))), (data(8, 6),)),
    ("x .- mean", x -> sum(abs2, x .- sum(x) / length(x)), (data(8, 6),)),
    ("column maxima", x -> sum(maximum(x; dims = 1) .* data(1, 6)), (data(8, 6),)),
    ("row minima", x -> sum(minimum(x; dims = 2)), (data(8, 6),)),
    ("products", x -> sum(prod(x; dims = 1)), (data(5, 3; scale = 0.4),)),
    ("softmax columns", (x, t) -> sum(softmax_columns(x) .* t), (data(8, 6), data(8, 6; scale = 0.7))),
    ("softmax rows", (x, t) -> sum(softmax_rows(x) .* t), (data(8, 6), data(8, 6; scale = 0.7))),
    ("RMSNorm", (x, w) -> sum(abs2, rmsnorm(x, w)), (data(8, 6), data(8; scale = 0.1))),
    ("layer norm", (x, w) -> sum(abs2, layernorm(x, w)), (data(8, 6), data(8; scale = 0.1))),
    ("rank 3 softmax", (x, t) -> sum(exp.(x .- T.logsumexp(x; dims = 3)) .* t), (data(4, 3, 5), data(4, 3, 5; scale = 0.2))),
    ("MLP", (x, w1, b1, w2, b2) -> sum(abs2, mlp(x, w1, b1, w2, b2)),
     (data(6, 10), data(7, 6; scale = 0.1), data(7), data(6, 7; scale = 0.2), data(6))),
    ("adjoint product", (a, b) -> sum(tanh.(a' * b)), (data(6, 4), data(6, 5; scale = 0.3))),
    ("batched product", (a, b) -> sum(abs2, T.batched_mul(a, b)), (data(3, 4, 2), data(4, 5, 2; scale = 0.3))),
    ("matrix vector", (a, v) -> sum(abs2, a * v), (data(5, 4), data(4; scale = 0.5))),
    ("swish", x -> sum(swish.(x)), (data(8, 6),)),
    ("opaque leaky with a rule", x -> sum(abs2, leaky.(x)), (data(8, 6),)),
    ("powers", x -> sum(x .^ 2 .+ x .^ 3 .+ sqrt.(abs.(x) .+ 1)), (data(8, 6),)),
    ("clamp and max", x -> sum(clamp.(x, -0.5, 0.5) .+ max.(x, 0.1)), (data(8, 6),)),
    ("permutation and reshape", x -> sum(abs2, reshape(permutedims(x, (2, 1, 3)), 3, 20) .* data(3, 20)), (data(4, 3, 5),)),
    ("strided view and index", x -> sum(abs2, view(x, 2:2:8, :) .+ x[3, 2]), (data(8, 6),)),
    ("gather by a constant list", x -> sum(abs2, x[:, [3, 1, 3, 6]]), (data(8, 6),)),
    ("vec and dropdims", x -> sum(abs2, vec(dropdims(sum(x; dims = 1); dims = 1))), (data(8, 6),)),
    ("scalar input", (x, s) -> sum(abs2, x .* s .+ s^2), (data(8, 6), 0.7)),
]

function finite_difference(f, args, k)
    x = args[k]
    if x isa Number
        e = 1.0e-6
        plus = Base.setindex(args, x + e, k)
        minus = Base.setindex(args, x - e, k)
        return (f(plus...) - f(minus...)) / 2e
    end
    g = similar(x)
    for i in eachindex(x)
        e = 1.0e-6 * max(1.0, abs(x[i]))
        xp = copy(x); xp[i] += e
        xm = copy(x); xm[i] -= e
        g[i] = (f(Base.setindex(args, xp, k)...) - f(Base.setindex(args, xm, k)...)) / 2e
    end
    g
end

@testset "program level" begin
    @testset "$name" for (name, f, args) in PROGRAMS
        p = trace(f, args...)
        @test interpret(p, args...) ≈ f(args...)
        d = differentiate(p)
        loss, gradients... = interpret(d, args...)
        @test loss ≈ f(args...)
        for k in eachindex(args)
            @test isapprox(gradients[k], finite_difference(f, args, k); rtol = 1.0e-5, atol = 1.0e-7)
        end
    end
    @test any(n -> n.op isa T.Map && n.op.f === leaky, trace(x -> leaky.(x), data(3)).nodes)
    @test !any(n -> n.op isa T.Map && n.op.f === swish, trace(x -> swish.(x), data(3)).nodes)
end
