# The program level: traced programs agree with plain Julia, and gradients of
# differentiated programs agree with finite differences.
using Test
include("cases.jl")

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
        # Every node is read by a result, or is a call's result beside one.
        read = T.live(d.nodes, d.results)
        @test all(id -> read[id] || id in d.arguments || d.nodes[id].op isa T.Call, eachindex(d.nodes))
        loss, gradients... = interpret(d, args...)
        @test loss ≈ f(args...)
        for k in eachindex(args)
            args[k] isa Union{AbstractFloat, AbstractArray{<:AbstractFloat}} || continue
            @test isapprox(gradients[k], finite_difference(f, args, k); rtol = 1.0e-5, atol = 1.0e-7)
        end
    end
    @test any(n -> n.op isa T.Map && n.op.f === leaky, trace(x -> leaky.(x), data(3)).nodes)
    @test !any(n -> n.op isa T.Map && n.op.f === swish, trace(x -> swish.(x), data(3)).nodes)
end

# A gradient in traced code is the differentiated program inlined where it is
# called; what the function closes over is a constant to it, and an argument
# passed twice gets a gradient for each place.
@testset "gradients in traced code" begin
    same(a, b) = a === nothing ? b === nothing : isapprox(a, b; rtol = 1.0e-12)
    @testset "$name" for (name, f, args) in PROGRAMS
        expected = T.leaves(interpret(differentiate(trace(f, args...)), args...))
        @test all(map(same, expected, T.leaves(interpret(trace((xs...) -> T.gradient(f, xs...), args...), args...))))
        @test all(map(same, expected, T.leaves(T.gradient(f, args...))))
    end
    x, y = data(8, 6), data(8, 6; scale = 0.3)
    loss, dx = interpret(trace((x, y) -> T.gradient(a -> sum(a .* y .* x), x), x, y), x, y)
    @test loss ≈ sum(x .* y .* x) && dx ≈ y .* x
    loss, da, db = interpret(trace(x -> T.gradient((a, b) -> sum(a .* b .* b), x, x), x), x)
    @test da ≈ x .* x && db ≈ 2 .* x .* x
    # Structured arguments, and Julia arrays among them.
    w, b = data(8, 8; scale = 0.2), data(8)
    loss, g = interpret(trace((w, x) -> T.gradient(q -> sum(abs2, q.w * x .+ q.b), (; w, b)), w, x), w, x)
    @test g.w ≈ 2 .* (w * x .+ b) * x' && g.b ≈ vec(sum(2 .* (w * x .+ b); dims = 2))
end
