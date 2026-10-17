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
