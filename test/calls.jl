# Calls: recorded per result, evaluated through the reference, expanded when
# nobody claims them, and, when claimed, one opaque group that may absorb the
# pointwise work after a result as an epilogue.
using Test
include("cases.jl")

# A claim on the CPU that realizes attention through its reference on the host.
struct HostAttention <: T.Implementation end
T.claims_call(::HostAttention, ::T.CPUTarget, f, kwargs, inputs) = f in (T.attention_forward, T.attention_backward)
T.epilogue_results(::HostAttention, ::typeof(T.attention_forward), kwargs) = (1,)
T.build_call(::HostAttention, p, g, buffers, target) = T.opaque_launch(p, g, buffers)

q, k, v = data(4, 4, 6, 2; scale = 0.3), data(4, 2, 6, 2; scale = 0.2), data(4, 2, 6, 2; scale = 0.1)
x = data(4, 4, 6, 2; scale = 0.5)
y = data(4, 6, 4, 2; scale = 0.7)
residual(q, k, v, x) = sum(abs2, T.attention(q, k, v; causal = true) .+ x)
permuted(q, k, v, y) = sum(abs2, permutedims(T.attention(q, k, v; causal = true), (1, 3, 2, 4)) .+ y)

calls(p) = [n for n in p.nodes if n.op isa T.Call]
opaque_groups(ex) = [g for g in ex.plan.groups if g.kind === :opaque]
agree(a, b) = all(map((x, y) -> isapprox(x, y; rtol = 1.0e-10, atol = 1.0e-12), a, b))
results(ex) = map(r -> ndims(r) == 0 ? r[] : r, filter(!isnothing, T.leaves(execute!(ex))))

@testset "calls" begin
    p = trace(residual, q, k, v, x)
    @test length(calls(p)) == 2 # attention_forward: o and lse
    @test interpret(p, q, k, v, x) ≈ residual(q, k, v, x)
    @test !any(v -> v.definition isa T.Opaque, T.lower(p).values)

    push!(T.EXTRA_IMPLEMENTATIONS, HostAttention())
    try
        ex = prepare(p, q, k, v, x)
        g = only(opaque_groups(ex))
        o = only(keys(g.epilogues))
        @test !ex.plan.stored[o] # the residual is written in the output's place
        @test agree(results(ex), [residual(q, k, v, x)])

        # A permuted read does not align with the output tile: no epilogue.
        ex = prepare(trace(permuted, q, k, v, y), q, k, v, y)
        @test isempty(only(opaque_groups(ex)).epilogues)
        @test agree(results(ex), [permuted(q, k, v, y)])

        # Training claims the backward verb too.
        d = differentiate(p)
        @test length(calls(d)) == 5 # o, lse; dq, dk, dv
        ex = prepare(d, q, k, v, x)
        @test length(opaque_groups(ex)) == 2
        @test agree(results(ex), filter(!isnothing, T.leaves(interpret(d, q, k, v, x))))
    finally
        filter!(i -> !(i isa HostAttention), T.EXTRA_IMPLEMENTATIONS)
    end
end
