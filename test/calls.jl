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

    T.register!(@__MODULE__, HostAttention())
    try
        ex = prepare(p, q, k, v, x)
        g = only(opaque_groups(ex))
        o = only(keys(g.epilogues))
        @test !ex.plan.stored[o] # the residual is written in the output's place
        @test agree(results(ex), [residual(q, k, v, x)])

        # A result nobody reads is not stored.
        ex = prepare(trace((q, k, v) -> T.attention(q, k, v; causal = true), q, k, v), q, k, v)
        @test count(v -> ex.plan.stored[v], only(opaque_groups(ex)).outputs) == 1
        @test agree(results(ex), [T.attention(q, k, v; causal = true)])

        # A permuted read does not align with the output tile: no epilogue.
        ex = prepare(trace(permuted, q, k, v, y), q, k, v, y)
        @test isempty(only(opaque_groups(ex)).epilogues)
        @test agree(results(ex), [permuted(q, k, v, y)])

        # Training claims the backward verb too.
        d = differentiate(p)
        @test length(calls(d)) == 5 # o, lse; dq, dk, dv
        ex = prepare(d, q, k, v, x)
        @test length(opaque_groups(ex)) == 2
        # The zero cotangent of the log-sum-exp is one buffer for all three results.
        @test count(v -> v.definition isa T.Pointwise && all(o -> o isa T.Literal, v.definition.operands), ex.plan.graph.values) == 1
        @test agree(results(ex), filter(!isnothing, T.leaves(interpret(d, q, k, v, x))))
    finally
        T.register!(@__MODULE__)
    end
end

using CUDACore, cuTile
if CUDACore.functional()
    @testset "cuTile attention claims" begin
        single(dims...; scale) = Float32.(data(dims...; scale))
        q, k, v = single(16, 4, 64, 2; scale = 0.1), single(16, 2, 64, 2; scale = 0.2), single(16, 2, 64, 2; scale = 0.3)
        x = single(16, 4, 64, 2; scale = 0.5)
        device(xs) = map(CuArray, xs)
        # Untraced, the verbs run their references on device arrays.
        @test Array(T.attention(device((q, k, v))...; causal = true)) ≈ T.attention(q, k, v; causal = true)
        seed = Int32[7]
        @test Array(T.dropout(CuArray(x), CuArray(seed); rate = 0.25)) == T.dropout(x, seed; rate = 0.25)
        close(a, b; rtol) = all(map((x, y) -> isapprox(ndims(x) == 0 ? Array(x)[] : Array(x), y; rtol), a, b))
        # Full precision keeps the reference; TF32 takes the kernels.
        p = trace(residual, q, k, v, x)
        ex = prepare(p, device((q, k, v, x))...)
        @test isempty(opaque_groups(ex))
        @test close(filter(!isnothing, T.leaves(execute!(ex))), [interpret(p, q, k, v, x)]; rtol = 1.0e-4)
        for (target, rtol) in ((T.CUDATarget(precision = :tf32), 5.0e-3),)
            p = trace(residual, q, k, v, x)
            ex = prepare(p, device((q, k, v, x))...; target)
            g = only(opaque_groups(ex))
            @test length(g.epilogues) == 1
            @test any(d -> d isa Base.get_extension(Tempest, :cuTileExt).BuiltAttention, ex.kernels)
            @test close(filter(!isnothing, T.leaves(execute!(ex))), [interpret(p, q, k, v, x)]; rtol)
            # The kernel leaves the log-sum-exp unwritten when nobody reads it.
            f = trace((q, k, v) -> T.attention(q, k, v; causal = true), q, k, v)
            ex = prepare(f, device((q, k, v))...; target)
            @test count(v -> ex.plan.stored[v], only(opaque_groups(ex)).outputs) == 1
            @test close(filter(!isnothing, T.leaves(execute!(ex))), [interpret(f, q, k, v)]; rtol)
            d = differentiate(p)
            ex = prepare(d, device((q, k, v, x))...; target)
            @test length(opaque_groups(ex)) == 2
            @test close(filter(!isnothing, T.leaves(execute!(ex))), filter(!isnothing, T.leaves(interpret(d, q, k, v, x))); rtol)
        end
    end
end
