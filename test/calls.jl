# Operations: functions that keep their names under tracing. A call is
# recorded per result and evaluated through its reference (an opaque one
# through its own method); at the index level it is its reference expanded,
# and an implementation's claim on it, one opaque group that may absorb the
# pointwise work after a result as an epilogue, competes with that expansion
# by cost.
using Test
include("cases.jl")

# A claim on the CPU that realizes attention through its reference on the host.
struct HostAttention <: T.Implementation end
T.claims_call(::HostAttention, ::T.CPUTarget, f, kwargs, inputs) = f in (T.attention_forward, T.attention_backward)
T.epilogue_results(::HostAttention, ::typeof(T.attention_forward), kwargs) = (1,)
T.build_call(::HostAttention, p, g, buffers, target) = T.opaque_launch(p, g, buffers)

# A claim on attention estimated far slower than its reference.
struct SlowAttention <: T.Implementation end
T.claims_call(::SlowAttention, ::T.CPUTarget, f, kwargs, inputs) = f === T.attention_forward
T.estimate(::SlowAttention, claim, target, p, g) = [T.Work(1.0e12)]
T.build_call(::SlowAttention, p, g, buffers, target) = T.opaque_launch(p, g, buffers)

q, k, v = data(4, 4, 6, 2; scale = 0.3), data(4, 2, 6, 2; scale = 0.2), data(4, 2, 6, 2; scale = 0.1)
x = data(4, 4, 6, 2; scale = 0.5)
y = data(4, 6, 4, 2; scale = 0.7)
residual(q, k, v, x) = sum(abs2, T.attention(q, k, v; causal = true) .+ x)
permuted(q, k, v, y) = sum(abs2, permutedims(T.attention(q, k, v; causal = true), (1, 3, 2, 4)) .+ y)

calls(p) = [n for n in p.nodes if n.op isa T.Call]
opaque_groups(ex) = [g for g in ex.plan.groups if g.kind === :opaque]
agree(a, b) = all(map((x, y) -> isapprox(x, y; rtol = 1.0e-10, atol = 1.0e-12), a, b))
results(ex) = map(r -> ndims(r) == 0 ? r[] : r, filter(!isnothing, T.leaves(execute!(ex))))
gradients(p, args...) = filter(!isnothing, T.leaves(interpret(differentiate(p), args...)))
with_implementations(f, implementations...) = (T.register!(@__MODULE__, implementations...); try f() finally T.register!(@__MODULE__) end)
message(f) = try f(); "" catch e; sprint(showerror, e) end

@testset "calls" begin
    p = trace(residual, q, k, v, x)
    @test length(calls(p)) == 2 # attention_forward: o and lse
    @test interpret(p, q, k, v, x) ≈ residual(q, k, v, x)
    @test !any(v -> v.definition isa T.Opaque, T.lower(p).values)

    with_implementations(HostAttention()) do
        ex = prepare(p, q, k, v, x)
        g = only(opaque_groups(ex))
        o = only(keys(g.epilogues))
        @test !ex.plan.stored[o] # the residual is written in the output's place
        @test agree(results(ex), [residual(q, k, v, x)])
        @test occursin("call of attention_forward (%", sprint(explain, ex))
        @test occursin("HostAttention's claim with an epilogue replaces its reference expanded", sprint(explain, ex))

        # A result nobody reads is not stored.
        ex = prepare(trace((q, k, v) -> T.attention(q, k, v; causal = true), q, k, v), q, k, v)
        @test count(v -> ex.plan.stored[v], only(opaque_groups(ex)).outputs) == 1
        @test agree(results(ex), [T.attention(q, k, v; causal = true)])

        # A permuted read does not align with the output tile: no epilogue.
        ex = prepare(trace(permuted, q, k, v, y), q, k, v, y)
        @test isempty(only(opaque_groups(ex)).epilogues)
        @test agree(results(ex), [permuted(q, k, v, y)])

        # Training claims the backward operation too.
        d = differentiate(p)
        @test length(calls(d)) == 5 # o, lse; dq, dk, dv
        ex = prepare(d, q, k, v, x)
        @test length(opaque_groups(ex)) == 2
        # The zero cotangent of the log-sum-exp is one buffer for all three results.
        @test count(v -> v.definition isa T.Pointwise && all(o -> o isa T.Literal, v.definition.operands), ex.plan.graph.values) == 1
        @test agree(results(ex), filter(!isnothing, T.leaves(interpret(d, q, k, v, x))))
    end

    # A claim estimated slower than the reference fused into its neighbours
    # loses to it, and `explain` says so.
    with_implementations(SlowAttention()) do
        ex = prepare(p, q, k, v, x)
        @test isempty(opaque_groups(ex))
        @test occursin("its reference expanded is kept over SlowAttention's claim", sprint(explain, ex))
        @test agree(results(ex), [residual(q, k, v, x)])
    end
end

# An operation from a generic function: its method is its reference.
rms(x, w; eps = 1.0e-5) = x ./ sqrt.(sum(abs2, x; dims = 1) ./ size(x, 1) .+ eps) .* w
@operation rms(x::AbstractMatrix, w::AbstractVector; eps = 1.0e-5)

@testset "operations" begin
    a, w = data(8, 6), data(8; scale = 0.1)
    p = trace(rms, a, w)
    @test only(calls(p)).op.f === rms
    @test interpret(p, a, w) ≈ rms(a, w)
    ex = prepare(p, a, w)
    @test length(ex.plan.groups) == 1
    @test agree(results(ex), [rms(a, w)])
    # A Julia array among the arguments is a constant; keywords are part of
    # the call.
    p = trace(a -> rms(a, w; eps = 0.1), a)
    @test only(calls(p)).op.kwargs == (eps = 0.1,)
    @test interpret(p, a) ≈ rms(a, w; eps = 0.1)
    p = trace(w -> rms(a, w), w)
    @test only(calls(p)).op.f === rms && interpret(p, w) ≈ rms(a, w)
    # Gradients go through the reference.
    @test all(map(≈, gradients(trace((a, w) -> sum(abs2, rms(a, w)), a, w), a, w),
        gradients(trace((a, w) -> sum(abs2, rmsnorm(a, w)), a, w), a, w)))
    # The methods on traced arrays add no ambiguity.
    for f in (rms, T.attention_forward, T.attention_backward, T.dropout)
        ms = collect(methods(f))
        @test !any(Base.isambiguous(m, n) for m in ms for n in ms if m !== n)
    end
end

# An opaque operation: its body loops over scalars, and it declares its
# results' shapes and types.
function running_sum(x::AbstractVector)
    y, total = similar(x), zero(eltype(x))
    for i in eachindex(x)
        total += x[i]
        y[i] = total
    end
    y
end
@operation running_sum(x::AbstractVector)
T.declared_results(::typeof(running_sum), ((dims, T),)) = (dims, T)

# A CPU implementation of it, which applies the work after it as an epilogue.
struct HostRunningSum <: T.Implementation end
T.claims_call(::HostRunningSum, ::T.CPUTarget, f, kwargs, inputs) = f === running_sum
T.epilogue_results(::HostRunningSum, ::typeof(running_sum), kwargs) = (1,)
T.build_call(::HostRunningSum, p, g, buffers, target) = T.opaque_launch(p, g, buffers)

prefix(x) = running_sum(x .* 2) .+ 1
prefix_loss(x) = sum(abs2, prefix(x))

@testset "opaque operations" begin
    a = data(12)
    p = trace(prefix_loss, a)
    @test only(calls(p)).op.f === running_sum
    @test interpret(p, a) ≈ prefix_loss(a) # through its own method
    # Without an implementation, preparation fails and names it; without a
    # rule, so does differentiation.
    @test occursin(r"no implementation on .*CPUTarget\(\) runs running_sum", message(() -> prepare(p, a)))
    @test occursin("running_sum is an opaque operation without a derivative rule", message(() -> differentiate(p)))
    with_implementations(HostRunningSum()) do
        ex = prepare(p, a)
        @test length(opaque_groups(ex)) == 1
        @test agree(results(ex), [prefix_loss(a)])
        # The + 1 is written in the result's place, which saves a kernel.
        ex = prepare(trace(prefix, a), a)
        @test length(only(opaque_groups(ex)).epilogues) == 1 && length(ex.kernels) == 2
        @test occursin("HostRunningSum's claim with an epilogue replaces its claim without an epilogue", sprint(explain, ex))
        @test agree(results(ex), [prefix(a)])
    end
end

# With a rule, it differentiates: each element's cotangent is the sum of the
# cotangents from it on.
T.has_pullback(::typeof(running_sum)) = true
T.verb_pullback(::typeof(running_sum), (x,), y, dy) = (sum(dy) .- running_sum(dy) .+ dy,)

@testset "opaque operations with a rule" begin
    a = data(12)
    d = differentiate(trace(prefix_loss, a))
    g = 2 .* (cumsum(2 .* a) .+ 1)
    @test last(interpret(d, a)) ≈ 2 .* reverse(cumsum(reverse(g)))
    with_implementations(HostRunningSum()) do
        @test agree(results(prepare(d, a)), filter(!isnothing, T.leaves(interpret(d, a))))
    end
end

using CUDACore, cuTile
if CUDACore.functional()
    close(a, b; rtol) = all(map((x, y) -> isapprox(ndims(x) == 0 ? Array(x)[] : Array(x), y; rtol), a, b))
    @testset "cuTile attention claims" begin
        single(dims...; scale) = Float32.(data(dims...; scale))
        q, k, v = single(16, 4, 64, 2; scale = 0.1), single(16, 2, 64, 2; scale = 0.2), single(16, 2, 64, 2; scale = 0.3)
        x = single(16, 4, 64, 2; scale = 0.5)
        device(xs) = map(CuArray, xs)
        # Untraced, the operations run their own methods on device arrays.
        @test Array(T.attention(device((q, k, v))...; causal = true)) ≈ T.attention(q, k, v; causal = true)
        seed = Int32[7]
        @test Array(T.dropout(CuArray(x), CuArray(seed); rate = 0.25)) == T.dropout(x, seed; rate = 0.25)
        # Full precision keeps the reference.
        p = trace(residual, q, k, v, x)
        ex = prepare(p, device((q, k, v, x))...)
        @test isempty(opaque_groups(ex))
        @test close(filter(!isnothing, T.leaves(execute!(ex))), [interpret(p, q, k, v, x)]; rtol = 1.0e-4)
        # Under TF32 the hand-written kernels claim attention, and compete with
        # the attention generated from the reference. In training they win by
        # their estimates.
        target, rtol = T.CUDATarget(precision = :tf32), 5.0e-3
        d = differentiate(p)
        ex = prepare(d, device((q, k, v, x))...; target)
        @test length(opaque_groups(ex)) == 2
        # The log-sum-exp's cotangent, a constant, is generated too.
        @test !any(d -> d isa T.FloorKernels, ex.kernels)
        @test close(filter(!isnothing, T.leaves(execute!(ex))), filter(!isnothing, T.leaves(interpret(d, q, k, v, x))); rtol)
        # The forward pass alone is a close call, measured: whichever the
        # tuning record says is faster runs.
        f = trace((q, k, v) -> T.attention(q, k, v; causal = true), q, k, v)
        prepare(p, device((q, k, v, x))...; target)
        prepare(f, device((q, k, v))...; target)
        path = T.record_path(target)
        lines = readlines(path)
        rewrite(t) = write(path, join((string(key, '\t', occursin("cuTileAttention", key) ? t : 1.0e-5) for (key, _) in split.(lines, '\t')), '\n') * '\n')
        try
            for (t, claimed) in ((1.0e-7, true), (1.0e-3, false))
                rewrite(t)
                ex = prepare(p, device((q, k, v, x))...; target)
                @test isempty(opaque_groups(ex)) != claimed
                @test occursin(claimed ? "cuTileAttention's claim with an epilogue replaces its reference expanded" :
                    "its reference expanded is kept over cuTileAttention's claim", sprint(explain, ex))
                if claimed
                    @test length(only(opaque_groups(ex)).epilogues) == 1
                    @test any(d -> d isa Base.get_extension(Tempest, :cuTileExt).BuiltAttention, ex.kernels)
                end
                @test close(filter(!isnothing, T.leaves(execute!(ex))), [interpret(p, q, k, v, x)]; rtol)
                # The kernel leaves the log-sum-exp unwritten when nobody reads it.
                ex = prepare(f, device((q, k, v))...; target)
                @test isempty(opaque_groups(ex)) != claimed
                claimed && @test count(v -> ex.plan.stored[v], only(opaque_groups(ex)).outputs) == 1
                @test close(filter(!isnothing, T.leaves(execute!(ex))), [interpret(f, q, k, v)]; rtol)
            end
        finally
            write(path, join(lines, '\n') * '\n')
        end
    end
end
