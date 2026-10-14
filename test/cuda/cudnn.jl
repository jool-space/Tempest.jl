using Tempest: attention, attention_forward
using LinearAlgebra: norm

# Deterministic (D, heads, positions, batch) operands; keys and values have kv heads.
function attention_operands(D, heads, kv, positions, batch)
    shaped(s, h) = reshape(Float32.(sin.((1:(D * h * positions * batch)) .* s)), D, h, positions, batch)
    (shaped(0.3, heads), shaped(0.7, kv), shaped(1.1, kv))
end

relative_error(a, b) = norm(Float64.(Array(a)) .- b) / norm(b)

# The Float64 CPU reference of a loss and its gradients.
function reference_gradients(loss, arrays...)
    wide = map(x -> Float64.(x), arrays)
    value_and_gradient(differentiate(trace(loss, wide...)), wide...)
end

explanation(ex) = sprint(explain, ex)

@testset "cuDNN attention on half-precision operands" begin
    @test_throws ArgumentError CuDNNKernels(float32_cast = Float64)
    for T in (Float16, Core.BFloat16), causal in (false, true), (heads, kv) in ((4, 2), (2, 2))
        tolerance = T === Float16 ? 2e-3 : 2e-2
        q, k, v = map(x -> T.(x), attention_operands(64, heads, kv, 96, 2))
        forward(q, k, v) = attention_forward(q, k, v; causal, scale = 0.125f0)
        ex = prepare_forward(forward, CuArray(q), CuArray(k), CuArray(v); implementations = (CuDNNKernels(),))
        @test ex.providers == [:cuDNN]
        y = execute!(ex)
        expected = forward(Float64.(q), Float64.(k), Float64.(v))
        @test eltype(y.o) === eltype(y.lse) === T
        @test relative_error(y.o, expected.o) < tolerance
        # cuDNN's statistic is Tempest's natural-log log-sum-exp of the scaled scores.
        @test relative_error(y.lse, expected.lse) < tolerance

        loss(q, k, v) = sum(abs2, attention(q, k, v; causal))
        ex = prepare(loss, CuArray(q), CuArray(k), CuArray(v); implementations = (CuDNNKernels(),))
        text = explanation(ex)
        @test occursin("semantic attention_forward: reverse rule cuDNN", text)
        @test occursin("call sdpa_forward [cuDNN]", text)
        @test occursin("call sdpa_backward [cuDNN]", text)
        got = execute!(ex)
        value, gradients = reference_gradients(loss, q, k, v)
        @test Float64(Array(got[1])[]) ≈ value rtol = tolerance
        for i = 1:3
            @test relative_error(got[i + 1], gradients[i]) < 2tolerance
        end
    end
end

@testset "cuDNN attention with explicitly rounded Float32 operands" begin
    implementation = CuDNNKernels(float32_cast = Core.BFloat16)
    for causal in (false, true)
        q, k, v = attention_operands(64, 4, 2, 128, 2)
        # Compare with the reference on the rounded operands: the rounding is
        # the requested approximation, not the implementation's error.
        rounded = map(x -> Float32.(Core.BFloat16.(x)), (q, k, v))
        forward(q, k, v) = attention_forward(q, k, v; causal, scale = 0.125f0)
        ex = prepare_forward(forward, CuArray(q), CuArray(k), CuArray(v); implementations = (implementation,))
        @test ex.providers == [:cuDNN]
        @test occursin("Float32 operands rounded to BFloat16", explanation(ex))
        y = execute!(ex)
        expected = forward(map(x -> Float64.(x), rounded)...)
        @test eltype(y.o) === eltype(y.lse) === Float32
        # The probabilities enter the second product in BFloat16, too.
        @test relative_error(y.o, expected.o) < 2e-2
        @test relative_error(y.lse, expected.lse) < 1e-5

        loss(q, k, v) = sum(abs2, attention(q, k, v; causal) .* 0.5f0)
        ex = prepare(loss, CuArray(q), CuArray(k), CuArray(v); implementations = (implementation,))
        @test count(==(:cuDNN), ex.providers) == 2
        got = execute!(ex)
        value, gradients = reference_gradients(loss, rounded...)
        @test Array(got[1])[] ≈ value rtol = 1e-2
        for i = 1:3
            @test relative_error(got[i + 1], gradients[i]) < 3e-2
        end
        # The backward engine is deterministic.
        @test Array(execute!(ex)[2]) == Array(execute!(ex)[2])
    end
end

@testset "cuDNN attention against finite differences" begin
    implementation = CuDNNKernels(float32_cast = Float16)
    for causal in (false, true)
        q, k, v = attention_operands(8, 2, 1, 5, 2)
        loss(q, k, v) = sum(abs2, attention(q, k, v; causal) .- 0.25)
        ex = prepare(loss, CuArray(q), CuArray(k), CuArray(v); implementations = (implementation,))
        @test occursin("call sdpa_backward [cuDNN]", explanation(ex))
        got = execute!(ex)
        wide = map(x -> Float64.(x), (q, k, v))
        @test Array(got[1])[] ≈ loss(wide...) rtol = 2e-3
        for i = 1:3
            @test relative_error(got[i + 1], finite_difference(loss, wide, i)) < 5e-3
        end
    end
end

@testset "cuDNN attention capture and replay with changed inputs" begin
    q, k, v = attention_operands(64, 4, 1, 64, 2)
    loss(q, k, v) = sum(abs2, attention(q, k, v; causal = true))
    ex = prepare(loss, CuArray(q), CuArray(k), CuArray(v); implementations = (CuDNNKernels(float32_cast = Float16),))
    pointers = map(pointer, ex.buffers)
    capture!(ex)
    for scale in (0.5f0, 1.3f0)
        bind!(ex, scale .* q, k, scale .* v)
        GC.gc()
        got = replay!(ex)
        value, gradients = reference_gradients(loss, scale .* q, k, scale .* v)
        @test Array(got[1])[] ≈ value rtol = 2e-3
        for i = 1:3
            @test relative_error(got[i + 1], gradients[i]) < 1e-2
        end
        @test map(pointer, ex.buffers) == pointers
    end
    @test occursin("(captured)", explanation(ex))
end

@testset "cuDNN attention falls back with reasons" begin
    tiled = Base.get_extension(Tempest, :cuTileExt) !== nothing
    q, k, v = attention_operands(64, 2, 2, 64, 1)
    loss(q, k, v) = sum(abs2, attention(q, k, v; causal = true))
    gradients = reference_gradients(loss, q, k, v)[2]
    check(ex) = all(relative_error(execute!(ex)[i + 1], gradients[i]) < 1e-2 for i = 1:3)

    # Float32 without an explicit rounding: cuTile under TF32, else the composed form.
    for precision in (:full, :tf32)
        target = CUDATarget(matmul_precision = precision)
        ex = prepare(loss, CuArray(q), CuArray(k), CuArray(v); target, implementations = (CuDNNKernels(),))
        text = explanation(ex)
        @test :cuDNN ∉ ex.providers
        @test occursin("cuDNN has no Float32 attention engine", text)
        @test occursin("call attention_forward [cutile]", text) == (tiled && precision === :tf32)
        @test check(ex)
    end

    # Float64 is never rounded.
    doubles = map(x -> CuArray(Float64.(x)), (q, k, v))
    ex = prepare(loss, doubles...; implementations = (CuDNNKernels(float32_cast = Float16),))
    @test :cuDNN ∉ ex.providers
    @test occursin("not Float64", explanation(ex))

    # Twelve features: no cuDNN layout; the reference runs.
    narrow = attention_operands(12, 2, 2, 16, 1)
    ex = prepare(loss, CuArray.(narrow)...; implementations = (CuDNNKernels(float32_cast = Float16),))
    @test :cuDNN ∉ ex.providers
    @test occursin("multiple of 8 features", explanation(ex))
    expected = reference_gradients(loss, narrow...)[2]
    @test all(relative_error(execute!(ex)[i + 1], expected[i]) < 1e-4 for i = 1:3)

    # 160 features: on GB10 the forward builds but the backward has no engine,
    # so the pair is rejected and the public forward pairs with the reference
    # pullback. Other devices may build both.
    wide = attention_operands(160, 2, 2, 64, 1)
    ex = prepare(loss, CuArray.(wide)...; implementations = (CuDNNKernels(float32_cast = Float16),))
    text = explanation(ex)
    if occursin("call sdpa_backward [cuDNN]", text)
        @test occursin("call sdpa_forward [cuDNN]", text)
    else
        @test occursin("rejected as a pair: cuDNN: cuDNN has no engine for attention backward", text)
        @test occursin("call attention_forward [cuDNN]", text)
    end
    expected = reference_gradients(loss, wide...)[2]
    @test all(relative_error(execute!(ex)[i + 1], expected[i]) < 1e-2 for i = 1:3)
end

@testset "cuDNN attention with a used log-sum-exp" begin
    # cuDNN's backward has no input for the lse cotangent: the rule hands it to
    # the public backward, which cuTile serves under TF32. Without one, the
    # pair is rejected and the cuDNN forward pairs with the reference pullback.
    tiled = Base.get_extension(Tempest, :cuTileExt) !== nothing
    q, k, v = attention_operands(64, 2, 1, 128, 1)
    function loss(q, k, v)
        y = attention_forward(q, k, v; causal = false, scale = 0.125f0)
        sum(abs2, y.o) + sum(sin.(y.lse))
    end
    value, gradients = reference_gradients(loss, q, k, v)
    for precision in (:full, :tf32)
        target = CUDATarget(matmul_precision = precision)
        ex = prepare(loss, CuArray(q), CuArray(k), CuArray(v); target, implementations = (CuDNNKernels(float32_cast = Float16),))
        text = explanation(ex)
        if tiled && precision === :tf32
            @test occursin("call sdpa_forward [cuDNN]", text)
            @test occursin("call attention_backward [cutile]", text)
        else
            @test occursin("rejected as a pair", text)
            @test occursin("call attention_forward [cuDNN]", text)
        end
        got = execute!(ex)
        @test Array(got[1])[] ≈ value rtol = 2e-3
        @test all(relative_error(got[i + 1], gradients[i]) < 1e-2 for i = 1:3)
    end
end
