# cuTile's fused attention serves TF32 products; full precision stays composed.
@testset "Fused attention under TF32, composed attention in full precision" begin
    # Four query heads over two key/value heads, and the three-dimensional form.
    q = reshape(Float32.(sin.((1:(64 * 4 * 128 * 3)) .* 0.3)), 64, 4, 128, 3)
    k, v = (reshape(Float32.(sin.((1:(64 * 2 * 128 * 3)) .* s)), 64, 2, 128, 3) for s in (0.7, 1.1))
    g(q, k, v) = sum(abs2, Tempest.attention(q, k, v; causal = true))
    d = differentiate(trace(g, q, k, v))
    expected = execute!(prepare(d, q, k, v))
    # The CPU sums long reductions sequentially in Float32; compare the loss
    # with a Float64 evaluation instead.
    loss = g(Float64.(q), Float64.(k), Float64.(v))
    for precision in (:full, :tf32)
        ex = prepare(d, q, k, v; target = CUDATarget(matmul_precision = precision))
        got = execute!(ex)
        fused = occursin("call attention_forward [cutile]", sprint(explain, ex))
        @test fused == (precision === :tf32)
        tolerance = precision === :tf32 ? 5e-3 : 1e-4
        @test Array(got[1])[] ≈ loss rtol = tolerance
        for i = 1:3
            @test Array(got[i + 1]) ≈ expected[i + 1] rtol = 10tolerance atol = 10tolerance
        end
        # Deterministic: no atomics in either backward.
        @test Array(execute!(ex)[2]) == Array(execute!(ex)[2])
    end
    # Unsupported shapes fall back to the composed form.
    short = map(x -> x[:, 1, 1:50, :], (q, k, v))
    ex = prepare(differentiate(trace(g, short...)), short...; target = CUDATarget(matmul_precision = :tf32))
    @test !occursin("call attention_forward [cutile]", sprint(explain, ex))
end
