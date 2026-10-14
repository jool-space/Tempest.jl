@testset "Built-in batched product and logsumexp" begin
    # Every transpose combination, through the prepared CPU schedule and the
    # interpreter, with gradients from the verb pullbacks.
    for (ta, tb) in ((false, false), (true, false), (false, true), (true, true))
        a = randn(ta ? (3, 4, 5) : (4, 3, 5))
        b = randn(tb ? (6, 3, 5) : (3, 6, 5))
        f(a, b) = sum(abs2, Tempest.batched_mul(a, b; transpose_a = ta, transpose_b = tb))
        compare_prepared(f, a, b)
        @test Tempest.batched_mul(a, b; transpose_a = ta, transpose_b = tb)[:, :, 2] ≈
              (ta ? a[:, :, 2]' : a[:, :, 2]) * (tb ? b[:, :, 2]' : b[:, :, 2])
    end
    @test_throws DimensionMismatch trace((a, b) -> Tempest.batched_mul(a, b), ones(2, 3, 4), ones(2, 3, 4))
    @test_throws DimensionMismatch trace((a, b) -> Tempest.batched_mul(a, b), ones(2, 3, 4), ones(3, 3, 5))
    @test_throws ArgumentError trace((a, b) -> Tempest.batched_mul(a, b), ones(2, 3), ones(3, 3))

    x = 30 .* randn(7, 9)
    compare_prepared(x -> sum(abs2, Tempest.logsumexp(x; dims = 1)), x)
    compare_prepared(x -> sum(abs2, exp.(x .- Tempest.logsumexp(x))), x)
    @test Tempest.logsumexp([1000.0, 1000.0]) ≈ [1000 + log(2)]
    @test Tempest.logsumexp([-Inf, -Inf]) == [-Inf]
    @test Tempest.logsumexp(reshape(1:6.0, 2, 3); dims = 2) ≈
          log.(sum(exp.(reshape(1:6.0, 2, 3)); dims = 2))

    # Large inputs stay finite through the CPU implementation.
    big = [800.0 0.0; 801.0 1.0]
    ex = prepare_forward(x -> Tempest.logsumexp(x), big)
    @test Array(execute!(ex)) ≈ [801 + log1p(exp(-1.0)) log1p(exp(1.0))]

end

@testset "Runtime integer indices" begin
    table = randn(5, 7)
    indices = Int32[3 1 3; 7 2 3]  # repeated indices accumulate
    f(t, i) = sum(abs2, t[:, i] .* 2)
    p = trace(f, table, indices)
    d = differentiate(p; wrt = (1,))
    loss, (dt,) = value_and_gradient(d, table, indices)
    @test loss ≈ f(table, indices)
    @test dt ≈ finite_difference(t -> f(t, indices), (table,), 1) atol = 1e-6
    ex = prepare(d, table, indices)
    @test execute!(ex)[1][] ≈ loss
    @test outputs(ex)[2] ≈ dt
    bind_argument!(ex, 2, Int32[1 1 1; 1 1 1])
    @test execute!(ex)[1][] ≈ f(table, Int32[1 1 1; 1 1 1])

    x = randn(4, 6)
    rows = Int32[1, 4, 2, 2, 3, 4]
    g(x, r) = sum(abs2, Tempest.logsumexp(x) .- Tempest.take_rows(x, r))
    q = differentiate(trace(g, x, rows); wrt = (1,))
    value, (dx,) = value_and_gradient(q, x, rows)
    @test dx ≈ finite_difference(x -> g(x, rows), (x,), 1) atol = 1e-6
    @test execute!(prepare(q, x, rows))[2] ≈ dx
    # Integer arguments carry a zero gradient when selected.
    @test all(iszero, value_and_gradient(differentiate(p), table, indices)[2][2])
    @test_throws ArgumentError trace(x -> sum(x), [true, false])
    @test_throws DimensionMismatch trace((x, r) -> Tempest.take_rows(x, r), x, Int32[1, 2])
end

@testset "Counter-based dropout" begin
    x = randn(Float32, 64, 128)
    seed = Int32[7]
    y = Tempest.dropout(x, seed; rate = 0.25)
    @test 0.22 < count(iszero, y) / length(y) < 0.28
    @test y == Tempest.dropout(x, seed; rate = 0.25)
    @test y != Tempest.dropout(x, Int32[8]; rate = 0.25)
    @test y != Tempest.dropout(x, seed; rate = 0.25, stream = 1)
    @test all((y .== 0) .| (y .≈ x ./ 0.75f0))
    f(x, s) = sum(abs2, Tempest.dropout(tanh.(x), s; rate = 0.3))
    d = differentiate(trace(f, x, seed); wrt = (1,))
    value, (dx,) = value_and_gradient(d, x, seed)
    @test value ≈ f(x, seed)
    @test dx ≈ finite_difference(x -> f(x, seed), (Float64.(x),), 1) rtol = 1e-3 atol = 1e-4
    @test execute!(prepare(d, x, seed))[2] ≈ dx
    @test_throws ArgumentError trace(x -> Tempest.dropout(x, seed; rate = 1), x)
end

@testset "Attention verb" begin
    q, k, v = (reshape(sin.(Float64.(1:(16 * 12 * 3)) .* s), 16, 12, 3) for s in (0.3, 0.7, 1.1))
    for causal in (false, true)
        f(q, k, v) = sum(abs2, Tempest.attention(q, k, v; causal) .* 0.5)
        d = differentiate(trace(f, q, k, v))
        value, gradients = value_and_gradient(d, q, k, v)
        @test value ≈ f(q, k, v)
        for i = 1:3
            @test gradients[i] ≈ finite_difference(f, (q, k, v), i) rtol = 1e-5 atol = 1e-7
        end
        @test execute!(prepare(d, q, k, v))[1][] ≈ value
    end
    # Without masking every key is visible; with it, the first query sees one key.
    o = Tempest.attention(q, k, v; causal = true)
    @test o[:, 1, :] ≈ v[:, 1, :]
    @test Tempest.causal_fill(ones(3, 3); value = 0) == [1 1 1; 0 1 1; 0 0 1]
    @test_throws DimensionMismatch trace((q, k) -> Tempest.attention(q, k, k; causal = true), q, k[:, 1:5, :])
end
