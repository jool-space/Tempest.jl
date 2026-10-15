include("../fixtures/folds.jl")

@testset "Fold reference and CPU execution" begin
    x = 50 .* reshape(sin.(1.0:35.0) .+ cos.(0.3 .* (1:35)), 7, 5)
    for dims in (1, 2, (1, 2))
        @test Tempest.fold(LogSumExpFold(), x; dims) ≈ lse_reference(x; dims)
        @test Tempest.fold(MaxFold(), x; dims) == maximum(x; dims)
        @test Tempest.fold(WelfordVariance(), x; dims) ≈ variance_reference(x; dims)
        @test Tempest.fold(range_fold, x; dims) == range_reference(x; dims)
        f(x) = (
            lse = Tempest.fold(LogSumExpFold(), x; dims),
            max = maximum(x; dims),
            min = minimum(abs2, x; dims),
            var = Tempest.fold(WelfordVariance(), x; dims),
            range = Tempest.fold(range_fold, x; dims),
            tempered = Tempest.fold(TemperedLogSumExp(2.0f0), x; dims),
        )
        got = execute!(prepare_forward(f, x))
        @test got.lse ≈ lse_reference(x; dims)
        @test got.max == maximum(x; dims)
        @test got.min == minimum(abs2, x; dims)
        @test got.var ≈ variance_reference(x; dims)
        @test got.range == range_reference(x; dims)
        @test got.tempered ≈ tempered_reference(x, 2; dims)
    end
    ex = prepare_forward(x -> Tempest.fold(LogSumExpFold(), x), x)
    @test only(execute!(ex)) ≈ only(lse_reference(x; dims = (1, 2)))
    @test occursin("logsumexp [cpu]", sprint(explain, ex))
    # dims = : gives a number, (1, 2) a 1×1 array, as for sum.
    @test Tempest.logsumexp(x; dims = :) ≈ only(lse_reference(x; dims = (1, 2)))
    @test Tempest.logsumexp(x; dims = (1, 2)) ≈ lse_reference(x; dims = (1, 2))
    p = trace(x -> Tempest.logsumexp(x; dims = :), x)
    @test p.nodes[p.output].scalar
    @test value_and_gradient(differentiate(p), x)[2][1] ≈ exp.(x .- Tempest.logsumexp(x; dims = :))
    @test trace(x -> Tempest.fold(MaxFold(), x; dims = (1, 2)), x).nodes[end].dims == (1, 1)
    @test Tempest.fold(LogSumExpFold(), [-Inf, -Inf]) == -Inf
    @test Tempest.fold(LogSumExpFold(), [1.0, Inf]) == Inf
    @test Tempest.fold(LogSumExpFold(), [1000.0, 1000.0]) ≈ 1000 + log(2)
    # Integer extrema start from typemin/typemax, which equal ∓Inf for floats.
    indices = Int32[3 1; 4 1; 5 9]
    for (f, g) in ((maximum, MaxFold()), (minimum, MinFold())), dims in (:, (1,), (2,))
        p = trace(i -> f(i; dims), indices)
        @test only(Tempest.interpret(Tempest.ForwardProgram(p), indices)) == f(indices; dims)
        @test Tempest.fold(g, indices; dims) == f(indices; dims)
    end
    # logsumexp is the stable fold itself, not a call.
    big = 1000 .* x
    p = trace(x -> Tempest.logsumexp(x; dims = 2), big)
    @test only(Tempest.interpret(Tempest.ForwardProgram(p), big)) ≈ lse_reference(big; dims = 2)
    @test any(node -> node.op isa Tempest.FoldReduction{LogSumExpFold}, p.nodes)
    @test !any(node -> node.op isa Tempest.CallResult, p.nodes)
    # Fold functions must keep the state type of fold_init.
    widening = GenericFold(; init = T -> (zero(T),), combine = (a, b) -> (a[1] + b[1],), lift = x -> (Float64(x),))
    @test_throws ArgumentError trace(x -> Tempest.fold(widening, x), Float32.(x))
end

@testset "Fold derivatives" begin
    y = reshape(cos.(0.7 .* (1:24)), 6, 4)
    for f in (
        y -> sum(abs2, Tempest.fold(LogSumExpFold(), y; dims = 1)),
        y -> Tempest.fold(LogSumExpFold(), y) ^ 2,
        y -> sum(sin, maximum(y; dims = 1)),
        y -> sum(abs2, y .- maximum(y; dims = 2)),
        y -> sum(abs2, Tempest.fold(WelfordVariance(), y; dims = 1)),
        y -> Tempest.fold(WelfordVariance(), y),
    )
        p = trace(f, y)
        loss, (dy,) = value_and_gradient(p, y)
        @test loss ≈ f(y)
        @test dy ≈ finite_difference(f, (copy(y),), 1) atol = 1e-6
        @test execute!(prepare(differentiate(p), y))[2] ≈ dy
    end
    # Tied extrema share the cotangent equally.
    _, (dc,) = value_and_gradient(trace(maximum, fill(0.5, 4)), fill(0.5, 4))
    @test dc ≈ fill(0.25, 4)
    # A fold without a rule runs forward and fails only at differentiate.
    loss(y) = sum(Tempest.fold(range_fold, y; dims = 1))
    p = trace(loss, y)
    error = try
        differentiate(p)
        nothing
    catch e
        e
    end
    @test error isa ArgumentError
    @test occursin("no pullback for fold genericfold", sprint(showerror, error))
    @test only(execute!(prepare_forward(loss, y))) ≈ loss(y)
    # A rule passed to GenericFold.
    ruled = GenericFold(;
        init = T -> (zero(T),),
        lift = x -> (x * x,),
        combine = (a, b) -> (a[1] + b[1],),
        pullback = (x, y, g, axes) -> g .* (2 .* x),
    )
    h(y) = sum(Tempest.fold(ruled, y; dims = 2))
    _, (dh,) = value_and_gradient(trace(h, y), y)
    @test dh ≈ 2 .* y
end

@testset "Products and Base reductions" begin
    x = reshape(0.9 .+ 0.3 .* sin.(1.0:20.0), 4, 5)
    with_zeros = copy(x)
    with_zeros[2, 2] = 0             # one zero in column 2 and row 2
    with_zeros[1, 3] = with_zeros[4, 3] = 0  # two zeros in column 3
    with_zeros[4, 5] = 0             # row 4: zeros in columns 3 and 5
    for dims in (:, (1,), (2,)), y in (x, with_zeros)
        run(f) = (got = execute!(prepare_forward(f, y)); dims isa Colon ? only(got) : got)
        @test run(x -> prod(x; dims)) ≈ prod(y; dims)
        @test run(x -> prod(abs2, x; dims)) ≈ prod(abs2, y; dims)
        for (op, reference) in ((+, sum), (*, prod), (max, maximum), (min, minimum))
            @test run(x -> reduce(op, x; dims)) ≈ reference(y; dims)
            # A given `init` combines once with the result, as in Base.
            @test run(x -> mapreduce(abs2, op, x; dims, init = 0.5)) ≈
                  mapreduce(abs2, op, y; dims, init = 0.5)
        end
        # Gradients through the product and maximum rules, against finite
        # differences, with no zero, one zero, or several zeros in a slice.
        for f in (y -> sum(sin, prod(y; dims)), y -> sum(sin, reduce(max, y; dims)))
            loss, (dy,) = value_and_gradient(trace(f, y), y)
            @test loss ≈ f(y)
            @test dy ≈ finite_difference(f, (copy(y),), 1) atol = 1e-6
        end
    end
    @test Tempest.fold(ProdFold(), with_zeros; dims = 1) ≈ prod(with_zeros; dims = 1)
    # Small integers widen as in Base.
    indices = Int32[3 1; 4 2]
    p = trace(i -> prod(i; dims = 1), indices)
    @test only(Tempest.interpret(Tempest.ForwardProgram(p), indices)) == prod(indices; dims = 1)

    # Other operators need `init`; with it they run forward as a fold.
    error = try
        trace(x -> reduce(skewed_combine, x), x)
        nothing
    catch e
        e
    end
    @test error isa ArgumentError
    @test occursin("needs `init`", sprint(showerror, error))
    @test_throws ArgumentError trace(x -> mapreduce(abs, norm_combine, x; dims = 1), x)
    for dims in (:, (1,), (2,))
        run(f) = (got = execute!(prepare_forward(f, x)); dims isa Colon ? only(got) : got)
        # CPU kernels fold left in index order, like Base.
        @test run(x -> reduce(skewed_combine, x; dims, init = 0.0)) ≈
              reduce(skewed_combine, x; dims, init = 0.0)
        @test run(x -> mapreduce(abs, norm_combine, x; dims, init = 0.0)) ≈
              mapreduce(abs, norm_combine, x; dims, init = 0.0)
    end
    # Folds from a singleton operator and a common identity are zero-size; an
    # identity that the element type cannot rebuild is captured.
    reduction_fold(f) = only(node.op.fold for node in trace(f, x).nodes if node.op isa Tempest.FoldReduction)
    @test Base.issingletontype(typeof(reduction_fold(x -> reduce(norm_combine, x; init = 0))))
    @test Base.issingletontype(typeof(reduction_fold(x -> reduce(smaller, x; init = Inf))))
    @test !Base.issingletontype(typeof(reduction_fold(x -> reduce(norm_combine, x; init = 0.5))))
    # The integer `init` promotes as in Base; the fold state keeps the element type.
    @test trace(x -> reduce(norm_combine, x; init = 0), Float32.(x)).nodes[end].dtype == Float32
end

@testset "Half-precision folds accumulate in Float32" begin
    for T in (Float16, Core.BFloat16)
        x = half_fold_input(T)
        @test Tempest.fold_accumulator(Tempest.SumFold(), T) === Float32
        @test Tempest.fold_accumulator(MaxFold(), T) === T
        @test Tempest.fold_accumulator(range_fold, T) === T
        # Accumulating in the element type would be far off.
        expected = half_fold_reference(HALF_FOLD_CASES.full_sum, x)
        @test abs(Float64(foldl(+, x)) - Float64(expected)) > 100 * eps(expected)
        for case in HALF_FOLD_CASES
            elements, reduce = case
            expected = half_fold_reference(case, x)
            f(x) = reduce(elements(x))
            # The interpreter's reference folds and the CPU kernels.
            interpreted = only(Tempest.interpret(Tempest.ForwardProgram(trace(f, x)), x))
            check_half_fold(interpreted isa Number ? fill(interpreted) : interpreted, expected)
            check_half_fold(execute!(prepare_forward(f, x)), expected)
        end
    end
end
