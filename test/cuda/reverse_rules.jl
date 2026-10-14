include("../../examples/kernel_composition/rmsnorm_cuda_reference.jl")

function check_cuda_norm(implementation, name)

    @testset "Concrete CUDA wrappers and saved statistics" begin
        @test CUDACore.functional()
        for (rows, columns) in ((7, 5), (259, 3), (1024, 8)), reuse in (false, true)
            x = reshape(sin.(Float32.(1:(rows * columns))), rows, columns)
            w = fill(0.7f0, rows)
            loss(x, w) = sum(abs2, rmsnorm(x, w; eps = 0.03f0) .+ x)
            ex = prepare(
                loss,
                CuArray(x),
                CuArray(w);
                reuse,
                implementations = (implementation,),
            )
            @test count(==(name), ex.providers) == 2
            expected = execute!(prepare(loss, x, w))
            @test all(
                isapprox(Array(a), b; rtol = 2e-4, atol = 2e-4) for
                (a, b) in zip(execute!(ex), expected)
            )
            pointers = map(pointer, ex.buffers)
            capture!(ex)
            for scale in (0.7f0, 1.2f0)
                bind!(ex, scale * x, w)
                expected = execute!(prepare(loss, scale * x, w))
                GC.gc()
                actual = replay!(ex)
                @test all(
                    isapprox(Array(a), b; rtol = 2e-4, atol = 2e-4) for
                    (a, b) in zip(actual, expected)
                )
                @test map(pointer, ex.buffers) == pointers
            end
        end

        # A CUDA target alone is insufficient: unsupported dtypes keep the reference.
        x = reshape(sin.(Float64.(1:35)), 7, 5)
        w = fill(0.7, 7)
        fallback_loss(x, w) = sum(abs2, rmsnorm(x, w; eps = 0.03))
        ex = prepare(fallback_loss, CuArray(x), CuArray(w); implementations = (implementation,))
        @test name ∉ ex.providers
        @test occursin("Float32 only", sprint(explain, ex))
        expected = execute!(prepare(fallback_loss, x, w))
        @test all(isapprox(Array(a), b; rtol = 1e-10) for (a, b) in zip(execute!(ex), expected))


        # Logical views are materialized at the concrete wrapper boundary; their
        # gradients still scatter back into the original input shapes.
        view_x, view_w = Float32.(x), Float32.(w)
        view_loss(x, w) = sum(abs2, rmsnorm(view(x, 1:2:7, :), view(w, 1:2:7); eps = 0.03))
        viewed = prepare(
            view_loss,
            CuArray(view_x),
            CuArray(view_w);
            implementations = (implementation,),
        )
        @test count(==(name), viewed.providers) == 2
        capture!(viewed)
        bind!(viewed, 0.7f0 * view_x, view_w)
        expected = execute!(prepare(view_loss, 0.7f0 * view_x, view_w))
        @test all(
            isapprox(Array(a), b; rtol = 2e-4, atol = 2e-4) for
            (a, b) in zip(replay!(viewed), expected)
        )
    end

    @testset "Captured Adam through private RMSNorm forward/backward pairs" begin
        rows, columns = 7, 5
        params = (
            w1 = reshape(0.08f0 .* sin.(Float32.(1:49)), 7, 7),
            w2 = reshape(0.05f0 .* cos.(Float32.(1:49)), 7, 7),
            gain1 = fill(0.7f0, rows),
            gain2 = fill(0.8f0, rows),
        )
        opt = (first = map(zero, params), second = map(zero, params), step = 0.0f0)
        batch =
            (x = reshape(sin.(Float32.(1:35)), rows, columns), y = fill(0.1f0, rows, columns))
        function loss(p, unused, b)
            h = rmsnorm(p.w1 * b.x .+ b.x, p.gain1; eps = 0.03f0)
            y = rmsnorm(p.w2 * h .+ h, p.gain2; eps = 0.03f0)
            sum(abs2, y .- b.y) / length(b.y)
        end
        derivative = differentiate(trace(loss, params, opt, batch); wrt = (1,))
        step = with_update(derivative) do value, gradients, p, o, b
            next_params, next_opt = adam_update(p, o, only(gradients); rate = 0.001f0)
            (next_params, next_opt, value)
        end
        ex = prepare(
            step,
            params,
            opt,
            batch;
            target = CUDATarget(),
            donate = (1, 2),
            implementations = (implementation,),
        )
        @test count(==(name), ex.providers) == 4
        original = map(a -> copy(Array(a)), ex.inputs)
        capture!(ex)
        @test all(Array(a) == b for (a, b) in zip(ex.inputs, original))
        expected_params, expected_opt = params, opt
        for iteration = 1:5
            next_batch = (x = batch.x .* (0.8f0 + 0.1f0 * iteration), y = batch.y)
            value, (gradients,) =
                value_and_gradient(derivative, expected_params, expected_opt, next_batch)
            expected_params, expected_opt =
                adam_update(expected_params, expected_opt, gradients; rate = 0.001f0)
            bind_argument!(ex, 3, next_batch)
            actual_params, actual_opt, actual_loss = replay!(ex)
            @test Array(actual_loss)[] ≈ value rtol=5e-4 atol=2e-5
            test_state_close(actual_params, expected_params; rtol = 5e-4, atol = 3e-5)
            test_state_close(actual_opt, expected_opt; rtol = 1e-3, atol = 3e-5)
            if iteration == 3
                state = map(a -> copy(Array(a)), ex.inputs)
                capture!(ex)
                @test all(Array(a) == b for (a, b) in zip(ex.inputs, state))
            end
        end
    end

end

check_cuda_norm(ReferenceRMSNorm(), :reference_rmsnorm)

if haskey(ENV, "TEMPEST_RMS_SOURCE")
    include("../../examples/kernel_composition/rmsnorm_cuda.jl")
    check_cuda_norm(ArchivedRMSNorm(), :archived_rmsnorm)
end
