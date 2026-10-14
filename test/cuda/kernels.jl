include("../fixtures/planning.jl")
using CUDACore

function node_types(graph)
    n = Ref{Csize_t}(0)
    CUDACore.cuGraphGetNodes(graph, C_NULL, n)
    nodes = Vector{CUDACore.CUgraphNode}(undef, n[])
    CUDACore.cuGraphGetNodes(graph, nodes, n)
    map(nodes) do node
        t = Ref{CUDACore.CUgraphNodeType}()
        CUDACore.cuGraphNodeGetType(node, t)
        t[]
    end
end

@testset "CUDA kernels and repeated graph replay" begin
    CUDACore.allowscalar(false)
    for n in (1, 2, 133, 256, 257, 1025, 65537)
        # Reduction tail masking must happen after the scalar expression.
        x = reshape(Float32.(range(-0.3; step = 0.7 / max(n - 1, 1), length = n)), 1, n)
        compare_prepared(
            x -> sum(exp.(0 * x)),
            x;
            target = CUDATarget(),
            atol = 1e-5,
            rtol = 1e-5,
        )
        compare_prepared(x -> sum(abs2, x), x; target = CUDATarget(), atol = 1e-5, rtol = 1e-5)
    end
    x = reshape(Float32.(range(-0.6, 0.7; length = 95)), 5, 19)
    w = reshape(Float32.(range(-0.4, 0.5; length = 35)), 5, 7)
    f(x, w) = sum(abs2, tanh.(w' * x)) / length(x)
    ex = compare_prepared(f, x, w; target = CUDATarget(), atol = 2e-5, rtol = 2e-5)
    capture!(ex)
    @test all(
        t ->
            t != CUDACore.CU_GRAPH_NODE_TYPE_MEM_ALLOC &&
            t != CUDACore.CU_GRAPH_NODE_TYPE_MEM_FREE,
        node_types(ex.runtime.graph),
    )
    pointers = map(pointer, ex.buffers)
    for scale in (1.0f0, 2.0f0, -0.5f0, 0.0f0, 1.3f0)
        xx = scale * x
        bind!(ex, xx, w)
        got = replay!(ex)
        ref = value_and_gradient(ex.storage.program, xx, w)
        @test Array(got[1])[] ≈ ref[1] atol=2e-5 rtol=2e-5
        @test Array(got[2]) ≈ ref[2][1] atol=2e-5 rtol=2e-5
        @test Array(got[3]) ≈ ref[2][2] atol=2e-5 rtol=2e-5
        @test map(pointer, ex.buffers) == pointers
    end
    # Same scalar vocabulary, with CUDA indexing for a transposed leaf.
    compare_prepared(x -> sum(abs2, x' * x), x; target = CUDATarget(), atol = 1e-4, rtol = 2e-5)
    compare_prepared(x -> sum(ripple.(x')), x; target = CUDATarget(), atol = 1e-4, rtol = 2e-5)
    compare_prepared(
        x -> sum(x .* sum(abs2, x)),
        x;
        target = CUDATarget(),
        atol = 1e-4,
        rtol = 2e-5,
    )
    compare_prepared(
        (x, u) -> sum(abs2, x),
        x,
        w;
        target = CUDATarget(),
        atol = 1e-4,
        rtol = 2e-5,
    )
    compare_prepared(
        (x, w) -> sum(abs2, ripple.(w' * x)),
        x,
        w;
        target = CUDATarget(),
        atol = 1e-4,
        rtol = 2e-5,
    )

    function chain(x, w)
        for _ = 1:4
            x = tanh.(w' * x)
        end
        sum(abs2, x)
    end
    a=reshape(Float32.(range(-0.1, 0.2; length = 256)), 16, 16)
    reused=compare_prepared(chain, a, a; target = CUDATarget(), atol = 1e-4, rtol = 2e-5)
    unique=compare_prepared(
        chain,
        a,
        a;
        target = CUDATarget(),
        reuse = false,
        atol = 1e-4,
        rtol = 2e-5,
    )
    @test length(reused.buffers) < length(unique.buffers)
    capture!(reused)
    for _ = 1:3
        replay!(reused)
        @test Array(outputs(reused)[1])[] ≈ Array(outputs(unique)[1])[] atol=1e-4 rtol=2e-5
    end
    # A composed residual gated MLP also exercises binary pointwise derivatives.
    small(m, n, s) = reshape(s .* sin.(Float32.(1:(m * n))), m, n)
    mlpargs = (
        small(3, 5, 0.2f0),
        small(3, 2, 0.3f0),
        small(3, 2, 0.4f0),
        small(2, 3, 0.3f0),
        small(3, 5, 0.1f0),
    )
    mlp = compare_prepared(
        gated_loss,
        mlpargs...;
        target = CUDATarget(),
        atol = 1e-6,
        rtol = 2e-5,
    )
    capture!(mlp)
    for _ = 1:3
        replay!(mlp)
        @test Array(outputs(mlp)[1])[] ≈ gated_loss(mlpargs...) atol=1e-6 rtol=2e-5
    end
    # Float64 uses the same generated expressions, not downcast inputs.
    compare_prepared(
        x -> sum(abs2, tanh.(x)),
        Float64.(x);
        target = CUDATarget(),
        atol = 1e-9,
        rtol = 1e-8,
    )
    # The CUDA scalar lowering can be selected explicitly without changing IR.
    alt = prepare(f, x, w; target = CUDATarget(kernels = :cuda))
    execute!(alt)
    @test Array(outputs(alt)[1])[] ≈ f(x, w) atol=2e-5 rtol=2e-5
    capture!(alt)
    for _ = 1:3
        replay!(alt)
        @test Array(outputs(alt)[1])[] ≈ f(x, w) atol=2e-5 rtol=2e-5
    end
    @test :cuda in alt.providers
    CUDACore.synchronize()
end

@testset "Generated CUDA kernels resolve reads at build time" begin
    target = CUDATarget(kernels = :cuda)
    x = reshape(Float32.(sin.(1:(37 * 19 * 5))), 37, 19, 5)
    b = reshape(Float32.(cos.(1:19)), 1, 19)
    # Both axis-sum schedules, broadcasts, transposed reads and mixed axes; the
    # gradients add broadcast reads of the reduced values.
    check(f, args...) = compare_prepared(f, args...; target, atol = 1e-3, rtol = 1e-5)
    check(x -> sum(abs2, sum(abs2, x; dims = 1)), x)
    check(x -> sum(abs2, sum(abs2, x; dims = 2)), x)
    check(x -> sum(abs2, sum(x; dims = (1, 3))), x)
    check(x -> sum(abs2, sum(x; dims = (2, 3))), x)
    check((x, b) -> sum(abs2, sum(abs2, x[:, :, 2]' .* b'; dims = 2)), x, b)
    y = reshape(Float32.(sin.(1:(1024 * 64))) ./ 32, 1024, 64)
    check(y -> sum(abs2, sum(y; dims = 2)), y)
    check(y -> sum(abs2, y), y)
    check(y -> sum(abs2, sum(tanh.(y'); dims = 1)), y)

    # Index metadata stays in registers: a local-memory spill made these kernels
    # two orders of magnitude slower than the traffic bound. `--check-bounds=yes`
    # (the Pkg.test default) ignores `@inbounds`, and the bounds-error path then
    # puts the indices on the stack, so the check only holds for normal builds.
    forced_bounds_checks = Base.JLOptions().check_bounds == 1
    ext = Base.get_extension(Tempest, :CUDAExt)
    h = CuArray(y)
    ex = prepare_forward((h, b) -> tanh.(h .* sum(abs2, h; dims = 1) .+ b'), h, CuArray(b[:, 1:1]); target)
    for (step, call) in zip(ex.storage.steps, ex.calls)
        step.region isa Tempest.Elementwise || continue
        runtime, out, arrays = call.owners
        e = ext.device_expression(runtime, step.region)
        kernel = CUDACore.@cuda launch=false ext.cuda_pointwise_kernel(out, e, arrays, length(out))
        @test CUDACore.memory(kernel).var"local" == 0 skip = forced_bounds_checks
    end
end
