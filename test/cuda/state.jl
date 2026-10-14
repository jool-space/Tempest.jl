@testset "Captured Adam preserves and advances owned state" begin
    kernels = Base.get_extension(Tempest, :cuTileExt) === nothing ? (:cuda,) : (:cuda, :auto)
    for provider in kernels
        (; params, opt, batch, differentiated, step) = adam_problem()
        device(x) = x isa AbstractArray ? CuArray(x) : x
        inputs = Tempest.tree_map(device, (params, opt, batch))
        ex = prepare(step, inputs...; target = CUDATarget(kernels = provider), donate = (1, 2))
        state_outputs = outputs(ex)[1:2]
        addresses = map(pointer, Tempest.tree_leaves(outputs(ex)))
        capture!(ex)
        test_state_close(state_outputs, (params, opt))
        for iteration = 1:5
            next_batch = (x = batch.x .* Float32(iteration), y = batch.y)
            loss, gradients = adam_reference_gradient(params, next_batch)
            params, opt = adam_update(params, opt, gradients)
            bind_argument!(ex, 3, Tempest.tree_map(device, next_batch))
            actual = replay!(ex)
            test_state_close(actual, (params, opt, loss); rtol = 2.0f-4, atol = 3.0f-6)
            @test map(pointer, Tempest.tree_leaves(actual)) == addresses
            if iteration == 3
                capture!(ex)
                test_state_close(state_outputs, (params, opt); rtol = 2.0f-4, atol = 3.0f-6)
            end
        end
        @test inputs[2].step == 0.0f0
        types = Base.get_extension(Tempest, :CUDAExt).graph_node_types(ex.runtime.graph)
        @test !(CUDACore.CU_GRAPH_NODE_TYPE_MEM_ALLOC in types)
        @test !(CUDACore.CU_GRAPH_NODE_TYPE_MEM_FREE in types)
        # A failing warmup must also restore current state, including writes made
        # by a launch before it throws. No graph capture begins in this case.
        push!(ex.calls, () -> begin
            fill!(first(ex.inputs), 42.0f0)
            error("warmup failure")
        end)
        @test_throws ErrorException capture!(ex)
        test_state_close(state_outputs, (params, opt); rtol = 2.0f-4, atol = 3.0f-6)
    end
end

@testset "Captured simultaneous state commits" begin
    a, b = Float32[1, 2], Float32[3, 4]
    ex = prepare_forward(
        (a, b) -> (b, a, transpose(reshape(a, 1, 2))),
        CuArray(a),
        CuArray(b);
        donate = (1, 2),
    )
    capture!(ex)
    @test Array(outputs(ex)[1]) == a
    @test Array(outputs(ex)[2]) == b
    replay!(ex)
    @test Array(outputs(ex)[1]) == b
    @test Array(outputs(ex)[2]) == a
    @test vec(Array(outputs(ex)[3])) == a
    replay!(ex)
    @test Array(outputs(ex)[1]) == a
    @test Array(outputs(ex)[2]) == b
    @test vec(Array(outputs(ex)[3])) == b
end
