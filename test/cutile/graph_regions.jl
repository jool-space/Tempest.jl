# cuTile graph regions on CUDA; runtests.jl drops cutile/ when cuTile is not
# loaded. All fixtures here use production Prepared executables.
struct GraphWarpParameters
    positive::Float32
    negative::Float32
end

struct GraphWarp
    parameters::GraphWarpParameters
end

(f::GraphWarp)(x) = exp(ifelse(x > 0.0f0, f.parameters.positive, f.parameters.negative) * x)
Tempest.recomputable(::GraphWarp) = true

const graph_warp = GraphWarp(GraphWarpParameters(0.25f0, 0.125f0))

function graph_normalization(x, residual, weight, epsilon)
    h = graph_warp.(x .+ residual)
    mean = sum(h; dims = 1) ./ size(h, 1)
    centered = h .- mean
    variance = sum(abs2, centered; dims = 1) ./ size(h, 1)
    y = centered ./ sqrt.(variance .+ epsilon) .* weight
    (y = y, shared = h, mean = mean, variance = variance)
end

# This callable is device-compilable, but has not granted recomputation.
struct UnprovenGraphWarp end
(::UnprovenGraphWarp)(x) = exp(0.2f0 * x)

function graph_test_inputs(rows, columns = 3)
    indices = reshape(Float32.(1:(rows * columns)), rows, columns)
    x = sin.(0.07f0 .* indices)
    residual = 0.2f0 .* cos.(0.11f0 .* indices)
    weight = 0.8f0 .+ 0.1f0 .* sin.(Float32.(1:rows))
    (x, residual, weight, 0.03f0)
end

function check_graph_outputs(actual, expected; rtol = 4.0f-4, atol = 4.0f-5)
    @test Tempest.same_structure(actual, expected)
    for (device, reference) in zip(Tempest.tree_leaves(actual), Tempest.tree_leaves(expected))
        observed = reference isa Number ? only(Array(device)) : Array(device)
        @test size(observed) == size(reference)
        @test eltype(observed) === eltype(reference)
        @test all(isfinite, observed isa Number ? (observed,) : observed)
        @test isapprox(observed, reference; rtol, atol)
    end
end

graph_steps(ex) = filter(step -> step.region isa Tempest.GraphRegion, ex.storage.steps)
graph_addresses(ex) = map(pointer, (ex.inputs..., ex.buffers...))

function captured_kernel_count(ex)
    extension = Base.get_extension(Tempest, :CUDAExt)
    kinds = extension.graph_node_types(ex.runtime.graph)
    @test all(
        kind ->
            kind ∉
            (CUDACore.CU_GRAPH_NODE_TYPE_MEM_ALLOC, CUDACore.CU_GRAPH_NODE_TYPE_MEM_FREE),
        kinds,
    )
    count(==(CUDACore.CU_GRAPH_NODE_TYPE_KERNEL), kinds)
end

@testset "Production cuTile graph regions" begin
    @testset "Custom callable, dependent sums, and scalar replay: $rows rows" for rows in (
        259,
        8193,
    )
        inputs = graph_test_inputs(rows)
        device_inputs = map(a -> a isa AbstractArray ? CuArray(a) : a, inputs)
        ex = prepare_forward(graph_normalization, device_inputs...)
        @test length(graph_steps(ex)) == 1
        @test length(ex.storage.steps) == 1
        @test ex.providers == [:cutile]
        @test occursin("graph_region", sprint(explain, ex))
        check_graph_outputs(execute!(ex), graph_normalization(inputs...))
        pointers = graph_addresses(ex)
        capture!(ex)
        # 8193 rows split the column sums across blocks: partial and finishing
        # launches per phase, then the outputs.
        @test captured_kernel_count(ex) == only(ex.calls).launch.kernels ==
              (rows == 259 ? 1 : 5)
        for (scale, epsilon) in ((0.7f0, 0.01f0), (1.2f0, 0.2f0))
            changed = (scale .* inputs[1], inputs[2], inputs[3], epsilon)
            bind!(ex, changed...)
            foreach(array -> fill!(array, NaN32), Tempest.tree_leaves(outputs(ex)))
            GC.gc()
            check_graph_outputs(replay!(ex), graph_normalization(changed...))
            @test graph_addresses(ex) == pointers
        end
    end

    @testset "Escaping aliases and a preceding matmul" begin
        inputs = graph_test_inputs(259)
        function aliases(args...)
            values = graph_normalization(args...)
            (
                y = values.y,
                shared = values.shared,
                linear = reshape(values.shared, :),
                window = view(values.shared, 1:2:259, :),
            )
        end
        ex = prepare_forward(aliases, inputs...; target = CUDATarget())
        @test !isempty(graph_steps(ex))
        actual = execute!(ex)
        check_graph_outputs(actual, aliases(inputs...))
        @test pointer(actual.shared) == pointer(actual.linear)
        capture!(ex)
        changed = (0.4f0 .* inputs[1], inputs[2], inputs[3], 0.04f0)
        bind!(ex, changed...)
        check_graph_outputs(replay!(ex), aliases(changed...))

        function mixed(a, b, weight, epsilon)
            product = a * b
            values = graph_normalization(product, product, weight, epsilon)
            (normalized = values, product = product)
        end
        a = reshape(sin.(Float32.(1:(17 * 11))), 17, 11) .* 0.1f0
        b = reshape(cos.(Float32.(1:(11 * 3))), 11, 3) .* 0.1f0
        args = (a, b, ones(Float32, 17), 0.03f0)
        ex = prepare_forward(mixed, args...; target = CUDATarget())
        @test !isempty(graph_steps(ex))
        @test :cuBLAS in ex.providers
        check_graph_outputs(execute!(ex), mixed(args...))
        capture!(ex)
        changed = (1.3f0 .* a, b, args[3], 0.07f0)
        bind!(ex, changed...)
        check_graph_outputs(replay!(ex), mixed(changed...))
    end

    @testset "Equivalent regions share one compiled kernel" begin
        # The first layer reads a program argument and later layers read
        # matmul results, so their inputs appear in different graph orders;
        # the last layer writes its commutative operands in reverse order.
        function graph_rmsnorm(x, gain, epsilon)
            scale = inv.(sqrt.(sum(abs2, x; dims = 1) ./ Float32(size(x, 1)) .+ epsilon))
            x .* scale .* gain
        end
        function graph_rmsnorm_commuted(x, gain, epsilon)
            scale = inv.(sqrt.(epsilon .+ sum(abs2, x; dims = 1) ./ Float32(size(x, 1))))
            gain .* (scale .* x)
        end
        function layers(x, gains, weights, epsilon)
            h = graph_rmsnorm(x, gains[1], epsilon)
            h = graph_rmsnorm(weights[1] * h, gains[2], epsilon)
            graph_rmsnorm_commuted(weights[2] * h, gains[3], epsilon)
        end
        rows = 259
        x = graph_test_inputs(rows, 5)[1]
        gains = Tuple(1.0f0 .+ 0.1f0 .* sin.(Float32.(1:rows) .* k) for k = 1:3)
        weights = Tuple(
            reshape(cos.(Float32.(1:(rows * rows)) .* k), rows, rows) ./ 16 for k = 1:2
        )
        args = (x, gains, weights, 0.03f0)
        device_args = (CuArray(x), map(CuArray, gains), map(CuArray, weights), 0.03f0)
        ex = prepare_forward(layers, device_args...)
        extension = Base.get_extension(Tempest, :cuTileExt)
        launches = [call.launch for call in ex.calls if call.launch isa extension.GraphLaunch]
        @test length(launches) == 3
        # A launcher's type includes its compiled kernel, program, and arrays.
        @test allequal(typeof(launch.launch) for launch in launches)
        check_graph_outputs(execute!(ex), layers(args...); rtol = 1.0f-3)
    end

    @testset "Fallback is explicit and executable" begin
        input = graph_test_inputs(17)[1]
        unproven = UnprovenGraphWarp()
        function without_permission(x)
            h = unproven.(x)
            h ./ sqrt.(sum(abs2, h; dims = 1) .+ 0.03f0)
        end
        ex = prepare_forward(without_permission, CuArray(input))
        region = only(graph_steps(ex)).region
        @test only(ex.storage.program.results) in region.outputs
        @test all(graph_steps(ex)) do step
            all(step.region.members) do id
                operation = ex.storage.program.nodes[id].op
                !(operation isa Tempest.Pointwise && operation.f === unproven)
            end
        end
        @test occursin(r"recomput|purity", sprint(explain, ex))
        check_graph_outputs(execute!(ex), without_permission(input))

        normalized(x) = x ./ sqrt.(sum(abs2, x; dims = 1) .+ eltype(x)(0.03))
        wide = Float64.(input)
        ex = prepare_forward(normalized, CuArray(wide))
        @test isempty(graph_steps(ex))
        @test occursin("Float32", sprint(explain, ex))
        check_graph_outputs(execute!(ex), normalized(wide); rtol = 1e-10, atol = 1e-11)

        args = graph_test_inputs(17)
        old = prepare_forward(
            graph_normalization,
            args...;
            target = CUDATarget(),
            fuse_regions = false,
        )
        @test isempty(graph_steps(old))
        check_graph_outputs(execute!(old), graph_normalization(args...))
    end

    @testset "Forward values survive into ordinary differentiation" begin
        x = graph_test_inputs(7)[1]
        weight = fill(0.8f0, 7)
        projection = reshape(sin.(Float32.(1:49)), 7, 7) .* 0.1f0
        function loss(x, weight, projection)
            y = x ./ sqrt.(sum(abs2, x; dims = 1) ./ size(x, 1) .+ 0.03f0) .* weight
            sum(abs2, projection * y)
        end
        primal = trace(loss, x, weight, projection)
        derivative = differentiate(primal; wrt = (1, 2))
        ex = prepare(derivative, CuArray(x), CuArray(weight), CuArray(projection))
        @test !isempty(graph_steps(ex))
        forward_count = length(primal.nodes)
        saved = [
            id for step in graph_steps(ex) for id in step.outputs if
            id <= forward_count && any(enumerate(ex.storage.program.nodes)) do pair
                index, node = pair
                index > forward_count && index ∉ step.region.members && id in node.inputs
            end
        ]
        @test !isempty(saved)
        reference, gradients = value_and_gradient(derivative, x, weight, projection)
        check_graph_outputs(
            execute!(ex),
            (reference, gradients...);
            rtol = 8.0f-4,
            atol = 5.0f-5,
        )
        pointers = graph_addresses(ex)
        capture!(ex)
        changed = (0.6f0 .* x, 1.1f0 .* weight, projection)
        bind!(ex, changed...)
        reference, gradients = value_and_gradient(derivative, changed...)
        check_graph_outputs(
            replay!(ex),
            (reference, gradients...);
            rtol = 8.0f-4,
            atol = 5.0f-5,
        )
        @test graph_addresses(ex) == pointers
    end

    @testset "Captured functional update with staged donation" begin
        params = graph_test_inputs(7)[1]
        moments = zero(params)
        weight = fill(0.8f0, 7)
        function loss(p, unused_moments, weight)
            y = p ./ sqrt.(sum(abs2, p; dims = 1) ./ size(p, 1) .+ 0.03f0) .* weight
            sum(abs2, y)
        end
        derivative = differentiate(trace(loss, params, moments, weight); wrt = (1,))
        step = with_update(derivative) do value, gradients, p, m, w
            next_moments = 0.9f0 .* m .+ only(gradients)
            (params = p .- 0.01f0 .* next_moments, moments = next_moments, value = value)
        end
        ex = prepare(
            step,
            params,
            moments,
            weight;
            target = CUDATarget(),
            donate = (1, 2),
            inplace = false,
        )
        @test !isempty(graph_steps(ex))
        @test isempty(ex.storage.donations)
        @test length(ex.state.commits) == 2
        original = map(Array, ex.inputs)
        capture!(ex)
        @test map(Array, ex.inputs) == original
        pointers = graph_addresses(ex)
        expected_params, expected_moments = copy(params), copy(moments)
        for scale in (0.7f0, 1.1f0, 0.9f0)
            changed_weight = scale .* weight
            value, (gradient,) = value_and_gradient(
                derivative,
                expected_params,
                expected_moments,
                changed_weight,
            )
            expected_moments = 0.9f0 .* expected_moments .+ gradient
            expected_params = expected_params .- 0.01f0 .* expected_moments
            bind_argument!(ex, 3, changed_weight)
            expected = (params = expected_params, moments = expected_moments, value = value)
            check_graph_outputs(replay!(ex), expected; rtol = 1.0f-3, atol = 8.0f-5)
            @test graph_addresses(ex) == pointers
        end
        current_state = map(Array, ex.inputs)
        capture!(ex)
        @test map(Array, ex.inputs) == current_state
    end
end

function graph_row_normalization(x, bias, epsilon)
    h = graph_warp.(x .+ bias)
    mean = sum(h; dims = 2) ./ Float32(size(h, 2))
    centered = h .- mean
    variance = sum(abs2, centered; dims = 2) ./ Float32(size(h, 2))
    (; y = centered ./ sqrt.(variance .+ epsilon), h, mean, variance)
end

function graph_global_normalization(x, epsilon)
    h = graph_warp.(x)
    mean = sum(h) / Float32(length(h))
    energy = sum(abs2, h)
    centered = h .- mean
    variance = sum(abs2, centered) / Float32(length(h))
    (; y = centered ./ sqrt(variance + epsilon), h, mean, variance, energy)
end

@testset "cuTile graph reduction domains" begin
    @testset "Second axis and singleton broadcasts: $dims" for dims in (
        (1, 19),
        (17, 1),
        (33, 19),
        (65, 263),
    )
        x = reshape(0.3f0 .* sin.(Float32.(1:prod(dims))), dims)
        bias = reshape(0.1f0 .* cos.(Float32.(1:dims[2])), 1, :)
        ex = prepare_forward(graph_row_normalization, CuArray(x), CuArray(bias), 0.1f0)
        @test length(graph_steps(ex)) == length(ex.storage.steps) == 1
        # Only the widest domain has enough reduced tiles to split; a split
        # schedule completes its sums in shared workspace.
        split = occursin("split", sprint(explain, ex))
        @test split == (dims == (65, 263))
        @test isempty(ex.scratch) == !split
        @test occursin("over axis 2", sprint(explain, ex))
        check_graph_outputs(execute!(ex), graph_row_normalization(x, bias, 0.1f0))
        capture!(ex)
        @test captured_kernel_count(ex) == only(ex.calls).launch.kernels == (split ? 5 : 1)
        pointers = graph_addresses(ex)
        bind!(ex, 0.6f0 .* x, bias, 0.2f0)
        check_graph_outputs(replay!(ex), graph_row_normalization(0.6f0 .* x, bias, 0.2f0))
        @test graph_addresses(ex) == pointers
    end

    @testset "Dependent full reductions and scalar outputs: $dims" for dims in (
        (1, 1),
        (1, 259),
        (17, 1),
        (33, 19),
        (259,),
    )
        x = reshape(0.3f0 .* sin.(Float32.(1:prod(dims))), dims)
        ex = prepare_forward(graph_global_normalization, CuArray(x), 0.1f0)
        @test length(graph_steps(ex)) == length(ex.storage.steps) == 1
        @test length(ex.scratch) == 2
        @test occursin("2 phases, 5 kernels", sprint(explain, ex))
        check_graph_outputs(execute!(ex), graph_global_normalization(x, 0.1f0))
        capture!(ex)
        @test captured_kernel_count(ex) == 5
        pointers = map(pointer, (ex.inputs..., ex.buffers..., ex.scratch...))
        bind!(ex, 0.7f0 .* x, 0.2f0)
        foreach(array -> fill!(array, NaN32), ex.scratch)
        foreach(array -> fill!(array, NaN32), Tempest.tree_leaves(outputs(ex)))
        GC.gc()
        check_graph_outputs(replay!(ex), graph_global_normalization(0.7f0 .* x, 0.2f0))
        @test map(pointer, (ex.inputs..., ex.buffers..., ex.scratch...)) == pointers
        @test ndims(outputs(ex).mean) == 0
    end

    @testset "Full-reduction workspace is shared across regions" begin
        x = reshape(0.2f0 .* sin.(Float32.(1:(33 * 17))), 33, 17)
        w = reshape(0.1f0 .* cos.(Float32.(1:(13 * 33))), 13, 33)
        function two_norms(x, w)
            first = graph_global_normalization(x, 0.1f0)
            second = graph_global_normalization(w * first.y, 0.2f0)
            (; first, second)
        end
        ex = prepare_forward(two_norms, CuArray(x), CuArray(w))
        indices = findall(step -> step.region isa GraphRegion, ex.storage.steps)
        @test length(indices) == 2
        @test ex.workspace_slots[indices[1]] == ex.workspace_slots[indices[2]]
        check_graph_outputs(execute!(ex), two_norms(x, w))
        capture!(ex)
        bind!(ex, 0.6f0 .* x, w)
        foreach(array -> eltype(array) === Float32 && fill!(array, NaN32), ex.scratch)
        check_graph_outputs(replay!(ex), two_norms(0.6f0 .* x, w))
    end

    @testset "Padding, nonfinite inputs, and retained reduction dimensions" begin
        x = fill(2.0f0, 33, 19)
        for axes in ((1,), (2,), (1, 2), :)
            f(x) = sum(log.(x); dims = axes)
            ex = prepare_forward(f, CuArray(x))
            check_graph_outputs(execute!(ex), f(x))
        end
        scalar = fill(2.0f0)
        ex = prepare_forward(x -> sum(exp.(x)), CuArray(scalar))
        @test only(Array(execute!(ex))) ≈ exp(2.0f0)

        for axes in ((2,), :), bad in (NaN32, Inf32, -Inf32)
            values = copy(x)
            values[1] = bad
            f(x) = sum(x; dims = axes)
            ex = prepare_forward(f, CuArray(values))
            observed = Array(execute!(ex))
            expected = f(values)
            expected = expected isa Number ? fill(expected) : expected
            @test isequal(observed, expected)
        end
    end

    @testset "Mixed axes remain separate and differentiate correctly" begin
        x = reshape(0.2f0 .* sin.(Float32.(1:35)), 5, 7)
        function loss(x)
            h = exp.(x)
            row = sum(h; dims = 2)
            column = sum(h; dims = 1)
            sum(abs2, h ./ (row .+ column))
        end
        primal = trace(loss, x)
        ex = prepare(primal, CuArray(x))
        @test length(graph_steps(ex)) >= 2
        for step in graph_steps(ex)
            axes = [
                node.op.axes isa Colon ? (1, 2) : node.op.axes for
                id in step.region.members for node in (ex.storage.program.nodes[id],) if
                node.op isa Tempest.SumReduction
            ]
            @test length(unique(axes)) == 1
        end
        value, (gradient,) = value_and_gradient(primal, x)
        check_graph_outputs(execute!(ex), (value, gradient); rtol = 1.0f-3, atol = 5.0f-5)
        numerical = finite_difference(loss, (x,), 1; h = 2.0f-3)
        @test Array(outputs(ex)[2]) ≈ numerical rtol=2.0f-2 atol=5.0f-4
        capture!(ex)
        changed = 0.7f0 .* x
        bind!(ex, changed)
        value, (gradient,) = value_and_gradient(primal, changed)
        check_graph_outputs(replay!(ex), (value, gradient); rtol = 1.0f-3, atol = 5.0f-5)
    end

    @testset "Domain limits remain explicit" begin
        for dims in ((0, 7), (7, 0))
            @test_throws ArgumentError trace(x -> sum(abs2, x), zeros(Float32, dims))
        end
        # Any rank collapses when the reduced axes are all, leading or trailing;
        # an interior axis stays with the CUDA kernels.
        x = fill(0.2f0, 3, 5, 7)
        ex = prepare_forward(x -> sum(abs2, x), CuArray(x))
        @test length(graph_steps(ex)) == 1
        @test only(Array(execute!(ex))) ≈ sum(abs2, x)
        ex = prepare_forward(x -> sum(abs2, x; dims = 2), CuArray(x))
        @test isempty(graph_steps(ex))
        @test any(note -> occursin("leading or trailing axes", note), ex.semantic_notes)
        @test Array(execute!(ex)) ≈ sum(abs2, x; dims = 2)
    end

    # Scalar callables broadcast over the two-axis tiles of column reductions:
    # `sign`, and selects and constants of other shapes than their lanes.
    @testset "Scalar callables over two-axis tiles: $dims" for dims in ((64, 4), (300, 5))
        x = reshape(sin.(0.37f0 .* Float32.(1:prod(dims))), dims)
        f(x) = sum(sign.(x) .* ifelse.(x .> 0.0f0, 0.25f0, 0.125f0) .* exp.(x); dims = 1)
        ex = prepare_forward(f, CuArray(x))
        @test length(graph_steps(ex)) == 1
        @test only(unique(ex.providers)) === :cutile
        @test Array(execute!(ex)) ≈ f(x)
    end
end

@testset "Packed and split cuTile schedules" begin
    summaries(ex) = join(filter(!isnothing, map(Tempest.implementation_summary, ex.calls)), "; ")
    # Short columns share a block; odd extents exercise both lane masks.
    for (rows, columns) in ((100, 1000), (32, 4099), (128, 1024))
        inputs = graph_test_inputs(rows, columns)
        ex = prepare_forward(graph_normalization, map(a -> a isa AbstractArray ? CuArray(a) : a, inputs)...)
        packed = match(r"resident tile \d+×\d+, (\d+) columns per block", summaries(ex))
        @test packed !== nothing && parse(Int, packed[1]) > 1
        if (rows, columns) == (128, 1024)
            @test summaries(ex) ==
                  "cuTile reductions over axis 1 of 128×1024: resident tile 128×4, " *
                  "4 columns per block, grid 256; 2 phases, 1 kernel"
        end
        check_graph_outputs(execute!(ex), graph_normalization(inputs...))
    end
    # Few kept tiles and a long reduced axis split across blocks, with a
    # dependent second phase (the variance reads the mean). Split column
    # tiles shorten from 1024 rows until the grid holds 128 blocks.
    expected_summaries = Dict(
        (8193, 5) => "cuTile reductions over axis 1 of 8193×5: streamed tile 256×1, " *
                     "split 32 ways, grid 5×32; 2 phases, 5 kernels",
        (16384, 8) => "cuTile reductions over axis 1 of 16384×8: streamed tile 1024×1, " *
                      "split 16 ways, grid 8×16; 2 phases, 5 kernels",
    )
    for (rows, columns) in ((8193, 5), (16384, 8), (6000, 20))
        inputs = graph_test_inputs(rows, columns)
        ex = prepare_forward(graph_normalization, map(a -> a isa AbstractArray ? CuArray(a) : a, inputs)...)
        @test occursin("split", summaries(ex))
        if haskey(expected_summaries, (rows, columns))
            @test summaries(ex) == expected_summaries[(rows, columns)]
        end
        check_graph_outputs(execute!(ex), graph_normalization(inputs...))
        capture!(ex)
        check_graph_outputs(replay!(ex), graph_normalization(inputs...))
    end
    # 32 or more columns stream unsplit, one block each.
    inputs = graph_test_inputs(8193, 32)
    ex = prepare_forward(graph_normalization, map(a -> a isa AbstractArray ? CuArray(a) : a, inputs)...)
    @test summaries(ex) ==
          "cuTile reductions over axis 1 of 8193×32: streamed tile 1024×1, grid 32; " *
          "2 phases, 1 kernel"
    check_graph_outputs(execute!(ex), graph_normalization(inputs...))
    for (rows, columns) in ((97, 1500), (128, 1024), (3, 7000))
        x = reshape(sin.(0.01f0 .* Float32.(1:(rows * columns))), rows, columns)
        bias = reshape(cos.(Float32.(1:rows)), rows, 1)
        ex = prepare_forward(graph_row_normalization, CuArray(x), CuArray(bias), 0.05f0)
        @test occursin("split", summaries(ex))
        check_graph_outputs(execute!(ex), graph_row_normalization(x, bias, 0.05f0))
    end
end
