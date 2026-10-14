include("../fixtures/graph_regions.jl")

function graph_test_normalization(x)
    h = sin.(x) .+ 0.2
    mean = sum(h; dims = 1) ./ size(h, 1)
    centered = h .- mean
    variance = sum(abs2, centered; dims = 1) ./ size(h, 1)
    y = centered ./ sqrt.(variance .+ 0.1)
    (; y, h, mean, variance)
end

@testset "Graph regions expose original IDs and complete boundaries" begin
    x = reshape(collect(1.0:20.0), 5, 4) ./ 17
    f(x) = begin
        h = sin.(x)
        y = h ./ (sum(abs2, h; dims = 1) .+ 0.1)
        (y, reshape(h, 4, 5), h)
    end
    primal = trace(f, x)
    program = Tempest.ForwardProgram(primal)
    members = findall(
        n -> n.op isa Union{Tempest.Pointwise, Tempest.SumReduction, Tempest.Constant},
        program.nodes,
    )
    region = Tempest.graph_region(program, members)
    @test region.program === program
    @test region.members == sort(members)
    @test length(region.outputs) == length(unique(region.outputs))
    @test region.inputs == primal.arguments
    @test primal.output[1] in region.outputs
    @test primal.output[3] in region.outputs
    @test primal.output[2] ∉ region.members # the reshape stays outside
    @test all(
        id ->
            program.nodes[id].op isa
            Union{Tempest.Pointwise, Tempest.SumReduction, Tempest.Constant},
        region.members,
    )
    @test_throws ArgumentError Tempest.graph_region(program, [primal.arguments[1]])
    @test_throws ArgumentError Tempest.graph_region(program, [primal.output[2]])
    @test_throws ArgumentError Tempest.graph_region(program, [length(program.nodes) + 1])

    # Collapsing these nodes would create region -> transpose -> region. The
    # boundary must reject that cycle before any buffer allocation or build.
    crossing(x) = begin
        h = sin.(x)
        sum(h .+ transpose(h); dims = 1)
    end
    crossed = Tempest.ForwardProgram(trace(crossing, ones(3, 3)))
    crossing_members =
        findall(n -> n.op isa Union{Tempest.Pointwise, Tempest.SumReduction}, crossed.nodes)
    @test_throws ArgumentError Tempest.graph_region(crossed, crossing_members)

    impl = GraphTestImplementation()
    ex = prepare_forward(primal, x; implementations = (impl,))
    actual = execute!(ex)
    check_graph_test_outputs(actual, f(x))
    @test Base.mightalias(actual[2], actual[3])
    @test !isempty(impl.built)
    @test any(r -> primal.output[3] in r.region.outputs, impl.built)
    @test all(r -> all(id -> id in eachindex(primal.nodes), r.region.members), impl.built)
end

@testset "Ordinary graphs select multi-output implementations" begin
    x = reshape(collect(range(-0.6, 0.8; length = 20)), 5, 4)
    for reuse in (false, true)
        impl = GraphTestImplementation()
        ex = prepare_forward(graph_test_normalization, x; implementations = (impl,), reuse)
        @test :graph_test in ex.providers
        @test any(r -> length(r.region.outputs) > 1, impl.built)
        check_graph_test_outputs(execute!(ex), graph_test_normalization(x))
        changed = 0.7x .+ 0.1
        bind!(ex, changed)
        check_graph_test_outputs(execute!(ex), graph_test_normalization(changed))
        @test occursin("graph_test", sprint(explain, ex))
    end
    disabled = GraphTestImplementation()
    ordinary = prepare_forward(
        graph_test_normalization,
        x;
        implementations = (disabled,),
        fuse_regions = false,
    )
    @test isempty(disabled.requests)
    @test :graph_test ∉ ordinary.providers
    check_graph_test_outputs(execute!(ordinary), graph_test_normalization(x))
end

@testset "Graph selection preserves layout and matrix boundaries" begin
    x = reshape(collect(range(-0.4, 0.7; length = 20)), 5, 4)
    w = reshape(collect(range(0.2, 0.8; length = 15)), 3, 5)
    function mixed(w, x)
        h = w * x
        centered = h .- sum(h; dims = 1) ./ size(h, 1)
        y = centered .* centered
        (y * transpose(x), sum(view(y, 1:2, :); dims = 1), y)
    end
    impl = GraphTestImplementation()
    ex = prepare_forward(mixed, w, x; implementations = (impl,))
    check_graph_test_outputs(execute!(ex), mixed(w, x))
    @test :graph_test in ex.providers
    @test any(s -> s.region isa Tempest.GemmRegion, ex.storage.steps)
    @test all(
        r -> all(
            id -> !(
                r.region.program.nodes[id].op isa
                Union{Tempest.LayoutOperation, Tempest.MatMul}
            ),
            r.region.members,
        ),
        impl.built,
    )
    @test any(
        r -> any(id -> r.region.program.nodes[id].op isa Tempest.MatMul, r.region.inputs),
        impl.built,
    )
end

@testset "Rejected graphs leave smaller overlapping candidates available" begin
    x = reshape(collect(range(-0.4, 0.8; length = 12)), 3, 4)
    f(x) = begin
        h = sin.(x) .+ 0.2
        total = sum(h .* h; dims = 1)
        (h ./ (total .+ 0.1), h .+ total)
    end
    impl = GraphTestImplementation(member_limit = 3)
    ex = prepare_forward(f, x; implementations = (impl,))
    check_graph_test_outputs(execute!(ex), f(x))
    rejected = filter(r -> length(r.region.members) > impl.member_limit, impl.attempts)
    @test !isempty(rejected)
    @test !isempty(impl.built)
    @test any(
        large -> any(
            small -> issubset(Set(small.region.members), Set(large.region.members)),
            impl.built,
        ),
        rejected,
    )
    @test occursin("graph test construction rejection", sprint(explain, ex))
end

@testset "Graph fallback terminates and unexpected failures propagate" begin
    x = reshape(collect(range(0.2, 0.9; length = 12)), 3, 4)
    f(x) = x ./ (sum(abs2, x; dims = 1) .+ 0.1)
    for mode in (:reject, :late_reject)
        impl = GraphTestImplementation(; mode)
        ex = prepare_forward(f, x; implementations = (impl,))
        @test :graph_test ∉ ex.providers
        check_graph_test_outputs(execute!(ex), f(x))
        @test !isempty(impl.requests)
        if mode === :late_reject
            ids = map(r -> Tuple(r.region.members), impl.attempts)
            @test !isempty(ids)
            @test length(ids) == length(unique(ids))
            @test occursin("graph test construction rejection", sprint(explain, ex))
        else
            @test isempty(impl.attempts)
            @test occursin("graph test metadata rejection", sprint(explain, ex))
        end
    end
    broken = GraphTestImplementation(mode = :broken)
    @test_throws "graph test unexpected compiler failure" prepare_forward(
        f,
        x;
        implementations = (broken,),
    )
    @test length(broken.attempts) == 1
end

@testset "Graph regions compose with differentiation and staged donation" begin
    x = reshape(collect(range(-0.4, 0.6; length = 6)), 3, 2)
    loss(x) = sum(abs2, graph_test_normalization(x).y)
    impl = GraphTestImplementation()
    ex = prepare(loss, x; implementations = (impl,))
    actual = execute!(ex)
    @test :graph_test in ex.providers
    @test only(actual[1]) ≈ loss(x)
    @test actual[2] ≈ finite_difference(loss, (copy(x),), 1) rtol=2e-6 atol=2e-6

    update(x) = (x .+ sum(abs2, x; dims = 1), sum(x; dims = 1))
    initial = copy(x)
    state = copy(x)
    impl = GraphTestImplementation()
    ex = prepare_forward(update, x; implementations = (impl,), donate = (1,))
    @test :graph_test in ex.providers
    @test isempty(ex.storage.donations)
    @test !isempty(ex.state.commits)
    for _ = 1:3
        expected = update(state)
        check_graph_test_outputs(execute!(ex), expected)
        state = expected[1]
    end
    @test x == initial
end

@testset "Multiple graph regions remain jointly acyclic" begin
    x = reshape(collect(range(-0.2, 0.7; length = 12)), 3, 4)
    function crossed_reductions(x)
        a = sin.(x)
        b = cos.(x)
        (sum(a .+ b; dims = 1), sum(a .* b; dims = 1))
    end
    # Each three-node cluster can be valid in isolation. Selecting both may
    # create a cycle through their shared producers, so the second must shrink.
    impl = GraphTestImplementation(member_limit = 3)
    ex = prepare_forward(crossed_reductions, x; implementations = (impl,))
    check_graph_test_outputs(execute!(ex), crossed_reductions(x))
    regions = [step.region for step in ex.storage.steps if step.region isa Tempest.GraphRegion]
    @test length(regions) == 2
    @test all(r -> length(r.members) <= 3, regions)
    @test isempty(intersect(regions[1].members, regions[2].members))
end

@testset "Graph nodes retain intermediate element conversions" begin
    x = reshape(Float32.(range(0.2, 0.9; length = 12)), 3, 4)
    function converted_normalization(x)
        h = Float32.(Float64.(x) .+ 1e-8)
        (h ./ sum(abs2, h; dims = 1), h)
    end
    impl = GraphTestImplementation()
    ex = prepare_forward(converted_normalization, x; implementations = (impl,))
    actual = execute!(ex)
    expected = converted_normalization(x)
    @test :graph_test in ex.providers
    check_graph_test_outputs(actual, expected)
    @test actual[2] == expected[2]
    @test any(
        r -> any(id -> r.region.program.nodes[id].dtype === Float64, r.region.members),
        impl.built,
    )
end


struct GraphUnknownActivation end
(::GraphUnknownActivation)(x) = exp(0.2 * x)

function graph_centered_normalization(x)
    mean = sum(x; dims = 1) ./ size(x, 1)
    centered = x .- mean
    variance = sum(abs2, centered; dims = 1) ./ size(x, 1)
    centered ./ sqrt.(variance .+ 0.03)
end

@testset "Unsupported neighbors do not fragment a supported normalization" begin
    x = reshape(collect(range(0.1, 0.8; length = 20)), 5, 4)
    before(x) = graph_centered_normalization(GraphUnknownActivation().(x))
    after(x) = GraphUnknownActivation().(graph_centered_normalization(x))
    for f in (before, after)
        impl = GraphTestImplementation(mode = :row_sums)
        ex = prepare_forward(f, x; implementations = (impl,))
        check_graph_test_outputs(execute!(ex), f(x))
        region = only(impl.built).region
        nodes = region.program.nodes
        @test count(id -> nodes[id].op isa Tempest.SumReduction, region.members) == 2
        @test all(
            id -> !(
                nodes[id].op isa Tempest.Pointwise &&
                nodes[id].op.f isa GraphUnknownActivation
            ),
            region.members,
        )
        # Both placements preserve the whole normalization, including its final
        # division. The unsupported callable remains a separate execution unit.
        @test length(region.members) == 12
        @test count(step -> step.region isa Tempest.Elementwise, ex.storage.steps) == 1
        @test occursin("has not granted recomputation", sprint(explain, ex))
        @test all(r -> all(s -> s.dtype <: AbstractFloat, r.inputs), impl.requests)
    end
end

@testset "Other reduction axes do not cut off a supported epilogue" begin
    x = reshape(collect(range(0.1, 0.8; length = 20)), 5, 4)
    function branched(x)
        h = sin.(x)
        column = sum(h; dims = 1)
        row = sum(h; dims = 2)
        total = sum(h)
        y = x .* column
        (; y, row, total)
    end
    impl = GraphTestImplementation(mode = :row_sums)
    ex = prepare_forward(branched, x; implementations = (impl,))
    check_graph_test_outputs(execute!(ex), branched(x))
    region = only(impl.built).region
    @test ex.storage.program.primal.output.y in region.outputs
    @test all(
        id ->
            !(region.program.nodes[id].op isa Tempest.SumReduction) ||
            region.program.nodes[id].op.axes == (1,),
        region.members,
    )
end

@testset "Complete formulas remain eligible when their fragments are unsupported" begin
    x = reshape(collect(range(0.1, 0.8; length = 20)), 5, 4)
    impl = GraphTestImplementation(mode = :two_sums)
    ex = prepare_forward(graph_centered_normalization, x; implementations = (impl,))
    check_graph_test_outputs(execute!(ex), graph_centered_normalization(x))
    region = only(impl.built).region
    @test count(id -> region.program.nodes[id].op isa Tempest.SumReduction, region.members) ==
          2
end
