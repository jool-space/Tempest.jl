# A CPU provider that proposes matrix products with their pointwise and sum
# epilogues, the shape of a library epilogue fusion. Its interpreter is
# independent of scheduling and storage, so these tests exercise how proposals
# are selected, bounded, built and replaced by the default partition.
include("../fixtures/graph_regions.jl")
using Tempest: RegionProposal, ProposalRequest, ProposedRegion, ProposedRequest
using LinearAlgebra: Transpose

mutable struct ProposingTestImplementation <: Implementation
    name::Symbol
    mode::Symbol
    member_limit::Int
    rounds::Int
    proposals::Vector{Vector{Int}}
    supported::Vector{Any}
    built::Vector{Any}
end

ProposingTestImplementation(; name = :proposing_test, mode = :accept, member_limit = typemax(Int)) =
    ProposingTestImplementation(name, mode, member_limit, 0, Vector{Int}[], Any[], Any[])
Tempest.implementation_name(impl::ProposingTestImplementation) = impl.name

epilogue_operation(op) = op isa Union{Tempest.Pointwise, Tempest.SumReduction}

# Grow from each product along consumers, offering every prefix. Pointwise
# operands outside the region become its inputs; a sum ends the chain.
function Tempest.propose_regions(impl::ProposingTestImplementation, request::ProposalRequest)
    nodes = request.program.nodes
    impl.rounds += 1
    proposals = RegionProposal[]
    for (anchor, node) in enumerate(nodes)
        request.live[anchor] && node.op isa Tempest.MatMul || continue
        members = [anchor]
        open = [anchor]
        while !isempty(open)
            id = popfirst!(open)
            for user in request.users[id]
                user in members && continue
                epilogue_operation(nodes[user].op) || continue
                push!(members, user)
                nodes[user].op isa Tempest.SumReduction || push!(open, user)
                push!(proposals, RegionProposal(members; label = "matmul epilogue"))
            end
        end
    end
    impl.mode === :reversed && reverse!(proposals)
    append!(impl.proposals, (p.members for p in proposals))
    proposals
end

function Tempest.support(impl::ProposingTestImplementation, request::ProposedRequest)
    push!(impl.supported, request)
    request.target isa CPUTarget || return Unsupported("proposal test provider requires CPU")
    request.region.implementation === impl || return Unsupported("proposed by another implementation")
    impl.mode === :decline && return Unsupported("proposal test metadata rejection")
    if impl.mode === :no_sums
        any(id -> request.region.program.nodes[id].op isa Tempest.SumReduction, request.region.members) &&
            return Unsupported("proposal test declines sums")
    end
    Supported()
end

function Tempest.build_implementation(
    impl::ProposingTestImplementation,
    request::ProposedRequest,
    runtime,
    outputs,
    inputs,
)
    impl.mode === :broken && error("proposal test unexpected compiler failure")
    if impl.mode === :late_reject || length(request.region.members) > impl.member_limit
        return Unsupported("proposal test construction rejection")
    end
    region = request.region
    @test length(inputs) == length(region.inputs) == length(request.inputs)
    @test length(outputs) == length(region.outputs) == length(request.outputs)
    for (array, spec) in zip((inputs..., outputs...), (request.inputs..., request.outputs...))
        @test size(array) == spec.dims
        @test eltype(array) === spec.dtype
    end
    @test all(!Base.mightalias(out, input) for out in outputs for input in inputs)
    push!(impl.built, (request, map(typeof, inputs)))
    launch = () -> begin
        values = graph_test_values(region, inputs)
        for (output, id) in zip(outputs, region.outputs)
            region.program.nodes[id].scalar ? (output[] = values[id]) : copyto!(output, values[id])
        end
        nothing
    end
    BuiltImplementation(launch, (region, inputs, outputs), 0)
end

proposal_mlp(w1, b1, w2, b2, x) = begin
    h = tanh.(w1 * x .+ b1)
    y = tanh.(w2 * h .+ b2)
    (; y, row = sum(w2 * h; dims = 2))
end
proposal_loss(w1, b1, w2, b2, x) = sum(abs2, proposal_mlp(w1, b1, w2, b2, x).y)

proposal_arguments() = (
    reshape(collect(range(-0.5, 0.6; length = 12)), 4, 3),
    collect(range(-0.2, 0.1; length = 4)),
    reshape(collect(range(0.3, -0.4; length = 8)), 2, 4),
    [0.1, -0.05],
    reshape(collect(range(-0.7, 0.9; length = 15)), 3, 5),
)

proposed_steps(ex) = filter(step -> step.region isa ProposedRegion, ex.storage.steps)
partition(ex) = [(Tempest.region_name(s.region), s.region isa Tempest.SelectedRegion ? s.region.members : s.outputs) for s in ex.storage.steps]

@testset "Proposals claim matrix products and their epilogues" begin
    args = proposal_arguments()
    impl = ProposingTestImplementation()
    ex = prepare_forward(proposal_mlp, args...; implementations = (impl,))
    check_graph_test_outputs(execute!(ex), proposal_mlp(args...))
    steps = proposed_steps(ex)
    nodes = ex.storage.program.nodes
    # Both forward layers and the separate row sum product are claimed whole.
    @test length(steps) == 3
    @test all(s -> count(id -> nodes[id].op isa Tempest.MatMul, s.region.members) == 1, steps)
    @test !any(s -> s.region isa Tempest.GemmRegion, ex.storage.steps)
    @test any(s -> any(id -> nodes[id].op isa Tempest.SumReduction, s.region.members), steps)
    @test all(s -> s.region.implementation === impl, steps)
    report = sprint(explain, ex)
    @test occursin("Region proposals:", report)
    @test occursin("proposing_test matmul epilogue", report)
    @test occursin("selected as region", report)
    @test occursin("proposed_region [proposing_test] matmul epilogue", report)

    disabled = ProposingTestImplementation()
    ordinary = prepare_forward(proposal_mlp, args...; implementations = (disabled,), fuse_regions = false)
    @test isempty(disabled.proposals)
    @test isempty(proposed_steps(ordinary))
    check_graph_test_outputs(execute!(ordinary), proposal_mlp(args...))
end

@testset "Proposals cover backward work and keep values backward needs" begin
    args = proposal_arguments()
    impl = ProposingTestImplementation()
    ex = prepare(proposal_loss, args...; implementations = (impl,))
    reference = value_and_gradient(differentiate(trace(proposal_loss, args...)), args...)
    actual = execute!(ex)
    @test only(actual[1]) ≈ reference[1]
    for (observed, expected) in zip(actual[2:end], reference[2])
        @test observed ≈ expected
    end
    @test only(actual[1]) ≈ proposal_loss(args...)
    @test actual[1 + 1] ≈ finite_difference(proposal_loss, map(copy, args), 1) rtol=1e-5 atol=1e-6

    program = ex.storage.program
    primal_count = length(program.primal.nodes)
    steps = proposed_steps(ex)
    # The forward activation is an escaping output: backward reads it.
    @test any(s -> any(id -> id <= primal_count, s.region.members) && length(s.region.outputs) >= 1, steps)
    @test any(s -> all(id -> id > primal_count, s.region.members), steps)
    # The backward product reads a transposed weight natively; operands read by
    # pointwise members are dense buffers.
    @test any(((request, types),) -> any(T -> T <: Transpose, types), impl.built)
    for (request, types) in impl.built
        region = request.region
        for (id, T) in zip(region.inputs, types)
            if !Tempest.native_region_input(region, id)
                @test !(T <: Transpose)
            end
        end
    end
end

@testset "Support and construction rejections fall back to the default partition" begin
    args = proposal_arguments()
    expected = proposal_mlp(args...)

    declined = ProposingTestImplementation(mode = :decline)
    ex = prepare_forward(proposal_mlp, args...; implementations = (declined,))
    check_graph_test_outputs(execute!(ex), expected)
    @test isempty(proposed_steps(ex))
    @test isempty(declined.built)
    @test count(s -> s.region isa Tempest.GemmRegion, ex.storage.steps) == 3
    @test occursin("declined: proposal test metadata rejection", sprint(explain, ex))

    # Larger alternatives that are declined leave nested ones eligible.
    no_sums = ProposingTestImplementation(mode = :no_sums)
    ex = prepare_forward(proposal_mlp, args...; implementations = (no_sums,))
    check_graph_test_outputs(execute!(ex), expected)
    nodes = ex.storage.program.nodes
    @test length(proposed_steps(ex)) == 2
    @test count(s -> s.region isa Tempest.GemmRegion, ex.storage.steps) == 1
    @test !any(s -> any(id -> nodes[id].op isa Tempest.SumReduction, s.region.members), proposed_steps(ex))

    late = ProposingTestImplementation(mode = :late_reject)
    ex = prepare_forward(proposal_mlp, args...; implementations = (late,))
    check_graph_test_outputs(execute!(ex), expected)
    @test isempty(proposed_steps(ex))
    @test :proposing_test ∉ ex.providers
    report = sprint(explain, ex)
    @test occursin("fell back after construction", report)
    @test occursin("proposal test construction rejection", report)
    # Each replan excludes every region that failed to build: the three largest
    # proposals, then the two nested ones, then none remain.
    @test late.rounds == 3

    # A failed build excludes that exact member set; smaller proposals remain.
    limited = ProposingTestImplementation(member_limit = 2)
    ex = prepare_forward(proposal_mlp, args...; implementations = (limited,))
    check_graph_test_outputs(execute!(ex), expected)
    @test !isempty(proposed_steps(ex))
    @test all(s -> length(s.region.members) <= 2, proposed_steps(ex))
    @test occursin("fell back after construction", sprint(explain, ex))

    broken = ProposingTestImplementation(mode = :broken)
    @test_throws "proposal test unexpected compiler failure" prepare_forward(
        proposal_mlp, args...; implementations = (broken,),
    )
end

@testset "Priority, overlap and cycles decide between proposals" begin
    args = proposal_arguments()
    first = ProposingTestImplementation(name = :first)
    second = ProposingTestImplementation(name = :second)
    ex = prepare_forward(proposal_mlp, args...; implementations = (first, second))
    check_graph_test_outputs(execute!(ex), proposal_mlp(args...))
    @test all(s -> s.region.implementation === first, proposed_steps(ex))
    @test isempty(second.built)
    report = sprint(explain, ex)
    @test occursin("second matmul epilogue", report)
    @test occursin("conflict: overlaps selected proposal", report)

    # The first implementation's priority holds even when it declines: the
    # second then realizes the same patterns.
    declining = ProposingTestImplementation(name = :first, mode = :decline)
    second = ProposingTestImplementation(name = :second)
    ex = prepare_forward(proposal_mlp, args...; implementations = (declining, second))
    @test !isempty(proposed_steps(ex))
    @test all(s -> s.region.implementation === second, proposed_steps(ex))

    # u = w*x feeds v = u .+ (u * w')', which reads u through another product.
    # A region {w*x, +} would then depend on itself through that product.
    crossing(w, x) = begin
        u = w * x
        u .+ transpose(transpose(u) * w)
    end
    w = reshape(collect(range(0.1, 0.9; length = 9)), 3, 3)
    x = reshape(collect(range(-0.3, 0.5; length = 9)), 3, 3)
    impl = ProposingTestImplementation()
    ex = prepare_forward(crossing, w, x; implementations = (impl,))
    check_graph_test_outputs(execute!(ex), crossing(w, x))
    @test occursin("conflict: would create a dependency cycle", sprint(explain, ex))
end

@testset "Partitions are deterministic" begin
    args = proposal_arguments()
    runs = map((:accept, :accept, :reversed)) do mode
        impl = ProposingTestImplementation(; mode)
        partition(prepare(proposal_loss, args...; implementations = (impl,)))
    end
    @test runs[1] == runs[2] == runs[3]
end

@testset "Proposals coexist with growth-based graph regions" begin
    args = proposal_arguments()
    f(w1, b1, w2, b2, x) = begin
        h = tanh.(w1 * x .+ b1)
        g = sin.(x)
        (; h, g = g ./ sum(abs2, g; dims = 1))
    end
    proposer = ProposingTestImplementation()
    grower = GraphTestImplementation()
    ex = prepare_forward(f, args...; implementations = (proposer, grower))
    check_graph_test_outputs(execute!(ex), f(args...))
    @test length(proposed_steps(ex)) == 1
    @test count(s -> s.region isa Tempest.GraphRegion, ex.storage.steps) == 1
end

struct MalformedProposer <: Implementation
    pick::Symbol
end

function Tempest.propose_regions(impl::MalformedProposer, request::ProposalRequest)
    nodes = request.program.nodes
    id = impl.pick === :argument ? findfirst(n -> n.op isa Tempest.Argument, nodes) :
        findfirst(n -> n.op isa Tempest.TransposeOp, nodes)
    [RegionProposal([id])]
end

@testset "Malformed proposals are errors" begin
    w = ones(2, 2)
    @test_throws "a argument value" prepare_forward(w -> w * w, w; implementations = (MalformedProposer(:argument),))
    @test_throws "a transpose value" prepare_forward(w -> transpose(w) * w, w; implementations = (MalformedProposer(:transpose),))
    @test_throws ArgumentError RegionProposal(Int[])
end

@testset "Equivalent proposed regions share a standard numbering" begin
    layers(w, x) = tanh.(w * tanh.(w * x .+ 0.5) .+ 0.5)
    w = reshape(collect(range(-0.4, 0.4; length = 9)), 3, 3)
    x = reshape(collect(range(0.2, 0.6; length = 6)), 3, 2)
    impl = ProposingTestImplementation()
    ex = prepare_forward(layers, w, x; implementations = (impl,))
    regions = [s.region for s in proposed_steps(ex)]
    @test length(regions) == 2
    function signature(region)
        (; local_id, operands) = Tempest.standard_numbering(region)
        nodes = region.program.nodes
        sort!([(local_id[id], id in region.inputs ? nothing : nodes[id].op,
                nodes[id].dims, Tuple(local_id[i] for i in (id in region.inputs ? () : operands(id))))
               for id in keys(local_id)]; by = first)
    end
    @test signature(regions[1]) == signature(regions[2])
end
