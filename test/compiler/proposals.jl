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
    pick::Any # program -> member vectors to propose instead of matching
    rounds::Int
    proposals::Vector{Vector{Int}}
    supported::Vector{Any}
    built::Vector{Any}
end

ProposingTestImplementation(;
    name = :proposing_test,
    mode = :accept,
    member_limit = typemax(Int),
    pick = nothing,
) = ProposingTestImplementation(name, mode, member_limit, pick, 0, Vector{Int}[], Any[], Any[])
Tempest.implementation_name(impl::ProposingTestImplementation) = impl.name

epilogue_operation(op) = op isa Union{Tempest.Pointwise, Tempest.SumReduction}

# Grow from each product along consumers, offering every prefix. Pointwise
# operands outside the region become its inputs; a sum ends the chain.
function Tempest.propose_regions(impl::ProposingTestImplementation, request::ProposalRequest)
    nodes = request.program.nodes
    impl.rounds += 1
    if impl.pick !== nothing
        return [RegionProposal(members; label = "picked") for members in impl.pick(request.program)]
    end
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
    # Larger alternatives of the forward product that claim backward work
    # save less traffic than the backward product's own epilogue.
    @test occursin("ranked higher", sprint(explain, ex))
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

@testset "Proposals that save more traffic win overlaps" begin
    # h feeds a product of its own and s, which only t reads. A claims h's
    # product and s; s then escapes to t, a write the default plan avoids by
    # inlining s into t. B claims t's product with s and t, keeping the product
    # internal. A has more members; B saves more traffic, so B is selected.
    function traffic(w, x, v, y)
        h = tanh.(2.0 .* (w * x))
        s = 1.0 .- h
        (h, (v * y) .* s)
    end
    w, x, v, y = (reshape(collect(range(-0.5, 0.5; length = 9)) .* k, 3, 3) for k in 1:4)
    function pick(program)
        nodes = program.nodes
        products = findall(n -> n.op isa Tempest.MatMul, nodes)
        operation(f) = findfirst(n -> n.op isa Tempest.Pointwise && n.op.f === f && length(n.inputs) == 2, nodes)
        scale, s = operation(*), operation(-)
        t = findlast(n -> n.op isa Tempest.Pointwise && n.op.f === (*), nodes)
        tanh_id = findfirst(n -> n.op isa Tempest.Pointwise && n.op.f === tanh, nodes)
        [[products[1], scale, tanh_id, s], [products[2], s, t]]
    end
    impl = ProposingTestImplementation(; pick)
    ex = prepare_forward(traffic, w, x, v, y; implementations = (impl,))
    check_graph_test_outputs(execute!(ex), traffic(w, x, v, y))
    region = only(proposed_steps(ex)).region
    nodes = ex.storage.program.nodes
    products = findall(n -> n.op isa Tempest.MatMul, nodes)
    @test products[2] in region.members
    @test products[1] ∉ region.members
    @test length(region.outputs) == 1
    @test occursin("conflict: overlaps selected proposal", sprint(explain, ex))
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

# A named call claimed with its epilogue, the shape of an attention kernel that
# also applies the pointwise work on its output. The call must survive semantic
# resolution on its own (some implementation supports it alone); a proposal can
# then claim it together with its consumers, as a product is claimed.
wave_pair(x) = Tempest.reference(wave_pair, x)
wave_pair(x::Tempest.TracedArray) = Tempest.record_call(wave_pair, x)
Tempest.output_spec(::typeof(wave_pair), x) = (; s = Tempest.TensorSpec(x), c = Tempest.TensorSpec(x))
Tempest.reference(::typeof(wave_pair), x) = (; s = sin.(x), c = cos.(x))
Tempest.verb_pullback(::typeof(wave_pair), (x,), y, dy) = (dy.s .* y.c .- dy.c .* y.s,)

mutable struct CallEpilogueKernels <: Implementation
    mode::Symbol
end
Tempest.implementation_name(::CallEpilogueKernels) = :call_epilogue

function Tempest.support(::CallEpilogueKernels, request::Tempest.CallRequest)
    request.call.f === wave_pair ? Supported() : Unsupported("not a wave pair")
end

function Tempest.build_implementation(::CallEpilogueKernels, ::Tempest.CallRequest, runtime, out, inputs)
    x = only(inputs)
    BuiltImplementation(() -> (out.s .= sin.(x); out.c .= cos.(x); nothing), (out, inputs), 0)
end

# Propose the call's `s` result alone with the pointwise work that consumes the
# call; the partitioner adds the sibling result `c`.
function Tempest.propose_regions(::CallEpilogueKernels, request::ProposalRequest)
    nodes = request.program.nodes
    proposals = RegionProposal[]
    for (id, node) in enumerate(nodes)
        request.live[id] && node.op isa Tempest.CallResult && node.op.call.f === wave_pair || continue
        leaves = Tempest.call_ids(node.op.call)
        id == first(leaves) || continue
        members = [id]
        frontier = [leaves...]
        while !isempty(frontier)
            for user in request.users[popfirst!(frontier)]
                user in members && continue
                nodes[user].op isa Tempest.Pointwise || continue
                push!(members, user)
                push!(frontier, user)
                push!(proposals, RegionProposal(members; label = "wave pair epilogue"))
            end
        end
    end
    proposals
end

function Tempest.support(impl::CallEpilogueKernels, request::ProposedRequest)
    impl.mode === :decline && return Unsupported("call epilogue declined")
    Supported()
end

function Tempest.build_implementation(::CallEpilogueKernels, request::ProposedRequest, runtime, outputs, inputs)
    region = request.region
    nodes = region.program.nodes
    launch = () -> begin
        values = Dict{Int, Any}(zip(region.inputs, inputs))
        for id in region.members
            haskey(values, id) && continue
            node = nodes[id]
            operands = map(i -> values[i], node.inputs)
            if node.op isa Tempest.CallResult
                call = node.op.call
                result = Tempest.reference(call.f, operands...; call.kwargs...)
                foreach((leaf, value) -> values[leaf] = value, Tempest.call_ids(call), Tempest.tree_leaves(result))
            else
                values[id] = Tempest.evaluate(node.op, node, operands)
            end
        end
        foreach((output, id) -> copyto!(output, values[id]), outputs, region.outputs)
        nothing
    end
    BuiltImplementation(launch, (region, inputs, outputs), 0)
end

wave_epilogue(x) = (w = wave_pair(x); tanh.(w.s .* w.c .+ 0.5) .* 2)

@testset "Proposals claim semantic calls with their epilogues" begin
    x = reshape(collect(range(-1.0, 1.2; length = 12)), 3, 4)
    expected = tanh.(sin.(x) .* cos.(x) .+ 0.5) .* 2
    ex = prepare_forward(wave_epilogue, x; implementations = (CallEpilogueKernels(:accept),))
    @test execute!(ex) ≈ expected
    @test length(ex.storage.steps) == 1
    region = only(proposed_steps(ex)).region
    @test count(id -> region.program.nodes[id].op isa Tempest.CallResult, region.members) == 2
    @test occursin("wave pair epilogue", sprint(explain, ex))
    # Declined: the call runs alone and the epilogue in the default partition.
    ex = prepare_forward(wave_epilogue, x; implementations = (CallEpilogueKernels(:decline),))
    @test execute!(ex) ≈ expected
    @test [Tempest.region_name(s.region) for s in ex.storage.steps] == [:call, :pointwise]
    # Training: the region writes the call results that the pullback reads.
    loss(x) = sum(abs2, wave_epilogue(x))
    d = differentiate(trace(loss, x))
    value, (dx,) = value_and_gradient(d, x)
    ex = prepare(loss, x; implementations = (CallEpilogueKernels(:accept),))
    got = execute!(ex)
    @test got[1][] ≈ value
    @test got[2] ≈ dx
    @test !isempty(proposed_steps(ex))
end

# A proposer that estimates its launches is held to the cost check of graph
# regions; one that gives no estimate is not (src/compiler/costs.jl).
struct PricedProposalTarget <: Tempest.ExecutionTarget end
Tempest.cost_rates(::PricedProposalTarget) = Tempest.CostRates(1.0e9, 1.0e-6, 1.0e-6, "test rates")

struct PricedProposer <: Implementation
    kernels::Any
end
Tempest.propose_regions(::PricedProposer, request::ProposalRequest) =
    [RegionProposal(findall(n -> n.op isa Union{Tempest.MatMul, Tempest.Pointwise}, request.program.nodes))]
Tempest.support(::PricedProposer, ::ProposedRequest) = Supported()
Tempest.region_cost(proposer::PricedProposer, ::ProposedRequest) = proposer.kernels
Tempest.support(::PricedProposer, ::RegionRequest) = Supported()
Tempest.region_cost(::PricedProposer, ::RegionRequest) = [Tempest.KernelCost(1000, 1.0e9)]

@testset "Proposals that estimate their launches are checked" begin
    x, w = reshape(collect(1.0:12.0), 3, 4) ./ 7, reshape(collect(1.0:6.0), 2, 3) ./ 5
    program = Tempest.ForwardProgram(trace((x, w) -> tanh.(w * x) .+ 1.0, x, w))
    select(kernels) = begin
        records = Tempest.ProposalRecord[]
        regions = Tempest.select_proposed_regions(
            program, PricedProposalTarget(), (PricedProposer(kernels),), Dict{Tuple, String}(), records,
        )
        (regions, records)
    end
    regions, records = select([Tempest.KernelCost(1000, 1.0e9)])
    @test length(regions) == 1
    @test occursin("kept by estimate", only(records).reason)
    regions, records = select([Tempest.KernelCost(1.0e9, 1.0e9)])
    @test isempty(regions)
    @test only(records).status === :declined
    @test occursin("declined by estimate", only(records).reason)
    regions, records = select(nothing)
    @test length(regions) == 1 && isempty(only(records).reason)
end
