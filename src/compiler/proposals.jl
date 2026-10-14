# Implementation-proposed regions. Growth-based graph regions are formed by
# Tempest and stop at contractions. A proposal instead comes from an
# implementation that recognized a pattern in the prepared program, such as a
# matrix product with its epilogue, and asks to realize exactly that set of
# nodes. The partitioner chooses among proposals; whatever it does not select,
# and every proposal that fails, stays with the default partition.

"""
    RegionProposal(members; label = "")

A set of original value IDs an implementation offers to realize as one region.
Members may be pointwise operations, constants, folds and matrix products, and
must form a dependency-closed unit: every path from one member to another stays
inside the proposal, which the partitioner checks. Constants used only by
members join the proposal. Inputs and escaping outputs follow from the program,
as for graph regions. `label` names the pattern in `explain`.
"""
struct RegionProposal
    members::Vector{Int}
    label::String

    function RegionProposal(members; label::AbstractString = "")
        members = sort!(unique!(Int[members...]))
        isempty(members) && throw(ArgumentError("a region proposal needs at least one member"))
        new(members, String(label))
    end
end

"""
    ProposalRequest

The target and the prepared program offered to `propose_regions`. `program` is
the program after differentiation and semantic resolution, so forward and
backward work are both visible. `live` marks the values the program needs and
`users` lists each value's live consumers.
"""
struct ProposalRequest
    target::ExecutionTarget
    program::ExecutionProgram
    live::BitVector
    users::Vector{Vector{Int}}
end

"""
    propose_regions(implementation, request::ProposalRequest) -> collection of RegionProposal

Patterns `implementation` offers to realize, before storage is planned. Return
nested alternatives freely: the partitioner prefers larger proposals and then
asks `support(implementation, ::ProposedRequest)` about each with its complete
boundary. Defining this method is the participation mechanism. The default
proposes nothing. Proposals must depend only on the request, so that the same
program always gives the same partition.
"""
propose_regions(::Implementation, ::ProposalRequest) = RegionProposal[]

"""
A region selected from an implementation's proposal. It has the fields of a
`GraphRegion` and, in addition, the `implementation` that proposed it, which
alone builds it, and the proposal's `label`. Members may include matrix
products. An input read only by matrix products keeps its native presentation,
dense or a transpose wrapper; every other input is a dense buffer.
"""
struct ProposedRegion <: Region
    program::ExecutionProgram
    members::Vector{Int}
    inputs::Vector{Int}
    outputs::Vector{Int}
    implementation::Implementation
    label::String
end

region_name(::ProposedRegion) = :proposed_region

"""
    ProposedRequest

A `ProposedRegion` offered through `support` and `build_implementation` to the
implementation that proposed it. `inputs` and `outputs` are ordered TensorSpecs.
Construction receives matching tuples of logically shaped buffers, as for
`GraphRequest`; an input read only by matrix products may be a `Transpose` of a
dense buffer. Every output must be written, including values that escape only
to backward. Unsupported from support or construction leaves the members to the
default partition.
"""
struct ProposedRequest <: ImplementationRequest
    target::ExecutionTarget
    region::ProposedRegion
    inputs::Tuple
    outputs::Tuple
end

const SelectedRegion = Union{GraphRegion, ProposedRegion}

"""
The outcome of one proposal in the final partition, for `explain`. `status` is
`:selected`, `:declined` (support rejected it), `:conflict` (it overlaps a
preferred proposal or would create a dependency cycle) or `:fell_back`
(construction rejected it and preparation replanned without it).
"""
struct ProposalRecord
    implementation::Symbol
    label::String
    members::Vector{Int}
    status::Symbol
    reason::String
end

proposal_operation(op) = op isa Union{Pointwise, Constant, FoldReduction, MatMul}

# An input stays native when every member reading it accepts a native
# presentation without requiring a dense one.
function native_region_input(region::ProposedRegion, id)
    nodes = region.program.nodes
    all(region.members) do member
        op = nodes[member].op
        id in nodes[member].inputs || return true
        requires_native_inputs(op) && !requires_dense_inputs(op)
    end
end

function proposed_region(program, members, implementation, label, live)
    boundary = region_boundary(program, members, live)
    ProposedRegion(program, boundary.members, boundary.inputs, boundary.outputs, implementation, label)
end

function proposed_request(target, region::ProposedRegion)
    nodes = region.program.nodes
    spec(id) = TensorSpec(nodes[id].dtype, nodes[id].dims; scalar = nodes[id].scalar)
    ProposedRequest(target, region, Tuple(spec.(region.inputs)), Tuple(spec.(region.outputs)))
end

function proposal_users(program, live)
    users = [Int[] for _ in program.nodes]
    for (id, node) in enumerate(program.nodes)
        live[id] || continue
        foreach(input -> push!(users[input], id), node.inputs)
    end
    users
end

# Like graph providers, an implementation participates by defining the method.
function proposal_implementations(implementations)
    fallback = which(propose_regions, Tuple{Implementation, ProposalRequest})
    filter(implementations) do implementation
        which(propose_regions, Tuple{typeof(implementation), ProposalRequest}) !== fallback
    end
end

# A malformed proposal is a defect in its implementation, not a reason to fall back.
function check_proposal(program, live, proposal, implementation)
    proposal isa RegionProposal ||
        error("propose_regions must return RegionProposals, got $(typeof(proposal))")
    for id in proposal.members
        1 <= id <= length(program.nodes) ||
            throw(ArgumentError("$(implementation_name(implementation)) proposed invalid value %$id"))
        live[id] ||
            throw(ArgumentError("$(implementation_name(implementation)) proposed dead value %$id"))
        proposal_operation(program.nodes[id].op) || throw(
            ArgumentError(
                "$(implementation_name(implementation)) proposed %$id, " *
                "a $(operation_name(program.nodes[id].op)) value, which a region cannot contain",
            ),
        )
    end
    proposal
end

# Bytes of memory traffic a region saves against the default partition: the
# values the default schedule writes to memory (its fusion roots) that the
# region keeps internal, minus values the default would have inlined that the
# region must write because they escape it. This is the only benefit estimate;
# it compares proposals without a cost model of their implementations.
function proposal_benefit(program, members, live, roots)
    outputs = region_boundary(program, members, live).outputs
    bytes(id) = sizeof(program.nodes[id].dtype) * prod(program.nodes[id].dims; init = 1)
    saved = sum(id -> roots[id] && id ∉ outputs ? bytes(id) : 0, members; init = 0)
    added = sum(id -> roots[id] ? 0 : bytes(id), outputs; init = 0)
    saved - added
end

function default_roots(program, live)
    (; layouts, materialized) = plan_layouts(program, live)
    fusion_roots(program.nodes, live, materialized, layouts, program.results)
end

"""
    select_proposed_regions(program, target, implementations, blocked, records)

Collect every implementation's proposals and choose a non-overlapping, acyclic
subset. Implementations keep their candidate order as priority. Within one
implementation, proposals that save more memory traffic against the default
partition come first (`proposal_benefit`, in bytes), then larger ones, then
lower value IDs. Each proposal is then accepted when it is not blocked by an
earlier construction failure, overlaps no accepted proposal, keeps the
partition acyclic, and its proposer supports the complete request. Every
outcome is appended to `records`.
"""
function select_proposed_regions(program, target, implementations, blocked, records)
    proposers = proposal_implementations(implementations)
    isempty(proposers) && return ProposedRegion[]
    live = live_nodes(program.nodes, program.results)
    request = ProposalRequest(target, program, live, proposal_users(program, live))
    candidates = Tuple{Int, Any, RegionProposal}[]
    unclaimed = falses(length(program.nodes))
    for (priority, implementation) in enumerate(proposers)
        seen = Set{Vector{Int}}()
        for proposal in propose_regions(implementation, request)
            check_proposal(program, live, proposal, implementation)
            # Private literals enter with their consumers, as in graph regions.
            members = include_private_literals(program, proposal.members, request.users, unclaimed)
            proposal = RegionProposal(members; proposal.label)
            proposal.members in seen && continue
            push!(seen, proposal.members)
            push!(candidates, (priority, implementation, proposal))
        end
    end
    roots = default_roots(program, live)
    ranks = map(candidates) do (priority, _, p)
        (priority, -proposal_benefit(program, p.members, live, roots), -length(p.members), p.members)
    end
    candidates = candidates[sortperm(ranks)]

    claimed = zeros(Int, length(program.nodes))
    regions = ProposedRegion[]
    for (priority, implementation, proposal) in candidates
        name = implementation_name(implementation)
        members = proposal.members
        record(status, reason = "") =
            push!(records, ProposalRecord(name, proposal.label, members, status, reason))
        owners = unique(filter(!iszero, claimed[members]))
        if !isempty(owners)
            selected = regions[first(owners)]
            if selected.implementation === implementation
                # A smaller alternative of a selected proposal is no conflict.
                issubset(members, selected.members) && continue
                if issubset(selected.members, members)
                    record(:conflict, "its alternative $(selected.members) ranked higher")
                    continue
                end
            end
            record(:conflict, "overlaps selected proposal $(selected.members)")
            continue
        end
        if haskey(blocked, Tuple(members))
            record(:fell_back, blocked[Tuple(members)])
            continue
        end
        groups = [[region.members for region in regions]; [members]]
        if graph_partition_order(program, groups, live) === nothing
            record(:conflict, "would create a dependency cycle")
            continue
        end
        region = proposed_region(program, members, implementation, proposal.label, live)
        verdict = checked_support(implementation, proposed_request(target, region))
        if verdict isa Unsupported
            record(:declined, verdict.reason)
            continue
        end
        push!(regions, region)
        claimed[members] .= length(regions)
        record(:selected)
    end
    regions
end

function explain_proposals(io, records, steps)
    isempty(records) && return
    println(io, "Region proposals:")
    for record in records
        text = "  $(record.implementation) $(record.label) $(record.members): "
        if record.status === :selected
            index = findfirst(step -> step.region isa ProposedRegion &&
                                  step.region.members == record.members, steps)
            text *= index === nothing ? "selected" : "selected as region $index"
        elseif record.status === :fell_back
            text *= "fell back after construction: $(record.reason)"
        else
            text *= "$(record.status): $(record.reason)"
        end
        println(io, text)
    end
end
