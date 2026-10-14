# Automatic research candidates on Tempest's semantic graph. No kernel or
# backend eligibility is implied by membership in one of these regions.
if !isdefined(@__MODULE__, :GraphRegions)
    include("graph_regions.jl")
end

module GraphCandidates

using Tempest
using ..GraphRegions
const D = Tempest

export reduction_candidates, standalone_program

candidate_operation(op) = op isa Union{D.Pointwise, D.SumReduction, D.Constant}

function candidate_neighbors(program, live)
    neighbors = [Int[] for _ in program.nodes]
    for (id, node) in enumerate(program.nodes)
        live[id] && candidate_operation(node.op) || continue
        for input in node.inputs
            candidate_operation(program.nodes[input].op) || continue
            push!(neighbors[id], input)
            push!(neighbors[input], id)
        end
    end
    neighbors
end

# Check the entire proposed partition. Individually convex regions can still
# produce a cycle when several are contracted together.
function partition_units(program, groups, live)
    owners = Dict(id => group for group in groups for id in group)
    assigned = falses(length(program.nodes))
    units = Vector{Int}[]
    for (id, node) in enumerate(program.nodes)
        (!live[id] || assigned[id] || node.op isa D.Argument) && continue
        unit = if haskey(owners, id)
            owners[id]
        elseif node.op isa D.CallResult
            sort(D.call_ids(node.op.call))
        else
            [id]
        end
        push!(units, unit)
        assigned[unit] .= true
    end
    units
end

function can_contract(program, groups, live)
    units = partition_units(program, groups, live)
    try
        GraphRegions.order_units(program.nodes, units)
        true
    catch err
        # order_units has one deliberate ArgumentError: a contraction cycle.
        # No provider, user function, or layout transform is called here.
        err isa ArgumentError || rethrow()
        false
    end
end

"""
    reduction_candidates(program) -> Vector{RegionBoundary}

Greedily grow connected regions from sum reductions using direct graph edges.
Pointwise operations and constants may join a region; arguments, layouts,
matmul, semantic calls, and other operations remain boundaries. Dead work is
ignored. A scalar callable is retained without an operation-name whitelist.

The resulting selections are disjoint and jointly acyclic, returned in a valid
execution order. Growth may revisit a neighbor after another addition makes its
contraction legal. This small-graph experiment does not promise a maximal or
optimal partition, prove purity, or certify any backend's reduction schedule.
"""
function reduction_candidates(source)
    program = GraphRegions.execution_program(source)
    live = D.live_nodes(program.nodes, program.results)
    neighbors = candidate_neighbors(program, live)
    claimed = falses(length(program.nodes))
    groups = Vector{Int}[]

    for (seed, node) in enumerate(program.nodes)
        live[seed] && !claimed[seed] && node.op isa D.SumReduction || continue
        members = [seed]
        changed = true
        while changed
            changed = false
            frontier = sort!(unique!([
                next for id in members for next in neighbors[id]
                if !claimed[next] && next ∉ members
            ]))
            for next in frontier
                proposed = sort!([members; next])
                can_contract(program, [groups; [proposed]], live) || continue
                members = proposed
                changed = true
            end
        end
        push!(groups, members)
        claimed[members] .= true
    end

    units = partition_units(program, groups, live)
    order = GraphRegions.order_units(program.nodes, units)
    heads = Set(first(group) for group in groups)
    [region_boundary(program, unit) for unit in order if first(unit) in heads]
end

"""
    standalone_program(boundary)

Return `(program, original_to_local, local_to_original)`. Each ordered boundary
input becomes one positional argument; outputs are always an ordered tuple.
Every body node retains its original Operation object, including Julia callable
values, and only its operand IDs are translated. Boundary input operations are
replaced with Argument nodes. No expression is retraced or rewritten.

Automatic candidates contain no semantic calls. Arbitrary hand-selected calls
are rejected here because their result IDs also live inside SemanticCall; they
would require a separate atomic-call remapping, not just changing Node.inputs.
"""
function standalone_program(boundary::RegionBoundary)
    source = boundary.program.nodes
    any(id -> source[id].op isa D.CallResult, boundary.members) &&
        throw(ArgumentError("standalone cloning of semantic calls is outside this probe"))
    original_ids = [boundary.inputs; boundary.members]
    original_to_local = Dict(id => local_id for (local_id, id) in enumerate(original_ids))
    nodes = D.Node[]
    for id in boundary.inputs
        node = source[id]
        push!(nodes, D.Node(D.Argument(), (), node.dims, node.dtype, node.scalar))
    end
    for id in boundary.members
        node = source[id]
        inputs = map(input -> original_to_local[input], node.inputs)
        push!(nodes, D.Node(node.op, inputs, node.dims, node.dtype, node.scalar))
    end
    arguments = collect(1:length(boundary.inputs))
    output = Tuple(original_to_local[id] for id in boundary.outputs)
    signatures = Tuple{DataType, Tuple}[(nodes[id].dtype, nodes[id].dims) for id in arguments]
    program = D.TensorProgram(nodes, arguments, output, signatures, Tuple(arguments))
    (; program, original_to_local, local_to_original = original_ids)
end

end # module
