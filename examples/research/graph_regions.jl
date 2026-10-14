# A CPU-only probe of region boundaries on Tempest's existing semantic graph.
# It adds no operation vocabulary, kernel schedule, or production interface.
module GraphRegions

using Tempest
const D = Tempest

export RegionBoundary, region_boundary, execute_partition, explain_boundary

"""
A graph selection and its boundary, retaining original logical value IDs.

`order` is an acyclic order of execution units after collapsing the selection.
Semantic calls remain atomic across their result leaves. This proves structural
schedulability only: it does not establish purity, backend support, layout
compatibility, legal floating-point transformations, or profitability.
"""
struct RegionBoundary{P <: D.ExecutionProgram}
    program::P
    members::Vector{Int}
    inputs::Vector{Int}
    outputs::Vector{Int}
    mappings::Dict{Int, Tuple{Int, D.IndexMap}}
    order::Vector{Vector{Int}}
end

execution_program(program::D.TensorProgram) = D.ForwardProgram(program)
execution_program(program::D.ExecutionProgram) = program

function check_members(program, members, live)
    isempty(members) && throw(ArgumentError("a region needs at least one operation"))
    for id in members
        checkbounds(program.nodes, id)
        live[id] || throw(ArgumentError("region contains dead value %$id"))
        node = program.nodes[id]
        node.op isa D.Argument && throw(ArgumentError("arguments are region inputs"))
        if node.op isa D.CallResult
            siblings = D.call_ids(node.op.call)
            all(in(members), siblings) ||
                throw(ArgumentError("a semantic call's result leaves must stay together"))
        end
    end
end

function execution_units(program, members, live)
    units = Vector{Int}[]
    assigned = falses(length(program.nodes))
    for (id, node) in enumerate(program.nodes)
        (!live[id] || assigned[id] || node.op isa D.Argument) && continue
        unit = if id in members
            members
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

function order_units(nodes, units)
    owner = zeros(Int, length(nodes))
    for (index, unit) in enumerate(units)
        owner[unit] .= index
    end
    dependencies = map(eachindex(units)) do index
        sort!(unique!([owner[input] for id in units[index] for input in nodes[id].inputs
                       if owner[input] != 0 && owner[input] != index]))
    end

    # Deliberately simple for the small research graphs. Earliest original
    # operation breaks ties; a production planner would use a ready queue.
    complete = falses(length(units))
    order = Vector{Int}[]
    while !all(complete)
        next = findfirst(eachindex(units)) do index
            !complete[index] && all(dep -> complete[dep], dependencies[index])
        end
        next === nothing && throw(ArgumentError(
            "collapsing this region creates a dependency cycle through outside operations"))
        push!(order, units[next])
        complete[next] = true
    end
    order
end

"""
    region_boundary(program, selected_ids)

Find logical external inputs and all outputs used outside the selected region.
A selection may span nonadjacent node IDs. A path that leaves and re-enters it
creates a cycle when collapsed and is rejected. Layout nodes remain logical
operations; neither an escaping alias nor a shared value implies a copy here.
An output view can refer to an internal logical root: a physical implementation
must later retain that storage or materialize a suitable output presentation.
"""
function region_boundary(source, selected_ids)
    program = execution_program(source)
    members = sort!(unique!(Int[selected_ids...]))
    live = D.live_nodes(program.nodes, program.results)
    check_members(program, members, live)
    selected = Set(members)

    inputs = sort!(unique!([input for id in members for input in program.nodes[id].inputs
                           if input ∉ selected]))
    escaping = Set(program.results)
    for (id, node) in enumerate(program.nodes)
        live[id] && id ∉ selected && union!(escaping, node.inputs)
    end
    outputs = filter(in(escaping), members)
    units = execution_units(program, members, live)
    order = order_units(program.nodes, units)
    # Maps are relative to logical producers, before any decision to copy or
    # assign storage. A square transpose still differs from its parent.
    mappings = Dict{Int, Tuple{Int, D.IndexMap}}()
    for id in eachindex(program.nodes)
        live[id] || continue
        mappings[id] = D.source_layout(program.nodes, mappings, id)
    end
    RegionBoundary(program, members, inputs, outputs, mappings, order)
end

function evaluate_unit(nodes, unit, external)
    local_values = Dict{Int, Any}()
    lookup(id) = haskey(local_values, id) ? local_values[id] : external[id]
    for id in unit
        haskey(local_values, id) && continue
        node = nodes[id]
        inputs = map(lookup, node.inputs)
        if node.op isa D.CallResult
            call = node.op.call
            result = D.check_verb_result(call, D.reference(call.f, inputs...; call.kwargs...))
            for (output, leaf) in zip(D.call_ids(call), D.tree_leaves(result))
                local_values[output] = leaf
            end
        else
            local_values[id] = D.evaluate(node.op, node, inputs)
        end
    end
    local_values
end

"""
Execute a proposed partition on CPU using Tempest's operation interpreter.
Only declared region inputs enter its local environment, and only declared
outputs escape. This tests boundary accounting independently of GPU lowering.
No tile, memory-reuse, or performance claims follow from this interpreter.
"""
function execute_partition(boundary::RegionBoundary, args...)
    program = boundary.program
    flat_args = D.validate(program.primal, args)
    values = Dict{Int, Any}(zip(program.primal.arguments, flat_args))
    for unit in boundary.order
        selected = unit == boundary.members
        external = selected ? Dict(id => values[id] for id in boundary.inputs) : values
        local_values = evaluate_unit(program.nodes, unit, external)
        exports = selected ? boundary.outputs : unit
        for id in exports
            values[id] = local_values[id]
        end
    end
    D.result_structure(program, id -> values[id])
end

function explain_boundary(io::IO, boundary::RegionBoundary)
    names(ids) = join(("%$id" for id in ids), ", ")
    println(io, "Region members: ", names(boundary.members))
    println(io, "External inputs: ", names(boundary.inputs))
    println(io, "Escaping outputs: ", names(boundary.outputs))
    println(io, "Execution units: ", join(("[" * names(unit) * "]" for unit in boundary.order), " → "))
end

end # module
