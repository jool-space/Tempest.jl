struct RejectedCall
    call::SemanticCall
    reasons::Vector{String}
end

# State shared while copying a graph and any reference graphs it introduces.
# A rejected call's origin survives copying, so preparation retries expand that
# same call rather than repeatedly trying to construct its unavailable kernel.
struct SemanticLowering
    nodes::Vector{Node}
    target::ExecutionTarget
    implementations::Tuple
    excluded::IdDict{Any, Vector{String}}
    notes::Vector{String}
end

function call_support(lowering::SemanticLowering, call, inputs)
    rejected = get(lowering.excluded, call.origin, nothing)
    if rejected !== nothing
        return Unsupported(join(rejected, "; "))
    end

    request = CallRequest(lowering.target, call, map(TensorSpec, inputs))
    reasons = String[]
    for implementation in lowering.implementations
        verdict = checked_support(implementation, request)
        if verdict isa Supported
            return verdict
        end
        push!(reasons, "$(implementation_name(implementation)): $(verdict.reason)")
    end
    Unsupported(join(reasons, "; "))
end

function trace_reference(call, inputs, reason)
    nodes = Node[]
    arguments = map(inputs) do input
        emit(nodes, Argument(), (), size(input), eltype(input); scalar = input isa TracedScalar)
    end
    if !applicable(reference, call.f, arguments...)
        throw(ArgumentError("no supported implementation or reference for $(call.f): $reason"))
    end

    result = check_verb_result(call, reference(call.f, arguments...; call.kwargs...))
    if !all(x -> x isa Value && x.nodes === nodes, tree_leaves(result))
        throw(ArgumentError("reference returned values from another trace"))
    end
    (; nodes, arguments, result)
end

function lower_call!(lowering::SemanticLowering, call, inputs, active)
    verdict = call_support(lowering, call, inputs)
    if verdict isa Supported
        return emit_call(
            lowering.nodes,
            call.f,
            inputs,
            call.kwargs,
            call.specs;
            origin = call.origin,
        )
    end

    if call.f in active
        throw(ArgumentError("recursive reference decomposition for $(call.f)"))
    end
    note = "$(call.f): expanded reference"
    if !isempty(verdict.reason)
        note *= " ($(verdict.reason))"
    end
    push!(lowering.notes, note)

    traced = trace_reference(call, inputs, verdict.reason)
    result_ids = Int[x.id for x in tree_leaves(traced.result)]
    argument_ids = [x.id for x in traced.arguments]
    mapped = lower_graph!(
        lowering,
        traced.nodes,
        result_ids,
        argument_ids,
        inputs,
        (active..., call.f),
    )
    tree_map(x -> mapped[x.id], traced.result)
end

# Copy live nodes in dependency order. `mapped` translates source node IDs to
# values in the destination graph; reference arguments are bound to existing
# destination values instead of being copied as new graph inputs.
function lower_graph!(lowering::SemanticLowering, source, results, arguments, bindings, active)
    live = live_nodes(source, results)
    mapped = Dict{Int, Value}(zip(arguments, bindings))

    for (id, node) in enumerate(source)
        if !live[id] || haskey(mapped, id)
            continue
        end
        inputs = map(i -> mapped[i], node.inputs)

        if node.op isa CallResult
            call = node.op.call
            outputs = lower_call!(lowering, call, inputs, active)
            for (old_id, output) in zip(call_ids(call), tree_leaves(outputs))
                mapped[old_id] = output
            end
        else
            mapped[id] = emit(
                lowering.nodes,
                node.op,
                Tuple(x.id for x in inputs),
                node.dims,
                node.dtype;
                scalar = node.scalar,
            )
        end
    end
    mapped
end

# Expand unavailable calls before fusion and storage planning. This pass traces
# references symbolically; it neither allocates tensor data nor launches kernels.
function resolve_semantics(program::ExecutionProgram, target, candidates, excluded, notes)
    if !any(node -> node.op isa CallResult, program.nodes)
        return program
    end
    lowering = SemanticLowering(Node[], target, Tuple(candidates), excluded, notes)

    # Unused arguments still belong to the input binding contract.
    arguments = map(program.primal.arguments) do id
        node = program.primal.nodes[id]
        emit(lowering.nodes, Argument(), (), node.dims, node.dtype; scalar = node.scalar)
    end
    mapped = lower_graph!(
        lowering,
        program.nodes,
        program.results,
        program.primal.arguments,
        arguments,
        (),
    )
    remap_program(program, lowering.nodes, id -> mapped[id].id)
end
