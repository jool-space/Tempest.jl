"""
    prepare(f, examples...; wrt=Tuple(eachindex(examples)), target=nothing, reuse=true, implementations=())
    prepare(program, examples...; target=nothing, reuse=true, donate=())

Prepare a fixed-shape forward/backward executable. Infer placement from all array
inputs by default; floating scalar inputs do not choose the device. Mixed array
placement requires an explicit target. Scalar-only programs default to CPU.
Use `CPUTarget()` or `CUDATarget(...)` to explicitly copy inputs to that target.

Inputs are copied into owned stable buffers; `bind!` updates their contents.
For a forward/update program, `donate=(1, 2)` maps output groups 1 and 2 onto
owned input groups 1 and 2. Shapes, element types, and structures must match.
Safe elementwise results can write directly into donated input storage; other
state commits follow the kernels. Use `inplace=false` to retain staged copies and
`grouped=false` to disable independent-array grouping. Implementations can accept
semantic graph regions, or propose regions of their own, before storage planning;
`fuse_regions=false` retains the ordinary scalar-region planner. Rejection tries smaller graph candidates and then
the ordinary decomposition. Repeated execution advances state. `bind_argument!`
can replace a batch without resetting other groups.
Results (including scalar gradients) are borrowed arrays, overwritten by the next
execution. Provider selection and workspace provisioning happen during preparation.
Optional `implementations` are tried in order for existing regions; rejection
reasons appear in `explain`. Semantic calls without a supported realization
expand into their reference before region/storage planning.
Executables are single-execution, not thread-safe. CUDA executables must be used
on their original current device/context.
"""
function prepare(f, args...; wrt = Tuple(eachindex(args)), kwargs...)
    prepare(differentiate(trace(f, args...); wrt), args...; kwargs...)
end

function prepare(
    p::TensorProgram,
    args...;
    wrt = Tuple(eachindex(p.input_structure)),
    kwargs...,
)
    prepare(differentiate(p; wrt), args...; kwargs...)
end

"""Prepare a forward-only executable; preserve nested tuple/named-tuple outputs as borrowed arrays."""
Base.@nospecializeinfer function prepare_forward(f, args...; kwargs...)
    @nospecialize args
    prepare(ForwardProgram(trace(f, args...)), args...; kwargs...)
end

Base.@nospecializeinfer function prepare_forward(p::TensorProgram, args...; kwargs...)
    @nospecialize args
    prepare(ForwardProgram(p), args...; kwargs...)
end

Base.@nospecializeinfer function prepare(
    p::ExecutionProgram,
    args...;
    target = nothing,
    reuse = true,
    implementations = (),
    donate = (),
    inplace = true,
    grouped = true,
    fuse_regions = true,
)
    @nospecialize args
    if !all(x -> x isa Implementation, implementations)
        throw(ArgumentError("implementations must contain Implementation instances"))
    end
    flat_args = validate(p.primal, args)
    validate_donation(p, args, donate)
    if target === nothing
        target = infer_target(flat_args)
    end
    if !(target isa ExecutionTarget)
        throw(ArgumentError("target must be an ExecutionTarget"))
    end
    defaults = default_implementations(target)
    if !(defaults isa Tuple && all(x -> x isa Implementation, defaults))
        throw(
            ArgumentError(
                "default_implementations must return a tuple of Implementation instances",
            ),
        )
    end
    candidates = (Tuple(implementations)..., defaults...)
    excluded = IdDict{Any, Vector{String}}()
    notes = String[]
    rejected_rules = IdDict{Any, Set{Int}}()
    current, rule_owners = prepare_reverse_rules(p, target, candidates, rejected_rules, notes)
    blocked = Set{Int}()
    blocked_regions = Dict{Tuple, String}()
    current = resolve_semantics(current, target, candidates, excluded, notes)
    while true
        result = prepare_schedule(
            current,
            args...;
            target,
            reuse,
            implementations = candidates,
            donate,
            inplace,
            blocked,
            grouped,
            fuse_regions,
            blocked_regions,
            fusion_notes = notes,
        )
        if result isa Vector{RejectedGraphRegion}
            for rejection in result
                reason = join(rejection.reasons, "; ")
                blocked_regions[Tuple(rejection.members)] = reason
                # A proposal's outcome is reported with the other proposals.
                rejection.proposed || push!(
                    notes,
                    "graph region $(rejection.members) declined after construction: $reason",
                )
            end
            continue
        end
        if result isa RejectedDonation
            push!(blocked, result.input)
            push!(
                notes,
                "input %$(result.input) retains staged donation: $(join(result.reasons, "; "))",
            )
            continue
        end
        if result isa RejectedCall
            owner = get(rule_owners, result.call.origin, nothing)
            if owner !== nothing
                reject_reverse_rule!(rejected_rules, owner, join(result.reasons, "; "), notes)
                current, rule_owners =
                    prepare_reverse_rules(p, target, candidates, rejected_rules, notes)
                # Region and value IDs belong to the discarded specialization.
                empty!(excluded)
                empty!(blocked)
                empty!(blocked_regions)
                current = resolve_semantics(current, target, candidates, excluded, notes)
                continue
            end
            excluded[result.call.origin] = result.reasons
            # Nested call identities persist, but expansion can renumber values.
            # Candidate rejections belong to the previous graph snapshot.
            current = resolve_semantics(current, target, candidates, excluded, notes)
            empty!(blocked_regions)
            continue
        end
        prepare_state!(result, donate)
        append!(result.semantic_notes, notes)
        return result
    end
end

# Allocate physical storage once. Logical aliases are resolved separately and
# cached in `values`; they must never acquire an independent storage slot here.
Base.@nospecializeinfer function allocate_storage(runtime, storage, @nospecialize(args))
    inputs = Any[allocate(runtime, eltype(arg), size(arg)) for arg in tree_leaves(args)]
    buffers = Any[allocate(runtime, T, (count,)) for (T, count) in storage.specifications]
    values = Dict{Int, Any}(zip(storage.program.primal.arguments, inputs))
    for (output, input) in storage.donations
        values[output] = values[input]
    end
    for (id, slot) in storage.slots
        values[id] = reshape(buffers[slot], storage.program.nodes[id].dims)
    end
    (; inputs, buffers, values)
end

function resolve_value!(values, storage, id)
    if haskey(values, id)
        return values[id]
    end

    nodes = storage.program.nodes
    root, mapping = storage.layouts[id]
    dims = nodes[id].dims
    kind = native_layout(mapping, dims, nodes[root].dims)
    if kind === :mapped
        error("mapped value %$id needs materialization")
    end

    parent = resolve_value!(values, storage, root)
    values[id] = if kind === :dense
        reshape(parent, dims)
    else
        transpose(reshape(parent, reverse(dims)))
    end
end

# Opaque calls see logical arrays and their declared result tree. GEMM retains
# native transpose wrappers. Scalar kernels use flat storage and explicit maps.
function bind_step(program, step, lookup)
    if step.region isa SelectedRegion
        output = Tuple(lookup(id) for id in step.region.outputs)
        inputs = Tuple(lookup(id) for id in step.region.inputs)
    elseif step.region isa OpaqueCall
        output = tree_map(lookup, step.region.call.results)
        inputs = map(lookup, program.nodes[step.output].inputs)
    elseif step.region isa GemmRegion
        output = lookup(step.output)
        inputs = map(lookup, program.nodes[step.output].inputs)
    else
        output = vec(lookup(step.output))
        inputs = Tuple(vec(lookup(id)) for id in step.dependencies)
    end
    (; output, inputs)
end

function implementation_request(target, program, step, inputs)
    node = program.nodes[step.output]
    if step.region isa GraphRegion
        return graph_request(target, step.region)
    end
    if step.region isa ProposedRegion
        return proposed_request(target, step.region)
    end
    if step.region isa OpaqueCall
        specs = map(node.inputs) do id
            input = program.nodes[id]
            TensorSpec(input.dtype, input.dims; scalar = input.scalar)
        end
        return CallRequest(target, step.region.call, specs)
    end
    RegionRequest(target, step, node.dtype, node.dims, map(eltype, inputs), map(size, inputs))
end

# The public prepare entry point validates inputs and resolves semantic calls.
# This stage binds the resulting schedule. A rejected opaque call sends its
# identity back to prepare, which expands its reference and plans again.
Base.@nospecializeinfer function prepare_schedule(
    program::ExecutionProgram,
    args...;
    target,
    reuse,
    implementations,
    donate = (),
    inplace = true,
    blocked = Set{Int}(),
    grouped = true,
    fuse_regions = true,
    blocked_regions = Dict{Tuple, String}(),
    fusion_notes = String[],
)
    @nospecialize args
    # Proposals claim their members first; growth-based regions form around them.
    records = ProposalRecord[]
    regions = Region[]
    if fuse_regions
        proposed =
            select_proposed_regions(program, target, implementations, blocked_regions, records)
        grown = select_graph_regions(
            program,
            target,
            implementations,
            blocked_regions,
            fusion_notes;
            selected = proposed,
        )
        append!(regions, proposed, grown)
    end
    storage = plan(
        program;
        reuse,
        donate,
        inplace,
        blocked,
        graph_regions = regions,
    )
    runtime = make_runtime(target, storage, args)
    (; inputs, buffers, values) = allocate_storage(runtime, storage, args)
    lookup(id) = resolve_value!(values, storage, id)

    bindings = map(step -> bind_step(program, step, lookup), storage.steps)
    selection =
        select_schedule(implementations, target, program, storage, runtime, bindings; grouped)
    if selection isa Union{RejectedCall, RejectedDonation, Vector{RejectedGraphRegion}}
        return selection
    end
    bound = bind_implementations(runtime, selection.chosen)
    providers = Vector{Symbol}(undef, length(storage.steps))
    workspace_slots = Vector{Vector{Int}}(undef, length(storage.steps))
    for (launch, indices) in enumerate(selection.groups), index in indices
        providers[index] = bound.names[launch]
        workspace_slots[index] = bound.workspace_slots[launch]
    end
    result = result_structure(program, lookup)
    executable = Prepared(
        storage,
        runtime,
        inputs,
        buffers,
        values,
        bound.buffers,
        bound.calls,
        providers,
        result,
        selection.notes,
        String[],
        map(implementation_name, collect(implementations)),
        workspace_slots,
        StateBindings(),
        selection.groups,
        records,
    )
    bind!(executable, args...)
end
