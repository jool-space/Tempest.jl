# A layout's source presentation uses already-decided parent storage. For a
# materialized node this remains the map read by its explicit copy step.
function source_layout(nodes, layouts, id)
    node = nodes[id]
    if !is_layout(node)
        return (id, dense_map(node.dims))
    end
    parent = only(node.inputs)
    root, mapping = layouts[parent]
    (root, transform_layout(node, nodes[parent].dims, mapping))
end

requires_native_inputs(::Operation) = false
requires_native_inputs(::Union{MatMul, BatchedMatMul, CallResult}) = true
requires_dense_inputs(::Operation) = false
requires_dense_inputs(::Union{BatchedMatMul, CallResult}) = true

# Unknown operations are boundaries until they provide an explicit lowering.
# Constant fills and layout aliases have their own treatment below.
fusion_boundary(::Operation) = true
fusion_boundary(::Union{Pointwise, Constant, LayoutOperation}) = false

function native_requirements(program, live, regions)
    native = falses(length(program.nodes))
    dense = falses(length(program.nodes))
    for (id, node) in enumerate(program.nodes)
        if !live[id]
            continue
        end
        if requires_native_inputs(node.op)
            for input in node.inputs
                native[input] = true
                dense[input] |= requires_dense_inputs(node.op)
            end
        end
    end
    for region in regions, id in region.inputs
        native[id] = true
        dense[id] |= dense_region_input(region, id)
    end
    native[program.results] .= true
    (; native, dense)
end

dense_region_input(::GraphRegion, id) = true
dense_region_input(region::ProposedRegion, id) = !native_region_input(region, id)

function plan_layouts(program, live, regions = ())
    nodes = program.nodes
    (; native, dense) = native_requirements(program, live, regions)
    materialized = falses(length(nodes))
    layouts = Dict{Int, Tuple{Int, IndexMap}}()

    # Inputs precede consumers. Collecting all requirements first lets us decide
    # each node once: later consumers see its final storage root. No stale layout
    # entries or cache invalidation are needed, even across nested views.
    for id in eachindex(nodes)
        if !live[id]
            continue
        end
        root, mapping = source_layout(nodes, layouts, id)
        if native[id]
            kind = native_layout(mapping, nodes[id].dims, nodes[root].dims)
            if kind === :mapped || (dense[id] && kind !== :dense)
                materialized[id] = true
                root, mapping = id, dense_map(nodes[id].dims)
            end
        end
        layouts[id] = (root, mapping)
    end
    (; layouts, materialized)
end

# Shared pointwise values and storage boundaries stop scalar inlining. This is
# a deterministic fusion policy; provider support does not choose these roots.
function fusion_roots(nodes, live, materialized, layouts, results)
    uses = zeros(Int, length(nodes))
    for (id, node) in enumerate(nodes)
        if !live[id]
            continue
        end
        for a in node.inputs
            uses[a] += 1
        end
    end
    roots = [
        live[i] && (
            materialized[i] ||
            fusion_boundary(node.op) ||
            (node.op isa Pointwise && uses[i] > 1)
        ) for (i, node) in enumerate(nodes)
    ]
    for (id, node) in enumerate(nodes)
        if !live[id]
            continue
        end
        if requires_native_inputs(node.op) || is_layout(node)
            for a in node.inputs
                roots[first(layouts[a])] = true
            end
        end
    end
    for id in results
        roots[first(layouts[id])] = true
    end
    roots
end

function schedule_steps(program, live, layouts, materialized, regions = ())
    nodes = program.nodes
    roots = fusion_roots(nodes, live, materialized, layouts, program.results)
    selected = falses(length(nodes))
    for region in regions
        selected[region.members] .= true
        for id in (region.inputs..., region.outputs...)
            roots[first(layouts[id])] = true
        end
    end
    layout(id; force = false) = force ? source_layout(nodes, layouts, id) : layouts[id]

    function read_expression(i, dependencies, shape; force = false)
        root, mapping = layout(i; force)
        j = findfirst(==(root), dependencies)
        if j === nothing
            push!(dependencies, root)
            j = length(dependencies)
        end
        Read{j}(mapping, nodes[i].dims, shape)
    end
    function expression(i, dependencies, shape; force = false)
        node = nodes[i]
        if node.op isa Constant
            return Literal(node.op.value)
        end
        if is_layout(node) || (!force && roots[i])
            return read_expression(i, dependencies, shape; force)
        end
        if !(node.op isa Pointwise)
            error("cannot inline $(node.op)")
        end
        ScalarCall(node.op.f, map(a -> expression(a, dependencies, shape), node.inputs))
    end
    steps = Step[]
    for region in regions
        dependencies = unique([first(layouts[id]) for id in region.inputs])
        push!(steps, Step(region, first(region.outputs), dependencies, region.outputs))
    end
    for (i, node) in enumerate(nodes)
        if !(live[i] && roots[i]) || selected[i]
            continue
        end
        if node.op isa Argument
            continue
        end
        if is_layout(node) && !materialized[i]
            continue
        end
        dependencies = Int[]
        outputs = node.op isa CallResult ? call_ids(node.op.call) : [i]
        i == first(outputs) || continue
        if requires_native_inputs(node.op)
            dependencies = unique([first(layout(a)) for a in node.inputs])
        end
        region = schedule_region(node.op, nodes, i, dependencies, expression)
        push!(steps, Step(region, i, dependencies, outputs))
    end
    isempty(regions) ? steps : order_graph_steps(steps)
end

schedule_region(::MatMul, nodes, id, dependencies, expression) = GemmRegion()
schedule_region(op::BatchedMatMul, nodes, id, dependencies, expression) =
    BatchedGemmRegion(op.transpose_a, op.transpose_b)
schedule_region(op::CallResult, nodes, id, dependencies, expression) = OpaqueCall(op.call)

function schedule_region(
    ::Union{Pointwise, Constant, LayoutOperation},
    nodes,
    id,
    dependencies,
    expression,
)
    shape = nodes[id].dims
    e = expression(id, dependencies, shape; force = true)
    Elementwise(e, prod(shape))
end

function schedule_region(
    op::Union{FoldReduction, ScatterOp},
    nodes,
    id,
    dependencies,
    expression,
)
    input = only(nodes[id].inputs)
    shape = nodes[input].dims
    e = expression(input, dependencies, shape)
    reduction_region(op, e, shape, nodes[input].dtype)
end

# A reduction region's expression yields elements in the fold's accumulator, so
# every scalar backend lifts and combines in it; the state starts from that
# type's identity, and the store rounds the observed value to the output type.
function accumulated(op::FoldReduction, expression, T)
    A = fold_accumulator(op.fold, T)
    contributions = A === T ? expression : ScalarCall(Cast{A}(), (expression,))
    (contributions, fold_init(op.fold, A))
end

function reduction_region(op::FoldReduction{<:Fold, Colon}, expression, shape, T)
    contributions, init = accumulated(op, expression, T)
    FullReduction(contributions, prod(shape), op.fold, init)
end

function reduction_region(op::FoldReduction{<:Fold, <:Tuple}, expression, shape, T)
    contributions, init = accumulated(op, expression, T)
    AxisReductionRegion(contributions, prod(shape), AxisReduction(shape, op.axes), op.fold, init)
end
reduction_region(op::ScatterOp, expression, shape, T) =
    ScatterRegion(expression, prod(shape), op.selection)

# Results retain their storage past the final launch. Aliases extend the
# lifetime of their storage root, not an independently allocated logical value.
function storage_lifetimes(steps, layouts, results)
    lastuse = Dict(out => i for (i, s) in enumerate(steps) for out in s.outputs)
    for (i, s) in enumerate(steps), d in s.dependencies
        lastuse[d] = max(get(lastuse, d, 0), i)
    end
    for out in results
        lastuse[first(layouts[out])] = length(steps) + 1
    end
    lastuse
end

function assign_slots(nodes, steps, lastuse; reuse, donations = Dict{Int, Int}())
    slots = Dict{Int, Int}()
    specifications = Tuple{DataType, Int}[]
    occupied_until = Int[]
    naive_bytes = 0
    for (time, s) in enumerate(steps), out in s.outputs
        haskey(donations, out) && continue
        node = nodes[out]
        spec = (node.dtype, prod(node.dims))
        naive_bytes += sizeof(node.dtype) * spec[2]
        # Exact-size reuse first: no slicing/realignment policy is needed yet.
        slot = if reuse
            findfirst(
                j -> specifications[j] == spec && occupied_until[j] < time,
                eachindex(specifications),
            )
        else
            nothing
        end
        if slot === nothing
            push!(specifications, spec)
            push!(occupied_until, 0)
            slot = length(specifications)
        end
        occupied_until[slot] = lastuse[out]
        slots[out] = slot
    end
    (; slots, specifications, naive_bytes)
end

function plan(
    program::ExecutionProgram;
    reuse = true,
    donate = (),
    inplace = true,
    blocked = Set{Int}(),
    graph_regions = (),
)
    live = live_nodes(program.nodes, program.results)
    (; layouts, materialized) = plan_layouts(program, live, graph_regions)
    steps = schedule_steps(program, live, layouts, materialized, graph_regions)
    lastuse = storage_lifetimes(steps, layouts, program.results)
    donations =
        inplace ? donation_candidates(program, steps, layouts, lastuse, donate, blocked) :
        Dict{Int, Int}()
    storage = assign_slots(program.nodes, steps, lastuse; reuse, donations)

    StoragePlan(
        program,
        steps,
        storage.slots,
        storage.specifications,
        lastuse,
        storage.naive_bytes,
        reuse,
        layouts,
        donations,
    )
end

function explain(io::IO, plan::StoragePlan; providers = nothing)
    mode = plan.program isa TrainingProgram ? "forward + backward" : "forward"
    println(io, "Prepared schedule: $(length(plan.steps)) regions ($mode)")

    for (index, step) in enumerate(plan.steps)
        provider = providers === nothing ? "" : " [$(providers[index])]"
        if step.region isa SelectedRegion
            region = step.region
            label = region isa ProposedRegion ? " $(region.label)" : ""
            println(
                io,
                "  $index: $(step.outputs) = $(region_name(region))$provider$label",
                " covers $(region.members); reads $(step.dependencies)",
            )
            println(
                io,
                "    logical inputs $(region.inputs); escaping outputs $(region.outputs)",
                region isa GraphRegion ? "; dense boundary buffers" : "",
            )
        elseif step.region isa OpaqueCall
            slots = [plan.slots[id] for id in step.outputs]
            println(
                io,
                "  $index: $(step.outputs) = call $(step.region.call.f)$provider",
                " reads $(step.dependencies) → slots $slots",
            )
        else
            destination = if haskey(plan.donations, step.output)
                "input %$(plan.donations[step.output])"
            else
                "slot $(plan.slots[step.output])"
            end
            println(
                io,
                "  $index: %$(step.output) = $(region_name(step.region))$provider reads $(step.dependencies)",
                " → $destination (last use $(plan.lastuse[step.output]))",
            )
        end
    end

    for id in sort!(collect(keys(plan.layouts)))
        if !is_layout(plan.program.nodes[id])
            continue
        end
        root, mapping = plan.layouts[id]
        disposition = haskey(plan.slots, id) ? "materialized for consumer" : "alias"
        println(io, "  layout %$id → %$root [$(nameof(typeof(mapping)))] ($disposition)")
    end

    bytes = sum(spec -> sizeof(spec[1]) * spec[2], plan.specifications; init = 0)
    println(
        io,
        "Value storage: $bytes bytes in $(length(plan.specifications)) slots;",
        " without reuse $(plan.naive_bytes) bytes",
    )
    println(
        io,
        "Excludes persistent input copies, reduction scratch, and vendor-internal workspace.",
    )
end

explain(plan::StoragePlan) = explain(stdout, plan)
