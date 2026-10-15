"""
A selection from the semantic graph, before scalar lowering or storage planning.
Members retain their original operation objects and value IDs. Inputs and outputs
are ordered logical values; every value used outside the selection must escape.
The initial boundary contract supplies dense buffers with those logical shapes.
Layout operations, contractions, and semantic calls remain outside this region.
"""
struct GraphRegion <: Region
    program::ExecutionProgram
    members::Vector{Int}
    inputs::Vector{Int}
    outputs::Vector{Int}
end

region_name(::GraphRegion) = :graph_region

"""
    GraphRequest

A target and semantic GraphRegion offered through `support` before storage is
assigned. `inputs` and `outputs` are ordered TensorSpecs. Construction receives
matching tuples of dense, logically shaped buffers; scalars use 0D buffers.
Inputs are read-only and do not overlap outputs. Outputs may be needed only by
backward or by an escaping view and must still be written completely.

An implementation must preserve each node's logical element conversion. Every
fold accumulates in `fold_accumulator(fold, T)` of its input element type `T`:
elements convert to it before lifting, and the observed value rounds once to the
node's element type. For Float16 and BFloat16 sums, products and logsumexps that
is Float32. Fold association may differ, as for the existing parallel reduction
providers; this contract does not authorize algebraic rewrites or any other
accumulation type.
Reevaluating a scalar function requires its `recomputable` contract.
"""
struct GraphRequest <: ImplementationRequest
    target::ExecutionTarget
    region::GraphRegion
    inputs::Tuple
    outputs::Tuple
end

"""
    recomputable(f) -> Bool

Whether scalar `f` may be repeated or moved within a generated region. Declaring
true promises deterministic, side-effect-free evaluation whose reads depend only
on explicit operands and immutable scalar captures. Array captures must first
be exposed as graph operands. Device compilability and a derivative rule do not
establish this permission. Unknown functions return false.

The declaration is independent of backend support. A backend can reject a
recomputable function it cannot compile, or keep an unproven function outside a
region that requires recomputation.
"""
recomputable(f) = f in PURE_BASE_FUNCTIONS
recomputable(::Union{LiteralPower, Cast}) = true

# Base's scalar arithmetic and elementary functions are pure. Listing all of
# them keeps the recomputation permission from depending on spelling: `x .^ -0.5f0`
# and `inv.(sqrt.(x))`, or a broadcast `identity` emitted by a sum's pullback,
# plan alike. A provider may still decline a function it cannot compile.
const PURE_BASE_FUNCTIONS = (
    identity, +, -, *, /, \, ^, inv, abs, abs2, sign, copysign, flipsign, min, max,
    clamp, muladd, fma, ifelse, sqrt, cbrt, hypot, exp, exp2, exp10, expm1, log,
    log2, log10, log1p, sin, cos, tan, sinh, cosh, tanh, asin, acos, atan, asinh,
    acosh, atanh, sinpi, cospi, sec, csc, cot, deg2rad, rad2deg, float,
)

# Construction rejected a selected region, growth-based or proposed. Preparation
# excludes this exact member set and replans.
struct RejectedGraphRegion
    members::Vector{Int}
    reasons::Vector{String}
    proposed::Bool
end

graph_operation(op) = op isa Union{Pointwise, Constant, FoldReduction}

# Return execution units for a proposed partition. Calls stay atomic across all
# their result leaves, even though they cannot join these primitive regions.
function graph_units(program, groups, live)
    owner = Dict(id => group for group in groups for id in group)
    assigned = falses(length(program.nodes))
    units = Vector{Int}[]
    for (id, node) in enumerate(program.nodes)
        (!live[id] || assigned[id] || node.op isa Argument) && continue
        unit = if haskey(owner, id)
            owner[id]
        elseif node.op isa CallResult
            sort(call_ids(node.op.call))
        else
            [id]
        end
        push!(units, unit)
        assigned[unit] .= true
    end
    units
end

# A stable topological order for units with explicit produced and consumed IDs.
# Return nothing on a cycle so speculative selection does not use exceptions.
function dependency_order(produced, consumed)
    owner = Dict(id => unit for (unit, ids) in enumerate(produced) for id in ids)
    dependencies = [
        unique([owner[id] for id in inputs if haskey(owner, id) && owner[id] != unit]) for
        (unit, inputs) in enumerate(consumed)
    ]
    consumers = [Int[] for _ in produced]
    pending = length.(dependencies)
    for (unit, inputs) in enumerate(dependencies), input in inputs
        push!(consumers[input], unit)
    end
    ready = findall(iszero, pending)
    order = Int[]
    while !isempty(ready)
        unit = popfirst!(ready)
        push!(order, unit)
        for next in consumers[unit]
            pending[next] -= 1
            if pending[next] == 0
                insert!(ready, searchsortedfirst(ready, next), next)
            end
        end
    end
    length(order) == length(produced) ? order : nothing
end

function graph_partition_order(program, groups, live)
    units = graph_units(program, groups, live)
    inputs = [[input for id in unit for input in program.nodes[id].inputs] for unit in units]
    order = dependency_order(units, inputs)
    order === nothing ? nothing : units[order]
end

function graph_region(program::ExecutionProgram, members)
    members = sort!(unique!(Int[members...]))
    isempty(members) && throw(ArgumentError("a graph region needs at least one operation"))
    live = live_nodes(program.nodes, program.results)
    for id in members
        1 <= id <= length(program.nodes) || throw(ArgumentError("invalid graph value %$id"))
        live[id] || throw(ArgumentError("graph region contains dead value %$id"))
        graph_operation(program.nodes[id].op) ||
            throw(ArgumentError("value %$id is not a primitive region operation"))
    end
    graph_partition_order(program, [members], live) === nothing &&
        throw(ArgumentError("graph region introduces a dependency cycle"))
    region_boundary(program, members, live)
end

graph_region(program::TensorProgram, members) = graph_region(ForwardProgram(program), members)

function region_boundary(program, members, live)
    selected = Set(members)
    inputs = sort!(
        unique!([
            input for id in members for input in program.nodes[id].inputs if input ∉ selected
        ]),
    )
    escaping = Set(program.results)
    for (id, node) in enumerate(program.nodes)
        live[id] && id ∉ selected && union!(escaping, node.inputs)
    end
    outputs = filter(in(escaping), members)
    GraphRegion(program, members, inputs, outputs)
end

function graph_request(target, region::GraphRegion)
    nodes = region.program.nodes
    spec(id) = TensorSpec(nodes[id].dtype, nodes[id].dims; scalar = nodes[id].scalar)
    GraphRequest(target, region, Tuple(spec.(region.inputs)), Tuple(spec.(region.outputs)))
end

function graph_adjacency(program, live)
    neighbors = [Int[] for _ in program.nodes]
    users = [Int[] for _ in program.nodes]
    for (id, node) in enumerate(program.nodes)
        live[id] || continue
        for input in node.inputs
            push!(users[input], id)
            graph_operation(node.op) && graph_operation(program.nodes[input].op) || continue
            push!(neighbors[id], input)
            push!(neighbors[input], id)
        end
    end
    (; neighbors, users)
end

# Offer private literals together with their consumer. An isolated scalar output
# or integer input can be unsupported even when the combined expression is valid.
# Shared and returned constants retain their ordinary boundary obligations.
function include_private_literals(program, members, users, claimed)
    selected = Set(members)
    for id in members, input in program.nodes[id].inputs
        program.nodes[input].op isa Constant || continue
        claimed[input] && continue
        input in program.results && continue
        all(in(selected), users[input]) && push!(selected, input)
    end
    sort!(collect(selected))
end

function grow_graph_region(
    program,
    initial,
    neighbors,
    claimed,
    groups,
    live;
    accept = Returns(true),
    users = nothing,
    limit = 64,
)
    members = initial
    history = [members]
    declined = falses(length(program.nodes))
    changed = true
    while changed && length(members) < limit
        changed = false
        frontier = sort!(
            unique!([
                next for id in members for
                next in neighbors[id] if !claimed[next] && !declined[next] && next ∉ members
            ]),
        )
        for next in frontier
            # Constants enter with consumers during support-guided growth.
            users !== nothing && program.nodes[next].op isa Constant && continue
            proposed = sort!([members; next])
            if users !== nothing
                proposed = include_private_literals(program, proposed, users, claimed)
            end
            length(proposed) <= limit || continue
            graph_partition_order(program, [groups; [proposed]], live) === nothing && continue
            if !accept(proposed)
                declined[next] = true
                continue
            end
            members = proposed
            push!(history, members)
            changed = true
            length(members) == limit && break
        end
    end
    history
end

# The first implementation that supports the region, or Unsupported with
# every candidate's reason.
function graph_support(implementations, target, region)
    nodes = region.program.nodes
    boundary = (region.inputs..., region.outputs...)
    all(id -> nodes[id].dtype <: AbstractFloat, boundary) ||
        return Unsupported("boundary values must be floating")
    request = graph_request(target, region)
    reasons = String[]
    for implementation in implementations
        verdict = checked_support(implementation, request)
        verdict isa Supported && return implementation
        push!(reasons, "$(implementation_name(implementation)): $(verdict.reason)")
    end
    Unsupported(join(reasons, "; "))
end

# First try the complete structural candidate: support need not be monotonic,
# and a provider may recognize a complete formula without accepting its pieces.
# On rejection, grow from a supported anchor and keep declined neighbors outside.
# This remains greedy; a declined merge is not a proof against every superset.
function supported_graph_growth(accept, program, seed, neighbors, users, claimed, groups, live)
    history = grow_graph_region(program, [seed], neighbors, claimed, groups, live)
    accept(last(history)) && return history

    initial = include_private_literals(program, [seed], users, claimed)
    if !accept(initial)
        anchor = findlast(accept, history)
        anchor === nothing && return Vector{Int}[]
        initial = history[anchor]
    end
    grow_graph_region(program, initial, neighbors, claimed, groups, live; accept, users)
end

# Defining support for GraphRequest is the participation mechanism. Avoid graph
# searches when every implementation inherits the generic unsupported method;
# there is no separate registration list or capability trait to keep in sync.
function graph_implementations(implementations)
    fallback = which(support, Tuple{Implementation, ImplementationRequest})
    filter(implementations) do implementation
        which(support, Tuple{typeof(implementation), GraphRequest}) !== fallback
    end
end

# Growth starts after proposals are selected: their members are already claimed
# and take part in the acyclicity check of every growth candidate. Of each
# seed's growth, the largest unblocked candidate that is supported and, by
# estimate, faster than the default kernels for its members is selected
# (`region_profit`, src/compiler/costs.jl).
function select_graph_regions(program, target, implementations, blocked, notes; selected = ())
    candidates = graph_implementations(implementations)
    isempty(candidates) && return GraphRegion[]
    live = live_nodes(program.nodes, program.results)
    (; neighbors, users) = graph_adjacency(program, live)
    claimed = falses(length(program.nodes))
    regions = GraphRegion[]
    groups = Vector{Int}[region.members for region in selected]
    foreach(region -> claimed[region.members] .= true, selected)
    verdicts = Dict{Tuple, Any}()
    function supporter(members)
        get!(verdicts, Tuple(members)) do
            region = region_boundary(program, members, live)
            verdict = graph_support(candidates, target, region)
            if verdict isa Unsupported
                push!(notes, "graph region $members declined: $(verdict.reason)")
            end
            verdict
        end
    end
    accepted(members) = !(supporter(members) isa Unsupported)
    rates = REGION_COSTS[] ? cost_rates(target) : nothing
    defaults = rates === nothing ? nothing : default_kernels(program, target, implementations, live)
    if rates !== nothing
        line = "region estimates: $(rates.detail); a region is kept unless estimated " *
            "more than $(round(Int, 100 * (REGION_MARGIN - 1)))% slower than the default kernels"
        line in notes || push!(notes, line)
    end

    for (seed, node) in enumerate(program.nodes)
        live[seed] && !claimed[seed] && node.op isa FoldReduction || continue
        history = supported_graph_growth(
            accepted,
            program,
            seed,
            neighbors,
            users,
            claimed,
            groups,
            live,
        )
        # A construction rejection excludes this exact set, not its nodes. Keep
        # growing through it, then select the largest unblocked supported prefix
        # that pays.
        for members in Iterators.reverse(history)
            haskey(blocked, Tuple(members)) && continue
            accepted(members) || continue
            region = region_boundary(program, members, live)
            keep, line = region_profit(region, supporter(members), graph_request(target, region), defaults, rates, claimed)
            line === nothing || push!(notes, line)
            keep || continue
            push!(regions, region)
            push!(groups, members)
            claimed[members] .= true
            break
        end
    end
    regions
end

function order_graph_steps(steps)
    anchor(step) = step.region isa SelectedRegion ? first(step.region.members) : step.output
    steps = sort(steps; by = anchor)
    order = dependency_order([s.outputs for s in steps], [s.dependencies for s in steps])
    order === nothing && error("selected graph regions produced a cyclic storage schedule")
    steps[order]
end

# Swapping the operands of these binary calls leaves every floating-point result
# unchanged. Wider calls keep their order: `+(a, b, c)` rounds left to right.
commutative(::Pointwise{typeof(+)}, arity) = arity == 2
commutative(::Pointwise{typeof(*)}, arity) = arity == 2
commutative(operation, arity) = false

"""
    standard_numbering(region) -> (; local_id, operands)

Number a selected region's values locally in a standard form, so equivalent
regions can share one compiled implementation: the same computation in every
layer or in the backward pass, whichever of its inputs the graph created first,
and with commutative operands in either order. `local_id` maps original value
IDs to local numbers, which order every value after its operands; `operands(id)`
returns a member's operands in the order the numbering visited them.

Numbering is a depth-first walk from the outputs. Commutative operands are
visited in the order of a structural hash of their producers, which ignores
graph IDs. A hash collision or an unusual sharing pattern can only cost a missed
reuse, provided an implementation computes with the operand order `operands`
reports.
"""
function standard_numbering(region::Region)
    nodes = region.program.nodes
    structure = Dict{Int, UInt}()
    function structural_hash(id)
        haskey(structure, id) && return structure[id]
        node = nodes[id]
        structure[id] = if id in region.inputs
            hash((Argument, node.dims, node.dtype))
        else
            producers = map(structural_hash, node.inputs)
            if commutative(node.op, length(producers))
                producers = Tuple(sort!(collect(producers)))
            end
            hash((node.op, node.dims, node.dtype, producers))
        end
    end
    function operands(id)
        node = nodes[id]
        commutative(node.op, length(node.inputs)) || return node.inputs
        Tuple(sort!(collect(node.inputs); by = structural_hash))
    end
    local_id = Dict{Int, Int}()
    function visit(id)
        haskey(local_id, id) && return
        id in region.inputs || foreach(visit, operands(id))
        local_id[id] = length(local_id) + 1
    end
    foreach(visit, region.outputs)
    foreach(visit, sort(region.members))
    foreach(visit, region.inputs)
    (; local_id, operands)
end
