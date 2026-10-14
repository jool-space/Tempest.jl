# Host-side analysis of a semantic region. The schedule records an execution
# choice; original operations remain in the graph and in the typed bindings.
struct GraphReductionSchedule
    dims::Tuple{Int, Int}
    axes::Tuple{Vararg{Int}}
    tile::Tuple{Int, Int}
    groups::Tuple
    # Blocks sharing one kept tile along the reduced axis; 1 means unsplit.
    splits::Int
    # The fold of each reduction in `groups`, with the same nesting.
    folds::Tuple
end

GraphReductionSchedule(dims, axes, tile, groups, folds) =
    GraphReductionSchedule(dims, axes, tile, groups, 1, folds)

# Partials hold every state component of a phase's folds, one column each.
phase_widths(schedule) = map(folds -> sum(f -> fold_width(f, Float32), folds), schedule.folds)
summed_phase(schedule, index) = all(f -> f isa SumFold, schedule.folds[index])

# `Init` is a fold's identity state, `()` for other values. Generated code
# masks padded lanes with it and sizes accumulators by its length. `Resident`
# records whether the fold has a lowering for slices held in registers.
struct GraphTileBinding{ID, Operation, Inputs, Shape, T, Init, Resident}
    op::Operation
end

struct GraphSweepProgram{Bindings, Inputs, Outputs, Groups, Axes, Tile, Resident, Splits}
    bindings::Bindings
end

# A column that fits one tile stays resident: every value is loaded or computed
# once and shared by all reduction phases and outputs. Longer columns stream
# bounded tiles and recompute pointwise producers in each sweep.
const RESIDENT_ROWS = 4096

# Streamed columns load 1024 rows per step. Measured on GB10 with the residual
# RMSNorm fixture (warm/cold us), against 256-row tiles: 8192 x 64 took 12.0/40.5
# instead of 25.4/46.9, 16384 x 32 10.6/39.9 instead of 13.6/42.4 (both unsplit
# against split 8 ways). 2048-row tiles matched 1024 on power-of-two columns but
# lost on ragged ones, which pad the last tile: 8193 x 32 took 10.6/28.0 against
# 9.5/25.5, and 4096 rows 12.1/30.0.
const STREAMED_ROWS = 1024

resident_schedule(schedule) = schedule.axes == (1,) && schedule.tile[1] >= schedule.dims[1]

# Short columns share a block: columns of at most PACKED_ROWS rows are packed
# into tiles of about PACKED_ELEMENTS values, keeping at least MINIMUM_BLOCKS
# blocks. Measured on GB10 (cold, share of copy bandwidth): 128 x 1024 rose
# from 34% to 60%, 64 x 4096 from 20% to 70%, 32 x 8192 from 11% to 63%.
# Packing 256-row columns fell from 62% to about 40%, and larger tiles were
# slower than 512 values at every packed shape.
const PACKED_ROWS = 128
const PACKED_ELEMENTS = 512
const MINIMUM_BLOCKS = 128

function resident_tile(domain)
    rows = nextpow(2, domain[1])
    rows > PACKED_ROWS && return (rows, 1)
    packed = max(1, PACKED_ELEMENTS ÷ rows)
    spread = max(1, prevpow(2, max(1, domain[2] ÷ MINIMUM_BLOCKS)))
    (rows, min(packed, spread))
end

matrix_shape(shape) = (get(shape, 1, 1), get(shape, 2, 1))

function fits_graph_domain(shape, domain)
    length(shape) <= 2 && all(i -> shape[i] == 1 || shape[i] == domain[i], eachindex(shape))
end

# Captured arrays would hide reads from lifetime planning and AD. The callable
# may carry scalar data; every tensor operand must enter through the graph.
graph_scalar_capture(::Union{Ptr, CuPtr, AbstractArray, Ref}) = false
function graph_scalar_capture(value)
    isbits(value) &&
        all(i -> graph_scalar_capture(getfield(value, i)), 1:fieldcount(typeof(value)))
end

function graph_pointwise_support(op::Pointwise, arity)
    recomputable(op.f) || return Unsupported(
        "reduction scheduling requires a recomputable scalar function: $(op.f)",
    )
    graph_scalar_capture(op.f) ||
        return Unsupported("reduction scalar functions require scalar-only captures")
    expression = ScalarCall(op.f, ntuple(_ -> Literal(0.0f0), arity))
    tile_expression_support(expression)
end

function graph_value_support(node, domain, id)
    fits_graph_domain(node.dims, domain) ||
        return Unsupported("value %$id is outside the matrix reduction domain")
    if node.op isa Constant
        node.op.value isa Union{Float32, Integer} &&
        isbits(node.op.value) &&
        sizeof(node.op.value) <= 8 ||
            return Unsupported("constant %$id requires unsupported scalar precision")
        return Supported()
    end
    node.dtype === Float32 ||
        return Unsupported("cuTile graph values must be Float32; value %$id is $(node.dtype)")
    if node.op isa Pointwise
        return graph_pointwise_support(node.op, length(node.inputs))
    end
    node.op isa FoldReduction ||
        return Unsupported("cuTile graph reductions do not lower $(typeof(node.op))")
    graph_fold_support(node.op.fold)
end

# Generated kernels take the fold from its type: cuTile's reduction body and
# lane-mapped calls accept only zero-size callables, so a fold carrying data runs
# on the CUDA kernels instead. Folds are pure by contract; no `recomputable`
# declaration is needed to reevaluate their functions per lane.
function graph_fold_support(fold)
    Base.issingletontype(typeof(fold)) ||
        return Unsupported("cuTile reductions need a zero-size fold; $(typeof(fold)) carries data")
    all(value -> value isa Union{Float32, Int32}, fold_init(fold, Float32)) ||
        return Unsupported("cuTile fold states must be Float32 or Int32: $(typeof(fold))")
    Supported()
end

reduction_axes(op::FoldReduction) = op.axes isa Colon ? (1, 2) : op.axes

function graph_reduction_domain(region)
    nodes = region.program.nodes
    position = findfirst(id -> nodes[id].op isa FoldReduction, region.members)
    position === nothing && return Unsupported("cuTile graph scheduling requires a reduction")
    reduction = nodes[region.members[position]]
    shape = nodes[only(reduction.inputs)].dims
    length(shape) <= 2 || return Unsupported("cuTile graph reductions support rank at most two")
    domain = matrix_shape(shape)
    all(>(0), domain) ||
        return Unsupported("cuTile graph reductions require nonempty dimensions")
    domain[1] <= div(typemax(Int32), domain[2]) ||
        return Unsupported("graph reduction domain exceeds Int32 indexing")
    axes = reduction_axes(reduction.op)
    axes in ((1,), (2,), (1, 2)) || return Unsupported("unsupported graph reduction axes $axes")
    (; domain, axes)
end

function graph_reduction_schedule(request::GraphRequest)
    region = request.region
    nodes = region.program.nodes
    geometry = graph_reduction_domain(region)
    geometry isa Unsupported && return geometry
    (; domain, axes) = geometry
    for (ids, specs) in ((region.inputs, request.inputs), (region.outputs, request.outputs))
        for (id, spec) in zip(ids, specs)
            spec.dtype === Float32 ||
                return Unsupported("cuTile graph boundaries must be Float32")
            fits_graph_domain(spec.dims, domain) || return Unsupported(
                "boundary value %$id cannot broadcast into the reduction domain",
            )
        end
    end

    phases = Dict(id => 0 for id in region.inputs)
    reductions = Int[]
    for id in sort(region.members)
        node = nodes[id]
        verdict = graph_value_support(node, domain, id)
        verdict isa Unsupported && return verdict
        all(input -> haskey(phases, input), node.inputs) ||
            return Unsupported("value %$id reads an undeclared region input")
        phase = maximum((phases[input] for input in node.inputs); init = 0)
        if node.op isa FoldReduction
            reduction_axes(node.op) == axes ||
                return Unsupported("mixed reduction axes require separate graph regions")
            matrix_shape(nodes[only(node.inputs)].dims) == domain ||
                return Unsupported("value %$id reduces a different iteration domain")
            phase += 1
            push!(reductions, id)
        end
        phases[id] = phase
    end
    groups = Tuple(
        Tuple(id for id in reductions if phases[id] == phase) for
        phase = 1:maximum(phases[id] for id in reductions)
    )
    folds = map(group -> map(id -> nodes[id].op.fold, group), groups)
    tile = if axes == (1,)
        domain[1] <= RESIDENT_ROWS ? resident_tile(domain) : (STREAMED_ROWS, 1)
    elseif axes == (2,)
        (min(32, nextpow(2, domain[1])), 8)
    else
        rows = min(256, nextpow(2, domain[1]))
        (rows, div(256, rows))
    end
    schedule = split_schedule(GraphReductionSchedule(domain, axes, tile, groups, folds))
    all(n -> n < (1 << 24), graph_grid(schedule)) ||
        return Unsupported("graph reduction launch exceeds cuTile's 24-bit grid limit")
    schedule
end

# A streamed single-axis reduction with too few kept tiles to fill the device
# splits its reduced axis across blocks: each block sums a chunk into partials,
# a second launch completes the statistics, and a fully parallel launch writes
# the outputs. On GB10 the 128 x 1024 row sums of a transformer's norm-weight
# gradients ran on 4 blocks and took 76 us.
#
# Columns split below 32 kept blocks. With 1024-row tiles, 32 unsplit columns
# already reach copy bandwidth on GB10's 48 SMs: residual RMSNorm at 8192 x 32
# took 6.3/20.2 us warm/cold unsplit against 7.0/22.6 split 4 ways, and 8192 x 48
# 7.7/30.1 against 10.0/34.7. Ragged columns lose a little: 8193 x 32 took
# 9.4/25.1 unsplit against 8.5/24.6 split. At 16 columns splitting still pays
# warm: 65536 x 16 took 34.8 us unsplit and 22 to 30 split 16 ways, with cold
# times within noise of each other (84 to 88 us).
const SPLIT_KEPT_BLOCKS = (32, 64)
const SPLIT_TARGET_BLOCKS = 256
const SPLIT_TILE = (STREAMED_ROWS, 1), (32, 32)

# A split column tile halves, down to SPLIT_MINIMUM_ROWS, until the grid holds
# MINIMUM_BLOCKS blocks. At 8193 x 5, 1024-row tiles gave 8 chunks per column and
# 40 blocks, 4.6 us warm; 256-row tiles gave 160 blocks and 3.7 us. Shortening
# every split tile to reach SPLIT_TARGET_BLOCKS cost 32768 x 4 1.7 us warm (512
# rows, 6.8 us), while 1024 rows and 128 blocks took 5.1 us.
const SPLIT_MINIMUM_ROWS = 256

function split_schedule(schedule)
    schedule.axes in ((1,), (2,)) && !resident_schedule(schedule) || return schedule
    axis = only(schedule.axes)
    tile = SPLIT_TILE[axis]
    tile = axis == 1 ? tile : (min(tile[1], nextpow(2, schedule.dims[1])), tile[2])
    kept = cld(schedule.dims[3 - axis], tile[3 - axis])
    kept < SPLIT_KEPT_BLOCKS[axis] || return schedule
    splits = split_count(schedule.dims[axis], tile[axis], kept)
    while axis == 1 && kept * splits < MINIMUM_BLOCKS && tile[1] > SPLIT_MINIMUM_ROWS
        tile = (tile[1] ÷ 2, 1)
        splits = split_count(schedule.dims[1], tile[1], kept)
    end
    splits > 1 || return schedule
    GraphReductionSchedule(schedule.dims, schedule.axes, tile, schedule.groups, splits, schedule.folds)
end

# Blocks sharing one kept tile: a power of two, at most one per tile of the
# reduced axis, and none below four tiles.
function split_count(extent, tile_extent, kept)
    reduced = cld(extent, tile_extent)
    reduced >= 4 || return 1
    min(prevpow(2, reduced), nextpow(2, cld(SPLIT_TARGET_BLOCKS, kept)))
end

split_axis(schedule) = schedule.splits > 1 ? only(schedule.axes) : 0

function graph_grid(schedule::GraphReductionSchedule)
    rows, columns = cld.(schedule.dims, schedule.tile)
    axis = split_axis(schedule)
    axis == 1 && return (columns, schedule.splits)
    axis == 2 && return (rows, schedule.splits)
    schedule.axes == (1,) && return (columns,)
    schedule.axes == (2,) && return (rows,)
    (rows, columns)
end

# One line for `explain` and `profile`. A resident tile holds whole columns, and
# a packed one several of them; a streamed tile loops along the reduced axis, and
# a split schedule shares that loop among several blocks per kept tile.
function schedule_description(schedule::GraphReductionSchedule)
    reduced = schedule.axes == (1, 2) ? "all axes" : "axis $(only(schedule.axes))"
    "over $reduced of $(shape_text(schedule.dims)): $(tile_description(schedule)), " *
    "grid $(shape_text(graph_grid(schedule)))"
end

shape_text(dims) = join(dims, "×")

function tile_description(schedule::GraphReductionSchedule)
    tile = shape_text(schedule.tile)
    if resident_schedule(schedule)
        packed = schedule.tile[2]
        packed == 1 ? "resident tile $tile" : "resident tile $tile, $packed columns per block"
    elseif schedule.axes == (1, 2)
        # Folds other than sums keep unobserved state tuples as partials.
        partials = all(i -> summed_phase(schedule, i), eachindex(schedule.groups)) ? "sums" : "states"
        "partial $partials over tile $tile"
    elseif schedule.splits > 1
        "streamed tile $tile, split $(schedule.splits) ways"
    else
        "streamed tile $tile"
    end
end

# Returns the program and the order in which the kernel takes the region's
# inputs; the caller passes input arrays in that order. Values use Tempest's
# standard numbering, so equivalent regions share one program type and one
# compiled kernel.
function graph_sweep_program(region::GraphRegion, schedule)
    nodes = region.program.nodes
    (; local_id, operands) = standard_numbering(region)
    relabel(values) = Tuple(local_id[id] for id in values)
    input_order = sortperm(region.inputs; by = id -> local_id[id])
    ids = sort!(collect(keys(local_id)); by = id -> local_id[id])
    bindings = Tuple(
        map(ids) do id
            node = nodes[id]
            external = id in region.inputs
            operation = external ? Argument() : node.op
            dependencies = external ? () : relabel(operands(id))
            init = operation isa FoldReduction ? fold_init(operation.fold, Float32) : ()
            resident = operation isa FoldReduction && has_resident_lowering(operation.fold)
            binding_type = GraphTileBinding{
                local_id[id],
                typeof(operation),
                dependencies,
                node.dims,
                node.dtype,
                init,
                resident,
            }
            binding_type(operation)
        end,
    )
    program_type = GraphSweepProgram{
        typeof(bindings),
        relabel(region.inputs[input_order]),
        relabel(region.outputs),
        Tuple(Tuple(sort!([local_id[id] for id in group])) for group in schedule.groups),
        schedule.axes,
        schedule.tile,
        resident_schedule(schedule),
        schedule.splits,
    }
    program_type(bindings), input_order
end
