# Host-side analysis of a semantic region. The schedule records an execution
# choice; original operations remain in the graph and in the typed bindings.
struct GraphReductionSchedule
    # The matrix domain the kernels iterate: the fold input's `leading` axes
    # collapsed into rows and the rest into columns.
    dims::Tuple{Int, Int}
    axes::Tuple{Vararg{Int}}
    tile::Tuple{Int, Int}
    groups::Tuple
    # Blocks sharing one kept tile along the reduced axis; 1 means unsplit.
    splits::Int
    # The fold of each reduction in `groups`, with the same nesting.
    folds::Tuple
    # The fold input's logical shape and how many of its leading axes form the
    # domain's rows (`collapsed_shape`).
    shape::Tuple{Vararg{Int}}
    leading::Int
end

GraphReductionSchedule(dims, axes, tile, groups, folds) =
    GraphReductionSchedule(dims, axes, tile, groups, 1, folds, dims, 1)

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

# Unsplit second-axis reductions stream STREAMED_ROW_COLUMNS columns at a time
# (fewer when the domain has fewer) over tiles of at most STREAMED_ROW_TILE
# rows. Measured on GB10 with row softmax, warm us at 32 / 128 / 256 rows and
# eight columns: 262144 x 16 (a rank-3 array reduced over its last axis)
# 498 / 149 / 149, 65536 x 64 207 / 153 / 148, 16384 x 256 166 / 157 / 171,
# 4096 x 4096 933 / 895 / 947. The CUDA kernels take 306 us on the first shape.
# Eight columns per step left the 2048 x 1024 bias gradient of an MLP at
# 35 us against 20 on the CUDA kernels, and row sums over 2048 x 8192 at 384
# against 336; 32 columns took 14 and 290, and kept the shapes above (softmax
# over 2048 x 4096 435 instead of 490). 64 columns gained nothing more.
const STREAMED_ROW_TILE = 128
const STREAMED_ROW_COLUMNS = 32

# Taller tiles leave fewer blocks: the unsplit schedule keeps at least
# SPLIT_KEPT_BLOCKS[2] blocks, the count below which second-axis reductions
# split, so a domain of 2048 rows runs 64 blocks of 32 rows, not 16 of 128.
function streamed_row_tile(rows)
    fitting = prevpow(2, max(1, rows ÷ SPLIT_KEPT_BLOCKS[2]))
    min(STREAMED_ROW_TILE, nextpow(2, rows), max(SPLIT_TILE[2][1], fitting))
end

# Short columns share a block: columns of at most PACKED_ROWS rows are packed
# into tiles of about PACKED_ELEMENTS values, keeping at least MINIMUM_BLOCKS
# blocks. Measured on GB10 (cold, share of copy bandwidth): 128 x 1024 rose
# from 34% to 60%, 64 x 4096 from 20% to 70%, 32 x 8192 from 11% to 63%.
# Packing 256-row columns fell from 62% to about 40%, and larger tiles were
# slower than 512 values at every packed shape.
const PACKED_ROWS = 128

# Full reductions give each block a tile of this many values, as tall as the
# domain's columns allow, and write one partial per block. With 256-value tiles
# (at most 256 rows) every full reduction lost to the CUDA kernels, measured on
# GB10 against `Tempest.deactivate(cuTile)`, warm us: `sum(a .* b)` over 2^20
# values 14.4 against 8.6, `sum(abs2, tanh.(x))` over 1024 x 4096 45 / 17, a
# full maximum and shifted sum 94 / 36. At 1024 values in 1024-row tiles: 7.6,
# 18.3 and 32. Tiles of 4096 values, or of 2048 rows, were slower again (fewer
# blocks than the device holds), and 512-row tiles of two columns lost a
# third. The test/benchmarks guard rails compare these against the CUDA kernels.
const FULL_TILE_ELEMENTS = 1024
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

# A value's shape in the reduction domain. The fold input's axes are grouped
# into the `leading` ones, the domain's rows, and the rest, its columns. A value
# takes part when, in each group, its axes equal the input's or are all
# singletons (Julia pads missing trailing axes with singletons); the group then
# collapses to the product of the input's extents or to one. A partial
# broadcast within a group, such as a bias over the second of three axes that
# merge into one column axis, does not collapse, and the value is declined.
# Boundary buffers are dense, so a collapsed value is a reshape of its buffer.
# A reduction over all axes has no axis semantics: any value with as many
# elements as the input is the same dense storage in the domain's shape, so
# `sum(abs2, vec(x)) + sum(x)` is one region.
function collapsed_shape(dims, shape, leading, full = false)
    if length(dims) <= length(shape)
        padded = ntuple(i -> get(dims, i, 1), length(shape))
        rows = collapse_group(padded, shape, 1:leading)
        columns = collapse_group(padded, shape, (leading + 1):length(shape))
        rows === nothing || columns === nothing || return (rows, columns)
    end
    full && prod(dims; init = 1) == prod(shape; init = 1) ? collapsed_shape(shape, shape, leading) : nothing
end

# A value's shape in the domain of a schedule or geometry.
value_shape(geometry, dims) =
    collapsed_shape(dims, geometry.shape, geometry.leading, geometry.axes == (1, 2))

function collapse_group(padded, shape, group)
    isempty(group) && return 1
    all(i -> padded[i] == shape[i], group) && return prod(shape[i] for i in group)
    all(i -> padded[i] == 1, group) ? 1 : nothing
end

# Which axis group a fold reduces: all of them, the leading ones, or the
# trailing ones. Each gives one matrix schedule; a reduction over interior
# axes, or over the leading and trailing but not the middle ones, has none.
# A vector reduced with `dims = 1` is a one-column matrix, as it was when the
# domains were matrices; `dims = :` over any rank is the all-axes schedule.
function fold_geometry(op::FoldReduction, shape)
    rank = length(shape)
    reduced = op.axes isa Colon ? Tuple(1:rank) : op.axes
    if reduced == Tuple(1:rank) && (op.axes isa Colon || rank >= 2)
        return (; leading = min(rank, 1), axes = (1, 2))
    elseif reduced == Tuple(1:length(reduced))
        return (; leading = length(reduced), axes = (1,))
    elseif reduced == Tuple((rank - length(reduced) + 1):rank)
        return (; leading = rank - length(reduced), axes = (2,))
    end
    Unsupported("cuTile graph reductions reduce leading or trailing axes, not $reduced of a rank-$rank array")
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

function graph_value_support(node, geometry, id)
    value_shape(geometry, node.dims) !== nothing ||
        return Unsupported("value %$id does not collapse into the reduction domain $(shape_text(geometry.shape))")
    if node.op isa Constant
        node.op.value isa Union{Float32, Integer} &&
        isbits(node.op.value) &&
        sizeof(node.op.value) <= 8 ||
            return Unsupported("constant %$id requires unsupported scalar precision")
        return Supported()
    end
    # Every value is Float32, so each fold already accumulates in its element
    # type, its `fold_accumulator`.
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

function graph_reduction_domain(region)
    nodes = region.program.nodes
    position = findfirst(id -> nodes[id].op isa FoldReduction, region.members)
    position === nothing && return Unsupported("cuTile graph scheduling requires a reduction")
    reduction = nodes[region.members[position]]
    shape = nodes[only(reduction.inputs)].dims
    all(>(0), shape) ||
        return Unsupported("cuTile graph reductions require nonempty dimensions")
    geometry = fold_geometry(reduction.op, shape)
    geometry isa Unsupported && return geometry
    (; leading, axes) = geometry
    domain = collapsed_shape(shape, shape, leading)
    domain[1] <= div(typemax(Int32), domain[2]) ||
        return Unsupported("graph reduction domain exceeds Int32 indexing")
    (; domain, axes, shape, leading)
end

function graph_reduction_schedule(request::GraphRequest)
    region = request.region
    nodes = region.program.nodes
    geometry = graph_reduction_domain(region)
    geometry isa Unsupported && return geometry
    (; domain, axes, shape, leading) = geometry
    for (ids, specs) in ((region.inputs, request.inputs), (region.outputs, request.outputs))
        for (id, spec) in zip(ids, specs)
            spec.dtype === Float32 ||
                return Unsupported("cuTile graph boundaries must be Float32")
            value_shape(geometry, spec.dims) !== nothing || return Unsupported(
                "boundary value %$id does not collapse into the reduction domain $(shape_text(shape))",
            )
        end
    end

    phases = Dict(id => 0 for id in region.inputs)
    reductions = Int[]
    for id in sort(region.members)
        node = nodes[id]
        verdict = graph_value_support(node, geometry, id)
        verdict isa Unsupported && return verdict
        all(input -> haskey(phases, input), node.inputs) ||
            return Unsupported("value %$id reads an undeclared region input")
        phase = maximum((phases[input] for input in node.inputs); init = 0)
        if node.op isa FoldReduction
            input_shape = nodes[only(node.inputs)].dims
            fold_geometry(node.op, input_shape) == (; leading, axes) ||
                return Unsupported("mixed reduction axes require separate graph regions")
            value_shape(geometry, input_shape) == domain ||
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
        (streamed_row_tile(domain[1]), min(STREAMED_ROW_COLUMNS, nextpow(2, domain[2])))
    else
        rows = min(FULL_TILE_ELEMENTS, nextpow(2, domain[1]))
        (rows, div(FULL_TILE_ELEMENTS, rows))
    end
    schedule = split_schedule(GraphReductionSchedule(domain, axes, tile, groups, 1, folds, shape, leading))
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
    GraphReductionSchedule(
        schedule.dims, schedule.axes, tile, schedule.groups, splits, schedule.folds,
        schedule.shape, schedule.leading,
    )
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
# Rank-two domains print as before; a collapsed domain names the logical
# shape and the matrix it became.
function schedule_description(schedule::GraphReductionSchedule)
    rank = length(schedule.shape)
    group = schedule.axes == (1,) ? (1:schedule.leading) : ((schedule.leading + 1):rank)
    reduced = schedule.axes == (1, 2) ? "all axes" :
        length(group) == 1 ? "axis $(only(group))" : "axes $(first(group)):$(last(group))"
    domain = rank <= 2 ? shape_text(schedule.dims) :
        "$(shape_text(schedule.shape)) as $(shape_text(schedule.dims))"
    "over $reduced of $domain: $(tile_description(schedule)), " *
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
                value_shape(schedule, node.dims),
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
