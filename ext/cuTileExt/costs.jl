# The launches of cuTile's kernels for Tempest's region estimates
# (src/compiler/costs.jl): the bytes each moves and the bytes its blocks request
# at once, a block's tile at a time. Completions read the partial sums or
# states, a tile at a time.

# Scalar regions (scalar.jl): pointwise and full sums over 256-value tiles,
# column sums with one block per column.
function Tempest.region_cost(::CuTileKernels, request::RegionRequest)
    region = request.step.region
    reads = sum(zip(request.input_types, request.input_dims); init = 0) do (T, dims)
        sizeof(T) * prod(dims; init = 1)
    end
    element = sizeof(request.dtype)
    bytes = reads + element * prod(request.dims; init = 1)
    region isa Elementwise && return [Tempest.KernelCost(bytes, element * region.count)]
    if region isa FullReduction
        # One block completes the partials, FINISH_TILE at a time.
        blocks = cld(region.count, REDUCTION_TILES * TILE_SIZE)
        return [
            Tempest.KernelCost(bytes, element * TILE_SIZE * blocks),
            Tempest.KernelCost(4 * blocks, 4 * FINISH_TILE),
        ]
    end
    if region isa AxisReductionRegion
        columns = prod(request.dims; init = 1)
        return [Tempest.KernelCost(bytes, element * TILE_SIZE * columns)]
    end
    nothing
end

# Graph regions (graph/build.jl). Every pass reads the region inputs its values
# need: a resident tile reads each once; a streamed or split schedule reads,
# for each phase, the inputs its reductions consume, and once more for the
# outputs. Graph values are Float32.
function Tempest.region_cost(::CuTileKernels, request::GraphRequest)
    schedule = graph_reduction_schedule(request)
    schedule isa Tempest.Unsupported && return nothing
    region = request.region
    nodes = region.program.nodes
    bytes(id) = 4 * prod(nodes[id].dims; init = 1)
    reductions = Set(id for group in schedule.groups for id in group)
    # The region inputs `roots` read, through members that are not reductions:
    # a reduction an earlier phase completed is read from its statistics.
    function read_bytes(roots)
        seen, found = Set{Int}(), Set{Int}()
        stack = collect(Int, roots)
        while !isempty(stack)
            id = pop!(stack)
            id in seen && continue
            push!(seen, id)
            if id in region.inputs
                push!(found, id)
            elseif !(id in reductions)
                append!(stack, nodes[id].inputs)
            end
        end
        sum(bytes, found; init = 0)
    end
    written = sum(bytes, region.outputs; init = 0)
    output_reads = read_bytes(filter(id -> !(id in reductions), region.outputs))
    phase_reads = [read_bytes(Set(only(nodes[id].inputs) for id in group)) for group in schedule.groups]
    blocks = prod(graph_grid(schedule))
    tile = 4 * prod(schedule.tile)
    parallel = blocks * tile
    if resident_schedule(schedule)
        return [Tempest.KernelCost(sum(bytes, region.inputs; init = 0) + written, parallel)]
    elseif schedule.axes != (1, 2) && schedule.splits == 1
        return [Tempest.KernelCost(sum(phase_reads) + output_reads + written, parallel)]
    end
    kernels = Tempest.KernelCost[]
    for (group, reads, width) in zip(schedule.groups, phase_reads, phase_widths(schedule))
        push!(kernels, Tempest.KernelCost(reads, parallel))
        if schedule.axes == (1, 2)
            # One block per statistic loops over the partials, 256 at a time.
            push!(kernels, Tempest.KernelCost(4 * blocks * width, 4 * 256 * length(group)))
        else
            # One tile of partials per kept tile and statistic.
            partials = 4 * blocks * width
            push!(kernels, Tempest.KernelCost(partials, partials))
        end
    end
    direct = schedule.axes == (1, 2) && direct_statistic(region, schedule)
    if !direct
        output_blocks = prod(cld.(matrix_shape_of(schedule, region), schedule.tile))
        push!(kernels, Tempest.KernelCost(output_reads + written, output_blocks * tile))
    end
    kernels
end

# The extent the output launch covers: the largest output's collapsed shape.
function matrix_shape_of(schedule, region)
    nodes = region.program.nodes
    shapes = [value_shape(schedule, nodes[id].dims) for id in region.outputs]
    (maximum(first, shapes), maximum(last, shapes))
end
