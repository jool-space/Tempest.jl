function support(::CuTileKernels, request::GraphRequest)
    request.target isa CUDATarget || return Unsupported("requires a CUDA target")
    schedule = graph_reduction_schedule(request)
    schedule isa Unsupported ? schedule : Supported()
end

# Kernels take each array as its own parameter. cuTile gives each its own alias
# class, so a store need not wait for loads from other arrays. A tuple of arrays
# once shared one class, which kept resident kernels at few registers, but since
# cuTile.jl#318 its arrays are separate too; the occupancy hint below now limits
# registers instead.
const graph_entry_points = Dict{Tuple{Function, Int}, Function}()
const graph_entry_lock = ReentrantLock()

function graph_flat_entry(kernel, count)
    lock(graph_entry_lock) do
        get!(graph_entry_points, (kernel, count)) do
            name = gensym(:graph_entry)
            arrays = [Symbol(:array_, i) for i = 1:count]
            Core.eval(
                @__MODULE__,
                quote
                    function $name(program, rows, columns, phase, $(arrays...))
                        $kernel(program, rows, columns, phase, $(arrays...))
                        return
                    end
                end,
            )
        end
    end
end

function graph_matrix(array, spec)
    reshape(array, matrix_shape(spec.dims))
end

struct CompiledGraphKernel{K}
    kernel::K
end

# Resident kernels compile for 4 blocks per SM. Without a hint, tileiras gives
# them 255 registers whenever their store is not ordered after their loads. On
# GB10 the residual RMSNorm fixture at 128 x 4096 then ran in 23.3/43.9 us
# warm/cold; with occupancy 4 (64 registers) in 9.0/38.8 us. A hint of 8 helped
# packed tiles slightly more but was ignored for 1024-row tiles, which kept 255
# registers and ran slower than with no hint.
const RESIDENT_OCCUPANCY = 4

function compile_graph_kernel(kernel, program, schedule, phase, arrays)
    rows, columns = Int32.(schedule.dims)
    occupancy = resident_schedule(schedule) ? RESIDENT_OCCUPANCY : nothing
    entry = graph_flat_entry(kernel, length(arrays))
    compiled = CUDACore.@cuda backend=ct launch=false occupancy=occupancy entry(program, rows, columns, phase, arrays...)
    CompiledGraphKernel(compiled)
end

function graph_launcher(compiled, program, schedule, phase, arrays; grid = graph_grid(schedule))
    rows, columns = Int32.(schedule.dims)
    kernel = compiled.kernel
    () -> begin
        # Conversion records array use on the current stream. Compilation and
        # device allocation have completed before any execution or capture.
        arguments = ct.cuTileconvert(arrays)
        kernel(program, rows, columns, phase, arguments...; blocks = grid)
        nothing
    end
end

struct GraphLaunch{F}
    launch::F
    schedule::GraphReductionSchedule
    kernels::Int
end

(call::GraphLaunch)() = call.launch()

function Tempest.implementation_summary(call::GraphLaunch)
    phases = counted(length(call.schedule.groups), "phase")
    "cuTile reductions $(schedule_description(call.schedule)); " *
    "$phases, $(counted(call.kernels, "kernel"))"
end

counted(count, noun) = count == 1 ? "1 $noun" : "$count $(noun)s"

function build_axis_graph(program, schedule, outputs, inputs)
    arrays = (outputs..., inputs...)
    phase = ct.Constant(Val(0))
    kernel = compile_graph_kernel(graph_axis_kernel, program, schedule, phase, arrays)
    launch = graph_launcher(kernel, program, schedule, phase, arrays)
    BuiltImplementation(GraphLaunch(launch, schedule, 1), (arrays, program, kernel), 0)
end

# Full reductions cross blocks. Each dependency phase writes partial sums or
# fold states, then completes them in a second launch. One final launch writes
# escaping values.
# Scratch is visible to the common workspace planner and never survives a call.
function build_full_graph(program, schedule, outputs, inputs)
    blocks = prod(graph_grid(schedule))
    width = maximum(phase_widths(schedule))
    count = sum(length, schedule.groups)
    specifications = (WorkspaceSpec(Float32, blocks * width), WorkspaceSpec(Float32, count))
    bind = function (workspace)
        partials = reshape(view(workspace[1], 1:(blocks * width)), blocks, width)
        statistics = view(workspace[2], 1:count)
        calls = Any[]
        owners = Any[]
        offset = 0
        for (index, group) in enumerate(schedule.groups)
            phase = ct.Constant(Val(index))
            arrays = (partials, statistics, inputs...)
            kernel =
                compile_graph_kernel(graph_partial_kernel, program, schedule, phase, arrays)
            push!(calls, graph_launcher(kernel, program, schedule, phase, arrays))
            push!(owners, kernel)
            # Bind a fresh offset for this phase; the loop advances it below.
            finish, finish_call = if summed_phase(schedule, index)
                finish = CUDACore.@cuda backend=ct launch=false finish_graph_reductions(
                    statistics,
                    partials,
                    Int32(blocks),
                    Int32(offset),
                )
                finish, let finish = finish, offset = Int32(offset), groups = length(group)
                    () -> begin
                        args = ct.cuTileconvert((statistics, partials))
                        finish(args..., Int32(blocks), offset; blocks = groups)
                        nothing
                    end
                end
            else
                finish = CUDACore.@cuda backend=ct launch=false graph_finish_kernel(
                    program,
                    Int32(blocks),
                    Int32(offset),
                    phase,
                    statistics,
                    partials,
                )
                finish, let finish = finish, offset = Int32(offset), groups = length(group), phase = phase
                    () -> begin
                        args = ct.cuTileconvert((statistics, partials))
                        finish(program, Int32(blocks), offset, phase, args...; blocks = groups)
                        nothing
                    end
                end
            end
            push!(calls, finish_call)
            push!(owners, finish)
            offset += length(group)
        end
        arrays = (statistics, outputs..., inputs...)
        phase = ct.Constant(Val(0))
        kernel = compile_graph_kernel(graph_output_kernel, program, schedule, phase, arrays)
        # Only visit tiles containing an escaping output. A scalar-only result
        # needs one block, regardless of the original reduction domain.
        output_grid = ntuple(2) do axis
            maximum(out -> cld(size(out, axis), schedule.tile[axis]), outputs)
        end
        push!(
            calls,
            graph_launcher(kernel, program, schedule, phase, arrays; grid = output_grid),
        )
        push!(owners, kernel)
        launch = () -> begin
            foreach(call -> call(), calls)
            nothing
        end
        built = GraphLaunch(launch, schedule, length(calls))
        BuiltImplementation(built, (program, outputs, inputs, workspace, owners, calls), 0)
    end
    ImplementationPlan(bind, specifications)
end

# A single-axis reduction split across blocks: per phase, partial sums or fold
# states over chunks of the reduced axis, then their completion into per-row or
# per-column statistics; a final launch writes the escaping values of every tile.
function build_split_graph(program, schedule, outputs, inputs)
    axis = only(schedule.axes)
    kept_tile = schedule.tile[3 - axis]
    kept_blocks = cld(schedule.dims[3 - axis], kept_tile)
    kept = kept_blocks * kept_tile
    width = maximum(phase_widths(schedule))
    count = sum(length, schedule.groups)
    splits = schedule.splits
    specifications =
        (WorkspaceSpec(Float32, kept * splits * width), WorkspaceSpec(Float32, kept * count))
    bind = function (workspace)
        partials = reshape(view(workspace[1], 1:(kept * splits * width)), kept, splits * width)
        statistics = reshape(view(workspace[2], 1:(kept * count)), kept, count)
        calls = Any[]
        owners = Any[]
        offset = 0
        for (index, group) in enumerate(schedule.groups)
            phase = ct.Constant(Val(index))
            arrays = (partials, statistics, inputs...)
            kernel = compile_graph_kernel(
                graph_split_partial_kernel,
                program,
                schedule,
                phase,
                arrays,
            )
            push!(calls, graph_launcher(kernel, program, schedule, phase, arrays))
            push!(owners, kernel)
            finish, finish_call = if summed_phase(schedule, index)
                shape = (ct.Constant(Val(kept_tile)), ct.Constant(Val(splits)))
                finish = CUDACore.@cuda backend=ct launch=false finish_split_reductions(
                    statistics,
                    partials,
                    Int32(offset),
                    shape...,
                )
                finish, let finish = finish, offset = Int32(offset), grid = (kept_blocks, length(group))
                    () -> begin
                        args = ct.cuTileconvert((statistics, partials))
                        finish(args..., offset, shape...; blocks = grid)
                        nothing
                    end
                end
            else
                rows = ct.Constant(Val(kept_tile))
                finish = CUDACore.@cuda backend=ct launch=false graph_split_finish_kernel(
                    program,
                    Int32(offset),
                    phase,
                    rows,
                    statistics,
                    partials,
                )
                finish, let finish = finish, offset = Int32(offset), grid = (kept_blocks, length(group)), phase = phase
                    () -> begin
                        args = ct.cuTileconvert((statistics, partials))
                        finish(program, offset, phase, rows, args...; blocks = grid)
                        nothing
                    end
                end
            end
            push!(calls, finish_call)
            push!(owners, finish)
            offset += length(group)
        end
        arrays = (statistics, outputs..., inputs...)
        phase = ct.Constant(Val(0))
        kernel = compile_graph_kernel(graph_output_kernel, program, schedule, phase, arrays)
        output_grid = ntuple(2) do axis
            maximum(out -> cld(size(out, axis), schedule.tile[axis]), outputs)
        end
        push!(
            calls,
            graph_launcher(kernel, program, schedule, phase, arrays; grid = output_grid),
        )
        push!(owners, kernel)
        launch = () -> begin
            foreach(call -> call(), calls)
            nothing
        end
        built = GraphLaunch(launch, schedule, length(calls))
        BuiltImplementation(built, (program, outputs, inputs, workspace, owners, calls), 0)
    end
    ImplementationPlan(bind, specifications)
end

function build_implementation(::CuTileKernels, request::GraphRequest, runtime, outputs, inputs)
    schedule = graph_reduction_schedule(request)
    schedule isa Unsupported && return schedule
    program, input_order = graph_sweep_program(request.region, schedule)
    program = ct.Constant(program)
    destinations = map(graph_matrix, outputs, request.outputs)
    sources = map(graph_matrix, inputs[input_order], request.inputs[input_order])
    if schedule.axes == (1, 2)
        return build_full_graph(program, schedule, destinations, sources)
    end
    if schedule.splits > 1
        return build_split_graph(program, schedule, destinations, sources)
    end
    build_axis_graph(program, schedule, destinations, sources)
end
