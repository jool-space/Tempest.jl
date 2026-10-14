# A native cuTile schedule for a deliberately bounded family of semantic graphs.
# The graph remains the specification. This file chooses repeated row sweeps;
# it does not recognize normalization formulas or replace callable identities.
module CuTileSweeps

using Tempest
using CUDACore
import cuTile as ct

include("reduction_scopes.jl")
using .ReductionScopes: analyze, pure_scalar
const D = Tempest

export prepare_sweeps, explain_sweeps, launch!, bind_inputs!

# These bindings are a compiled schedule, not a second semantic operation set.
# Operations and their functions are the objects from the original graph.
struct TileBinding{O, Inputs, Shape, T}
    op::O
end

struct SweepProgram{B, Arguments, Outputs, Groups}
    bindings::B
end

function schedule_program(analysis)
    graph = analysis.program
    bindings = Tuple(map(eachindex(graph.nodes)) do id
        # Keep IDs stable without making dead captures part of the kernel's
        # constant program or imposing backend requirements on dead operations.
        node = analysis.active[id] ? graph.nodes[id] :
               D.Node(D.Constant(0f0), (), (), Float32, true)
        TileBinding{typeof(node.op), node.inputs, node.dims, node.dtype}(node.op)
    end)
    groups = Tuple(
        Tuple(id for id in analysis.reductions if analysis.phases[id] == phase)
        for phase in 1:maximum(analysis.phases)
    )
    outputs = Tuple(unique(Int[D.tree_leaves(graph.output)...]))
    SweepProgram{typeof(bindings), Tuple(graph.arguments), outputs, groups}(bindings)
end

# Capture adaptation alone cannot establish dependencies or purity. Restrict
# captures to scalar data here; arrays must enter through explicit graph inputs.
scalar_capture(x::Union{Ptr, AbstractArray, Ref}) = false
function scalar_capture(x)
    isbits(x) && all(i -> scalar_capture(getfield(x, i)), 1:fieldcount(typeof(x)))
end

function check_schedule(analysis)
    graph = analysis.program
    length(analysis.domain) == 2 ||
        throw(ArgumentError("streamed schedule needs a matrix domain"))
    analysis.axes == (1,) || throw(ArgumentError("streamed schedule reduces rows only"))
    all(>(0), analysis.domain) || throw(ArgumentError("empty domains are not supported"))
    prod(analysis.domain) <= typemax(Int32) || throw(ArgumentError("domain exceeds Int32 indexing"))
    for (id, node) in enumerate(graph.nodes)
        analysis.active[id] || continue
        if node.op isa D.Constant
            node.op.value isa Union{Float32, Integer} ||
                throw(ArgumentError("constant %$id requires unsupported precision"))
        else
            node.dtype === Float32 || throw(ArgumentError("value %$id must be Float32"))
        end
        if node.op isa D.Pointwise
            pure_scalar(node.op.f) || throw(ArgumentError("value %$id has no purity contract"))
            scalar_capture(node.op.f) || throw(ArgumentError("value %$id has a non-scalar capture"))
        elseif !(node.op isa Union{D.Argument, D.Constant, D.SumReduction})
            throw(ArgumentError("unsupported schedule operation $(typeof(node.op))"))
        end
    end
    # A block owns one column. Avoid duplicated writers for outputs that omit
    # the column axis; extending ownership rules is separate from broadcasting.
    for id in D.tree_leaves(graph.output)
        shape = graph.nodes[id].dims
        length(shape) == 2 && shape[2] == analysis.domain[2] ||
            throw(ArgumentError("escaping value %$id needs the column axis"))
    end
end

# Generate only the dependency closure needed in this sweep. A completed sum is
# a scalar tile that survives the loop; every other value is local to this sweep.
# The cache shares common subexpressions among reductions or escaping outputs.
function emit_value!(statements, cache, bindings, arguments, id)
    haskey(cache, id) && return cache[id]
    operation, dependencies, shape, dtype = bindings[id].parameters
    if operation <: D.SumReduction
        return Symbol(:completed_, id)
    end
    name = gensym(:value)
    expression = if operation <: D.Argument
        position = findfirst(==(id), arguments)
        row_extent = isempty(shape) ? 1 : shape[1]
        column_extent = length(shape) < 2 ? 1 : shape[2]
        row_index = row_extent == 1 ? :(Int32(1)) : :block
        column_index = column_extent == 1 ? :(Int32(1)) : :column
        width = row_extent == 1 ? 1 : :tile
        array = Symbol(:input_, position)
        :(reshape(ct.load($array, ($row_index, $column_index), ($width, 1);
                          padding_mode = ct.PaddingMode.Zero), ($width,)))
    elseif operation <: D.Constant
        :(program.bindings[$id].op.value)
    elseif operation <: D.Pointwise
        operands = map(dependencies) do input
            emit_value!(statements, cache, bindings, arguments, input)
        end
        # Preserve each logical conversion even when no array is materialized.
        :($dtype.(program.bindings[$id].op.f.($(operands...))))
    else
        error("unexpected operation in checked schedule: $operation")
    end
    push!(statements, :($name = $expression))
    cache[id] = name
    name
end

@inline @generated function streamed_kernel(
    program::SweepProgram{B, A, O, G}, rows::Int32, tile::Int, arrays::Vararg{Any, N},
) where {B, A, O, G, N}
    bindings = B.parameters
    statements = Expr[]
    for position in eachindex(O)
        name = Symbol(:output_, position)
        push!(statements, :($name = arrays[$position]))
    end
    for position in eachindex(A)
        name = Symbol(:input_, position)
        push!(statements, :($name = arrays[$(length(O) + position)]))
    end
    for group in G
        body = Expr[]
        cache = Dict{Int, Symbol}()
        for id in group
            accumulator = Symbol(:accumulator_, id)
            push!(statements, :($accumulator = zeros(Float32, tile)))
            source = only(bindings[id].parameters[2])
            contribution = emit_value!(body, cache, bindings, A, source)
            # Padding the input with zero is insufficient: f(0) can be nonzero
            # or NaN. Mask the final contribution before accumulation instead.
            push!(body, :($accumulator = $accumulator .+ ifelse.(valid, $contribution, 0f0)))
        end
        push!(statements, quote
            for block = Int32(1):cld(rows, Int32(tile))
                # Compare against the remaining count so padded coordinates
                # cannot overflow Int32 at the end of a very large domain.
                remaining = rows - (block - Int32(1)) * Int32(tile)
                valid = ct.arange(tile) .<= remaining
                $(body...)
            end
        end)
        for id in group
            accumulator = Symbol(:accumulator_, id)
            completed = Symbol(:completed_, id)
            push!(statements, :($completed = sum($accumulator)))
        end
    end

    # Write invariant results once, then make a final sweep for element results.
    column_body = Expr[]
    element_body = Expr[]
    column_cache = Dict{Int, Symbol}()
    element_cache = Dict{Int, Symbol}()
    for (position, id) in enumerate(O)
        invariant = bindings[id].parameters[3][1] == 1
        body = invariant ? column_body : element_body
        cache = invariant ? column_cache : element_cache
        value = emit_value!(body, cache, bindings, A, id)
        destination = Symbol(:output_, position)
        if invariant
            push!(body, :(ct.store($destination, (Int32(1), column),
                                  reshape(ct.Tile($value), (1, 1)))))
        else
            push!(body, :(ct.store($destination, (block, column),
                                  reshape(ct.broadcast_to(ct.Tile($value), (tile,)), (tile, 1)))))
        end
    end
    append!(statements, column_body)
    if !isempty(element_body)
        push!(statements, quote
            for block = Int32(1):cld(rows, Int32(tile))
                $(element_body...)
            end
        end)
    end
    quote
        column = ct.bid(1)
        $(statements...)
        return
    end
end

# cuTile 1.0 miscompiles loop-carried tensor views when the kernel entry has
# tuple/vararg array parameters. A fixed-arity entry is sufficient; tuple-valued
# internals still specialize normally. Cache this adapter by argument count.
const entry_points = Dict{Int, Function}()

function entry_point(count)
    get!(entry_points, count) do
        name = gensym(:streamed_entry)
        arrays = [Symbol(:array_, i) for i in 1:count]
        Core.eval(@__MODULE__, quote
            function $name(program, rows, tile, $(arrays...))
                streamed_kernel(program, rows, tile, $(arrays...))
                return
            end
        end)
    end
end

struct SweepExecutable{A, P, I, O, K}
    analysis::A
    program::P
    inputs::I
    outputs::O
    kernel::K
    tile::Int
    compilation_seconds::Float64
end

"""
    prepare_sweeps(graph, device_inputs...; tile = 256)

Compile a matrix region with Float32 values and sums over its first axis.
Dependent sums run in successive sweeps inside one block per column. Scalar
functions must have a research purity contract; captures may contain scalar
values, while arrays must be explicit inputs. Preparation owns copies of the
inputs and allocates the escaping outputs. No device scratch is required.

This experiment specializes on the graph, shapes, callables, and tile width.
It changes reduction association and does not promise bitwise CPU agreement.
It is separate from Tempest's production `prepare` and storage planner.
"""
function prepare_sweeps(graph::D.TensorProgram, args...; tile = 256)
    tile > 0 && ispow2(tile) || throw(ArgumentError("tile must be a positive power of two"))
    tile <= 1024 || throw(ArgumentError("this experiment bounds tiles at 1024 rows"))
    analysis = analyze(graph)
    check_schedule(analysis)
    flat = D.validate(graph, args)
    all(x -> x isa CuArray, flat) || throw(ArgumentError("inputs must be dense CuArrays"))
    # Own stable storage so changed values can be bound without recompilation or
    # changing graph pointers. This research executable has no donation contract.
    # Active inputs have at most two axes. Flatten any unused trailing axes
    # as well, so dead inputs do not impose a schedule's presentation limits.
    inputs = Tuple(reshape(copy(x), size(x, 1), :) for x in flat)
    program = schedule_program(analysis)
    result_ids = typeof(program).parameters[3]
    outputs = Tuple(CuArray{Float32}(undef, graph.nodes[id].dims) for id in result_ids)
    compiled_program = ct.Constant(program)
    entry = entry_point(length(inputs) + length(outputs))
    compile_time = @elapsed kernel = CUDACore.@cuda backend=ct launch=false entry(
        compiled_program, Int32(analysis.domain[1]), ct.Constant(tile), outputs..., inputs...,
    )
    SweepExecutable(analysis, program, inputs, outputs, kernel, tile, compile_time)
end

"""
Replace input contents without recompiling or changing device addresses.
All signatures and cross-input overlaps are checked before any copy occurs.
"""
function bind_inputs!(ex::SweepExecutable, args...)
    flat = D.validate(ex.analysis.program, args)
    for (i, source) in enumerate(flat), (j, destination) in enumerate(ex.inputs)
        if i != j && Base.mightalias(source, destination)
            throw(ArgumentError("a binding source overlaps another executable input"))
        end
    end
    for (destination, source) in zip(ex.inputs, flat)
        copyto!(destination, reshape(source, size(destination)))
    end
    ex
end

"""
Launch the prepared kernel and return its structured, borrowed output arrays.
Subsequent launches or captured replays overwrite those arrays.
"""
function launch!(ex::SweepExecutable)
    # Convert at launch so CUDACore observes buffer use on the current stream.
    # Compilation and device allocation have already happened in preparation.
    inputs = ct.cuTileconvert(ex.inputs)
    outputs = ct.cuTileconvert(ex.outputs)
    ex.kernel(ct.Constant(ex.program), Int32(ex.analysis.domain[1]), ct.Constant(ex.tile),
              outputs..., inputs...; blocks = ex.analysis.domain[2])
    ids = typeof(ex.program).parameters[3]
    D.tree_map(ex.analysis.program.output) do id
        ex.outputs[findfirst(==(id), ids)]
    end
end

"""Print the selected sweeps, output storage, and measured compilation time."""
function explain_sweeps(io::IO, ex::SweepExecutable)
    groups = typeof(ex.program).parameters[4]
    println(io, "Native cuTile streamed region: ", ex.analysis.domain)
    println(io, "  one block per column, ", ex.tile, " rows per tile")
    println(io, "  reduction phases: ", join((join(g, ", ") for g in groups), " → "))
    println(io, "  one kernel; reduction scalars retained, pure pointwise work recomputed")
    println(io, "  output bytes: ", sum(sizeof, ex.outputs), "; device workspace bytes: 0")
    println(io, "  Julia/cuTile preparation: ", round(ex.compilation_seconds; digits = 3), " s")
end

end # module
