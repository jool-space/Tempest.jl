# A direct evaluation of the index level: every value materialized in its own
# shape, every read through its formula. It checks the lowering against the
# program-level interpreter, and is the CPU realization of last resort.

function read_operand(o::Read, x, arrays, graph)
    stored = arrays[o.value]
    graph.values[o.value].scalar && return stored
    index = map(e -> evaluate_index(e, x, (v, i) -> arrays[v][(i .+ 1)...]) + 1, o.index)
    stored[index...]
end

read_operand(o::Literal, x, arrays, graph) = o.value
read_operand(o::IndexOperand{T}, x, arrays, graph) where {T} =
    T(evaluate_index(o.expr, x, (v, i) -> arrays[v][(i .+ 1)...]))

# Every value of `operand` over `dims`, densely, as a Julia array (or a number for
# a rank-zero read of a scalar).
function materialize(o::Operand, dims, arrays, graph; scalar = false)
    if scalar
        return read_operand(o, (), arrays, graph)
    end
    first_value = read_operand(o, ntuple(_ -> 0, length(dims)), arrays, graph)
    out = Array{typeof(first_value)}(undef, dims)
    for I in CartesianIndices(dims)
        out[I] = read_operand(o, Tuple(I) .- 1, arrays, graph)
    end
    out
end

evaluate_value(d::Input, v, arrays, graph, arguments) = arguments[d.argument]
evaluate_value(d::ConstantArray, v, arrays, graph, arguments) = d.data

function evaluate_value(d::Map, v, arrays, graph, arguments)
    apply(x) = v.eltype(d.f(map(o -> read_operand(o, x, arrays, graph), d.operands)...))
    v.scalar && return apply(())
    out = Array{v.eltype}(undef, v.dims)
    for I in CartesianIndices(v.dims)
        out[I] = apply(Tuple(I) .- 1)
    end
    out
end

function evaluate_value(d::Fold, v, arrays, graph, arguments)
    input = materialize(d.operand, d.iteration, arrays, graph)
    axes = d.reduced == Tuple(1:length(d.iteration)) && v.scalar ? Colon() : d.reduced
    Tempest.fold_array(d.fold, input, axes)
end

function evaluate_value(d::Opaque, v, arrays, graph, arguments)
    xs = map((o, (dims, scalar)) -> materialize(o, dims, arrays, graph; scalar), d.operands, d.shapes)
    Tempest.evaluate(d.node.op, d.node, xs)
end

"""
    interpret(graph, arguments...) -> results

Evaluate an index graph on the CPU, value by value.
"""
function interpret(graph::IndexGraph, arguments...)
    arrays = Vector{Any}(undef, length(graph.values))
    for (id, v) in enumerate(graph.values)
        arrays[id] = evaluate_value(v.definition, v, arrays, graph, arguments)
    end
    map(eachindex(graph.results)) do k
        materialize(graph.results[k], graph.result_dims[k], arrays, graph; scalar = graph.result_scalar[k])
    end
end
