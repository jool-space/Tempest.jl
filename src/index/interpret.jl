# A direct evaluation of the index level: every value materialized in its own
# shape, every read through its formula. It checks the lowering against the
# program interpreter.

function read_operand(o::Read, x, arrays, graph)
    stored = arrays[o.value]
    graph.values[o.value].scalar && return stored
    gather = (v, i) -> arrays[v][(i .+ 1)...]
    stored[map(e -> evaluate_index(e, x, gather) + 1, o.index)...]
end

read_operand(o::Literal, x, arrays, graph) = o.value
read_operand(o::IndexOperand{T}, x, arrays, graph) where {T} =
    T(evaluate_index(o.expr, x, (v, i) -> arrays[v][(i .+ 1)...]))

function materialize(o::Operand, dims, arrays, graph; scalar = false)
    scalar && return read_operand(o, (), arrays, graph)
    sample = read_operand(o, ntuple(_ -> 0, length(dims)), arrays, graph)
    out = Array{typeof(sample)}(undef, dims)
    for I in CartesianIndices(dims)
        out[I] = read_operand(o, Tuple(I) .- 1, arrays, graph)
    end
    out
end

# An array in the shape of value `v`, whose domain has no axes of extent one.
shaped(x, v::Value) = v.scalar ? x : reshape(x, v.dims)

evaluate_value(d::Input, v, arrays, graph, arguments) = shaped(arguments[d.argument], v)
evaluate_value(d::ConstantArray, v, arrays, graph, arguments) = d.data

function evaluate_value(d::Pointwise, v, arrays, graph, arguments)
    apply(x) = convert(v.eltype, d.f(map(o -> read_operand(o, x, arrays, graph), d.operands)...))
    v.scalar && return apply(())
    out = Array{v.eltype}(undef, v.dims)
    for I in CartesianIndices(v.dims)
        out[I] = apply(Tuple(I) .- 1)
    end
    out
end

function evaluate_value(d::Reduction, v, arrays, graph, arguments)
    y = fold_array(d.fold, materialize(d.operand, d.iteration, arrays, graph), d.reduced)
    v.scalar ? only(y) : shaped(y, v)
end

function evaluate_value(d::Opaque, v, arrays, graph, arguments)
    xs = map((o, (dims, scalar)) -> materialize(o, dims, arrays, graph; scalar), d.operands, d.shapes)
    shaped(evaluate(d.node.op, d.node, xs), v)
end

"""
    interpret(graph::IndexGraph, inputs...) -> flat results

Evaluate an index graph on the CPU, value by value.
"""
function interpret(graph::IndexGraph, inputs...)
    arguments = filter(is_input, leaves(inputs))
    arrays = Vector{Any}(undef, length(graph.values))
    for (id, v) in enumerate(graph.values)
        arrays[id] = evaluate_value(v.definition, v, arrays, graph, arguments)
    end
    map(eachindex(graph.results)) do k
        materialize(graph.results[k], graph.result_dims[k], arrays, graph; scalar = graph.result_scalar[k])
    end
end
