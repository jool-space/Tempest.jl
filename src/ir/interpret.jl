# Intentionally allocating, independent of kernel scheduling and buffer reuse.
function interpret(p::ExecutionProgram, args...)
    flat_args = validate(p.primal, args)
    values = Vector{Any}(undef, length(p.nodes))
    for (id, a) in zip(p.primal.arguments, flat_args)
        values[id] = a
    end
    live = live_nodes(p.nodes, p.results)
    for (id, n) in enumerate(p.nodes)
        if !live[id]
            continue
        end
        if n.op isa Argument
            continue
        end
        xs = map(i -> values[i], n.inputs)
        if n.op isa CallResult
            call = n.op.call
            if id == first(call_ids(call))
                result = check_verb_result(call, reference(call.f, xs...; call.kwargs...))
                for (output, leaf) in zip(call_ids(call), tree_leaves(result))
                    values[output] = leaf
                end
            end
            continue
        end
        values[id] = evaluate(n.op, n, xs)
    end
    map(i -> values[i], p.results)
end

# Each method interprets one mathematical operation, independently of storage
# planning and kernel implementations. Result metadata preserves scalar/0D identity.
evaluate(op::Constant, node, xs) = node.scalar ? op.value : fill(op.value, node.dims)
evaluate(::MatMul, node, xs) = xs[1] * xs[2]
evaluate(::TransposeOp, node, xs) = transpose(only(xs))
evaluate(op::PermuteOp, node, xs) = permutedims(only(xs), op.permutation)

function evaluate(::ReshapeOp, node, xs)
    x = only(xs)
    if node.scalar
        x isa Number ? x : only(x)
    elseif x isa Number
        fill(x, node.dims)
    else
        reshape(x, node.dims)
    end
end

function evaluate(op::Union{ViewOp, SliceOp}, node, xs)
    # view preserves the zero-dimensional array case for integer indices.
    view(only(xs), native_indices(op.selection)...)
end

function evaluate(op::ScatterOp, node, xs)
    out = zeros(node.dtype, node.dims)
    selected = view(LinearIndices(out), native_indices(op.selection)...)
    for (i, g) in zip(selected, only(xs))
        out[i] += g
    end
    out
end

function evaluate(op::Pointwise, node, xs)
    result = op.f.(xs...)
    if node.scalar || (result isa AbstractArray && size(result) == node.dims)
        return result
    end
    out = Array{node.dtype}(undef, node.dims)
    out .= result
    out
end

"""
    value_and_gradient(program, args...) -> (loss, gradients)

Allocating reference execution of the explicit differentiated graph. This is a
correctness baseline, not the prepared, fused executor. Use CPU arrays here for
finite-difference comparisons without GPU scalar synchronization.
"""
function value_and_gradient(p::TrainingProgram, args...)
    vals = interpret(p, args...)
    mapped = Dict(zip(p.results, vals))
    result = result_structure(p, id -> mapped[id])
    first(result), Base.tail(result)
end

value_and_gradient(p::TensorProgram, args...) = value_and_gradient(differentiate(p), args...)
