# The reference interpreter: the meaning of a program, one node at a time, on
# Julia arrays. It allocates freely and ignores every plan.

evaluate(op::Constant, n, xs) = op.value isa AbstractArray ? op.value : n.scalar ? op.value : fill(op.value, n.dims)
function evaluate(op::Positions, n, xs)
    op.axis == 0 && return reshape(Int32.(1:prod(n.dims)), n.dims)
    shape = ntuple(k -> k == op.axis ? n.dims[k] : 1, length(n.dims))
    out = Array{Int32}(undef, n.dims)
    out .= reshape(Int32.(1:n.dims[op.axis]), shape)
    out
end
evaluate(op::Permute, n, xs) = permutedims(only(xs), op.perm)

function evaluate(op::Map, n, xs)
    n.scalar && return convert(n.eltype, op.f(xs...))
    out = Array{n.eltype}(undef, n.dims)
    out .= op.f.(xs...)
    out
end

function evaluate(op::Reduce, n, xs)
    y = fold_array(op.fold, only(xs), op.axes)
    n.scalar ? only(y) : y
end

function evaluate(::Reshape, n, xs)
    x = only(xs)
    n.scalar && return x isa Number ? x : only(x)
    x isa Number ? fill(x, n.dims) : reshape(x, n.dims)
end

function evaluate(op::Select, n, xs)
    y = only(xs)[op.indices...]
    n.scalar ? y : y isa Number ? fill(y) : y
end

function evaluate(op::Scatter, n, xs)
    out = zeros(n.eltype, n.dims)
    targets = view(LinearIndices(out), op.indices...)
    for (t, g) in zip(targets, only(xs))
        out[t] += g
    end
    out
end

# A product of half-precision operands accumulates in Float32 and rounds once.
function evaluate(op::Contract, n, xs)
    a, b = xs
    A = n.eltype <: HalfFloat ? Float32 : n.eltype
    n.eltype.(einsum(A.(a), op.a, A.(b), op.b, op.out))
end

# A two-operand einsum through matrix products: free axes of each operand,
# contracted axes and batch axes, in that order.
function einsum(a, la, b, lb, lo)
    batch = [l for l in lo if l in la && l in lb]
    summed = [l for l in la if l in lb && !(l in lo)]
    free_a = [l for l in la if !(l in lb)]
    free_b = [l for l in lb if !(l in la)]
    extent = Dict{Int, Int}()
    foreach((l, n) -> extent[l] = n, (la..., lb...), (size(a)..., size(b)...))
    position(labels, l) = findfirst(==(l), labels)
    total(ls) = prod(l -> extent[l], ls; init = 1)
    pa = permutedims(a, [position(la, l) for l in [free_a; summed; batch]])
    pb = permutedims(b, [position(lb, l) for l in [summed; free_b; batch]])
    ma = reshape(pa, total(free_a), total(summed), total(batch))
    mb = reshape(pb, total(summed), total(free_b), total(batch))
    mc = stack(k -> ma[:, :, k] * mb[:, :, k], 1:total(batch); dims = 3)
    c = reshape(mc, (map(l -> extent[l], [free_a; free_b; batch])...,))
    order = [free_a; free_b; batch]
    permutedims(c, [findfirst(==(l), order) for l in lo])
end

"""
    interpret(program, args...) -> results

Evaluate `program` on Julia arrays and numbers, in the structure of its results.
"""
function interpret(p::Program, args...)
    inputs = leaves(args)
    values = Vector{Any}(undef, length(p.nodes))
    for (k, id) in enumerate(p.arguments)
        values[id] = inputs[k]
    end
    keep = live(p.nodes, p.results)
    for (id, n) in enumerate(p.nodes)
        (keep[id] && !(n.op isa Argument)) || continue
        if n.op isa Call
            isassigned(values, id) || evaluate_call!(values, p.nodes, id)
            continue
        end
        values[id] = evaluate(n.op, n, map(i -> values[i], n.inputs))
    end
    rebuild(k -> k === nothing ? nothing : values[p.results[k]], p.output_structure)
end
