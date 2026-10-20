# The reference interpreter: the meaning of a program, one node at a time, on
# Julia arrays. It allocates freely and ignores every plan.

evaluate(op::Constant, n, xs) = op.value isa AbstractArray ? op.value : n.scalar ? op.value : fill(op.value, n.dims)
evaluate(op::Permute, n, xs) = permutedims(only(xs), op.perm)

function evaluate(op::Map, n, xs)
    n.scalar && return apply_scalar(op.f, n.eltype, xs)
    out = Array{n.eltype}(undef, n.dims)
    out .= op.f.(xs...)
    out
end

# A scalar node's inputs are numbers. Inferred with the arrays other nodes
# pass, `f(xs...)` would compile `f` of matrices (sqrt, exp, tanh) that never runs.
Base.@nospecializeinfer apply_scalar(@nospecialize(f), T, @nospecialize(xs)) = convert(T, f(xs...))

function evaluate(op::Reduce, n, xs)
    y = fold_array(op.fold, only(xs), op.axes)
    n.scalar ? only(y) : y
end

function evaluate(::Reshape, n, xs)
    x = only(xs)
    n.scalar && return x isa Number ? x : only(x)
    x isa Number ? fill(x, n.dims) : reshape(x, n.dims)
end

# A selection's or a scatter's indices, with its index arrays.
indices_of(op, xs) = map(i -> i isa IndexInput ? xs[i.input] : i, op.indices)

function evaluate(op::Select, n, xs)
    y = first(xs)[indices_of(op, xs)...]
    n.scalar ? y : y isa Number ? fill(y) : y
end

function evaluate(op::Scatter, n, xs)
    out = zeros(n.eltype, n.dims)
    targets = view(LinearIndices(out), indices_of(op, xs)...)
    for (t, g) in zip(targets, first(xs))
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

Evaluate `program` on Julia arrays and numbers, in the structure of its
results, node by node: the reference that prepared programs are tested
against, with no plan and no kernels.

```jldoctest
julia> interpret(trace(sum, [1.0, 2.0]), [3.0, 4.0])
7.0
```
"""
function interpret(p::Program, args...)
    inputs = argument_leaves(p, args)
    values = Vector{Any}(undef, length(p.nodes))
    for (k, id) in enumerate(p.arguments)
        values[id] = inputs[k]
    end
    keep = live(p.nodes, p.results)
    for (id, n) in enumerate(p.nodes)
        (keep[id] && !(n.op isa Argument)) || continue
        values[id] = evaluate(n.op, n, map(i -> values[i], n.inputs))
    end
    rebuild(k -> k === nothing ? nothing : values[p.results[k]], p.output_structure)
end
