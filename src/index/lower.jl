# From the program level to the index level. Layout operations become
# formulas; every other node becomes a value over its own domain.

# How a program node is read: a value through a formula over the node's own
# domain, a constant, or an integer computed from the index (positions).
abstract type Access end

struct ValueAccess <: Access
    value::Int
    index::Tuple{Vararg{Affine}}
end

struct LiteralAccess{T} <: Access
    value::T
end

struct IndexAccess <: Access
    expr::Affine
    eltype::DataType
end

# Each variable of a domain, read as itself; an axis of extent one reads zero,
# the canonical form of its only index.
identity_index(dims) = ntuple(k -> dims[k] == 1 ? constant(0) : var(k), length(dims))

# A value ranges over the axes of its node wider than one, so programs that
# differ by axes of extent one have the same values; the node reads its value
# at the variables of those axes.
wide_axes(dims) = Tuple(k for k in eachindex(dims) if dims[k] != 1)
wide_index(dims) = map(var, wide_axes(dims))

# Expressions over a shape as expressions over its axes `keep`: each kept axis
# its variable among them, any other (of extent one) zero; and their domain.
function narrowing(dims, keep)
    index = ntuple(k -> (j = findfirst(==(k), keep)) === nothing ? constant(0) : var(j), length(dims))
    (index, Domain(map(k -> dims[k], keep)))
end

narrow(o::Read, (index, d)) = Read(o.value, map(e -> substitute(e, index, d), o.index))
narrow(o::Literal, _) = o
narrow(o::IndexOperand{T}, (index, d)) where {T} = IndexOperand(substitute(o.expr, index, d), T)

# A definition over a node of shape `dims` as one over its value's domain. An
# opaque operation reads its operands in their own shapes.
narrow(d::Union{Input, Opaque}, dims) = d
narrow(d::ConstantArray, dims) = ConstantArray(reshape(d.data, map(k -> dims[k], wide_axes(dims))))
narrow(d::Pointwise, dims) = (n = narrowing(dims, wide_axes(dims)); Pointwise(d.f, map(o -> narrow(o, n), d.operands)))

# A fold ranges over the axes of its iteration wider than one, and over its
# reduced axes even of extent one when all of them are.
function narrow(d::Reduction, dims)
    s = d.iteration
    keep = all(k -> s[k] == 1, d.reduced) ? Tuple(k for k in eachindex(s) if s[k] != 1 || k in d.reduced) : wide_axes(s)
    n = narrowing(s, keep)
    Reduction(d.fold, narrow(d.operand, n), n[2].extents, Tuple(j for (j, k) in enumerate(keep) if k in d.reduced))
end

# Read a node of shape `dims` from a reader whose domain is `reader`, as Julia
# broadcasts: axes of extent one, and missing trailing axes, read index zero.
broadcast_index(dims) = ntuple(k -> dims[k] == 1 ? constant(0) : var(k), length(dims))

operand(a::ValueAccess, dims, reader::Domain) =
    Read(a.value, map(e -> substitute(e, broadcast_index(dims), reader), a.index))
operand(a::LiteralAccess, dims, ::Domain) = Literal(a.value)
operand(a::IndexAccess, dims, reader::Domain) =
    IndexOperand(substitute(a.expr, broadcast_index(dims), reader), a.eltype)

function linear_index(dims, d::Domain)
    linear = constant(0)
    stride = 1
    for (k, n) in enumerate(dims)
        linear = add(linear, scale(var(k), stride), d)
        stride *= n
    end
    linear
end

function delinearize(linear, dims, d::Domain)
    stride = 1
    map(dims) do n
        e = fmod(fdiv(linear, stride, d), n, d)
        stride *= n
        e
    end
end

# The index of a layout node's input, for each index of the node, as
# expressions over the node's domain. `held(i, axis, extent)` is the index
# that index array `i` of a selection holds where its axes are the node's from
# `axis + 1` on, and how many axes it has.
layout_index(::Reshape, dims, parentdims, d, held) = delinearize(linear_index(dims, d), parentdims, d)

function layout_index(op::Permute, dims, parentdims, d, held)
    index = Vector{Affine}(undef, length(parentdims))
    for (k, p) in enumerate(op.perm)
        index[p] = var(k)
    end
    Tuple(index)
end

function layout_index(op::Select, dims, parentdims, d, held)
    axis = 0
    ntuple(length(op.indices)) do p
        i = op.indices[p]
        if i isa Integer
            constant(i - 1)
        elseif i isa AbstractRange
            axis += 1
            add(scale(var(axis), step(i)), first(i) - 1, d)
        else
            e, rank = held(i, axis, parentdims[p])
            axis += rank
            e
        end
    end
end

# The zero-based index that an array of one-based indices into an axis of
# `extent` holds at `at`, its index over `d`. A value is read through an
# indirect term, a gather, whose entries lie in the axis (between a constant
# array's extremes); an index computed from the position is a formula.
function held_index(a::ValueAccess, at, d, values, extent)
    data = values[a.value].definition
    lo, hi = data isa ConstantArray && !isempty(data.data) ? extrema(data.data) : (1, extent)
    index = map(e -> substitute(e, at, d), a.index)
    add(normalize(0, (Indirect(a.value, index, lo, hi) => 1,), d), -1, d)
end
held_index(a::IndexAccess, at, d, values, extent) = add(substitute(a.expr, at, d), -1, d)
held_index(a::LiteralAccess, at, d, values, extent) = constant(a.value - 1)

compose(a::ValueAccess, index, d) = ValueAccess(a.value, map(e -> substitute(e, index, d), a.index))
compose(a::LiteralAccess, index, d) = a
compose(a::IndexAccess, index, d) = IndexAccess(substitute(a.expr, index, d), a.eltype)

# A node of shape `dims` filled out to a larger shape, as Julia broadcasts.
fill_out(a::ValueAccess, dims, d) = ValueAccess(a.value, map(e -> substitute(e, broadcast_index(dims), d), a.index))
fill_out(a::LiteralAccess, dims, d) = a
fill_out(a::IndexAccess, dims, d) = IndexAccess(substitute(a.expr, broadcast_index(dims), d), a.eltype)

# A product is a sum fold of a pointwise product: over the domain of the
# result's axes then the summed ones, the two operands (converted to Float32
# when they are half precision) are multiplied and summed over the summed axes.
# The result reads that reduction through a formula. Kernels recognize the
# pattern; nothing else needs to.
function contraction!(value!, op::Contract, inputs, input_dims, n)
    summed = [l for l in op.a if l in op.b && !(l in op.out)]
    labels = [collect(op.out); summed]
    extent = Dict{Int, Int}()
    foreach((l, e) -> extent[l] = e, (op.a..., op.b...), (input_dims[1]..., input_dims[2]...))
    iteration = Tuple(extent[l] for l in labels)
    d = Domain(iteration)
    read(access, ls) = Read(
        access.value,
        map(e -> substitute(e, Tuple(var(findfirst(==(l), labels)) for l in ls), d), access.index),
    )
    A = n.eltype <: HalfFloat ? Float32 : n.eltype
    function factor(access, ls)
        r = read(access, ls)
        A === n.eltype && return r
        cast = value!(Pointwise(Cast{A}(), (r,)), iteration, A)
        Read(cast.value, cast.index)
    end
    product = value!(Pointwise(*, (factor(inputs[1], op.a), factor(inputs[2], op.b))), iteration, A)
    reduced = Tuple((length(op.out) + 1):length(labels))
    kept = ntuple(k -> k in reduced ? 1 : iteration[k], length(iteration))
    # The result's axes come first, so its value is read at the same variables.
    value!(Reduction(SumFold(), Read(product.value, product.index), iteration, reduced), kept, n.eltype)
end

"""
    lower(program) -> IndexGraph

The program at the index level.
"""
function lower(p::Program; keep_call = (n, nodes) -> false, epilogue = (n, nodes) -> false)
    p = expand_calls(p, n -> keep_call(n, p.nodes))
    nodes = p.nodes
    keep = live(nodes, p.results)
    # A claimed call computes all its results, used or not.
    for (id, n) in enumerate(nodes)
        n.op isa Call && keep[id] && foreach(r -> keep[r] = true, results_of(nodes, n.op.instance))
    end
    values = Value[]
    access = Vector{Access}(undef, length(nodes))
    # A value defined over a node's shape, over the node's wide axes instead.
    function value!(definition, dims, T; scalar = false)
        push!(values, Value(narrow(definition, dims), map(k -> dims[k], wide_axes(dims)), T, scalar))
        ValueAccess(length(values), wide_index(dims))
    end
    value!(definition, n::Node) = value!(definition, n.dims, n.eltype; scalar = n.scalar)
    # Materialized call operands, one per node, shared by every call reading it.
    copies = Dict{Int, ValueAccess}()
    arguments = Int[]
    for (id, n) in enumerate(nodes)
        if n.op isa Argument
            access[id] = value!(Input(findfirst(==(id), p.arguments)), n)
            push!(arguments, access[id].value)
            continue
        end
        keep[id] || continue
        d = Domain(n.dims)
        op = n.op
        inputs = map(i -> access[i], n.inputs)
        input_dims = map(i -> nodes[i].dims, n.inputs)
        access[id] = if op isa Constant
            op.value isa AbstractArray ? value!(ConstantArray(op.value), n) : LiteralAccess(op.value)
        elseif op isa Positions
            IndexAccess(add(op.axis == 0 ? linear_index(n.dims, d) : identity_index(n.dims)[op.axis], 1, d), Int32)
        elseif op isa Map && all(a -> a isa LiteralAccess, inputs)
            # Constants fold: a map of constants is a constant.
            LiteralAccess(convert(n.eltype, op.f(map(a -> a.value, inputs)...)))
        elseif op isa Map && op.f === identity && nodes[only(n.inputs)].eltype === n.eltype
            # Filling out to a larger shape is a formula, as a layout is.
            fill_out(only(inputs), only(input_dims), d)
        elseif op isa Map
            value!(Pointwise(op.f, map((a, dims) -> operand(a, dims, d), inputs, input_dims)), n)
        elseif op isa Reduce
            parent = only(input_dims)
            value!(Reduction(op.fold, operand(only(inputs), parent, Domain(parent)), parent, op.axes), n)
        elseif op isa Contract
            all(a -> a isa ValueAccess, inputs) || error("a product of a constant is not supported yet")
            contraction!(value!, op, inputs, input_dims, n)
        elseif op isa Call
            # A claimed call reads its operands densely, in their program
            # shapes; anything else (a layout, a constant) is materialized.
            reads = map(inputs, n.inputs) do a, i
                dims = nodes[i].dims
                if a isa ValueAccess && values[a.value].scalar == nodes[i].scalar && reads_storage(values[a.value], a.index, dims)
                    return Read(a.value, a.index)
                end
                copy = get!(copies, i) do
                    value!(Pointwise(identity, (operand(a, dims, Domain(dims)),)), nodes[i])
                end
                Read(copy.value, copy.index)
            end
            shapes = map(i -> (nodes[i].dims, nodes[i].scalar), n.inputs)
            value!(Opaque(n, reads, shapes, epilogue(n, nodes)), n)
        elseif op isa Union{Reshape, Permute, Select}
            held = function (i, axis, extent)
                rank = length(input_dims[i.input])
                (held_index(inputs[i.input], ntuple(k -> var(axis + k), rank), d, values, extent), rank)
            end
            compose(first(inputs), layout_index(op, n.dims, first(input_dims), d, held), d)
        else
            reads = map((a, dims) -> operand(a, dims, Domain(dims)), inputs, input_dims)
            shapes = map(i -> (nodes[i].dims, nodes[i].scalar), n.inputs)
            value!(Opaque(n, reads, shapes), n)
        end
    end
    results = Operand[operand(access[r], nodes[r].dims, Domain(nodes[r].dims)) for r in p.results]
    IndexGraph(
        values,
        arguments,
        results,
        [nodes[r].dims for r in p.results],
        [nodes[r].scalar for r in p.results],
        [nodes[r].eltype for r in p.results],
    )
end
