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

# A scatter by integers and ranges places its input: each element lands at
# one position, and every other position is zero. At each position, a select
# between the input, read where the position came from, and zero. The read
# wraps around the input's axes, so that it stays inside them where it is not
# selected; a scatter that covers its result is a formula.
function placement!(value!, op::Scatter, input, n, d)
    i = identity_index(n.dims)
    conditions = ValueAccess[]
    condition!(f, e, c) = push!(conditions, value!(Pointwise(f, (IndexOperand(e, Int32), Literal(Int32(c)))), n.dims, Bool))
    source = Affine[]
    for (k, s) in enumerate(op.indices)
        if s isa Integer
            n.dims[k] == 1 || condition!(==, i[k], s - 1)
            continue
        end
        # The distance from the range's first position, along its step.
        a = abs(step(s))
        o = step(s) > 0 ? add(i[k], 1 - first(s), d) : add(scale(i[k], -1), first(s) - 1, d)
        push!(source, fmod(fdiv(o, a, d), length(s), d))
        lo, hi = affine_range(o, d)
        lo < 0 && condition!(>=, o, 0)
        hi > (length(s) - 1) * a && condition!(<=, o, (length(s) - 1) * a)
        a > 1 && condition!(==, fmod(o, a, d), 0)
    end
    placed = compose(input, Tuple(source), d)
    isempty(conditions) && return placed
    read(a) = operand(a, n.dims, d)
    mask = foldl((a, b) -> value!(Pointwise(&, (read(a), read(b))), n.dims, Bool), conditions)
    # A placed `true` is the mask itself.
    n.eltype === Bool && read(placed) == Literal(true) && return mask
    value!(Pointwise(ifelse, (read(mask), read(placed), Literal(zero(n.eltype)))), n)
end

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
    # A factor read at the labels `ls`: a value, or a constant or positions
    # filling the factor's shape.
    read(access, ls) = operand(compose(access, Tuple(var(findfirst(==(l), labels)) for l in ls), d), iteration, d)
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

The program at the index level. A call with a reference is expanded in place,
and its place kept for implementations to claim.
"""
function lower(p::Program)
    p, expansions = expand_calls(p, f -> !opaque(f))
    nodes = p.nodes
    # A call computes all its results, used or not, from all its arguments.
    keep = live(nodes, [p.results; [i for e in expansions for i in [e.inputs; e.results]]])
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
    # A call reads its arguments densely, in their program shapes; anything
    # else (a layout, a constant) is materialized, once per node.
    copies = Dict{Int, ValueAccess}()
    function dense(i)
        r = dense_read(values, access[i], nodes[i])
        r === nothing || return r
        dims = nodes[i].dims
        copy = get!(copies, i) do
            value!(Pointwise(identity, (operand(access[i], dims, Domain(dims)),)), nodes[i])
        end
        Read(copy.value, copy.index)
    end
    # Expansions by their first node, outermost first. Each one's arguments are
    # read densely (`arguments_of`) before the values it expands to, which
    # begin at `starts`.
    expanding = Dict{Int, Vector{Int}}()
    for (k, e) in enumerate(expansions)
        isempty(e.nodes) || push!(get!(expanding, first(e.nodes), Int[]), k)
    end
    arguments_of = Vector{Any}(undef, length(expansions))
    starts = zeros(Int, length(expansions))
    first_value = zeros(Int, length(nodes) + 1)
    arguments = Int[]
    for (id, n) in enumerate(nodes)
        first_value[id] = length(values) + 1
        for k in sort(get(expanding, id, Int[]); by = k -> -length(expansions[k].nodes))
            arguments_of[k] = map(dense, Tuple(expansions[k].inputs))
            starts[k] = length(values) + 1
        end
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
            contraction!(value!, op, inputs, input_dims, n)
        elseif op isa Call
            # An opaque call, which implementations alone realize.
            value!(Opaque(n, map(dense, n.inputs), map(i -> (nodes[i].dims, nodes[i].scalar), n.inputs)), n)
        elseif op isa Scatter && !any(i -> i isa IndexInput, op.indices)
            placement!(value!, op, only(inputs), n, d)
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
    # Where each call went: a call's results are nodes of their own here.
    first_value[end] = length(values) + 1
    calls = CallSite[]
    for (k, e) in enumerate(expansions)
        isempty(e.nodes) && continue
        returned = map(enumerate(e.results)) do (j, r)
            Node(Call(e.f, e.kwargs, j, length(nodes) + k, e.structure), Tuple(e.inputs), nodes[r].dims, nodes[r].eltype, nodes[r].scalar)
        end
        site = expanded_site(values, collect(starts[k]:(first_value[last(e.nodes) + 1] - 1)), arguments_of[k],
            map(i -> (nodes[i].dims, nodes[i].scalar), Tuple(e.inputs)), returned, access[e.results])
        site === nothing || push!(calls, site)
    end
    for (id, n) in enumerate(nodes)
        n.op isa Call && n.op.instance == id && keep[id] || continue
        rs = results_of(nodes, id)
        d = values[access[id].value].definition
        push!(calls, CallSite([access[r].value for r in rs], d.operands, d.shapes, nodes[rs],
            [operand(access[r], nodes[r].dims, Domain(nodes[r].dims)) for r in rs]))
    end
    sort!(calls; by = c -> (first(c.values), -length(c.values)))
    results = Operand[operand(access[r], nodes[r].dims, Domain(nodes[r].dims)) for r in p.results]
    IndexGraph(
        values,
        arguments,
        results,
        [nodes[r].dims for r in p.results],
        [nodes[r].scalar for r in p.results],
        [nodes[r].eltype for r in p.results],
        calls,
    )
end

# Node `n`, accessed as `a`, read as a value whose storage holds it densely in
# the node's shape; nothing when it is not.
function dense_read(values, a, n)
    a isa ValueAccess || return nothing
    v = values[a.value]
    r = operand(a, n.dims, Domain(n.dims))
    v.eltype === n.eltype && v.scalar == n.scalar && reads_storage(v, r.index, n.dims) ? r : nothing
end

# An expanded call's place, when implementations can take it: each result is
# read densely from a distinct value of the call's own.
function expanded_site(values, region, operands, shapes, nodes, accesses)
    reads = map((a, n) -> dense_read(values, a, n), accesses, nodes)
    all(r -> r !== nothing && r.value in region, reads) && allunique(r.value for r in reads) || return nothing
    CallSite(region, operands, shapes, nodes, Read[reads...])
end
