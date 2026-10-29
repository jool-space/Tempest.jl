# From the program level to the index level. Layout operations become
# formulas; every other node becomes a value over its own domain.

# How a program node is read: a value through a formula over the node's own
# domain, a constant, or an integer computed from the index (a range).
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

# A definition over a node of shape `dims` as one over its value's domain.
narrow(d::Input, dims) = d
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

# A fold by key keeps the key's formulas for its value's wide axes, and reads
# the contents it starts from over them; its iteration is narrowed where it is
# formed.
narrow(d::Keyed, dims) = Keyed(d.fold, d.operand, d.iteration, map(k -> d.key[k], wide_axes(dims)),
    d.into === nothing ? nothing : narrow(d.into, narrowing(dims, wide_axes(dims))))

# A scan ranges over its node's wide axes, as a map does; along an axis of
# extent one, each position is its own prefix.
function narrow(d::Scan, dims)
    keep = wide_axes(dims)
    n = narrowing(dims, keep)
    Scan(d.fold, narrow(d.operand, n), Tuple(j for (j, k) in enumerate(keep) if k in d.axes), d.reverse, d.exclusive)
end

# A recurrence's state keeps its axis among the wide ones: its steps are more than one.
function narrow(d::Carried, dims)
    n = narrowing(dims, wide_axes(dims))
    Carried(narrow(d.init, n), narrow(d.from, n), findfirst(==(d.axis), wide_axes(dims)))
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

# A read of a select by positions, where its condition is decided at every
# position the read reaches, is a read of the branch taken there: a slice of
# a write that misses what was written reads the contents written over, as
# XLA simplifies a slice of a dynamic update. So the cotangent of a
# concatenation reaches each piece as a slice of it, rather than through the
# masks of every piece written after.
function taken(a::ValueAccess, values, d::Domain)
    while true
        def = values[a.value].definition
        def isa Pointwise && def.f === ifelse || return a
        c = decided(values, def.operands[1], a.index, d)
        c === nothing && return a
        a = access(def.operands[c ? 2 : 3], a.index, d)
        a isa ValueAccess || return a
    end
end
taken(a::Access, values, d::Domain) = a

access(o::Read, index, d) = ValueAccess(o.value, map(e -> substitute(e, index, d), o.index))
access(o::Literal, index, d) = LiteralAccess(o.value)
access(o::IndexOperand{T}, index, d) where {T} = IndexAccess(substitute(o.expr, index, d), T)

# Whether a condition read at `index` holds at every position of `d`, at
# none, or neither (nothing): comparisons of positions, by the ranges of
# their formulas, and conjunctions, disjunctions and negations of those.
decided(values, o::Literal{Bool}, index, d::Domain) = o.value
decided(values, o::Operand, index, d::Domain) = nothing
function decided(values, o::Read, index, d::Domain)
    def = values[o.value].definition
    def isa Pointwise || return nothing
    at = map(e -> substitute(e, index, d), o.index)
    xs = def.operands
    if def.f === (&) || def.f === (|)
        cs = map(x -> decided(values, x, at, d), xs)
        all = def.f === (&)
        any(==(!all), cs) && return !all
        Base.all(==(all), cs) ? all : nothing
    elseif def.f === (!)
        c = decided(values, only(xs), at, d)
        c === nothing ? nothing : !c
    elseif def.f in (<, <=, >, >=, ==, !=) && length(xs) == 2
        es = map(x -> position_formula(x, at, d), xs)
        any(isnothing, es) && return nothing
        lo, hi = affine_range(add(es[1], scale(es[2], -1), d), d)
        f = def.f
        holds = f === (<) ? hi < 0 : f === (<=) ? hi <= 0 : f === (>) ? lo > 0 : f === (>=) ? lo >= 0 : lo == hi == 0
        fails = f === (<) ? lo >= 0 : f === (<=) ? lo > 0 : f === (>) ? hi <= 0 : f === (>=) ? hi < 0 : lo > 0 || hi < 0
        f === (!=) && ((holds, fails) = (fails, holds))
        holds ? true : fails ? false : nothing
    else
        nothing
    end
end

# A position compared, as a formula over `d`: one of an integer type its
# range fits, so that the comparison in that type is the formulas'.
function position_formula(o::IndexOperand{T}, at, d::Domain) where {T}
    T <: Integer || return nothing
    e = substitute(o.expr, at, d)
    lo, hi = affine_range(e, d)
    typemin(T) <= lo && hi <= typemax(T) ? e : nothing
end
position_formula(o::Literal{<:Integer}, at, d::Domain) = o.value isa Bool ? nothing : constant(o.value)
position_formula(o::Operand, at, d::Domain) = nothing

# A node of shape `dims` filled out to a larger shape, as Julia broadcasts.
fill_out(a::ValueAccess, dims, d) = ValueAccess(a.value, map(e -> substitute(e, broadcast_index(dims), d), a.index))
fill_out(a::LiteralAccess, dims, d) = a
fill_out(a::IndexAccess, dims, d) = IndexAccess(substitute(a.expr, broadcast_index(dims), d), a.eltype)

# A scatter or a write by integers and ranges places its input: each element
# lands at one position, and every other position is `otherwise`, zero or the
# contents written over. At each position, a select between the input, read
# where the position came from, and `otherwise`. The read wraps around the
# input's axes, so that it stays inside them where it is not selected; a
# scatter that covers its result is a formula.
function placement!(value!, indices, input, otherwise, n, d)
    i = identity_index(n.dims)
    conditions = ValueAccess[]
    condition!(f, e, c) = push!(conditions, value!(Pointwise(f, (IndexOperand(e, Int32), Literal(Int32(c)))), n.dims, Bool))
    source = Affine[]
    for (k, s) in enumerate(indices)
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
    # A `true` placed over `false` is the mask itself.
    otherwise == Literal(false) && read(placed) == Literal(true) && return mask
    value!(Pointwise(ifelse, (read(mask), read(placed), otherwise)), n)
end

# The elements of a scatter or a write by index arrays, input `k` of the
# node, each at the position a selection by `indices` reads: the elements over
# their wide axes, and the selection's formula over them, the key. Where the
# elements are a value read in its storage order, they are taken over that
# value's domain, so that a fold of them can join the kernel computing it.
function keyed_elements(indices, k, inputs, input_dims, n, values)
    dims = input_dims[k]
    d = Domain(dims)
    held = function (i, axis, extent)
        rank = length(input_dims[i.input])
        (held_index(inputs[i.input], ntuple(j -> var(axis + j), rank), d, values, extent), rank)
    end
    index, iteration = narrowing(dims, wide_axes(dims))
    key = map(e -> substitute(e, index, iteration), layout_index(Select(indices), dims, n.dims, d, held))
    elements = narrow(operand(inputs[k], dims, d), (index, iteration))
    if elements isa Read && values[elements.value].dims != iteration.extents &&
       reads_storage(values[elements.value], elements.index, iteration.extents)
        own = Domain(values[elements.value].dims)
        at = delinearize(linear_index(own.extents, own), iteration.extents, own)
        key = map(e -> substitute(e, at, own), key)
        elements = Read(elements.value, identity_index(own.extents))
        iteration = own
    end
    (elements, iteration.extents, key)
end

# A scatter by index arrays adds each element at the position the selection
# it is the adjoint of reads it from: a sum by key. Half-precision elements are
# summed in Float32 and rounded once, after the fold.
function scattered!(value!, op::Scatter, inputs, input_dims, n, values)
    elements, iteration, key = keyed_elements(op.indices, 1, inputs, input_dims, n, values)
    A = n.eltype <: HalfFloat ? Float32 : n.eltype
    sums = value!(Keyed(SumFold(), elements, iteration, key), n.dims, A; scalar = n.scalar)
    A === n.eltype && return sums
    value!(Pointwise(Cast{n.eltype}(), (operand(sums, n.dims, Domain(n.dims)),)), n)
end

# A write by index arrays places each element at the position the selection
# reads, over the contents written: an overwrite by key into them.
function written!(value!, op::Update, inputs, input_dims, n, values)
    elements, iteration, key = keyed_elements(op.indices, 2, inputs, input_dims, n, values)
    value!(Keyed(Overwrite(), elements, iteration, key, operand(inputs[1], n.dims, Domain(n.dims))), n)
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

# Integer arithmetic on ranges and literals stays a formula: `(1:n) .- 1` is
# the index shifted, `2 .* (1:n)` scaled, `fld.(1:n, 4)` divided. Sums,
# differences, multiples by a constant, division and remainder by a positive
# constant (truncating ones where the dividend is never negative), and
# conversions between integer types, as long as the result stays within its
# type.
function index_arithmetic(f, inputs, input_dims, n, d)
    n.eltype <: Integer && any(a -> a isa IndexAccess, inputs) || return nothing
    all(a -> a isa IndexAccess || a isa LiteralAccess && a.value isa Integer, inputs) || return nothing
    es = map((a, dims) -> a isa LiteralAccess ? constant(a.value) : substitute(a.expr, broadcast_index(dims), d), inputs, input_dims)
    binary = length(es) == 2
    e = if f === (+) && binary
        add(es[1], es[2], d)
    elseif f === (-) && binary
        add(es[1], mul(es[2], -1, d), d)
    elseif f === (-) && length(es) == 1
        mul(es[1], -1, d)
    elseif f === (*) && binary && (isconstant(es[1]) || isconstant(es[2]))
        isconstant(es[1]) ? mul(es[2], es[1].constant, d) : mul(es[1], es[2].constant, d)
    elseif f in (fld, mod, div, rem) && binary && isconstant(es[2]) && es[2].constant > 0 &&
           (f in (fld, mod) || affine_range(es[1], d)[1] >= 0)
        f in (fld, div) ? fdiv(es[1], es[2].constant, d) : fmod(es[1], es[2].constant, d)
    elseif f isa Cast && length(es) == 1
        es[1]
    else
        return nothing
    end
    # A formula is exact; integers wrap. Where the result could leave the
    # type, the arithmetic stays as written.
    lo, hi = affine_range(e, d)
    typemin(n.eltype) <= lo && hi <= typemax(n.eltype) || return nothing
    IndexAccess(e, n.eltype)
end

# A quotient by a constant whose reciprocal is exact in the result's type, a
# power of two, is the product by that reciprocal, which rounds alike: so
# `x / 8` and `x * 0.125` are one value.
function map_definition(f, operands, T)
    if f === (/) && T <: Union{Base.IEEEFloat, BFloat16} && length(operands) == 2 && operands[2] isa Literal && operands[2].value isa Real
        c = T(operands[2].value)
        r = inv(c)
        c == operands[2].value && isfinite(r) && !iszero(r) && abs(significand(c)) == 1 && abs(significand(r)) == 1 &&
            return Pointwise(*, (operands[1], Literal(r)))
    end
    Pointwise(f, operands)
end

"""
    lower(program) -> IndexGraph

The program at the index level.
"""
# Recurrences: each loop's step spliced into the program, every node of the
# step over its own shape and a last axis of the steps, as JAX batches a
# function over a mapped axis. A slice of a scanned array is the array, its
# sliced axis last; a value the step captures is the same at every step; a
# product takes the steps as a batch; and the state is a `Carry`, the value
# the step computes one step back. So below the program level a recurrence
# is values over its steps like any others, and one definition, `Carried`.
function inline_loops(p::Program)
    any(n -> n.op isa Loop, p.nodes) || return p
    nodes = Node[]
    new = zeros(Int, length(p.nodes))
    spliced = IdDict{LoopBody, Vector{Int}}()
    backward = BitSet()
    for (id, n) in enumerate(p.nodes)
        if n.op isa Loop
            new[id] = get!(spliced, n.op.body) do
                from = length(nodes) + 1
                results = splice_step!(nodes, n.op.body, map(i -> new[i], n.inputs))
                id in p.backward && union!(backward, from:length(nodes))
                results
            end[n.op.result]
        else
            push!(nodes, Node(n.op, map(i -> new[i], n.inputs), n.dims, n.eltype, n.scalar))
            new[id] = length(nodes)
            id in p.backward && push!(backward, new[id])
        end
    end
    Program(nodes, map(i -> new[i], p.arguments), map(i -> new[i], p.results), p.input_structure, p.output_structure, backward)
end

# The nodes of a loop's step over its steps, appended to `nodes`, the loop's
# inputs at `inputs`; the node of each of its results.
function splice_step!(nodes, body::LoopBody, inputs)
    steps, step = body.steps, body.program
    nstate, nscanned = length(body.carried), length(body.sliced)
    over(n::Node) = n.scalar ? (steps,) : (n.dims..., steps)
    emit!(op, ins, dims, T) = (push!(nodes, Node(op, Tuple(ins), Tuple(dims), T, false)); length(nodes))
    # A node of rank `r` over the steps, read where a node of rank `rank`
    # broadcasts it: its axes, then axes of extent one, then the steps.
    function broadcast(id, r, rank)
        n = nodes[id]
        r >= rank && return id
        emit!(Reshape(), (id,), (n.dims[1:(end - 1)]..., ntuple(_ -> 1, rank - r)..., steps), n.eltype)
    end
    rank(n::Node) = n.scalar ? 0 : length(n.dims)
    at = zeros(Int, length(step.nodes))
    carries = Int[]
    for (k, a) in enumerate(step.arguments)
        n = step.nodes[a]
        outer = nodes[inputs[k]]
        at[a] = if k <= nstate
            # One step: the state is where it starts.
            steps == 1 ? emit!(Reshape(), (inputs[k],), over(n), n.eltype) :
                (push!(carries, emit!(Carry(0), (inputs[k],), over(n), n.eltype)); last(carries))
        elseif k <= nstate + nscanned
            axis = body.sliced[k - nstate]
            m = length(outer.dims)
            axis == m ? inputs[k] : emit!(Permute(Tuple([1:(axis - 1); (axis + 1):m; axis])), (inputs[k],), over(n), n.eltype)
        else
            # A capture, the same at each step.
            shaped = emit!(Reshape(), (inputs[k],), outer.scalar ? (1,) : (outer.dims..., 1), outer.eltype)
            emit!(Map(identity), (shaped,), over(n), n.eltype)
        end
    end
    for (id, n) in enumerate(step.nodes)
        n.op isa Argument && continue
        op = n.op
        ins = map(i -> at[i], n.inputs)
        any(i -> i isa IndexInput, op isa Union{Select, Scatter, Update} ? op.indices : ()) &&
            throw(ArgumentError("a gather or scatter by index arrays in a recurrence's step is not supported yet"))
        at[id] = if op isa Map
            emit!(op, map((i, j) -> broadcast(i, rank(step.nodes[j]), rank(n)), ins, n.inputs), over(n), n.eltype)
        elseif op isa Reduce
            r = emit!(op, ins, n.scalar ? (ntuple(_ -> 1, length(step.nodes[only(n.inputs)].dims))..., steps) : over(n), n.eltype)
            n.scalar ? emit!(Reshape(), (r,), (steps,), n.eltype) : r
        elseif op isa Contract
            label = maximum((op.a..., op.b..., op.out...); init = 0) + 1
            emit!(Contract((op.a..., label), (op.b..., label), (op.out..., label)), ins, over(n), n.eltype)
        elseif op isa Permute
            emit!(Permute((op.perm..., length(op.perm) + 1)), ins, over(n), n.eltype)
        elseif op isa Select
            emit!(Select((op.indices..., 1:steps)), ins, over(n), n.eltype)
        elseif op isa Scatter
            emit!(Scatter((op.indices..., 1:steps)), ins, over(n), n.eltype)
        elseif op isa Update
            emit!(Update((op.indices..., 1:steps)), ins, over(n), n.eltype)
        elseif op isa Union{Reshape, Accumulate}
            emit!(op, ins, over(n), n.eltype)
        else
            throw(ArgumentError("a $(op_name(op)) in a recurrence's step is not supported yet"))
        end
    end
    for (k, c) in enumerate(carries)
        n = nodes[c]
        nodes[c] = Node(Carry(at[step.results[body.carried[k]]]), n.inputs, n.dims, n.eltype, n.scalar)
    end
    [at[r] for r in step.results]
end

function lower(p::Program)
    p = inline_loops(p)
    nodes = p.nodes
    keep = live(nodes, p.results)
    values = Value[]
    backward = BitVector()
    in_backward = false
    numbered = Dict{Value, Int}()
    access = Vector{Access}(undef, length(nodes))
    # A value defined over a node's shape, over the node's wide axes instead.
    # Two nodes with one definition are one value: a subexpression written
    # twice is computed once, and spellings that repeat it read the same value.
    function value!(definition, dims, T; scalar = false)
        v = Value(narrow(definition, dims), map(k -> dims[k], wide_axes(dims)), T, scalar)
        id = get!(numbered, v) do
            push!(values, v)
            push!(backward, in_backward)
            length(values)
        end
        ValueAccess(id, wide_index(dims))
    end
    value!(definition, n::Node) = value!(definition, n.dims, n.eltype; scalar = n.scalar)
    arguments = Int[]
    carried = Tuple{Int, Int}[]
    for (id, n) in enumerate(nodes)
        if n.op isa Argument
            access[id] = value!(Input(findfirst(==(id), p.arguments)), n)
            push!(arguments, access[id].value)
            continue
        end
        keep[id] || continue
        in_backward = id in p.backward
        d = Domain(n.dims)
        op = n.op
        inputs = map(i -> access[i], n.inputs)
        input_dims = map(i -> nodes[i].dims, n.inputs)
        access[id] = if op isa Constant && op.value isa AbstractRange
            # A range is the index itself: `first + step·i`.
            r = op.value
            IndexAccess(add(mul(only(identity_index(n.dims)), step(r), d), first(r), d), n.eltype)
        elseif op isa Constant
            op.value isa AbstractArray ? value!(ConstantArray(op.value), n) : LiteralAccess(op.value)
        elseif op isa Map && all(a -> a isa LiteralAccess, inputs)
            # Constants fold: a map of constants is a constant.
            LiteralAccess(convert(n.eltype, op.f(map(a -> a.value, inputs)...)))
        elseif op isa Map && (arithmetic = index_arithmetic(op.f, inputs, input_dims, n, d)) !== nothing
            arithmetic
        elseif op isa Map && op.f isa Union{typeof(identity), typeof(stopgrad)} && nodes[only(n.inputs)].eltype === n.eltype
            # Filling out to a larger shape is a formula, as a layout is. A
            # stopped gradient is its input once the program is differentiated.
            fill_out(only(inputs), only(input_dims), d)
        elseif op isa Map
            value!(map_definition(op.f, map((a, dims) -> operand(a, dims, d), inputs, input_dims), n.eltype), n)
        elseif op isa Reduce
            parent = only(input_dims)
            value!(Reduction(op.fold, operand(only(inputs), parent, Domain(parent)), parent, op.axes), n)
        elseif op isa Accumulate
            value!(Scan(op.fold, operand(only(inputs), only(input_dims), d), op.axes, op.reverse, op.exclusive), n)
        elseif op isa Carry
            # Defined once the state it carries is lowered, after it.
            push!(values, Value(Carried(Literal(nothing), Literal(nothing), 0), map(k -> n.dims[k], wide_axes(n.dims)), n.eltype, false))
            push!(backward, in_backward)
            push!(carried, (length(values), id))
            ValueAccess(length(values), wide_index(n.dims))
        elseif op isa Contract
            contraction!(value!, op, inputs, input_dims, n)
        elseif op isa Scatter && !any(i -> i isa IndexInput, op.indices)
            placement!(value!, op.indices, only(inputs), Literal(zero(n.eltype)), n, d)
        elseif op isa Update && !any(i -> i isa IndexInput, op.indices)
            placement!(value!, op.indices, inputs[2], operand(inputs[1], n.dims, d), n, d)
        elseif op isa Update
            written!(value!, op, inputs, input_dims, n, values)
        elseif op isa Union{Reshape, Permute, Select}
            held = function (i, axis, extent)
                rank = length(input_dims[i.input])
                (held_index(inputs[i.input], ntuple(k -> var(axis + k), rank), d, values, extent), rank)
            end
            taken(compose(first(inputs), layout_index(op, n.dims, first(input_dims), d, held), d), values, d)
        else
            scattered!(value!, op, inputs, input_dims, n, values)
        end
    end
    for (v, id) in carried
        n = nodes[id]
        d = Domain(n.dims)
        init = operand(access[only(n.inputs)], nodes[only(n.inputs)].dims, d)
        values[v] = Value(narrow(Carried(init, operand(access[n.op.from], n.dims, d), length(n.dims)), n.dims), values[v].dims, n.eltype, false)
    end
    results = Operand[operand(access[r], nodes[r].dims, Domain(nodes[r].dims)) for r in p.results]
    IndexGraph(
        values,
        arguments,
        results,
        [nodes[r].dims for r in p.results],
        [nodes[r].scalar for r in p.results],
        [nodes[r].eltype for r in p.results],
        Pair{Int, String}[],
        backward,
    )
end
