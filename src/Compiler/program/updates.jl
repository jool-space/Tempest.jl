# Writes into traced arrays, as functional updates. Any traced array may be
# written: each write records the array's new contents as a node and moves the
# array, with every array sharing its memory, to it. Later reads see the latest contents; reads before a write keep
# what they read. A write of part of an array is an `Update`: the old contents
# with the values written at the positions a selection by integers, ranges and
# arrays of integers, traced or not, names. So the program states which
# elements each write replaces, and its derivative masks the old contents and
# selects the values written. Where several land at one position, one of them
# does, unspecified which, as in JAX.
#
#
# A write into an input, or into a view, reshape or transpose of one, is a
# write into the caller's array: the input's contents when the program ends
# are a result that replaces it (see `trace`).

# Allocation. Julia leaves the contents of `similar` undefined; in a traced
# program they are zero, so reading them before a write is defined.
element_type(::Type{Traced{T}}) where {T} = T
element_type(T) = T

Base.similar(x::TracedArray, ::Type{T}, dims::Dims) where {T <: Number} =
    TracedArray(x.tape, record!(x.tape, Constant(zero(element_type(T))), (), dims, element_type(T)))

Base.fill(v::Traced, dims::Dims) = TracedArray(v.tape, filled(v.tape, v.id, dims))
Base.fill(v::Traced, dims::Tuple{}) = TracedArray(v.tape, filled(v.tape, v.id, dims))
Base.fill(v::Traced, dims::Integer...) = fill(v, Dims(dims))

"""
    write!(x, contents)

Make node `contents`, of the shape and element type of `x`, the contents of
`x`: the array that owns its memory (`x`, or the parent of the layouts that
make `x`) takes the contents written through those layouts, and every layout
made from it reads them.
"""
function write!(x::TracedArray, contents::Int)
    owner = x
    while owner.layout !== nothing
        owner = owner.layout.parent
    end
    isempty(owner.tape.steps) || owner.id > last(owner.tape.steps) || throw(ArgumentError(STEP_WRITE))
    while x.layout !== nothing
        (; parent, op, inputs) = x.layout
        contents = written_through(op, parent, inputs, contents)
        x = parent
    end
    x.id = contents
    moved!(x)
    x
end

# A layout's parent with the layout's elements replaced by node `c`.
written_through(::Reshape, p, inputs, c) = record!(p.tape, Reshape(), (c,), p.dims, eltype(p))
written_through(op::Permute, p, inputs, c) = record!(p.tape, Permute(Tuple(invperm(op.perm))), (c,), p.dims, eltype(p))
written_through(op::Select, p, inputs, c) = update(p, op.indices, inputs, c)

# The layouts made from `x` read its new contents. A layout nothing holds any
# more is dropped.
function moved!(x::TracedArray)
    filter!(r -> r.value !== nothing, x.views)
    for r in x.views
        y = r.value
        y === nothing && continue
        y.id = record!(x.tape, y.layout.op, (x.id, y.layout.inputs...), y.dims, eltype(y))
        moved!(y)
    end
end

"""
    joined_write(tape, x, indices, arrays, values, j, positions) -> node

The contents of node `x` with the writes of a loop's steps made at once, as
one write: step `s` writes, as `Update(indices)` does, its slice of node
`values`, the steps its last axis, with the index `j` at `positions[s]`, an
integer or a range, and each index array `IndexInput(k)` the node `first`
of `arrays(k)`, an array every step reads or, where `last` of it says so,
one with the steps its last axis, of which the step reads its slice. The
positions keep the steps' writes apart. The axes from the first to the last
whose index differs from step to step are merged into one, which the write
indexes by the linear index of each element's position along them (see
`linear_index`); the other axes it indexes as each step does.
"""
function joined_write(tape::Tape, x::Int, indices, arrays, values::Int, j, positions)
    nodes = tape.nodes
    A, T, P = nodes[x].dims, nodes[x].eltype, length(positions)
    stepwise(k) = indices[k] isa IndexInput && last(arrays(indices[k].input))
    # Each index's axes among a step's values, in order.
    function extent(k)
        i = indices[k]
        i isa Integer && return ()
        i isa IndexInput || return (length(i),)
        id = first(arrays(i.input))
        stepwise(k) ? nodes[id].dims[1:(end - 1)] : nodes[id].dims
    end
    blocks = [extent(k) for k in eachindex(indices)]
    a, b = extrema([j; filter(stepwise, eachindex(indices))])
    inner = Tuple(Iterators.flatten(blocks[a:b]))
    # The values in the order of the axes the write's index along them
    # gives: the steps beside the axes merged, before those of the indices
    # after them.
    outer = sum(length, blocks[1:(a - 1)]; init = 0)
    V = nodes[values].dims
    perm = (1:(outer + length(inner))..., length(V), (outer + length(inner) + 1):(length(V) - 1)...)
    values = issorted(perm) ? values : record!(tape, Permute(perm), (values,), map(d -> V[d], perm), nodes[values].eltype)
    written = (V[1:outer]..., prod(inner) * P, V[(outer + length(inner) + 1):(end - 1)]...)
    nodes[values].dims == written || (values = record!(tape, Reshape(), (values,), written, nodes[values].eltype))
    nodes[values].eltype === T || (values = record!(tape, Map(Cast{T}()), (values,), written, T))
    # The other indices as each step has them, their index arrays the write's.
    others = Int[]
    renumbered(i) = i isa IndexInput ? (push!(others, first(arrays(i.input))); IndexInput(1 + length(others))) : i
    head = map(renumbered, indices[1:(a - 1)])
    tail = map(renumbered, indices[(b + 1):end])
    # Where only `j` differs and the steps' positions follow one another,
    # the write is of their range.
    following = all(r -> r isa Integer || step(r) == 1, positions) && all(s -> first(positions[s + 1]) == last(positions[s]) + 1, 1:(P - 1))
    if a == b && following
        return update(TracedArray(tape, x), (head..., first(first(positions)):last(last(positions)), tail...), others, values)
    end
    # Else of the axes from `a` to `b` merged, by the linear index of each
    # element's position along them, its axes in their block, the steps last.
    before = 0
    components = map(a:b) do k
        i, block = indices[k], blocks[k]
        shape = (ntuple(_ -> 1, before)..., block..., ntuple(_ -> 1, length(inner) - before - length(block))...)
        before += length(block)
        k == j && return sliding_index(tape, positions, shape)
        i isa Integer && return i
        i isa IndexInput || return Traced(tape, constant!(tape, reshape(collect(i), shape..., 1)))
        id = first(arrays(i.input))
        Traced(tape, record!(tape, Reshape(), (id,), (shape..., stepwise(k) ? P : 1), nodes[id].eltype))
    end
    linear = linear_index(components, A[a:b])
    push!(others, record!(tape, Reshape(), (linear.id,), (prod(inner) * P,), node(linear).eltype))
    merged = TracedArray(tape, record!(tape, Reshape(), (x,), (A[1:(a - 1)]..., prod(A[a:b]), A[(b + 1):end]...), T))
    w = update(merged, (head..., IndexInput(1 + length(others)), tail...), others, values)
    record!(tape, Reshape(), (w,), A, T)
end

# The index `positions[s]` of each step `s` of a joined write, its elements
# along axes `shape` and the steps last: where each step's starts, stepping
# evenly over the steps, so that lowering sees its first index and the steps'.
function sliding_index(tape, positions, shape)
    P = length(positions)
    stride = first(positions[2]) - first(positions[1])
    steps = Traced(tape, record!(tape, Reshape(), (constant!(tape, 0:stride:((P - 1) * stride)),), (ntuple(_ -> 1, length(shape))..., P), Int))
    start = first(positions)
    start isa Integer && return steps + start
    steps + Traced(tape, record!(tape, Reshape(), (constant!(tape, start),), (shape..., 1), Int))
end

# The arrays that share the memory of `x`: the array that owns it and every
# layout made from it that something still holds, each at the node of its
# contents now.
function sharing(x::TracedArray)
    owner = x
    while owner.layout !== nothing
        owner = owner.layout.parent
    end
    found, pending = TracedArray[], TracedArray[owner]
    while !isempty(pending)
        y = pop!(pending)
        push!(found, y)
        foreach(r -> r.value === nothing || push!(pending, r.value), y.views)
    end
    found
end

"""
    update(x, indices, inputs, v) -> node

The contents of `x` with the elements that `indices` select replaced by those
of node `v`, an array of the selection's shape and the element type of `x`.
`indices` are a selection's, its index arrays the nodes `inputs`.
"""
function update(x::TracedArray, indices, inputs, v::Int)
    full = isempty(inputs) && length(indices) == ndims(x) &&
        all(k -> indices[k] isa AbstractRange && indices[k] == 1:size(x, k), 1:ndims(x))
    full && return v
    # The write's index arrays follow the contents written among its inputs.
    written = map(i -> i isa IndexInput ? IndexInput(i.input + 1) : i, Tuple(indices))
    record!(x.tape, Update(written), (x.id, v, inputs...), x.dims, eltype(x))
end

# The node holding `v`, a number, a handle or an array, with element type `T`.
function converted(tape, v, ::Type{T}) where {T}
    holds_traced(v) && throw(ArgumentError(HOST_CONTAINER))
    v isa Union{Traced, TracedArray} || return constant!(tape, v isa AbstractArray ? convert(AbstractArray{T}, v) : convert(T, v))
    n = node(v)
    n.eltype === T ? v.id : record!(tape, Map(Cast{T}()), (v.id,), n.dims, T; scalar = n.scalar)
end

function Base.setindex!(x::TracedArray, v, I...)
    holds_traced(v) && return written_each!(x, v, I)
    y, indices = selection(x, I)
    indices = map(i -> i isa AbstractVector{<:Integer} && !(i isa TracedArray) ? linear_range(i) : i, indices)
    for (k, i) in enumerate(indices)
        i isa TracedArray || all(j -> 1 <= j <= size(y, k), i) || throw(BoundsError(x, I))
    end
    dims = Tuple(n for i in indices if !(i isa Integer) for n in size(i))
    T = eltype(x)
    contents = if v isa AbstractArray
        Base.setindex_shape_check(v, map(length, indices)...)
        id = converted(x.tape, v, T)
        size(v) == dims ? id : record!(x.tape, Reshape(), (id,), dims, T)
    else
        isempty(dims) || throw(ArgumentError("indexed assignment with a single value to possibly \
            many locations is not supported; perhaps use broadcasting `.=` instead?"))
        converted(x.tape, v, T)
    end
    indices, inputs = index_inputs(x.tape, indices)
    write!(y, update(y, indices, inputs, contents))
    x
end

# `x .= ...`, `broadcast!` and `x ./= n`: the broadcast's result, filled out to
# the shape of `x` and converted to its element type, is its new contents.
function assign!(x::TracedArray, bc::Broadcast.Broadcasted)
    h = cast(eltype(x), broadcast_handle(bc, x.tape))
    write!(x, filled(x.tape, h.id, x.dims))
    x
end

Base.copyto!(x::TracedArray, bc::Broadcast.Broadcasted) = assign!(x, bc)
Base.copyto!(x::TracedArray, bc::Broadcast.Broadcasted{Nothing}) = assign!(x, bc)
Base.copyto!(x::TracedArray, bc::Broadcast.Broadcasted{<:Broadcast.AbstractArrayStyle{0}}) = assign!(x, bc)
Base.copyto!(x::TracedArray, bc::Broadcast.Broadcasted{TracedStyle}) = assign!(x, bc)

# An array the function allocates itself, as `zeros(Float32, n)` does, is a
# constant of the program and holds numbers: what a traced broadcast computes
# is the program's, and is written into arrays allocated from traced ones.
const PLAIN_WRITE = "a traced value cannot be written into an array the function allocates itself, such as `zeros(Float32, n)`; allocate it from a traced array: `similar(x, …)`, `zero(x)` or `fill!(similar(x, …), v)`"

Base.copyto!(::AbstractArray, ::Broadcast.Broadcasted{TracedStyle}) = throw(ArgumentError(PLAIN_WRITE))

# So is one traced number written into an element of such an array, one of
# plain numbers (see `plain_needed`); a view and a Cartesian index come here
# through the array's own integer indices.
function Base.setindex!(y::Array{T}, v::Traced, i::Int) where {T <: Number}
    v isa T || throw(ArgumentError(plain_needed(T, v)))
    invoke(setindex!, Tuple{Array{T}, Any, Int}, y, v, i)
end

function Base.setindex!(y::Array{T}, v::Traced, i::Int, j::Int, I::Int...) where {T <: Number}
    v isa T || throw(ArgumentError(plain_needed(T, v)))
    invoke(setindex!, Tuple{Array{T}, Any, Int, Int, Vararg{Int}}, y, v, i, j, I...)
end

# A host array of traced numbers (see `HOST_CONTAINER`) written into a
# traced array is written as Base's loop writes it, each number at its
# position: a write of one number a position, each a choice of that number
# at its position and the contents elsewhere, the choices one kernel's.
# Faster still is the array made by array code, which no loop of numbers is.
function written_each!(x::TracedArray, v, I)
    J = to_indices(x, I)
    Base.setindex_shape_check(v, (length(i) for i in J if !(i isa Integer))...)
    for (at, e) in zip(Iterators.product(map(i -> i isa Integer ? (i,) : i, J)...), v)
        x[at...] = e
    end
    x
end

function Base.copyto!(dest::TracedArray, src::AbstractArray)
    length(src) <= length(dest) || throw(ArgumentError("destination has fewer elements than required"))
    holds_traced(src) && return written_each!(dest, src, (1:length(src),))
    length(src) == length(dest) || return copyto!(dest, 1, src, 1, length(src))
    id = converted(dest.tape, src, eltype(dest))
    write!(dest, size(src) == size(dest) ? id : record!(dest.tape, Reshape(), (id,), dest.dims, eltype(dest)))
    dest
end

# The `n` elements of `src` from its `s`th on, written over those of `dest`
# from its `d`th on in linear order: an update of the contents of `dest` laid
# out along one axis, laid back. The layout is a node of this write alone,
# not a view of `dest`: a view is moved by every later write into `dest` until
# the collector drops it (see `moved!`), and Base's `stack` copies each slice
# so, which moved the views of all the slices before at each one, T²/2 nodes
# for a stack of T slices.
function Base.copyto!(dest::TracedArray, d::Integer, src::AbstractArray, s::Integer, n::Integer)
    n >= 0 || throw(ArgumentError("tried to copy n=$n elements, but n should be nonnegative"))
    n == 0 && return dest
    checkbounds(dest, d:(d + n - 1))
    elements = vec(src)[s:(s + n - 1)]
    tape, T = dest.tape, eltype(dest)
    flat = ndims(dest) == 1 ? dest : TracedArray(tape, record!(tape, Reshape(), (dest.id,), (length(dest),), T))
    written = update(flat, (linear_range(d:(d + n - 1)),), (), converted(tape, elements, T))
    write!(dest, ndims(dest) == 1 ? written : record!(tape, Reshape(), (written,), dest.dims, T))
    dest
end

function Base.fill!(x::TracedArray, v)
    write!(x, filled(x.tape, converted(x.tape, v, eltype(x)), x.dims))
    x
end

function Base.permutedims!(dest::TracedArray, src::AbstractArray, perm)
    p = permutedims(src, perm)
    size(p) == size(dest) || throw(DimensionMismatch("destination of size $(size(dest)) for a permutation of size $(size(p))"))
    write!(dest, converted(dest.tape, p, eltype(dest)))
    dest
end

function Base.map!(f, dest::TracedArray, xs::AbstractArray...)
    all(x -> size(x) == size(dest), xs) || throw(DimensionMismatch("map! over arrays of sizes $(map(size, (dest, xs...)))"))
    dest .= f.(xs...)
end
