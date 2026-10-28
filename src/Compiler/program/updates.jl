# Writes into traced arrays, as functional updates. An array the traced code
# allocates (with `similar`, `copy`, `zero` or `fill`, or as the result of any
# operation on traced arrays) may be written: each write records the array's
# new contents as a node and moves the array, with every array sharing its
# memory, to it. Later reads see the latest contents; reads before a write keep
# what they read. A write of part of an array is an `Update`: the old contents
# with the values written at the positions a selection by integers, ranges and
# arrays of integers, traced or not, names. So the program states which
# elements each write replaces, and its derivative masks the old contents and
# selects the values written. Where several land at one position, one of them
# does, unspecified which, as in JAX.
#
# The program's inputs, and the arrays sharing their memory, are never written.

const INPUT_WRITE = "mutation of the program's inputs is not supported: a traced program is a \
    function of its inputs, which it reads and never writes, so a write into an input, or into a view, \
    reshape or transpose of one, would change the caller's array without the program stating it. \
    Write into a copy, such as `y = copy(x)`, and return it"

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
    owner.input && throw(ArgumentError(INPUT_WRITE))
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
    v isa Union{Traced, TracedArray} || return constant!(tape, v isa AbstractArray ? convert(AbstractArray{T}, v) : convert(T, v))
    n = node(v)
    n.eltype === T ? v.id : record!(tape, Map(Cast{T}()), (v.id,), n.dims, T; scalar = n.scalar)
end

function Base.setindex!(x::TracedArray, v, I...)
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

function Base.copyto!(dest::TracedArray, src::AbstractArray)
    length(src) <= length(dest) || throw(ArgumentError("destination has fewer elements than required"))
    length(src) == length(dest) || return copyto!(dest, 1, src, 1, length(src))
    id = converted(dest.tape, src, eltype(dest))
    write!(dest, size(src) == size(dest) ? id : record!(dest.tape, Reshape(), (id,), dest.dims, eltype(dest)))
    dest
end

function Base.copyto!(dest::TracedArray, d::Integer, src::AbstractArray, s::Integer, n::Integer)
    n >= 0 || throw(ArgumentError("tried to copy n=$n elements, but n should be nonnegative"))
    n == 0 || (vec(dest)[d:(d + n - 1)] = vec(src)[s:(s + n - 1)])
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
