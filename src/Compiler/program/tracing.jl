# Tracing: ordinary Julia array code runs on stand-ins that record nodes.
#
# There is one scalar stand-in, `Traced`, a handle to a node that behaves as a
# number. Outside a broadcast it is a scalar result such as `sum(x)`. Inside a
# broadcast it stands for the element of an array node: the broadcast function
# runs on handles, and each primitive operation it performs records a `Map`
# whose shape is the broadcast of its inputs' shapes. So a user's scalar
# function enters the program as primitives, and derivative rules are written
# as ordinary Julia on handles. A function that cannot run on handles (it
# branches on a traced value, say) is recorded whole, as an opaque `Map`.
#
# A traced array is memory, as a Julia array is: an array the traced code
# allocates can be written, and each write is a functional update (see
# `write!`). Views, reshapes and transposes share the memory of the array they
# are made from.

public fold, logsumexp, TracedArray, contract

"""
How a traced array shares the memory of `parent`: it is `op`, a selection, a
reshape or a permutation, of `parent` and of the nodes `inputs` (a
selection's index arrays).
"""
struct Layout{A}
    parent::A
    op::Operation
    inputs::Tuple{Vararg{Int}}
end

"""
An array a program is written with: a stand-in that records what is done to
it as nodes of its tape. It is memory, as a Julia array is: an array the
traced code allocates can be written, and views, reshapes and transposes
share the memory of the array they are made from. A package that adds
functions programs call defines them on it, in Tempest's primitives.
"""
mutable struct TracedArray{T, N} <: AbstractArray{T, N}
    const tape::Tape
    # The node holding the array's contents. A write records a new node and
    # moves the array, and every array sharing its memory, to it.
    id::Int
    const dims::NTuple{N, Int}
    # The array whose memory this one shares, or nothing.
    const layout::Union{Nothing, Layout}
    # The program's inputs are never written.
    const input::Bool
    # The layouts made from this array, which read its writes.
    const views::Vector{WeakRef}
end

struct Traced{T} <: Real
    tape::Tape
    id::Int
end

TracedArray{T, N}(tape::Tape, id::Int, dims) where {T, N} = TracedArray{T, N}(tape, id, dims, nothing, false, WeakRef[])
TracedArray(tape::Tape, id::Int; input = false) =
    TracedArray{tape.nodes[id].eltype, length(tape.nodes[id].dims)}(tape, id, tape.nodes[id].dims, nothing, input, WeakRef[])
Traced(tape::Tape, id::Int) = Traced{tape.nodes[id].eltype}(tape, id)

node(x::Union{TracedArray, Traced}) = x.tape.nodes[x.id]

Base.size(x::TracedArray) = x.dims
Base.IndexStyle(::Type{<:TracedArray}) = IndexCartesian()
Base.show(io::IO, x::TracedArray{T, N}) where {T, N} = print(io, "TracedArray{$T}(%", x.id, ", ", x.dims, ")")
Base.show(io::IO, ::MIME"text/plain", x::TracedArray) = show(io, x)
Base.show(io::IO, x::Traced{T}) where {T} = print(io, "Traced{$T}(%", x.id, ")")

function broadcast_dims(@nospecialize(dims...))
    n = maximum(length, dims; init = 0)
    ntuple(n) do k
        extent = 1
        for d in dims
            e = k <= length(d) ? d[k] : 1
            e == 1 && continue
            extent == 1 || extent == e || throw(DimensionMismatch("cannot broadcast shapes $(dims)"))
            extent = e
        end
        extent
    end
end

constant!(tape::Tape, x::Number) = record!(tape, Constant(x), (), (), typeof(x); scalar = true)
constant!(tape::Tape, x::AbstractArray{<:Number}) = record!(tape, Constant(collect(x)), (), size(x), eltype(x))

# A range is the index itself: it stays a range, and lowers to a formula
# rather than an array in memory, as does a range reshaped or transposed. So
# `reshape(1:size(x, 2), 1, :)` broadcast against `x` is the position along
# its second axis, and `reshape(1:length(x), size(x))` the linear index.
constant!(tape::Tape, x::AbstractRange{<:Integer}) = record!(tape, Constant(x), (), size(x), eltype(x))
constant!(tape::Tape, x::Base.ReshapedArray{<:Integer, N, <:AbstractRange}) where {N} =
    record!(tape, Reshape(), (constant!(tape, parent(x)),), size(x), eltype(x))
constant!(tape::Tape, x::LinearAlgebra.AdjOrTrans{<:Integer, <:AbstractRange}) =
    record!(tape, Reshape(), (constant!(tape, parent(x)),), size(x), eltype(x))

operand_id(tape, x::Union{Traced, TracedArray}) = x.id
operand_id(tape, x::Number) = constant!(tape, x)

function tape_of(args...)
    for a in args
        a isa Union{Traced, TracedArray} && return a.tape
    end
    error("no traced operand")
end

# Exact identities: multiplying or dividing by one, adding or subtracting zero.
# They return an operand unchanged, so equal values stay one node.
# A plain number, or a handle to a scalar constant: its value; else nothing.
constant_value(x::Traced) = (op = node(x).op; op isa Constant && op.value isa Number ? op.value : nothing)
constant_value(x::Number) = x
constant_value(x) = nothing
plain(x) = constant_value(x) !== nothing
one_(x) = plain(x) && isone(constant_value(x))
zero_(x) = plain(x) && iszero(constant_value(x))
neutral(::typeof(*), a, b) = one_(b) ? a : one_(a) ? b : nothing
neutral(::typeof(/), a, b) = one_(b) ? a : nothing
neutral(::typeof(+), a, b) = zero_(b) ? a : zero_(a) ? b : nothing
neutral(::typeof(-), a, b) = zero_(b) ? a : nothing
neutral(f, args...) = nothing

function simplify(@nospecialize(f), @nospecialize(args...))
    length(args) == 2 && any(plain, args) || return nothing
    a = neutral(f, args...)
    a isa Traced || return nothing
    n = node(a)
    dims = broadcast_dims(map(x -> x isa Traced ? node(x).dims : (), args)...)
    T = Base.promote_op(f, map(x -> x isa Traced ? node(x).eltype : typeof(x), args)...)
    n.dims == dims && n.eltype === T && n.scalar == all(x -> !(x isa Traced) || node(x).scalar, args) ? a : nothing
end

# The tracer does not specialize on what it traces. A program's functions
# and shapes are its data, and Julia code specialized on them is compiled
# once per program, hundreds of instances for a transformer step, before a
# single node is recorded. So the tracer's entry points take their arguments
# unspecialized and dispatch at run time, which costs a traced operation
# about a microsecond.
"Record `f` applied elementwise to the operands, broadcast together."
function record_map(@nospecialize(f), @nospecialize(args...))
    simplified = simplify(f, args...)
    simplified === nothing || return simplified
    tape = tape_of(args...)
    ids = map(a -> operand_id(tape, a), args)
    nodes = tape.nodes
    dims = broadcast_dims(map(i -> nodes[i].dims, ids)...)
    T = Base.promote_op(f, map(i -> nodes[i].eltype, ids)...)
    scalar = all(i -> nodes[i].scalar, ids)
    Traced(tape, record!(tape, Map(f), ids, dims, T; scalar))
end

# Primitive scalar functions. Each has a derivative rule in `differentiate.jl`.
const UNARY = (
    :-, :abs, :abs2, :sign, :exp, :exp2, :exp10, :expm1, :log, :log2, :log10, :log1p,
    :sqrt, :cbrt, :sin, :cos, :tan, :sinh, :cosh, :tanh, :asin, :acos, :atan, :asinh,
    :acosh, :atanh, :inv, :floor, :ceil, :round, :trunc, :isnan, :isinf, :isfinite,
    :signbit, :!,
)
const BINARY = (
    :+, :-, :*, :/, :^, :max, :min, :atan, :copysign, :flipsign, :rem, :mod, :div, :fld,
    :cld, :<, :<=, :>, :>=, :(==), :!=, :isless, :&, :|, :xor,
)

for f in UNARY
    @eval Base.$f(x::Traced) = record_map(Base.$f, x)
end

for f in BINARY
    @eval begin
        Base.$f(x::Traced, y::Traced) = record_map(Base.$f, x, y)
        Base.$f(x::Traced, y::Real) = record_map(Base.$f, x, y)
        Base.$f(x::Real, y::Traced) = record_map(Base.$f, x, y)
        Base.$f(x::Traced, y::Bool) = record_map(Base.$f, x, y)
        Base.$f(x::Bool, y::Traced) = record_map(Base.$f, x, y)
    end
end

Base.:+(x::Traced) = x
Base.:^(x::Traced, p::Integer) = record_map(^, x, p)
Base.ifelse(c::Traced, a, b) = record_map(ifelse, c, a, b)
Base.ifelse(c::Traced, a::Real, b::Real) = record_map(ifelse, c, a, b)
Base.muladd(a::Traced, b::Real, c::Real) = record_map(muladd, a, b, c)
Base.muladd(a::Real, b::Traced, c::Real) = record_map(muladd, a, b, c)
Base.muladd(a::Real, b::Real, c::Traced) = record_map(muladd, a, b, c)
Base.muladd(a::Traced, b::Traced, c::Real) = record_map(muladd, a, b, c)
Base.muladd(a::Traced, b::Traced, c::Traced) = record_map(muladd, a, b, c)
Base.muladd(a::Traced, b::Real, c::Traced) = record_map(muladd, a, b, c)
Base.muladd(a::Real, b::Traced, c::Traced) = record_map(muladd, a, b, c)
Base.fma(a::Traced, b::Real, c::Real) = muladd(a, b, c)
Base.clamp(x::Traced, lo::Real, hi::Real) = min(max(x, lo), hi)

# `x^2` and other literal powers reach here from broadcasts and from code.
function Base.literal_pow(::typeof(^), x::Traced, ::Val{p}) where {p}
    p == 0 && return one(node(x).eltype)
    p == 1 && return x
    p == 2 && return x * x
    p == -1 && return inv(x)
    record_map(^, x, p)
end

cast(::Type{T}, x::Traced) where {T} = node(x).eltype === T ? x : record_map(Cast{T}(), x)

for T in (Float16, Float32, Float64, BFloat16, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Bool)
    @eval (::Type{$T})(x::Traced) = cast($T, x)
end

Base.convert(::Type{T}, x::Traced) where {T <: Number} = cast(T, x)
Base.convert(::Type{Traced{T}}, x::Traced{T}) where {T} = x
Base.float(x::Traced{T}) where {T} = cast(float(T), x)
Base.oftype(x::Traced{T}, y::Real) where {T} = convert(T, y)
Base.oftype(x::Traced{T}, y::Traced) where {T} = cast(T, y)

for f in (:zero, :one, :typemin, :typemax, :eps, :floatmin, :floatmax)
    @eval begin
        Base.$f(::Traced{T}) where {T} = $f(T)
        Base.$f(::Type{Traced{T}}) where {T} = $f(T)
    end
end

# Broadcasting over traced arrays.
struct TracedStyle <: Broadcast.BroadcastStyle end
Base.BroadcastStyle(::Type{<:TracedArray}) = TracedStyle()
Base.BroadcastStyle(s::TracedStyle, ::Broadcast.AbstractArrayStyle) = s

handle(x::TracedArray) = Traced(x.tape, x.id)
handle(x::TracedArray, tape) = handle(x)
handle(x::AbstractArray{<:Number}, tape) = Traced(tape, constant!(tape, x))
handle(x, tape) = handle(x)
handle(x::Traced) = x
handle(x::Number) = x
handle(x::Base.RefValue) = x[]
handle(x::Val) = x
handle(x::Function) = x
handle(x::Type) = x
handle(x) = throw(ArgumentError("cannot broadcast a $(typeof(x)) with traced arrays; pass arrays as inputs"))

operand_dims(x::TracedArray) = x.dims
operand_dims(x::Traced) = node(x).dims
operand_dims(x::AbstractArray{<:Number}) = size(x)
operand_dims(x) = ()

# Each function of a broadcast, nested ones included, runs on handles by
# itself, so a function that cannot be traced is recorded whole without taking
# the functions around it along.
function trace_broadcast(@nospecialize(f), @nospecialize(args), tape)
    mark = length(tape.nodes)
    result = try
        f(args...)
    catch
        resize!(tape.nodes, mark)
        nothing
    end
    result isa Traced && return result
    result isa Number && return Traced(tape, constant!(tape, result))
    # The function did not run on handles: record it whole, with its arguments
    # that are neither traced nor numbers bound into it.
    result === nothing || resize!(tape.nodes, mark)
    inputs = filter(a -> a isa Union{Traced, Number}, collect(args))
    g = length(inputs) == length(args) ? f : Bound(f, map(a -> a isa Union{Traced, Number} ? Slot() : a, args))
    record_map(g, inputs...)
end

broadcast_handle(@nospecialize(bc::Broadcast.Broadcasted), tape) =
    trace_broadcast(bc.f, map(a -> broadcast_handle(a, tape), bc.args), tape)
broadcast_handle(x, tape) = handle(x, tape)

broadcast_dims_of(@nospecialize(bc::Broadcast.Broadcasted)) = broadcast_dims(map(broadcast_dims_of, bc.args)...)
broadcast_dims_of(x) = operand_dims(x)

function tape_in(@nospecialize(bc::Broadcast.Broadcasted))
    for a in bc.args
        t = a isa Broadcast.Broadcasted ? tape_in(a) : a isa Union{Traced, TracedArray} ? a.tape : nothing
        t === nothing || return t
    end
    nothing
end

function Base.copy(@nospecialize(bc::Broadcast.Broadcasted{TracedStyle}))
    tape = tape_in(bc)
    TracedArray(tape, filled(tape, broadcast_handle(bc, tape).id, broadcast_dims_of(bc)))
end

# Node `id` as an array of shape `dims`: a result smaller than a broadcast, or
# a scalar, is filled out to it.
function filled(tape, id, dims)
    n = tape.nodes[id]
    n.dims != dims || n.scalar ? record!(tape, Map(identity), (id,), dims, n.eltype) : id
end

"A function with some arguments fixed; each `Slot` takes the next input."
struct Bound{F, A <: Tuple}
    f::F
    fixed::A
end

struct Slot end

fill_slots(::Tuple{}, xs) = ()
fill_slots(fixed::Tuple, xs) = first(fixed) isa Slot ?
    (first(xs), fill_slots(Base.tail(fixed), Base.tail(xs))...) :
    (first(fixed), fill_slots(Base.tail(fixed), xs)...)
(b::Bound)(xs...) = b.f(fill_slots(b.fixed, xs)...)
Base.show(io::IO, b::Bound) = print(io, b.f)

# Reductions.
function reduced_axes(x, dims)
    dims isa Colon && return Tuple(1:ndims(x))
    ds = dims isa Integer ? (dims,) : Tuple(dims)
    all(d -> d isa Integer && d > 0, ds) || throw(ArgumentError("reduction dimensions must be positive integers"))
    Tuple(sort!(unique!([Int(d) for d in ds if d <= ndims(x)])))
end

"""
    Tempest.fold(f::Fold, x; dims = :)

Reduce `x` over `dims` with the fold `f`.
"""
function fold(f::Fold, x::TracedArray; dims = :)
    axes = reduced_axes(x, dims)
    T = fold_eltype(f, eltype(x))
    if dims isa Colon
        return Traced(x.tape, record!(x.tape, Reduce(f, axes), (x.id,), (), T; scalar = true))
    end
    isempty(axes) && return copy(x)
    out = ntuple(d -> d in axes ? 1 : size(x, d), ndims(x))
    TracedArray(x.tape, record!(x.tape, Reduce(f, axes), (x.id,), out, T))
end

fold(f::Fold, x::AbstractArray; dims = :) = (y = fold_array(f, x, reduced_axes(x, dims)); dims isa Colon ? only(y) : y)

Base.sum(x::TracedArray; dims = :) = fold(SumFold(), x; dims)
Base.maximum(x::TracedArray; dims = :) = fold(MaxFold(), x; dims)
Base.minimum(x::TracedArray; dims = :) = fold(MinFold(), x; dims)
Base.prod(x::TracedArray; dims = :) = fold(ProdFold(), x; dims)
Base.sum(f, x::TracedArray; dims = :) = sum(f.(x); dims)
Base.maximum(f, x::TracedArray; dims = :) = maximum(f.(x); dims)
Base.minimum(f, x::TracedArray; dims = :) = minimum(f.(x); dims)
Base.prod(f, x::TracedArray; dims = :) = prod(f.(x); dims)

"""
    Tempest.logsumexp(x; dims = :)

`log.(sum(exp.(x); dims))`, computed stably in one pass.
"""
logsumexp(x::AbstractArray; dims = :) = fold(Relative(Exponential(), LOGSUMEXP), x; dims)

builtin_fold(::Union{typeof(+), typeof(Base.add_sum)}) = SumFold()
builtin_fold(::Union{typeof(*), typeof(Base.mul_prod)}) = ProdFold()
builtin_fold(::typeof(max)) = MaxFold()
builtin_fold(::typeof(min)) = MinFold()
builtin_fold(op) = nothing

struct OperatorCombine{O}
    op::O
end
(c::OperatorCombine)(a, b) = (c.op(a[1], b[1]),)

function Base.mapreduce(f, op, x::TracedArray; dims = :, init = nothing)
    y = f === identity ? x : f.(x)
    builtin = builtin_fold(op)
    if builtin !== nothing
        r = fold(builtin, y; dims)
        return init === nothing ? r : op.(init, r)
    end
    init === nothing && throw(ArgumentError("reducing a traced array by $op needs `init`, its identity"))
    T = promote_type(eltype(y), typeof(init))
    fold(GenericFold(; init = (convert(T, init),), combine = OperatorCombine(op)), T.(y); dims)
end

Base.reduce(op, x::TracedArray{<:Real}; dims = :, init = nothing) = mapreduce(identity, op, x; dims, init)

# Products. Julia arrays among the factors are constants of the program.

"""
    contract(a, b, la, lb, lo)

The product of `a` and `b` summed over shared axes, as an einsum: `la` and `lb`
label the axes of `a` and `b`, `lo` those of the result, and labels of both
factors missing from the result are summed over. Every product is one: `a * b`
of matrices is `contract(a, b, (1, 2), (2, 3), (1, 3))`, and so is each product
of an Einops `einsum`.

```jldoctest
julia> Tempest.Compiler.contract([1.0 2; 3 4], [5.0 6; 7 8], (1, 2), (2, 3), (1, 3))
2×2 Matrix{Float64}:
 19.0  22.0
 43.0  50.0
```
"""
function contract end

function contract(a::TracedArray, b::TracedArray, la, lb, lo)
    T = promote_type(eltype(a), eltype(b))
    labels = Dict{Int, Int}()
    for (l, n) in zip((la..., lb...), (size(a)..., size(b)...))
        get!(labels, l, n) == n || throw(DimensionMismatch("contracted axes differ: $(size(a)) and $(size(b))"))
    end
    dims = map(l -> labels[l], lo)
    TracedArray(a.tape, record!(a.tape, Contract(la, lb, lo), (a.id, b.id), dims, T))
end

contract(a::TracedArray, b::AbstractArray{<:Number}, la, lb, lo) = contract(a, TracedArray(a.tape, constant!(a.tape, b)), la, lb, lo)
contract(a::AbstractArray{<:Number}, b::TracedArray, la, lb, lo) = contract(TracedArray(b.tape, constant!(b.tape, a)), b, la, lb, lo)
contract(a::AbstractArray, b::AbstractArray, la, lb, lo) = einsum(a, la, b, lb, lo)

# Each product with a traced factor, the other traced or not.
for (A, B) in ((TracedArray, TracedArray), (TracedArray, AbstractArray), (AbstractArray, TracedArray))
    @eval begin
        Base.:*(a::$A{<:Real, 2}, b::$B{<:Real, 2}) = contract(a, b, (1, 2), (2, 3), (1, 3))
        Base.:*(a::$A{<:Real, 2}, b::$B{<:Real, 1}) = contract(a, b, (1, 2), (2,), (1,))
    end
end

# A row vector or a diagonal matrix times a traced array, as LinearAlgebra
# multiplies them: a row times a vector is a number.
Base.:*(a::LinearAlgebra.AdjOrTransAbsVec{<:Real}, b::TracedArray{<:Real, 2}) = contract(a, b, (1, 2), (2, 3), (1, 3))
Base.:*(a::LinearAlgebra.AdjOrTransAbsVec{<:Real}, b::TracedArray{<:Real, 1}) = sum(vec(a) .* b)
Base.:*(a::Transpose{T, <:AbstractVector}, b::TracedArray{T, 1}) where {T <: Real} = sum(vec(a) .* b)
Base.:*(a::LinearAlgebra.Diagonal{<:Real}, b::TracedArray{<:Real, 1}) = a.diag .* b

# Layout. As in Julia, `reshape`, `vec`, `view`, `transpose` and
# `PermutedDimsArray` share the memory of the array they are made from;
# `permutedims` and `getindex` copy.

"""
A layout of `x` that shares its memory: it reads later writes into `x`, and a
write into it writes into `x`. `op` is applied to `x` and the nodes `inputs`.
"""
function layout(x::TracedArray, op, inputs, dims)
    id = record!(x.tape, op, (x.id, inputs...), dims, eltype(x))
    y = TracedArray{eltype(x), length(dims)}(x.tape, id, Tuple(dims), Layout(x, op, Tuple(inputs)), false, WeakRef[])
    push!(x.views, WeakRef(y))
    y
end

"A copy of `x`: its contents now, in an array of its own."
Base.copy(x::TracedArray) = TracedArray(x.tape, x.id)

function Base.reshape(x::TracedArray, dims::Tuple{Vararg{Int}})
    prod(dims) == length(x) || throw(DimensionMismatch("cannot reshape $(size(x)) to $dims"))
    dims == size(x) && return x
    layout(x, Reshape(), (), dims)
end

function check_permutation(x, perm)
    perm = Tuple(perm)
    isperm(perm) && length(perm) == ndims(x) || throw(ArgumentError("$perm is not a permutation of $(ndims(x)) axes"))
    perm
end

function Base.permutedims(x::TracedArray, perm)
    perm = check_permutation(x, perm)
    perm == ntuple(identity, ndims(x)) && return copy(x)
    TracedArray(x.tape, record!(x.tape, Permute(perm), (x.id,), map(p -> size(x, p), perm), eltype(x)))
end

# A traced array, as a view of one is, rather than Base's wrapper.
function Base.PermutedDimsArray(x::TracedArray, perm)
    perm = check_permutation(x, perm)
    perm == ntuple(identity, ndims(x)) ? x : layout(x, Permute(perm), (), map(p -> size(x, p), perm))
end

Base.permutedims(x::TracedArray{<:Any, 2}) = permutedims(x, (2, 1))
Base.permutedims(x::TracedArray{<:Any, 1}) = reshape(x, (1, length(x)))
Base.transpose(x::TracedArray{<:Real, 2}) = PermutedDimsArray(x, (2, 1))
Base.transpose(x::TracedArray{<:Real, 1}) = reshape(x, (1, length(x)))
Base.adjoint(x::TracedArray{<:Real}) = transpose(x)

# Indexing: integers drop an axis; ranges keep one; an array of integers,
# traced or not, replaces it by its own axes (a gather). A traced integer is
# an array without axes.
selection_index(i::Integer) = Int(i)
selection_index(i::Base.Slice) = 1:length(i)
selection_index(i::AbstractRange{<:Integer}) = i
# A vector of integers that steps evenly is a range: a formula, not a gather.
selection_index(i::Vector{<:Integer}) = linear_range(i)
selection_index(i::AbstractArray{<:Integer}) = i
selection_index(i) = throw(ArgumentError("unsupported index $(typeof(i)) into a traced array"))

index_array(i) = i
index_array(i::Traced{T}) where {T <: Integer} = TracedArray{T, 0}(i.tape, i.id, ())
index_array(::Traced{Bool}) = mask_error()
index_array(::TracedArray{Bool}) = mask_error()
mask_error() = throw(ArgumentError("a traced mask selects a number of elements known only when the program runs"))

# Linear indices, as a range where they step evenly.
function linear_range(l)
    l isa AbstractVector && !isempty(l) || return l
    s = length(l) > 1 ? l[2] - l[1] : 1
    s != 0 && all(k -> l[k] - l[k - 1] == s, 2:length(l)) ? (first(l):s:last(l)) : l
end

"""
    selection(x, I) -> (array, indices)

What indexing `x` by `I` selects, and from which array: `x` or, for one
linear index into an array of several axes, `vec(x)`; by integers, ranges and
arrays of integers. A linear integer is its Cartesian index, and an array of
Cartesian indices the linear indices it holds.
"""
function selection(x::TracedArray, I)
    I = Base.to_indices(x, map(index_array, I))
    length(I) == 1 && ndims(x) != 1 || return x, map(selection_index, I)
    i = only(I)
    i isa Integer && return (checkbounds(x, i); (x, Tuple(CartesianIndices(x)[i])))
    i isa AbstractArray{<:CartesianIndex} && (i = linear_range(LinearIndices(x)[i]))
    vec(x), (selection_index(i),)
end

# With `scalar`, as `getindex`, a selection copies the elements, and one
# element is a number; otherwise it is a view that shares their memory.
function select(x::TracedArray, I; scalar)
    x, indices = selection(x, I)
    for (k, i) in enumerate(indices)
        i isa TracedArray || all(j -> 1 <= j <= size(x, k), i) || throw(BoundsError(x, I))
    end
    dims = Tuple(n for i in indices if !(i isa Integer) for n in size(i))
    full = length(indices) == ndims(x) &&
        all(k -> indices[k] isa AbstractRange && indices[k] == 1:size(x, k), 1:ndims(x))
    full && !(scalar && isempty(dims)) && return scalar ? copy(x) : x
    # Arrays of integers are inputs of the node, constants unless traced.
    inputs = Int[]
    indices = map(indices) do i
        i isa AbstractArray && !(i isa AbstractRange) || return i
        push!(inputs, i isa TracedArray ? i.id : constant!(x.tape, i))
        IndexInput(length(inputs) + 1)
    end
    scalar || return layout(x, Select(indices), inputs, dims)
    scalar = isempty(dims)
    id = record!(x.tape, Select(indices), (x.id, inputs...), dims, eltype(x); scalar)
    scalar ? Traced(x.tape, id) : TracedArray(x.tape, id)
end

Base.getindex(x::TracedArray, I...) = select(x, I; scalar = true)
Base.view(x::TracedArray, I...) = select(x, I; scalar = false)

# A reversal is a selection by reversed ranges, not a loop of swaps.
Base.reverse(x::TracedArray; dims = :) = x[ntuple(k -> dims isa Colon || k in dims ? (size(x, k):-1:1) : (1:size(x, k)), ndims(x))...]

# Triangles are masks of positions, not loops over elements.
LinearAlgebra.tril(x::TracedArray{<:Any, 2}, k::Integer = 0) =
    ifelse.(reshape(1:size(x, 1), :, 1) .- reshape(1:size(x, 2), 1, :) .>= -k, x, zero(eltype(x)))
LinearAlgebra.triu(x::TracedArray{<:Any, 2}, k::Integer = 0) =
    ifelse.(reshape(1:size(x, 1), :, 1) .- reshape(1:size(x, 2), 1, :) .<= -k, x, zero(eltype(x)))

# A scan is an ordered fold, which the language does not have.
const SCAN = "a scan (cumsum, cumprod, accumulate) is an ordered fold, which Tempest does not trace"
Base.cumsum(x::TracedArray; dims = :) = throw(ArgumentError(SCAN))
Base.cumprod(x::TracedArray; dims = :) = throw(ArgumentError(SCAN))
Base.accumulate(op, x::TracedArray; dims = :, kws...) = throw(ArgumentError(SCAN))

# `map` over arrays of one shape is a broadcast, and `dot` a sum of products,
# rather than loops over their elements.
function Base.map(f, x::TracedArray, xs::AbstractArray...)
    all(y -> size(y) == size(x), xs) || throw(DimensionMismatch("map over arrays of sizes $(map(size, (x, xs...)))"))
    f.(x, xs...)
end

function LinearAlgebra.dot(x::TracedArray{<:Real}, y::TracedArray{<:Real})
    length(x) == length(y) || throw(DimensionMismatch("dot of arrays of lengths $(length(x)) and $(length(y))"))
    sum(vec(x) .* vec(y))
end

# Tracing a function.

# What becomes a program's input among the leaves of its arguments.
is_input(leaf) = leaf isa AbstractArray{<:Number} || leaf isa Number && !(leaf isa Integer)

"""
    trace(f, args...) -> Program

Run `f` on stand-ins for `args` and record what it computes. Arrays and numbers
among `args`, also inside tuples, named tuples and immutable structs, become the
program's inputs, except integers: they are constants, so that they can be
sizes, and an integer that varies between executions is an array of one. The
results are arrays and numbers in the same kinds of structure, and `nothing`
stays `nothing`.

```jldoctest
julia> p = trace(x -> sum(abs2, x), [1.0, 2.0, 3.0])
Program with 3 nodes: Float64(3,) → Float64
  %1 = argument :: Float64(3,)
  %2 = map abs2 %1 :: Float64(3,)
  %3 = reduce sum over (1,) %2 :: Float64  [result]
```
"""
function trace(f, args...)
    tape = Tape()
    arguments = Int[]
    function stand_in(leaf)
        if leaf isa AbstractArray{<:Number}
            id = record!(tape, Argument(), (), size(leaf), eltype(leaf))
            push!(arguments, id)
            return TracedArray(tape, id; input = true)
        elseif is_input(leaf)
            id = record!(tape, Argument(), (), (), typeof(leaf); scalar = true)
            push!(arguments, id)
            return Traced(tape, id)
        end
        leaf
    end
    traced = rebuild(stand_in, args)
    position = Dict(id => k for (k, id) in enumerate(arguments))
    input_structure = structure_of(t -> t isa Union{Traced, TracedArray} ? position[t.id] : nothing, traced)
    output = f(traced...)
    results = Int[]
    output_structure = structure_of(output) do leaf
        # As a gradient has for an integer argument.
        leaf === nothing && return nothing
        id = if leaf isa Union{Traced, TracedArray}
            leaf.tape === tape || throw(ArgumentError("a result was traced by another program"))
            leaf.id
        elseif leaf isa Number
            constant!(tape, leaf)
        else
            throw(ArgumentError("a program's results are arrays and numbers, got $(typeof(leaf))"))
        end
        push!(results, id)
        length(results)
    end
    prune(Program(tape.nodes, arguments, results, input_structure, output_structure, tape.backward))
end
