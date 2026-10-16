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

struct TracedArray{T, N} <: AbstractArray{T, N}
    tape::Tape
    id::Int
    dims::NTuple{N, Int}
end

struct Traced{T} <: Real
    tape::Tape
    id::Int
end

TracedArray(tape::Tape, id::Int) = TracedArray{tape.nodes[id].eltype, length(tape.nodes[id].dims)}(tape, id, tape.nodes[id].dims)
Traced(tape::Tape, id::Int) = Traced{tape.nodes[id].eltype}(tape, id)

node(x::Union{TracedArray, Traced}) = x.tape.nodes[x.id]

Base.size(x::TracedArray) = x.dims
Base.IndexStyle(::Type{<:TracedArray}) = IndexCartesian()
Base.show(io::IO, x::TracedArray{T, N}) where {T, N} = print(io, "TracedArray{$T}(%", x.id, ", ", x.dims, ")")
Base.show(io::IO, ::MIME"text/plain", x::TracedArray) = show(io, x)
Base.show(io::IO, x::Traced{T}) where {T} = print(io, "Traced{$T}(%", x.id, ")")

function broadcast_dims(dims...)
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

function simplify(f, args...)
    length(args) == 2 && any(plain, args) || return nothing
    a = neutral(f, args...)
    a isa Traced || return nothing
    n = node(a)
    dims = broadcast_dims(map(x -> x isa Traced ? node(x).dims : (), args)...)
    T = Base.promote_op(f, map(x -> x isa Traced ? node(x).eltype : typeof(x), args)...)
    n.dims == dims && n.eltype === T && n.scalar == all(x -> !(x isa Traced) || node(x).scalar, args) ? a : nothing
end

"Record `f` applied elementwise to the operands, broadcast together."
function record_map(f, args...)
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

"A conversion to `T`, as a primitive with its own derivative rule."
struct Cast{T} end
(::Cast{T})(x) where {T} = T(x)
Base.show(io::IO, ::Cast{T}) where {T} = print(io, T)

cast(::Type{T}, x::Traced) where {T} = node(x).eltype === T ? x : record_map(Cast{T}(), x)

for T in (Float16, Float32, Float64, Core.BFloat16, Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Bool)
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
function trace_broadcast(f, args, tape)
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

broadcast_handle(bc::Broadcast.Broadcasted, tape) =
    trace_broadcast(bc.f, map(a -> broadcast_handle(a, tape), bc.args), tape)
broadcast_handle(x, tape) = handle(x, tape)

broadcast_dims_of(bc::Broadcast.Broadcasted) = broadcast_dims(map(broadcast_dims_of, bc.args)...)
broadcast_dims_of(x) = operand_dims(x)

function tape_in(bc::Broadcast.Broadcasted)
    for a in bc.args
        t = a isa Broadcast.Broadcasted ? tape_in(a) : a isa Union{Traced, TracedArray} ? a.tape : nothing
        t === nothing || return t
    end
    nothing
end

function Base.copy(bc::Broadcast.Broadcasted{TracedStyle})
    tape = tape_in(bc)
    dims = broadcast_dims_of(bc)
    id = broadcast_handle(bc, tape).id
    # A result smaller than the broadcast, or a scalar, is filled out to it.
    n = tape.nodes[id]
    if n.dims != dims || n.scalar
        id = record!(tape, Map(identity), (id,), dims, n.eltype)
    end
    TracedArray(tape, id)
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
    isempty(axes) && return x
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
logsumexp(x::AbstractArray; dims = :) = fold(LogSumExpFold(), x; dims)

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

# Products.
function contract(a::TracedArray, b::TracedArray, la, lb, lo)
    T = promote_type(eltype(a), eltype(b))
    labels = Dict{Int, Int}()
    for (l, n) in zip((la..., lb...), (size(a)..., size(b)...))
        get!(labels, l, n) == n || throw(DimensionMismatch("contracted axes differ: $(size(a)) and $(size(b))"))
    end
    dims = map(l -> labels[l], lo)
    TracedArray(a.tape, record!(a.tape, Contract(la, lb, lo), (a.id, b.id), dims, T))
end

Base.:*(a::TracedArray{<:Real, 2}, b::TracedArray{<:Real, 2}) = contract(a, b, (1, 2), (2, 3), (1, 3))
Base.:*(a::TracedArray{<:Real, 2}, b::TracedArray{<:Real, 1}) = contract(a, b, (1, 2), (2,), (1,))

"""
    Tempest.batched_mul(a, b)

`a[:, :, k] * b[:, :, k]` for each `k` along the third axis.
"""
batched_mul(a::TracedArray{<:Real, 3}, b::TracedArray{<:Real, 3}) = contract(a, b, (1, 2, 4), (2, 3, 4), (1, 3, 4))
function batched_mul(a::AbstractArray{<:Real, 3}, b::AbstractArray{<:Real, 3})
    stack(k -> a[:, :, k] * b[:, :, k], axes(a, 3); dims = 3)
end

# Layout.
function Base.reshape(x::TracedArray, dims::Tuple{Vararg{Int}})
    prod(dims) == length(x) || throw(DimensionMismatch("cannot reshape $(size(x)) to $dims"))
    dims == size(x) && return x
    TracedArray(x.tape, record!(x.tape, Reshape(), (x.id,), dims, eltype(x)))
end

function Base.permutedims(x::TracedArray, perm)
    perm = Tuple(perm)
    isperm(perm) && length(perm) == ndims(x) || throw(ArgumentError("$perm is not a permutation of $(ndims(x)) axes"))
    perm == ntuple(identity, ndims(x)) && return x
    dims = map(p -> size(x, p), perm)
    TracedArray(x.tape, record!(x.tape, Permute(perm), (x.id,), dims, eltype(x)))
end

Base.permutedims(x::TracedArray{<:Any, 2}) = permutedims(x, (2, 1))
Base.permutedims(x::TracedArray{<:Any, 1}) = reshape(x, (1, length(x)))
Base.transpose(x::TracedArray{<:Real, 2}) = permutedims(x, (2, 1))
Base.transpose(x::TracedArray{<:Real, 1}) = reshape(x, (1, length(x)))
Base.adjoint(x::TracedArray{<:Real}) = transpose(x)

# Indexing: integers drop an axis; ranges and integer vectors keep one.
selection_index(i::Integer) = Int(i)
selection_index(i::Base.Slice) = 1:length(i)
selection_index(i::AbstractRange{<:Integer}) = i
selection_index(i::AbstractVector{<:Integer}) = collect(Int, i)
selection_index(i::AbstractVector{Bool}) = findall(i)
selection_index(i) = throw(ArgumentError("unsupported index $(typeof(i)) into a traced array"))

function select(x::TracedArray, I; scalar)
    if length(I) == 1 && ndims(x) != 1
        return select(vec(x), I; scalar)
    end
    indices = map(selection_index, Base.to_indices(x, I))
    for (k, i) in enumerate(indices)
        all(j -> 1 <= j <= size(x, k), i) || throw(BoundsError(x, I))
    end
    kept = filter(i -> !(i isa Integer), collect(indices))
    dims = Tuple(map(length, kept))
    full = length(indices) == ndims(x) &&
        all(k -> indices[k] isa AbstractRange && indices[k] == 1:size(x, k), 1:ndims(x))
    full && return x
    scalar = scalar && isempty(dims)
    id = record!(x.tape, Select(indices), (x.id,), dims, eltype(x); scalar)
    scalar ? Traced(x.tape, id) : TracedArray(x.tape, id)
end

Base.getindex(x::TracedArray, I...) = select(x, I; scalar = true)
Base.view(x::TracedArray, I...) = select(x, I; scalar = false)

"The one-based linear index of each element of `x`, as `Int32`."
positions(x::TracedArray) = TracedArray(x.tape, record!(x.tape, Positions(), (), size(x), Int32))
positions(x::AbstractArray) = similar(x, Int32) .= reshape(Int32(1):Int32(length(x)), size(x))

"The one-based index of each element of `x` along `axis`, as `Int32`, in the shape of `x`."
axis_positions(x::TracedArray, axis::Integer) =
    TracedArray(x.tape, record!(x.tape, Positions(Int(axis)), (), size(x), Int32))
axis_positions(x::AbstractArray, axis::Integer) =
    similar(x, Int32) .= reshape(Int32(1):Int32(size(x, axis)), ntuple(k -> k == axis ? size(x, k) : 1, ndims(x)))

# Tracing a function.

"""
    trace(f, args...) -> Program

Run `f` on stand-ins for `args` and record what it computes. Arrays and numbers
among `args`, also inside tuples and named tuples, become the program's inputs.
"""
function trace(f, args...)
    tape = Tape()
    arguments = Int[]
    function stand_in(leaf)
        if leaf isa AbstractArray{<:Number}
            id = record!(tape, Argument(), (), size(leaf), eltype(leaf))
            push!(arguments, id)
            return TracedArray(tape, id)
        elseif leaf isa Number
            id = record!(tape, Argument(), (), (), typeof(leaf); scalar = true)
            push!(arguments, id)
            return Traced(tape, id)
        end
        leaf
    end
    traced = rebuild(stand_in, args)
    position = Dict(id => k for (k, id) in enumerate(arguments))
    input_structure = rebuild(t -> t isa Union{Traced, TracedArray} ? position[t.id] : nothing, traced)
    output = f(traced...)
    results = Int[]
    output_structure = rebuild(output) do leaf
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
    Program(tape.nodes, arguments, results, input_structure, output_structure)
end
