# Array-level operations: use Base's broadcast machinery for elementwise work,
# but never permit an unsupported generic method to scalar-iterate the trace.
function Base.:*(a::TracedArray, b::TracedArray)
    nodes = same_trace(a, b)
    if !(ndims(a) == ndims(b) == 2)
        throw(ArgumentError("matmul currently requires matrices"))
    end
    if size(a, 2) != size(b, 1)
        throw(DimensionMismatch("matmul inner dimensions differ"))
    end
    emit(
        nodes,
        MatMul(),
        (a.id, b.id),
        (size(a, 1), size(b, 2)),
        promote_type(eltype(a), eltype(b));
        scalar = false,
    )
end

for f in (:+, :-)
    @eval function Base.$f(a::TracedArray, b::TracedArray)
        Base.promote_shape(size(a), size(b)) # array addition is NOT broadcast addition
        pointwise($f, a, b)
    end
end

Base.:-(a::TracedArray) = pointwise(-, a)
Base.:+(a::TracedArray) = a
Base.:*(a::Real, b::TracedArray) = pointwise(*, a, b)
Base.:*(a::TracedArray, b::Real) = pointwise(*, a, b)
Base.:/(a::TracedArray, b::Real) = pointwise(/, a, b)

for f in (:+, :-, :*, :/, :^)
    @eval begin
        Base.$f(a::TracedScalar, b::TracedScalar) = pointwise($f, a, b)
        Base.$f(a::TracedScalar, b::Real) = pointwise($f, a, b)
        Base.$f(a::Real, b::TracedScalar) = pointwise($f, a, b)
    end
end

for f in (
    :-,
    :tanh,
    :sin,
    :cos,
    :exp,
    :log,
    :sqrt,
    :inv,
    :abs2,
    :expm1,
    :log1p,
    :sinh,
    :cosh,
    :tan,
    :asin,
    :acos,
    :atan,
    :abs,
    :sign,
)
    @eval Base.$f(a::TracedScalar) = pointwise($f, a)
end

Base.sincos(x::TracedScalar) = (sin(x), cos(x))

for f in (:atan, :hypot, :min, :max)
    @eval Base.$f(x::TracedScalar, y::TracedScalar) = pointwise($f, x, y)
end

Base.:+(a::TracedScalar) = a

function Base.literal_pow(::typeof(^), a::TracedScalar, ::Val{P}) where {P}
    pointwise(LiteralPower{P}(), a)
end

Base.:^(a::TracedScalar, p::Integer) = pointwise(LiteralPower{p}(), a)
Base.:^(a::TracedScalar, p::Rational) = pointwise(^, a, p)
Base.:^(::Irrational{:ℯ}, a::TracedScalar) = exp(a)
# Resolve intersections with scalar-real arithmetic above.
Base.:*(a::TracedScalar, b::TracedArray) = pointwise(*, a, b)
Base.:*(a::TracedArray, b::TracedScalar) = pointwise(*, a, b)
Base.:/(a::TracedArray, b::TracedScalar) = pointwise(/, a, b)

function branch_error()
    throw(
        ArgumentError(
            "data-dependent scalar comparison during tracing is unsupported; only metadata may control host branches",
        ),
    )
end

for f in (:(==), :(!=), :(<), :(<=), :(>), :(>=), :isless)
    @eval begin
        Base.$f(::TracedScalar, ::TracedScalar) = branch_error()
        Base.$f(::TracedScalar, ::Real) = branch_error()
        Base.$f(::Real, ::TracedScalar) = branch_error()
    end
end

Base.isless(::TracedScalar, ::AbstractFloat) = branch_error()
Base.isless(::AbstractFloat, ::TracedScalar) = branch_error()
Base.:(==)(::TracedScalar, ::AbstractIrrational) = branch_error()
Base.:(==)(::AbstractIrrational, ::TracedScalar) = branch_error()

function Base.transpose(v::TracedArray)
    if !(ndims(v) in (1, 2))
        throw(ArgumentError("transpose requires a vector or matrix"))
    end
    n = v.nodes[v.id]
    if n.op isa TransposeOp
        return value(v.nodes, only(n.inputs))
    end
    shape = ndims(v) == 1 ? (1, length(v)) : reverse(size(v))
    emit(v.nodes, TransposeOp(), (v.id,), shape, eltype(v); scalar = false)
end

Base.adjoint(v::TracedArray) = transpose(v) # only real tensors
Base.transpose(v::TracedScalar) = v
Base.adjoint(v::TracedScalar) = v

function present(v::Value, dims; scalar = false)
    if prod(dims) != length(v)
        throw(DimensionMismatch("reshape changes the number of elements"))
    end
    if size(v) == dims && v.nodes[v.id].scalar == scalar
        return v
    end
    emit(v.nodes, ReshapeOp(), (v.id,), dims, eltype(v); scalar)
end

function reshape_dims(v, dims)
    if !(count(x -> x isa Colon, dims) <= 1)
        throw(ArgumentError("only one inferred reshape dimension is allowed"))
    end
    if !all(x -> x isa Colon || (x isa Integer && x >= 0), dims)
        throw(ArgumentError("reshape dimensions must be nonnegative"))
    end
    known = prod((Int(x) for x in dims if !(x isa Colon)); init = 1)
    if known == 0
        return if any(x -> x isa Colon, dims)
            throw(ArgumentError("cannot infer a dimension from a zero product"))
        else
            present(v, Tuple(Int(x) for x in dims))
        end
    end
    if length(v) % known != 0
        throw(DimensionMismatch("reshape dimensions do not divide the length"))
    end
    shape = map(x -> x isa Colon ? length(v) ÷ known : Int(x), dims)
    present(v, shape)
end

Base.reshape(v::TracedArray, dims::Dims) = reshape_dims(v, dims)
Base.reshape(v::TracedArray, dims::Tuple{Vararg{Union{Int, Colon}}}) = reshape_dims(v, dims)
Base.reshape(v::TracedArray, dims::Union{Int, Colon}...) = reshape_dims(v, dims)
Base.reshape(v::TracedArray{T, 1}, ::Colon) where {T} = v
Base.reshape(v::TracedArray{T, 1}, ::Tuple{Colon}) where {T} = v

function Base.dropdims(v::TracedArray; dims)
    ds = dims isa Integer ? (dims,) : Tuple(dims)
    if !all(d -> d isa Integer && 1 <= d <= ndims(v) && size(v, d) == 1, ds)
        throw(ArgumentError("dropdims requires singleton dimensions in range"))
    end
    if length(unique(ds)) != length(ds)
        throw(ArgumentError("duplicate dropdims dimensions"))
    end
    present(v, Tuple(size(v, d) for d = 1:ndims(v) if !(d in ds)))
end

function Base.sum(v::TracedArray; dims = :)
    T = Base.promote_op(Base.add_sum, eltype(v), eltype(v))
    if dims isa Colon
        return emit(v.nodes, SumReduction(:), (v.id,), (), T; scalar = true)
    end
    ds = dims isa Integer ? (dims,) : Tuple(dims)
    if !all(d -> d isa Integer && d > 0, ds)
        throw(ArgumentError("reduction dimensions must be positive integers"))
    end
    axes = Tuple(sort!(unique!(Int[d for d in ds if d <= ndims(v)])))
    if isempty(axes)
        return eltype(v) == T ? v : pointwise(T, v)
    end
    shape = ntuple(d -> d in axes ? 1 : size(v, d), ndims(v))
    emit(v.nodes, SumReduction(axes), (v.id,), shape, T; scalar = false)
end

Base.sum(f, v::TracedArray; dims = :) = sum(pointwise(f, v); dims)
Base.map(f, v::TracedArray) = pointwise(f, v)
Base.mapreduce(f, ::typeof(+), v::TracedArray; dims = :) = sum(f, v; dims)
Base.sum(v::TracedScalar) = v
Base.sum(f, v::TracedScalar) = f(v)

# Sum all broadcasted axes, then remove the trailing singleton axes introduced
# by broadcasting. Preserve scalar vs 0D-array identity explicitly.
function sum_to(g::Value, target::Value)
    if target isa TracedScalar
        return g isa TracedScalar ? g : sum(g)
    end
    axes = Tuple(d for d = 1:ndims(g) if size(target, d) == 1 && size(g, d) != 1)
    r = isempty(axes) ? g : sum(g; dims = axes)
    present(r, size(target); scalar = false)
end

# Structural indices are fixed during tracing. Snapshot gathers so mutating a
# captured index vector later cannot silently alter a prepared graph.
freeze_index(i::Base.Slice) = i.indices
freeze_index(i::Integer) = Int(i)
freeze_index(i::AbstractRange{<:Integer}) = Int(first(i)):Int(step(i)):Int(last(i))
freeze_index(i::AbstractVector{<:Integer}) = IndexList(Int[x for x in i])

function traced_selection(a::TracedArray, I, op)
    if !(all(
        i ->
            i isa Colon ||
            i isa CartesianIndex ||
            (i isa Integer && !(i isa Bool) && !(i isa TracedScalar)) ||
            (i isa AbstractVector{<:Integer} && !(eltype(i) <: Bool)),
        I,
    ))
        throw(
            ArgumentError(
                "indices must be fixed integers, colons, integer ranges, or integer vectors",
            ),
        )
    end
    inds = Base.to_indices(a, I)
    Base.checkbounds(a, inds...)
    frozen = map(freeze_index, inds)
    parentdims = length(inds) == 1 ? (length(a),) : ntuple(d -> size(a, d), length(inds))
    dims = Tuple(index_length(i) for i in frozen if !(i isa Integer))
    unique_indices =
        all(i -> !(i isa IndexList) || length(unique(i.indices)) == length(i.indices), frozen)
    selection = Selection(parentdims, frozen, dims, unique_indices)
    emit(a.nodes, op(selection), (a.id,), dims, eltype(a); scalar = false)
end

Base.view(a::TracedArray, I...) = traced_selection(a, I, ViewOp)

function Base.permutedims(a::TracedArray, perm)
    ps = Tuple(Int(p) for p in perm)
    if sort(collect(ps)) != collect(1:ndims(a))
        throw(ArgumentError("invalid axis permutation"))
    end
    if ps == ntuple(identity, ndims(a))
        return a
    end
    emit(
        a.nodes,
        PermuteOp(ps),
        (a.id,),
        Tuple(size(a, d) for d in ps),
        eltype(a);
        scalar = false,
    )
end

Base.PermutedDimsArray(a::TracedArray, perm) = permutedims(a, perm)
Base.permutedims(a::TracedArray{T, 2}) where {T} = permutedims(a, (2, 1))
Base.permutedims(a::TracedArray{T, 1}) where {T} = present(a, (1, length(a)))
