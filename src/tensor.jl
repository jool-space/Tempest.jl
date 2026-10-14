"""Symbolic array frontend. Shape queries are concrete; elements are not host data."""
struct TracedArray{T, N} <: AbstractArray{T, N}
    nodes::Vector{Node}
    id::Int
end

"""Symbolic real scalar; unlike `TracedArray{T,0}`, this participates as a number."""
struct TracedScalar{T <: Real} <: Real
    nodes::Vector{Node}
    id::Int
end

const Value = Union{TracedArray, TracedScalar}

function value(nodes, id)
    n = nodes[id]
    if n.scalar
        TracedScalar{n.dtype}(nodes, id)
    else
        TracedArray{n.dtype, length(n.dims)}(nodes, id)
    end
end

Base.size(v::TracedArray) = v.nodes[v.id].dims
Base.axes(v::TracedArray) = map(Base.OneTo, size(v))
Base.IndexStyle(::Type{<:TracedArray}) = IndexCartesian()
Base.eltype(::Type{TracedScalar{T}}) where {T} = T
Base.eltype(::TracedScalar{T}) where {T} = T
Base.zero(::Type{TracedScalar{T}}) where {T} = zero(T)
Base.one(::Type{TracedScalar{T}}) where {T} = one(T)
Base.zero(::TracedScalar{T}) where {T} = zero(T)
Base.one(::TracedScalar{T}) where {T} = one(T)

function emit(nodes, op::Operation, inputs, dims, T; scalar = isempty(dims))
    if scalar && !isempty(dims)
        error("a scalar cannot have array dimensions")
    end
    push!(nodes, Node(op, inputs, dims, T, scalar))
    value(nodes, length(nodes))
end

constant(nodes, x::Real) = emit(nodes, Constant(x), (), (), typeof(x))

function same_trace(args::Value...)
    nodes = first(args).nodes
    if !all(a -> a.nodes === nodes, args)
        throw(ArgumentError("values belong to different traces"))
    end
    nodes
end

# Indexing is intentionally not a tracing fallback. In particular, Base's generic
# algorithms must not unroll an array into one scalar node per element.
function unsupported_index()
    throw(
        ArgumentError(
            "symbolic scalar indexing is unsupported; use array operations (or [] on a zero-dimensional array)",
        ),
    )
end

function Base.getindex(a::TracedArray, I...)
    if all(i -> i isa Integer || i isa CartesianIndex, I)
        return unsupported_index()
    end
    traced_selection(a, I, SliceOp)
end

Base.getindex(a::TracedArray, i::CartesianIndex) = getindex(a, Tuple(i)...)

function Base.getindex(a::TracedArray{T, 0}) where {T}
    emit(a.nodes, ReshapeOp(), (a.id,), (), T; scalar = true)
end

function Base.similar(::TracedArray, ::Type{T}, ::Dims) where {T}
    throw(
        ArgumentError(
            "similar on a symbolic array is not storage allocation; use a functional array operation",
        ),
    )
end

function mutation_error()
    throw(
        ArgumentError(
            "mutation of symbolic values is unsupported; write a functional tensor expression",
        ),
    )
end

Base.setindex!(::TracedArray, args...) = mutation_error()
Base.fill!(::TracedArray, x) = mutation_error()
Base.copyto!(::TracedArray, ::TracedArray) = mutation_error()
Base.copyto!(::TracedArray, ::Array) = mutation_error()
Base.copy(v::TracedArray) = v # immutable logical value; mutation is refused

function Base.zero(a::TracedArray{T}) where {T}
    emit(a.nodes, Constant(zero(T)), (), size(a), T; scalar = false)
end

function Base.show(io::IO, v::Value)
    print(
        io,
        v isa TracedScalar ? "TracedScalar" : "TracedArray",
        "{",
        eltype(v),
        "}(%",
        v.id,
        v isa TracedScalar ? ")" : ", $(size(v)))",
    )
end

Base.show(io::IO, ::MIME"text/plain", v::Value) = show(io, v)

"""
    trace(f, examples...) -> TensorProgram

Run `f` on `TracedArray{T,N} <: AbstractArray{T,N}` and `TracedScalar{T} <: Real`
stand-ins. No tensor data operations execute while tracing. Shapes/axes are
concrete, scalar indexing and mutation are refused. Results may be symbolic
values or nested tuples/named tuples of them. Examples may be nonempty,
one-based real floating arrays or floating scalars, nested in tuples/NamedTuples.
Integer arrays are accepted as runtime indices (for example `table[:, indices]`);
they carry no gradient. The positional argument structure is fixed at trace
time. Captured arrays are unsupported.
"""
runtime_eltype(T) = T <: AbstractFloat || (T <: Integer && !(T <: Bool) && isbitstype(T))

Base.@nospecializeinfer function trace(f, examples...)
    @nospecialize examples
    flat_examples = tree_leaves(examples)
    for a in flat_examples
        if a isa AbstractArray
            if !(runtime_eltype(eltype(a)) && all(>(0), size(a)))
                throw(
                    ArgumentError(
                        "examples must be nonempty real floating-point or integer-index arrays",
                    ),
                )
            end
            Base.require_one_based_indexing(a)
        elseif !(a isa AbstractFloat)
            throw(ArgumentError("examples must be floating-point arrays or scalars"))
        end
    end
    nodes = Node[]
    args = tree_map(examples) do a
        emit(nodes, Argument(), (), size(a), eltype(a); scalar = a isa Number)
    end
    result = f(args...)
    leaves = tree_leaves(result)
    if isempty(leaves) || !all(v -> v isa Value && v.nodes === nodes, leaves)
        throw(
            ArgumentError(
                "return symbolic values in a nonempty tuple/named tuple or a single symbolic value",
            ),
        )
    end
    TensorProgram(
        nodes,
        Int[a.id for a in tree_leaves(args)],
        tree_map(v -> v.id, result),
        Tuple{DataType, Tuple}[(eltype(a), size(a)) for a in flat_examples],
        tree_map(v -> v.id, args),
    )
end

Base.@nospecializeinfer function validate(p::TensorProgram, @nospecialize(args))
    if length(args) != length(p.input_structure)
        throw(ArgumentError("argument count differs from trace"))
    end
    if !same_structure(args, p.input_structure)
        throw(ArgumentError("argument tuple/named-tuple structure differs from trace"))
    end
    flat_args = tree_leaves(args)
    for (id, a) in zip(p.arguments, flat_args)
        validate_leaf(p.nodes[id], a)
    end
    flat_args
end

function validate_leaf(node::Node, argument)
    goodkind = node.scalar ? argument isa AbstractFloat : argument isa AbstractArray
    if !(goodkind && eltype(argument) == node.dtype && size(argument) == node.dims)
        throw(
            ArgumentError(
                "argument signature (including scalar/array kind) differs from trace",
            ),
        )
    end
    argument isa AbstractArray && Base.require_one_based_indexing(argument)
    nothing
end
