struct Cast{T} end
(::Cast{T})(x) where {T} = T(x)
struct LiteralPower{P} end
(::LiteralPower{P})(x) where {P} = x ^ P

function pointwise(f, args...; dims = nothing, scalar = nothing)
    if f isa Type
        f = Cast{f}()
    end
    first_symbolic = findfirst(a -> a isa Value, args)
    if first_symbolic === nothing
        throw(ArgumentError("a pointwise operation needs a traced operand"))
    end
    nodes = args[first_symbolic].nodes
    operands = map(args) do a
        if a isa Value
            if a.nodes !== nodes
                throw(ArgumentError("values belong to different traces"))
            end
            a
        elseif a isa Real
            constant(nodes, a)
        else
            throw(
                ArgumentError(
                    "broadcast operands must be traced values or real scalars; pass arrays as arguments",
                ),
            )
        end
    end
    shape = map(length, Base.Broadcast.broadcast_shape(map(axes, operands)...))
    if dims !== nothing
        expanded = map(
            length,
            Base.Broadcast.broadcast_shape(map(Base.OneTo, shape), map(Base.OneTo, dims)),
        )
        if expanded != dims
            throw(DimensionMismatch("cannot broadcast $shape to $dims"))
        end
        shape = dims
    end
    is_scalar =
        scalar === nothing ? (dims === nothing && all(a -> a isa TracedScalar, operands)) :
        scalar
    T = Base.promote_op(f, map(eltype, operands)...)
    if !(isconcretetype(T) && T <: Real)
        throw(ArgumentError("scalar return type must infer to a concrete real type, got $T"))
    end
    if all(v -> nodes[v.id].op isa Constant, operands)
        result = f(map(v -> nodes[v.id].op.value, operands)...)
        return emit(nodes, Constant(result), (), shape, T; scalar = is_scalar)
    end
    if length(operands) == 1 &&
       (f === identity || f isa Cast) &&
       size(operands[1]) == shape &&
       eltype(operands[1]) == T &&
       nodes[operands[1].id].scalar == is_scalar
        return operands[1]
    end
    if f === (*) && length(operands) == 2
        for (a, b) in ((operands[1], operands[2]), (operands[2], operands[1]))
            n = nodes[a.id]
            if n.op isa Constant &&
               isone(n.op.value) &&
               size(b) == shape &&
               eltype(b) == T &&
               nodes[b.id].scalar == is_scalar
                return b
            end
        end
    end
    emit(nodes, Pointwise(f), Tuple(a.id for a in operands), shape, T; scalar = is_scalar)
end

# Capture the broadcast tree at materialization, rather than hijacking individual
# arities of broadcasted. Base computes axes and dimension compatibility first.
struct TracedStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end
Base.BroadcastStyle(::Type{<:TracedArray{T, N}}) where {T, N} = TracedStyle{N}()
Base.BroadcastStyle(::Type{<:TracedScalar}) = TracedStyle{0}()
Base.broadcastable(x::TracedScalar) = x
Base.BroadcastStyle(::TracedStyle{N}, ::TracedStyle{M}) where {N, M} = TracedStyle{max(N, M)}()

function Base.BroadcastStyle(
    ::TracedStyle{N},
    ::Base.Broadcast.DefaultArrayStyle{M},
) where {N, M}
    TracedStyle{max(N, M)}()
end

function Base.BroadcastStyle(
    ::Base.Broadcast.DefaultArrayStyle{M},
    ::TracedStyle{N},
) where {N, M}
    TracedStyle{max(N, M)}()
end

find_nodes(v::Value) = v.nodes
find_nodes(::Any) = nothing
find_nodes(r::Ref) = find_nodes(r[])

function find_nodes(bc::Base.Broadcast.Broadcasted)
    for a in bc.args
        nodes = find_nodes(a)
        nodes === nothing || return nodes
    end
    nothing
end

function trace_broadcast(nodes, v::Value)
    if v.nodes !== nodes
        throw(ArgumentError("values belong to different traces"))
    end
    v
end

trace_broadcast(nodes, x::Real) = constant(nodes, x)
trace_broadcast(nodes, x::Ref) = trace_broadcast(nodes, x[])

function trace_broadcast(nodes, x)
    throw(
        ArgumentError(
            "cannot trace broadcast over $(typeof(x)); pass arrays as function arguments",
        ),
    )
end

power_value(::Val{P}) where {P} = P
power_value(r::Ref) = power_value(r[])

function trace_broadcast(nodes, bc::Base.Broadcast.Broadcasted)
    if bc.f === Base.literal_pow
        _, x, p = bc.args
        return pointwise(LiteralPower{power_value(p)}(), trace_broadcast(nodes, x))
    end
    pointwise(bc.f, map(a -> trace_broadcast(nodes, a), bc.args)...)
end

function Base.copy(bc::Base.Broadcast.Broadcasted{<:TracedStyle})
    nodes = find_nodes(bc)
    if nodes === nothing
        throw(ArgumentError("broadcast contains no traced values"))
    end
    # trace_broadcast retains the scalar vs array distinction, including zero-dimensional
    # arrays, which cannot be inferred from BroadcastStyle's dimensionality alone.
    trace_broadcast(nodes, bc)
end

Base.Broadcast.materialize!(::TracedArray, ::Base.Broadcast.Broadcasted) = mutation_error()
