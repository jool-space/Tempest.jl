"""Logical output metadata for a semantic verb. Scalars use scalar=true and dims=()."""
struct TensorSpec
    dtype::DataType
    dims::Tuple{Vararg{Int}}
    scalar::Bool

    function TensorSpec(T::Type, dims::Tuple; scalar = false)
        if !(isconcretetype(T) && runtime_eltype(T))
            throw(ArgumentError("verb values must have floating or integer element types"))
        end
        if !all(d -> d isa Integer && d >= 0, dims)
            throw(ArgumentError("invalid output dimensions"))
        end
        if scalar && !isempty(dims)
            throw(ArgumentError("scalar outputs must have empty dimensions"))
        end
        new(T, Tuple(Int.(dims)), scalar)
    end
end

TensorSpec(x::TracedArray) = TensorSpec(eltype(x), size(x))
TensorSpec(x::TracedScalar) = TensorSpec(eltype(x), (); scalar = true)
TensorSpec(x::AbstractArray) = TensorSpec(eltype(x), size(x))
TensorSpec(x::AbstractFloat) = TensorSpec(typeof(x), (); scalar = true)

"""Declare a verb's output TensorSpec or nested tuple/named tuple of specifications."""
function output_spec end

"""Independent mathematical decomposition of a semantic verb into array operations."""
function reference end

"""
    verb_pullback(f, inputs::Tuple, outputs, cotangents; kwargs...) -> Tuple

Return one cotangent (or nothing) per positional input. Outputs and cotangents
have the verb's declared result structure. Unused output cotangents are symbolic
zeros. The recipe can call other semantic verbs, including a custom backward.
"""
function verb_pullback end

call_ids(c::SemanticCall) = Int[tree_leaves(c.results)...]
static_keyword(x) = isbits(x) || x isa Symbol || x isa Type
static_keyword(x::Union{Tuple, NamedTuple}) = all(static_keyword, x)

function emit_call(nodes, f, args, kwargs, specs; origin = Ref(nothing))
    leaves = tree_leaves(specs)
    if isempty(leaves) || !all(s -> s isa TensorSpec, leaves)
        throw(
            ArgumentError(
                "output_spec must return TensorSpec leaves in a nonempty result tree",
            ),
        )
    end
    all(s -> s.dtype <: AbstractFloat, leaves) ||
        throw(ArgumentError("verb outputs must have floating element types"))
    # Reserve result IDs before constructing the call. Every result node refers
    # to the same complete, immutable call; no partially initialized object escapes.
    next_id = length(nodes)
    results = tree_map(specs) do _
        next_id += 1
    end
    call = SemanticCall(f, kwargs, specs, results, origin)
    inputs = Tuple(a.id for a in args)
    tree_map(specs) do spec
        emit(nodes, CallResult(call), inputs, spec.dims, spec.dtype; scalar = spec.scalar)
    end
end

"""
    record_call(f, args...; kwargs...)

Record a semantic call from a verb's TracedArray/TracedScalar method. Uses
output_spec without executing the reference or launcher. Positional tensor/scalar
inputs are data; keywords are immutable trace-time configuration. Mutation of
inputs and output aliases are not supported by this functional call contract.
"""
function record_call(f, args...; kwargs...)
    firstvalue = findfirst(a -> a isa Value, args)
    if firstvalue === nothing
        throw(ArgumentError("record_call needs a symbolic input"))
    end
    nodes = args[firstvalue].nodes
    xs = map(args) do a
        if a isa Value && a.nodes === nodes
            return a
        end
        if a isa AbstractFloat
            return constant(nodes, a)
        end
        throw(
            ArgumentError(
                "verb inputs must be symbolic values or floating constants from one trace",
            ),
        )
    end
    kws = NamedTuple(kwargs)
    if !all(static_keyword, kws)
        throw(
            ArgumentError(
                "verb keywords must be immutable configuration; pass runtime data positionally",
            ),
        )
    end
    emit_call(nodes, f, xs, kws, output_spec(f, xs...; kws...))
end

# Replacing leaves with `nothing` gives tuples and named tuples a canonical
# structure type, independent of the values or metadata stored at their leaves.
function same_result_structure(a, b)
    typeof(tree_map(_ -> nothing, a)) === typeof(tree_map(_ -> nothing, b))
end

matches_spec(x, spec::TensorSpec) = false

function matches_spec(x::Union{Value, AbstractArray, AbstractFloat}, spec::TensorSpec)
    eltype(x) === spec.dtype &&
        size(x) == spec.dims &&
        (x isa Union{TracedScalar, AbstractFloat}) == spec.scalar
end

function check_verb_result(call, result)
    if !same_result_structure(result, call.specs)
        throw(
            ArgumentError("reference result structure differs from output_spec for $(call.f)"),
        )
    end
    for (output, spec) in zip(tree_leaves(result), tree_leaves(call.specs))
        if !matches_spec(output, spec)
            throw(
                ArgumentError(
                    "reference result metadata differs from output_spec for $(call.f)",
                ),
            )
        end
    end
    result
end
