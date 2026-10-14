"""
    scalar_partials(f) -> tuple of scalar derivative callables

Provisional extension point for pure scalar functions. Each partial receives all
primal arguments. For example `scalar_partials(::typeof(cube)) = (x -> 3x*x,)`.
Callables used on the GPU must be device-compilable. This is explicit scalar AD,
not automatic differentiation of arbitrary Julia or quantization algorithms.
"""
function scalar_partials end

# Factored form also avoids a mixed-shape FMA verifier issue in cuTile 1.0.
scalar_partials(::typeof(tanh)) = (x -> (one(x) - tanh(x)) * (one(x) + tanh(x)),)
scalar_partials(::typeof(abs2)) = (x -> 2x,)
scalar_partials(::typeof(sin)) = (cos,)
scalar_partials(::typeof(cos)) = (x -> -sin(x),)
scalar_partials(::typeof(exp)) = (exp,)
scalar_partials(::typeof(log)) = (inv,)
scalar_partials(::typeof(sqrt)) = (x -> inv(2sqrt(x)),)
scalar_partials(::typeof(inv)) = (x -> -inv(x * x),)
scalar_partials(::typeof(^)) = ((x, p) -> p * x ^ (p - one(p)), (x, p) -> x ^ p * log(x))
scalar_partials(::LiteralPower{P}) where {P} = P == 0 ? (zero,) : (x -> P * x ^ (P - 1),)
scalar_partials(::typeof(identity)) = (one,)
scalar_partials(::Cast{T}) where {T <: AbstractFloat} = (one,)
scalar_partials(::typeof(+)) = ((x, y) -> one(x), (x, y) -> one(y))
scalar_partials(::typeof(*)) = ((x, y) -> y, (x, y) -> x)
scalar_partials(::typeof(/)) = ((x, y) -> inv(y), (x, y) -> -x / (y * y))
# Small mathematical recipes avoid fictitious dependencies: d(x+y)/dx does
# not read either array merely to obtain a correctly typed scalar one.
unit(v) = constant(v.nodes, one(eltype(v)))
local_partials(::typeof(+), xs, y) = map(unit, xs)

function local_partials(::typeof(-), xs, y)
    length(xs) == 1 ? (-unit(xs[1]),) : (unit(xs[1]), -unit(xs[2]))
end

function local_partials(::typeof(*), xs, y)
    ntuple(length(xs)) do index
        partial = unit(xs[index])
        for other in eachindex(xs)
            if other != index
                partial = partial .* xs[other]
            end
        end
        partial
    end
end

function local_partials(::typeof(/), xs, y)
    numerator, denominator = xs
    numerator_partial = unit(denominator) ./ denominator
    denominator_partial = -numerator ./ (denominator .* denominator)
    (numerator_partial, denominator_partial)
end

function local_partials(::typeof(^), xs, y)
    base, exponent = xs
    base_partial = exponent .* (base .^ (exponent .- unit(exponent)))
    exponent_partial = y .* log.(base)
    (base_partial, exponent_partial)
end

function local_partials(::LiteralPower{P}, xs, y) where {P}
    x = only(xs)
    if P == 0
        return (zero_value(x),)
    end
    (P .* pointwise(LiteralPower{P - 1}(), x),)
end

local_partials(::typeof(abs2), xs, y) = (2 * xs[1],)
local_partials(::typeof(identity), xs, y) = (unit(xs[1]),)
local_partials(::Cast{T}, xs, y) where {T <: AbstractFloat} = (unit(xs[1]),)
# Built-in recipes emit ordinary graph operations. Backends need to understand
# their arithmetic, not a separate vocabulary of AD-only scalar helpers.
local_partials(::typeof(sin), xs, y) = (cos.(xs[1]),)
local_partials(::typeof(cos), xs, y) = (-sin.(xs[1]),)
local_partials(::typeof(exp), xs, y) = (y,)
local_partials(::typeof(log), xs, y) = (inv.(xs[1]),)
local_partials(::typeof(sqrt), xs, y) = (inv.(2 .* y),)
local_partials(::typeof(inv), xs, y) = (-inv.(xs[1] .* xs[1]),)

function local_partials(::typeof(tanh), xs, y)
    # Reuse the forward result. The factored form avoids a mixed-shape FMA
    # verifier issue in cuTile 1.0, so keep it as two factors rather than 1 - y².
    one_y = unit(y)
    ((one_y .- y) .* (one_y .+ y),)
end

# Optional rule libraries provide this hook without becoming core dependencies.
# A missing rule returns nothing. Errors inside an available rule must propagate.
function external_scalar_partials end

function local_partials(f, xs, y)
    if applicable(scalar_partials, f)
        fs = scalar_partials(f)
        length(fs) == length(xs) || throw(ArgumentError("wrong number of scalar partials"))
        return map(df -> pointwise(df, xs...), fs)
    end
    if applicable(external_scalar_partials, f, xs, y)
        partials = external_scalar_partials(f, xs, y)
        partials === nothing || return partials
    end
    throw(
        ArgumentError(
            "no scalar derivative rule for $f; define Tempest.scalar_partials or load ChainRules",
        ),
    )
end

zero_value(v::TracedArray) = zero(v)
zero_value(v::TracedScalar) = constant(v.nodes, zero(eltype(v)))

# A semantic call is differentiated once, after all of its result cotangents
# have been accumulated. Missing cotangents represent unused result leaves.
function call_pullback(call, inputs, nodes, gradients)
    lookup(id) = value(nodes, id)
    xs = map(lookup, inputs)
    ys = tree_map(lookup, call.results)
    gs = tree_map(call.results) do id
        get(() -> zero_value(lookup(id)), gradients, id)
    end

    if !applicable(verb_pullback, call.f, xs, ys, gs)
        throw(
            ArgumentError(
                "no verb_pullback for $(call.f); define a recipe or write the reference decomposition directly",
            ),
        )
    end
    record_pullback(call, xs, ys, gs, nodes)
end

# Recipes return one cotangent per operand. Accumulation and dtype policy belong
# to the reverse traversal; the mathematical rules do not mutate that traversal.
pullback(::MatMul, xs, y, g) = (g * transpose(xs[2]), transpose(xs[1]) * g)

function pullback(::TransposeOp, xs, y, g)
    x = only(xs)
    (present(transpose(g), size(x); scalar = x isa TracedScalar),)
end

function pullback(::ReshapeOp, xs, y, g)
    x = only(xs)
    (present(g, size(x); scalar = x isa TracedScalar),)
end

pullback(op::PermuteOp, xs, y, g) = (permutedims(g, Tuple(invperm(collect(op.permutation)))),)

function pullback(op::Union{ViewOp, SliceOp}, xs, y, g)
    x = only(xs)
    result = emit(g.nodes, ScatterOp(op.selection), (g.id,), size(x), eltype(x); scalar = false)
    (result,)
end

function pullback(op::Pointwise, xs, y, g)
    partials = local_partials(op.f, xs, y)
    length(partials) == length(xs) ||
        throw(ArgumentError("wrong number of partials for $(op.f)"))
    map(xs, partials) do x, partial
        x.nodes[x.id].op isa Constant && return nothing
        sum_to(pointwise(*, g, partial), x)
    end
end

function selected_arguments(p::TensorProgram, wrt)
    wrt isa Tuple || throw(ArgumentError("wrt must be a tuple of positional argument indices"))
    all(i -> i isa Integer && !(i isa Bool) && 1 <= i <= length(p.input_structure), wrt) ||
        throw(ArgumentError("wrt contains an invalid argument index"))
    length(unique(wrt)) == length(wrt) || throw(ArgumentError("wrt contains duplicate indices"))
    Set(Int[id for index in wrt for id in tree_leaves(p.input_structure[index])])
end

"""
    differentiate(program; wrt=Tuple(eachindex(program.input_structure))) -> TrainingProgram

Construct an explicit reverse graph. Contributions are ordinary additions;
broadcast cotangents reduce to each operand's shape, rank, and scalar/array kind.
Transpose and reshape aliases get matching reverse presentations. Unreachable
work is skipped by interpretation/planning. The original trace is not mutated.
`wrt` selects unique positional argument indices, in gradient-result order. Each
gradient preserves its argument's tuple/NamedTuple structure. The default selects
all arguments; `wrt=()` constructs no derivatives.
"""
function differentiate(p::TensorProgram; wrt = Tuple(eachindex(p.input_structure)))
    if !(p.output isa Int && p.nodes[p.output].scalar)
        throw(
            ArgumentError(
                "differentiate requires a scalar-number loss; use prepare_forward for general outputs",
            ),
        )
    end
    selected = selected_arguments(p, wrt)
    active = falses(length(p.nodes))
    for (id, node) in enumerate(p.nodes)
        active[id] = id in selected || any(input -> active[input], node.inputs)
    end
    nodes = copy(p.nodes)
    v(i) = value(nodes, i)
    grads = Dict{Int, Value}(p.output => constant(nodes, one(nodes[p.output].dtype)))
    function contribute!(i, g)
        if !active[i] || nodes[i].op isa Constant
            return
        end
        # Prototype cotangent policy: storage follows the logical primal dtype.
        # Scalar derivatives such as inv(::Int) may otherwise widen a Float32
        # gradient and accidentally change the following GEMM's representation.
        T = nodes[i].dtype
        if eltype(g) != T
            g = pointwise(T, g)
        end
        grads[i] = haskey(grads, i) ? pointwise(+, grads[i], g) : g
    end
    for id in reverse(eachindex(p.nodes))
        n = nodes[id]
        active[id] || continue
        if n.op isa CallResult
            call = n.op.call
            ids = call_ids(call)
            if !(id == first(ids) && any(i -> haskey(grads, i), ids))
                continue
            end
            cotangents = call_pullback(call, n.inputs, nodes, grads)
            for (input, cotangent) in zip(n.inputs, cotangents)
                if cotangent !== nothing
                    contribute!(input, cotangent)
                end
            end
            continue
        end
        if !haskey(grads, id)
            continue
        end
        g, n = grads[id], nodes[id]
        if n.op isa Union{Argument, Constant}
            continue
        end
        xs = map(v, n.inputs)
        cotangents = pullback(n.op, xs, v(id), g)
        length(cotangents) == length(xs) ||
            throw(ArgumentError("wrong number of operation cotangents"))
        for (input, cotangent) in zip(n.inputs, cotangents)
            cotangent === nothing || contribute!(input, cotangent)
        end
    end
    function gradient_id(id)
        g = get(grads, id, nothing)
        if g === nothing
            g = pointwise(
                identity,
                constant(nodes, zero(nodes[id].dtype));
                dims = nodes[id].dims,
                scalar = nodes[id].scalar,
            )
        end
        g.id
    end
    gradients = map(index -> tree_map(gradient_id, p.input_structure[index]), Tuple(wrt))
    structure = (p.output, gradients...)
    TrainingProgram(p, nodes, Int[tree_leaves(structure)...], structure)
end

function Base.show(io::IO, ::MIME"text/plain", p::TrainingProgram)
    println(io, "TrainingProgram (explicit forward + backward)")
    show_nodes(io, p.nodes, p.results)
end

"""
    with_update(f, gradients::TrainingProgram) -> ForwardProgram

Append a functional update to an explicit backward graph. The callback receives
`(loss, gradients, original_arguments...)` as symbolic values. `gradients` is a
tuple in `wrt` order, preserving each selected argument's structure. Return the
next state and any observations as an ordinary tuple or NamedTuple.
"""
function with_update(f, program::TrainingProgram)
    nodes = copy(program.nodes)
    symbolic(id) = value(nodes, id)
    args = tree_map(symbolic, program.primal.input_structure)
    differentiated = result_structure(program, symbolic)
    result = f(first(differentiated), Base.tail(differentiated), args...)
    leaves = tree_leaves(result)
    if isempty(leaves) || !all(v -> v isa Value && v.nodes === nodes, leaves)
        throw(ArgumentError("an update must return symbolic values from its own graph"))
    end
    ForwardProgram(
        program.primal,
        nodes,
        Int[v.id for v in leaves],
        tree_map(v -> v.id, result),
    )
end
