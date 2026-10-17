# Operations: Julia functions that keep their names under tracing. Three
# things about an operation `f` are kept apart:
#
# - its meaning: `f`'s own methods, which run on Julia arrays as they always do;
# - its reference, the meaning in primitives: `f`'s method for the declared
#   argument types run on traced arrays, or a method of `reference`. An opaque
#   operation has none; it declares its results (`declared_results`) instead;
# - its implementations, which claim calls of `f` (see `claims_call`), and an
#   optional derivative rule, `verb_pullback`. Without a rule, differentiation
#   goes through the reference.
#
# `@operation` adds the methods of `f` on traced arrays that record a call. At
# the index level a call is its reference expanded, with the call's place kept,
# so it fuses like any other code; an implementation's claim on it competes by
# cost with the reference fused into its neighbours. An opaque call runs only
# where an implementation claims it.

"""
One result of a call of `f` on the node's inputs. A call with several results
has a node for each, consecutive, numbered by `output`; `instance` is the id of
the first, shared by all. `structure` arranges the results as the operation
returns them, with leaf positions.
"""
struct Call{F, K <: NamedTuple} <: Operation
    f::F
    kwargs::K
    output::Int
    instance::Int
    structure::Any
end

op_detail(op::Call) = string(" ", op.f, op.output > 1 || op.structure isa Union{Tuple, NamedTuple} ? " result $(op.output)" : "")

"""
    @operation f(x::AbstractArray, w::AbstractArray; eps = 1.0f-6)

Make `f` an operation: on traced arrays it records a call of `f` rather than
running its body, so the program keeps its name, an implementation can claim
it and `explain` shows it. Each positional argument is an array of the type
given (`AbstractArray` by default); keywords are passed on as written and are
part of the call. Julia arrays among the arguments become constants. Eager
execution is unchanged: Julia arrays still reach `f`'s own methods.

The reference, the call's meaning in primitives, is `f`'s method for the
declared argument types run on traced arrays, unless `Tempest.reference` has a
method for `f`. An operation whose body does not trace either gives a
reference, or declares its results with `Tempest.declared_results` and is
opaque.
"""
macro operation(signature)
    Meta.isexpr(signature, :call) ||
        throw(ArgumentError("@operation takes a signature, as in `@operation f(x::AbstractArray; dims = 1)`"))
    f = signature.args[1]
    keywords = [k for p in signature.args[2:end] if Meta.isexpr(p, :parameters) for k in p.args]
    positional = [a for a in signature.args[2:end] if !Meta.isexpr(a, :parameters)]
    all(a -> a isa Symbol || Meta.isexpr(a, :(::)), positional) ||
        throw(ArgumentError("an operation's positional arguments are `x`, `x::T` or `::T`, without defaults"))
    names = map(a -> a isa Symbol ? a : length(a.args) == 2 ? a.args[1] : gensym(), positional)
    types = map(a -> a isa Symbol ? AbstractArray : last(a.args), positional)
    keyword(k) = k isa Symbol ? k : keyword(first(k.args))
    passed = map(k -> Meta.isexpr(k, :...) ? k : Expr(:kw, keyword(k), keyword(k)), keywords)
    # A method for each nonempty set of traced arguments, the others of their
    # declared types: every mix of traced and Julia arrays reaches one, and
    # each is more specific than `f`'s own methods for the declared types.
    methods = map(1:(2^length(names) - 1)) do traced
        args = map(eachindex(names)) do k
            :($(names[k])::$(isodd(traced >> (k - 1)) ? :($traced_type($(types[k]))) : types[k]))
        end
        call = isempty(keywords) ? :($f($(args...))) : :($f($(args...); $(keywords...)))
        :($call = $record_call($f, $(names...); $(passed...)))
    end
    esc(quote
        $Tempest.operation_signature(::typeof($f)) = $operation_types($(types...))
        $(methods...)
        nothing
    end)
end

"The traced arrays among arrays of type `T`."
traced_type(T) = typeintersect(TracedArray, T)

function operation_types(types...)
    all(T -> T <: AbstractArray, types) ||
        throw(ArgumentError("an operation's positional arguments are arrays, got $(join(types, ", ")); pass other parameters as keywords"))
    Tuple{types...}
end

"The argument types `@operation` declared for `f`, or nothing."
operation_signature(f) = nothing

"""
    reference(f, args...; kwargs...)

The meaning of the operation `f` in primitives: by default, `f`'s own method
for the argument types `@operation` declared.
"""
function reference(f, args...; kwargs...)
    signature = operation_signature(f)
    signature === nothing && throw(ArgumentError("$f has no reference; declare it with @operation, or define Tempest.reference for it"))
    invoke(f, signature, args...; kwargs...)
end

"""
    declared_results(f, inputs; kwargs...) -> results

The results of a call of the opaque operation `f` whose arguments have the
`(dims, eltype)` in `inputs`: a `(dims, eltype)` per result, in the structure
`f` returns them (alone, or in a tuple or a named tuple). Defining it makes `f`
opaque: its body never runs on traced arrays, the interpreter runs its own
method, it differentiates only through a rule, and it runs where an
implementation claims it.
"""
function declared_results end

"Whether `f` is an operation without a reference: one that declares its results."
opaque(f) = hasmethod(declared_results, Tuple{typeof(f), Any})

"""
    verb_pullback(f, args, results, cotangents; kwargs...) -> tuple

The derivative rule of the operation `f`: the cotangents of `args` (each an
array or `nothing`), given the call's results and their cotangents, both in
the structure `f` returns them. Rules are traced code, and may call
operations. Declare one with `has_pullback(::typeof(f)) = true`.
"""
function verb_pullback end

has_pullback(f) = false

function record_call(f, args...; kwargs...)
    tape = tape_of(args...)
    # Julia arrays among the arguments are constants of the program.
    args = map(a -> a isa AbstractArray{<:Number} && !(a isa TracedArray) ? TracedArray(tape, constant!(tape, a)) : a, args)
    all(a -> a isa TracedArray && a.tape === tape, args) ||
        throw(ArgumentError("the arguments of a call of $f must be traced arrays of one program"))
    tape.expands(f) && return expand!(tape, f, args, values(kwargs))
    # Each result's shape and type: declared, or from the reference traced on
    # stand-ins.
    probe = if opaque(f)
        declared(declared_results(f, map(a -> (size(a), eltype(a)), args); kwargs...), f)
    else
        scratch = Tape()
        stand_ins = map(a -> TracedArray(scratch, record!(scratch, Argument(), (), size(a), eltype(a))), args)
        rebuild(node, reference(f, stand_ins...; kwargs...))
    end
    position = Ref(0)
    structure = rebuild(_ -> (position[] += 1), probe)
    instance = length(tape.nodes) + 1
    ids = map(enumerate(leaves(probe))) do (k, s)
        op = Call(f, values(kwargs), k, instance, structure)
        record!(tape, op, map(a -> a.id, args), s.dims, s.eltype; scalar = s.scalar)
    end
    rebuild(k -> handle_of(tape, ids[k]), structure)
end

# Declared results as the nodes they become: a leaf is a `(dims, eltype)` pair.
function declared(x, f)
    x isa Tuple{Tuple{Vararg{Integer}}, Type} && return Node(Argument(), (), map(Int, x[1]), x[2], false)
    x isa Union{Tuple, NamedTuple} && !isempty(x) ||
        throw(ArgumentError("declared_results of $f gives a (dims, eltype) per result, got $x"))
    map(r -> declared(r, f), x)
end

handle_of(tape, id) = tape.nodes[id].scalar ? Traced(tape, id) : TracedArray(tape, id)

results_of(p_nodes, instance) = begin
    k = instance
    while k + 1 <= length(p_nodes) && p_nodes[k + 1].op isa Call && p_nodes[k + 1].op.instance == instance
        k += 1
    end
    instance:k
end

"""
A call expanded where it was recorded: the operation, its keywords, inputs and
result structure, the nodes its reference recorded, and each result's node.
"""
struct Expansion
    f::Any
    kwargs::NamedTuple
    inputs::Vector{Int}
    structure::Any
    nodes::UnitRange{Int}
    results::Vector{Int}
end

function expand!(tape, f, args, kwargs)
    start = length(tape.nodes)
    out = reference(f, args...; kwargs...)
    position = Ref(0)
    structure = rebuild(_ -> (position[] += 1), out)
    push!(tape.expansions, Expansion(f, kwargs, [a.id for a in args], structure, (start + 1):length(tape.nodes), [r.id for r in leaves(out)]))
    out
end

"""
    expand_calls(program, expands) -> (program, expansions)

Replace every call of an operation `f` for which `expands(f)` holds by its
reference, traced in place, together with such calls its reference makes; each
`Expansion` keeps where a call went. Nodes that only unread results of an
expanded call read stay, unread.
"""
function expand_calls(p::Program, expands)
    any(n -> n.op isa Call && expands(n.op.f), p.nodes) || return p, Expansion[]
    tape = Tape()
    tape.expands = expands
    new = zeros(Int, length(p.nodes))
    for (id, n) in enumerate(p.nodes)
        op = n.op
        if op isa Call && expands(op.f)
            op.instance == id || continue
            args = map(i -> handle_of(tape, new[i]), n.inputs)
            out = leaves(expand!(tape, op.f, args, op.kwargs))
            for (k, r) in enumerate(results_of(p.nodes, id))
                new[r] = out[k].id
            end
            continue
        end
        if op isa Call
            instance = op.instance == id ? length(tape.nodes) + 1 : new[op.instance]
            op = Call(op.f, op.kwargs, op.output, instance, op.structure)
        end
        new[id] = record!(tape, op, map(i -> new[i], n.inputs), n.dims, n.eltype; scalar = n.scalar)
    end
    program = Program(tape.nodes, map(i -> new[i], p.arguments), map(i -> new[i], p.results), p.input_structure, p.output_structure)
    program, Expansion[tape.expansions...]
end

# A call's results on Julia arrays: an opaque operation's own method, else its
# reference.
evaluated(op::Call, xs) = leaves(opaque(op.f) ? op.f(xs...; op.kwargs...) : reference(op.f, xs...; op.kwargs...))

# One result of a call.
evaluate(op::Call, n, xs) = evaluated(op, xs)[op.output]

# The interpreter evaluates a call once for all its results.
function evaluate_call!(values, p_nodes, id)
    n = p_nodes[id]
    out = evaluated(n.op, map(i -> values[i], n.inputs))
    for (k, r) in enumerate(results_of(p_nodes, n.op.instance))
        values[r] = out[k]
    end
end

# Differentiating a call with a rule: gather the cotangents of all its results
# (zero where none), and hand them to the rule in the operation's structure.
function call_pullback(n, id, cotangents, tape)
    op = n.op
    has_pullback(op.f) ||
        throw(ArgumentError("$(op.f) is an opaque operation without a derivative rule; define Tempest.verb_pullback and Tempest.has_pullback for it"))
    ids = results_of(tape.nodes, op.instance)
    results = rebuild(k -> handle_of(tape, ids[k]), op.structure)
    gradient = rebuild(op.structure) do k
        r = ids[k]
        c = get(cotangents, r, nothing)
        rn = tape.nodes[r]
        c === nothing && return handle_of(tape, record!(tape, Constant(zero(rn.eltype)), (), rn.dims, rn.eltype; scalar = rn.scalar))
        rn.scalar ? c : TracedArray(tape, c.id)
    end
    args = map(i -> handle_of(tape, i), n.inputs)
    contributions = verb_pullback(op.f, args, results, gradient; op.kwargs...)
    map(c -> c === nothing ? nothing : Traced(tape, c.id), Tuple(contributions))
end
