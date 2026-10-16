# Calls: named operations with a reference. A verb is a Julia function `f`
# with three parts:
#
# - `reference(f, args...; kwargs...)`, its meaning in primitives. It runs on
#   Julia arrays and on traced arrays alike.
# - Optional realizations: implementations claim calls of `f` (see
#   `claims_call`), and realize them as kernels of their own.
# - An optional derivative rule, `verb_pullback`. Without one, differentiation
#   goes through the reference.
#
# A call nobody claims is replaced by its reference before planning, and then
# fuses like any other code. A claimed call is one kernel group, which may
# absorb pointwise work after it as an epilogue.

"""
One result of a call of `f` on the node's inputs. A call with several results
has a node for each, consecutive, numbered by `output`; `instance` is the id of
the first, shared by all. `structure` arranges the results as the reference
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
    reference(f, args...; kwargs...)

The meaning of the verb `f`, written in primitives.
"""
function reference end

"""
    verb_pullback(f, args, results, cotangents; kwargs...) -> tuple

The derivative rule of the verb `f`: the cotangents of `args` (each an array
or `nothing`), given the call's results and their cotangents, both in the
structure the reference returns. Rules are traced code, and may call verbs.
Declare one with `has_pullback(::typeof(f)) = true`.
"""
function verb_pullback end

has_pullback(f) = false

"""
    call(f, args...; kwargs...)

Call the verb `f`: on traced arrays, record a call; on Julia arrays, evaluate
its reference.
"""
function call(f, args...; kwargs...)
    any(a -> a isa TracedArray, args) || return reference(f, args...; kwargs...)
    record_call(f, args...; kwargs...)
end

function record_call(f, args...; kwargs...)
    tape = tape_of(args...)
    all(a -> a isa TracedArray && a.tape === tape, args) ||
        throw(ArgumentError("the arguments of a call of $f must be traced arrays of one program"))
    # The reference, traced on stand-ins, gives the results' shapes and types.
    scratch = Tape()
    stand_ins = map(a -> TracedArray(scratch, record!(scratch, Argument(), (), size(a), eltype(a))), args)
    probe = reference(f, stand_ins...; kwargs...)
    specs = map(node, leaves(probe))
    position = Ref(0)
    structure = rebuild(_ -> (position[] += 1), probe)
    instance = length(tape.nodes) + 1
    ids = map(enumerate(specs)) do (k, s)
        op = Call(f, values(kwargs), k, instance, structure)
        record!(tape, op, map(a -> a.id, args), s.dims, s.eltype; scalar = s.scalar)
    end
    rebuild(k -> handle_of(tape, ids[k]), structure)
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
    expand_calls(program, keep) -> Program

Replace every call whose first result node `n` gives `keep(n) == false` by its
reference, traced in place.
"""
function expand_calls(p::Program, keep)
    any(n -> n.op isa Call && !keep(p.nodes[n.op.instance]), p.nodes) || return p
    tape = Tape()
    new = zeros(Int, length(p.nodes))
    for (id, n) in enumerate(p.nodes)
        op = n.op
        if op isa Call && !keep(p.nodes[op.instance])
            op.instance == id || continue
            args = map(i -> handle_of(tape, new[i]), n.inputs)
            out = leaves(reference(op.f, args...; op.kwargs...))
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
    Program(tape.nodes, map(i -> new[i], p.arguments), map(i -> new[i], p.results), p.input_structure, p.output_structure)
end

# One result of a call, through the reference.
evaluate(op::Call, n, xs) = leaves(reference(op.f, xs...; op.kwargs...))[op.output]

# The interpreter evaluates a call through its reference, once for all results.
function evaluate_call!(values, p_nodes, id)
    n = p_nodes[id]
    op = n.op
    out = leaves(reference(op.f, map(i -> values[i], n.inputs)...; op.kwargs...))
    for (k, r) in enumerate(results_of(p_nodes, op.instance))
        values[r] = out[k]
    end
end

# Differentiating a call with a rule: gather the cotangents of all its results
# (zero where none), and hand them to the rule in the reference's structure.
function call_pullback(n, id, cotangents, tape)
    op = n.op
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
