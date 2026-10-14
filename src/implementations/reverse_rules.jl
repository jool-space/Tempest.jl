"""
    ReverseRule(rule)

A target-specific symbolic forward/backward recipe. `rule(inputs...; kwargs...)`
returns `(value, pullback)`, where `value` has the public verb's output structure
and `pullback(cotangents)` returns one cotangent (or `nothing`) per input.

The recipe runs during preparation, on symbolic values. It may call private
semantic verbs whose outputs include saved statistics. Captured symbolic values
become ordinary graph dependencies with planned storage and lifetimes; the
pullback closure itself is never called during execution or CUDA capture.
"""
struct ReverseRule{F}
    rule::F
end

"""
    reverse_rule(implementation, request::CallRequest)

Return a `ReverseRule` or `Unsupported(reason)`. Selection uses the actual target
and logical input metadata, after differentiation and before storage planning.
The ordinary `verb_pullback` remains the reference derivative. Forward-only
programs do not select reverse rules.

All semantic calls introduced by a rule must have supported implementations.
If a required call rejects construction, preparation retries another rule, then
the ordinary forward/pullback pair. Unexpected errors propagate.
"""
reverse_rule(::Implementation, ::CallRequest) = Unsupported("no specialized reverse rule")

# Keep the reference derivative as an executable graph, not a Julia closure over
# the original tape. The boundary lets preparation replace both halves together
# while reference execution remains independent of implementation selection.
struct CallPullback{C, P}
    primal::C
    recipe::TensorProgram
    positions::P
end

Base.show(io::IO, op::CallPullback) = print(io, "pullback(", op.primal.f, ")")

function splice_recipe!(nodes, recipe::TensorProgram, arguments)
    live = live_nodes(recipe.nodes, Int[tree_leaves(recipe.output)...])
    mapped = Dict{Int, Value}(zip(recipe.arguments, arguments))
    for (id, node) in enumerate(recipe.nodes)
        (!live[id] || haskey(mapped, id)) && continue
        inputs = map(i -> mapped[i], node.inputs)
        if node.op isa CallResult
            call = node.op.call
            outputs = emit_call(nodes, call.f, inputs, call.kwargs, call.specs)
            for (old, new) in zip(call_ids(call), tree_leaves(outputs))
                mapped[old] = new
            end
        else
            mapped[id] = emit(
                nodes,
                node.op,
                Tuple(x.id for x in inputs),
                node.dims,
                node.dtype;
                scalar = node.scalar,
            )
        end
    end
    tree_map(id -> mapped[id], recipe.output)
end

function reference(op::CallPullback, args...)
    if all(x -> x isa Value, args)
        return splice_recipe!(first(args).nodes, op.recipe, args)
    end
    Tuple(interpret(ForwardProgram(op.recipe), args...))
end

function symbolic_arguments(nodes, values)
    tree_map(values) do x
        emit(nodes, Argument(), (), size(x), eltype(x); scalar = x isa TracedScalar)
    end
end

function check_cotangents(xs, cotangents, nodes)
    cotangents isa Tuple && length(cotangents) == length(xs) ||
        throw(ArgumentError("pullback must return one cotangent per input"))
    for (x, g) in zip(xs, cotangents)
        g === nothing && continue
        g isa Value &&
        g.nodes === nodes &&
        size(g) == size(x) &&
        (g isa TracedScalar) == (x isa TracedScalar) ||
            throw(ArgumentError("pullback returned incompatible cotangent metadata"))
    end
    cotangents
end

function record_pullback(call, xs, ys, gs, nodes)
    recipe_nodes = Node[]
    arguments = (xs..., tree_leaves(ys)..., tree_leaves(gs)...)
    placeholders = symbolic_arguments(recipe_nodes, arguments)
    nx, ny = length(xs), length(tree_leaves(ys))
    recipe_xs = placeholders[1:nx]
    output_ids = Dict(zip(tree_leaves(call.results), 1:ny))
    recipe_ys = tree_map(id -> placeholders[nx + output_ids[id]], call.results)
    recipe_gs = tree_map(id -> placeholders[nx + ny + output_ids[id]], call.results)
    cotangents = verb_pullback(call.f, recipe_xs, recipe_ys, recipe_gs; call.kwargs...)
    check_cotangents(recipe_xs, cotangents, recipe_nodes)
    positions = Tuple(i for i in eachindex(xs) if cotangents[i] !== nothing)
    isempty(positions) && return cotangents

    results = map(i -> cotangents[i], positions)
    recipe = TensorProgram(
        recipe_nodes,
        Int[x.id for x in placeholders],
        map(x -> x.id, results),
        Tuple{DataType, Tuple}[(eltype(x), size(x)) for x in arguments],
        map(x -> x.id, placeholders),
    )
    op = CallPullback(call, recipe, positions)
    outputs = emit_call(nodes, op, arguments, (;), map(TensorSpec, results))
    by_position = Dict(zip(positions, outputs))
    ntuple(i -> get(by_position, i, nothing), nx)
end

struct RuleSelection
    origin::Base.RefValue{Nothing}
    candidate::Int
    name::Symbol
end

function remember_rule_calls!(owners, nodes, first_id, selection)
    for id = first_id:length(nodes)
        op = nodes[id].op
        op isa CallResult && (owners[op.call.origin] = selection)
    end
end

function select_reverse_rule(call, inputs, target, candidates, excluded, notes)
    request = CallRequest(target, call, map(TensorSpec, inputs))
    rejected = get(excluded, call.origin, Set{Int}())
    for (index, candidate) in enumerate(candidates)
        index in rejected && continue
        rule = reverse_rule(candidate, request)
        if rule isa Unsupported
            push!(
                notes,
                "$(call.f): $(implementation_name(candidate)) reverse rule: $(rule.reason)",
            )
            continue
        end
        rule isa ReverseRule || error("reverse_rule must return ReverseRule or Unsupported")
        name = implementation_name(candidate)
        push!(notes, "$(call.f): reverse rule $name")
        return rule, RuleSelection(call.origin, index, name)
    end
    nothing
end

struct SelectedPullback{F, X}
    back::F
    selection::RuleSelection
    inputs::X
end

function specialize_pullback!(nodes, call, inputs, selected, owners)
    op = call.f
    selected === nothing && return splice_recipe!(nodes, op.recipe, inputs)

    ny = length(tree_leaves(op.primal.results))
    cotangents = inputs[(end - ny + 1):end]
    positions = Dict(zip(tree_leaves(op.primal.results), 1:ny))
    gs = tree_map(i -> cotangents[positions[i]], op.primal.results)
    first_id = length(nodes) + 1
    xs = selected.inputs
    gradients = check_cotangents(xs, selected.back(gs), nodes)
    for index in eachindex(xs)
        if index ∉ op.positions && gradients[index] !== nothing
            throw(ArgumentError("reverse rule added a cotangent for a nondifferentiable input"))
        end
    end
    outputs = map(op.positions) do index
        g = gradients[index]
        g === nothing ? zero_value(xs[index]) : g
    end
    # Preserve the reference recipe's cotangent dtype. Accumulation conversions
    # downstream have already been traced against it.
    outputs = map(outputs, call.specs) do g, spec
        eltype(g) === spec.dtype ? g : pointwise(spec.dtype, g)
    end
    remember_rule_calls!(owners, nodes, first_id, selected.selection)
    outputs
end

function specialize_forward!(nodes, call, inputs, selected, pullbacks, owners)
    if selected === nothing
        return emit_call(nodes, call.f, inputs, call.kwargs, call.specs; origin = call.origin)
    end
    rule, selection = selected
    first_id = length(nodes) + 1
    result = rule.rule(inputs...; call.kwargs...)
    result isa Tuple && length(result) == 2 ||
        throw(ArgumentError("reverse rule must return (value, pullback)"))
    outputs, back = result
    check_verb_result(call, outputs)
    all(x -> x isa Value && x.nodes === nodes, tree_leaves(outputs)) ||
        throw(ArgumentError("reverse rule returned values from another trace"))
    pullbacks[call.origin] = SelectedPullback(back, selection, inputs)
    remember_rule_calls!(owners, nodes, first_id, selection)
    outputs
end

# A rule is chosen only when this graph contains the corresponding pullback.
# This also handles with_update: provenance travels with the derivative operation
# rather than with the enclosing program type.
function resolve_reverse_rules(program, target, candidates, excluded, notes)
    live = live_nodes(program.nodes, program.results)
    differentiated = IdDict{Any, Nothing}()
    for (id, node) in enumerate(program.nodes)
        live[id] && node.op isa CallResult || continue
        op = node.op.call.f
        op isa CallPullback && (differentiated[op.primal.origin] = nothing)
    end
    isempty(differentiated) && return program, IdDict{Any, RuleSelection}()

    nodes = Node[]
    mapped = Dict{Int, Value}()
    pullbacks = IdDict{Any, SelectedPullback}()
    owners = IdDict{Any, RuleSelection}()
    for id in program.primal.arguments
        node = program.nodes[id]
        mapped[id] = emit(nodes, Argument(), (), node.dims, node.dtype; scalar = node.scalar)
    end

    for (id, node) in enumerate(program.nodes)
        (!live[id] || haskey(mapped, id)) && continue
        inputs = map(i -> mapped[i], node.inputs)
        if !(node.op isa CallResult)
            mapped[id] = emit(
                nodes,
                node.op,
                Tuple(x.id for x in inputs),
                node.dims,
                node.dtype;
                scalar = node.scalar,
            )
            continue
        end

        call = node.op.call
        if call.f isa CallPullback
            selected = get(pullbacks, call.f.primal.origin, nothing)
            outputs = specialize_pullback!(nodes, call, inputs, selected, owners)
        else
            selected =
                haskey(differentiated, call.origin) ?
                select_reverse_rule(call, inputs, target, candidates, excluded, notes) : nothing
            outputs = specialize_forward!(nodes, call, inputs, selected, pullbacks, owners)
        end
        for (old, new) in zip(call_ids(call), tree_leaves(outputs))
            mapped[old] = new
        end
    end
    remap_program(program, nodes, id -> mapped[id].id), owners
end

function unsupported_rule_call(program, owners, target, candidates)
    live = live_nodes(program.nodes, program.results)
    for (id, node) in enumerate(program.nodes)
        live[id] && node.op isa CallResult || continue
        call = node.op.call
        id == first(call_ids(call)) || continue
        owner = get(owners, call.origin, nothing)
        owner === nothing && continue
        specs = map(node.inputs) do input
            n = program.nodes[input]
            TensorSpec(n.dtype, n.dims; scalar = n.scalar)
        end
        request = CallRequest(target, call, specs)
        reasons = String[]
        supported = false
        for candidate in candidates
            verdict = checked_support(candidate, request)
            if verdict isa Supported
                supported = true
                break
            end
            push!(reasons, "$(implementation_name(candidate)): $(verdict.reason)")
        end
        supported || return (owner, "$(call.f): $(join(reasons, "; "))")
    end
    nothing
end

function reject_reverse_rule!(excluded, owner, reason, notes)
    push!(get!(Set{Int}, excluded, owner.origin), owner.candidate)
    push!(notes, "reverse rule $(owner.name) rejected as a pair: $reason")
end

function prepare_reverse_rules(program, target, candidates, excluded, notes)
    while true
        current, owners = resolve_reverse_rules(program, target, candidates, excluded, notes)
        rejected = unsupported_rule_call(current, owners, target, candidates)
        rejected === nothing && return current, owners
        owner, reason = rejected
        reject_reverse_rule!(excluded, owner, reason, notes)
    end
end
