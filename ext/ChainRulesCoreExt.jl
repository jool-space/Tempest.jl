module ChainRulesCoreExt

using Tempest
using ChainRulesCore: rrule, unthunk, NoTangent, ZeroTangent
import Tempest: external_scalar_partials

function external_scalar_partials(f, inputs::Tuple, output::Tempest.Value)
    rule = trace_scalar_rule(f, map(eltype, inputs))
    rule === nothing && return nothing
    lift_partials(rule, inputs, output)
end

function trace_scalar_rule(f, types)
    nodes = Tempest.Node[]
    arguments = map(types) do T
        Tempest.emit(nodes, Tempest.Argument(), (), (), T; scalar = true)
    end
    rule = rrule(f, arguments...)
    rule === nothing && return nothing
    output, back = rule
    if output isa Real && !(output isa Tempest.TracedScalar)
        output = Tempest.constant(nodes, output)
    end
    output isa Tempest.TracedScalar && output.nodes === nodes ||
        throw(ArgumentError("scalar rrule must return a symbolic scalar output"))
    cotangents = back(one(output))
    cotangents isa Tuple || throw(ArgumentError("scalar rrule pullback must return a tuple"))
    cotangents = map(unthunk, cotangents)
    length(cotangents) == length(arguments) + 1 ||
        throw(ArgumentError("wrong number of rule cotangents"))
    first(cotangents) isa NoTangent ||
        throw(ArgumentError("captured function tangents need a separate contract"))
    partials = map(Base.tail(cotangents), types) do partial, T
        partial isa NoTangent && throw(
            ArgumentError(
                "scalar rrule returned NoTangent for a data input; use a semantic verb to declare nondifferentiable inputs",
            ),
        )
        partial isa ZeroTangent && return Tempest.constant(nodes, zero(T))
        if partial isa Tempest.TracedScalar
            partial.nodes === nodes || throw(ArgumentError("foreign derivative trace"))
            return partial
        end
        partial isa Real || throw(ArgumentError("expected a real scalar partial"))
        Tempest.constant(nodes, partial)
    end
    (; nodes, arguments, output, partials)
end

# Broadcast a scalar derivative graph over array inputs. Only scalar arithmetic
# is allowed here; semantic-call lowering preserves shapes and cannot substitute
# for this lifting pass. The existing forward output is an explicit binding.
function lift_partials(rule, inputs, output)
    nodes = output.nodes
    mapped = Dict{Int, Tempest.Value}(
        arg.id => input for (arg, input) in zip(rule.arguments, inputs)
    )
    if !haskey(mapped, rule.output.id)
        mapped[rule.output.id] = output
    end
    live = Tempest.live_nodes(rule.nodes, [x.id for x in rule.partials])
    for (id, node) in enumerate(rule.nodes)
        (!live[id] || haskey(mapped, id)) && continue
        mapped[id] = if node.op isa Tempest.Constant
            Tempest.constant(nodes, node.op.value)
        elseif node.op isa Tempest.Pointwise
            Tempest.pointwise(node.op.f, map(i -> mapped[i], node.inputs)...)
        else
            throw(ArgumentError("rule contains a non-scalar operation: $(typeof(node.op))"))
        end
    end
    map(x -> mapped[x.id], rule.partials)
end


end
