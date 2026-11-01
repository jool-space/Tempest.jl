# Packages state the derivatives of their scalar functions with ChainRulesCore's
# `@scalar_rule`, as partials given the result: Flux's `xlogy`, NNlib's
# activations, SpecialFunctions' `erf`. Tempest takes them as the functions'
# own where it has none of its own, for a function whose code it traces, whose
# derivative at a branch may not be the function's, and for one it records
# whole, whose code it cannot differentiate.
module ChainRulesCoreExt

using ChainRulesCore: AbstractZero, derivatives_given_output
import Tempest.Compiler: Traced, stated_derivatives

function stated_derivatives(f, y::Traced, xs::Union{Traced, Number}...)
    applicable(derivatives_given_output, y, f, xs...) || return nothing
    # One row of partials for each of the function's results; a scalar
    # function has one.
    rows = derivatives_given_output(y, f, xs...)
    length(rows) == 1 || return nothing
    map(d -> d isa AbstractZero ? nothing : d, only(rows))
end

end
