"""
The operations Tempest defines, `attention` and `dropout`. They are written
against `Tempest` alone, as an operation package would write its own.
"""
module Operations

using ..Tempest: @operation, contract, logsumexp, positions, axis_positions
import ..Tempest: has_pullback, verb_pullback, scalar_partials

export attention
include("attention.jl")

export dropout
include("dropout.jl")

end
