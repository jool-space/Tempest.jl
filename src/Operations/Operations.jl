"""
The array functions Tempest defines, `attention` and `dropout`. They are plain
code on Tempest's primitives, written against `Tempest` alone.
"""
module Operations

using ..Tempest: contract, logsumexp, positions, axis_positions
import ..Tempest: scalar_partials

export attention
public attention_forward
include("attention.jl")

export dropout
include("dropout.jl")

end
