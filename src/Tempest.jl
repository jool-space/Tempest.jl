"""
Tempest compiles traced Julia array programs into planned, fused kernels. See
DESIGN.md for the design this code implements.
"""
module Tempest

using LinearAlgebra

export trace, differentiate, interpret

# The program level.
include("program/program.jl")
include("program/folds.jl")
include("program/tracing.jl")
include("program/interpret.jl")
include("program/differentiate.jl")

# The index level.
include("index/expressions.jl")
include("index/graph.jl")
include("index/lower.jl")
include("index/interpret.jl")

end
