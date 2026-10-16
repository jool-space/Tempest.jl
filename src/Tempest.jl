"""
Tempest compiles traced Julia array programs into planned, fused kernels. See
DESIGN.md for the design this code implements.
"""
module Tempest

using LinearAlgebra

export trace, differentiate, interpret, explain, prepare, bind!, execute!, outputs

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

# Plans: what is stored, and the kernels.
include("plan/terms.jl")
include("plan/planner.jl")
include("plan/cpu.jl")
include("plan/explain.jl")

# Code generation and prepared programs.
include("codegen/scalar.jl")
include("runtime/runtime.jl")

end
