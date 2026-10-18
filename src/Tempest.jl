"""
Tempest compiles traced Julia array programs into planned, fused kernels. See
DESIGN.md for the design this code implements.

`Tempest` holds what models and operations are written with: `compile`, the
functions programs call beyond Base, and `@operation` with its rules.
`Tempest.Compiler` is how a program becomes kernels, with the levels beneath
`compile` and the interface for kernel packages. `Tempest.Operations` holds the
operations Tempest defines, written as an operation package would be.
"""
module Tempest

export compile, explain, @operation

module Compiler

using LinearAlgebra
using BFloat16s: BFloat16

export trace, differentiate, interpret, prepare, bind!, execute!, capture!, replay!, outputs

# The program level.
include("program/program.jl")
include("program/folds.jl")
include("program/tracing.jl")
include("program/updates.jl")
include("program/interpret.jl")
include("program/calls.jl")
include("program/differentiate.jl")

# The index level.
include("index/expressions.jl")
include("index/graph.jl")
include("index/lower.jl")
include("index/interpret.jl")

# Plans: what is stored, and the kernels.
include("plan/terms.jl")
include("plan/planner.jl")
include("plan/rewrites.jl")
include("plan/products.jl")
include("plan/cpu.jl")
include("plan/explain.jl")
include("plan/storage.jl")

# Code generation and prepared programs.
include("codegen/scalar.jl")
include("runtime/runtime.jl")
include("runtime/decide.jl")
include("runtime/cache.jl")
include("runtime/compile.jl")

end

# What models are written with: the front door, and the functions programs
# call beyond Base. What operations are written with: `@operation` and its rules.
using .Compiler: compile, Compiled, gradient, explain, CPUTarget, CUDATarget,
    contract, fold, GenericFold, logsumexp, positions, axis_positions, batched_mul,
    @operation, reference, declared_results, has_pullback, verb_pullback, scalar_partials

module Operations

using ..Tempest: @operation, contract, logsumexp, positions, axis_positions
import ..Tempest: has_pullback, verb_pullback, scalar_partials

export attention, dropout

include("verbs/attention.jl")
include("verbs/dropout.jl")

end

include("precompile.jl")

end
