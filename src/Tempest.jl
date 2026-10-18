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

include("Compiler/Compiler.jl")

# What models are written with: the front door, and the functions programs
# call beyond Base. What operations are written with: `@operation` and its rules.
using .Compiler: compile, Compiled, gradient, explain, CPUTarget, CUDATarget,
    contract, fold, GenericFold, logsumexp, positions, axis_positions, batched_mul,
    @operation, reference, declared_results, has_pullback, verb_pullback, scalar_partials

include("Operations/Operations.jl")

include("precompile.jl")

end
