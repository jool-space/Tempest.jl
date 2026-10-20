"""
Tempest compiles traced Julia array programs into planned, fused kernels. See
DESIGN.md for the design this code implements.

`Tempest` holds what models are written with: `compile`, and the functions
programs call beyond Base. `Tempest.Compiler` is how a program becomes
kernels, with the levels beneath `compile`. `Tempest.IR` is the vocabulary the
levels share and implementations read, and `Tempest.Implementations` the
interface kernel packages implement. `Tempest.Operations` holds the array
functions Tempest defines, written as a user would write them.
"""
module Tempest

using Republic: @republic

export compile, explain
public Compiler, IR, Implementations, Operations

include("IR/IR.jl")
include("Implementations/Implementations.jl")
include("Compiler/Compiler.jl")

# What models are written with: the front door, and the functions programs
# call beyond Base. What kernel packages are written with: `register!`.
@republic using .Compiler: compile, Compiled, gradient, explain, CPUTarget, CUDATarget,
    contract, fold, logsumexp, positions, axis_positions, batched_mul, scalar_partials
@republic using .IR: GenericFold
@republic using .Implementations: register!

include("Operations/Operations.jl")

include("precompile.jl")

end
