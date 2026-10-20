"""
Tempest compiles traced Julia array programs into planned, fused kernels.

`Tempest` holds what models are written with: `compile`, and the functions
programs call beyond Base. `Tempest.Compiler` is how a program becomes
kernels, with the levels beneath `compile`. `Tempest.IR` is the vocabulary the
levels share and implementations read, and `Tempest.Implementations` the
interface kernel packages implement.
"""
module Tempest

using Republic: @republic

export compile, explain
public Compiler, IR, Implementations

include("IR/IR.jl")
include("Implementations/Implementations.jl")
include("Compiler/Compiler.jl")

# What models are written with: the front door, and the functions programs
# call beyond Base. What kernel packages are written with: `register!`.
@republic using .Compiler: compile, Compiled, gradient, explain, CPUTarget, CUDATarget,
    contract, fold, logsumexp, batched_mul, scalar_partials
@republic using .IR: GenericFold
@republic using .Implementations: register!

include("precompile.jl")

end
