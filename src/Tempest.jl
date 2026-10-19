"""
Tempest compiles traced Julia array programs into planned, fused kernels. See
DESIGN.md for the design this code implements.

`Tempest` holds what models and operations are written with: `compile`, the
functions programs call beyond Base, and `@operation` with its rules.
`Tempest.Compiler` is how a program becomes kernels, with the levels beneath
`compile`. `Tempest.IR` is the vocabulary the levels share and implementations
read, and `Tempest.Implementations` the interface kernel packages implement.
`Tempest.Operations` holds the operations Tempest defines, written as an
operation package would be.
"""
module Tempest

using Republic: @republic

export compile, explain, @operation
public Compiler, IR, Implementations, Operations

include("IR/IR.jl")
include("Implementations/Implementations.jl")
include("Compiler/Compiler.jl")

# What models are written with: the front door, and the functions programs
# call beyond Base. What operations are written with: `@operation` and its rules.
# What kernel packages are written with: `Spec`, `Kernel`, `pointwise` and `register!`.
@republic using .Compiler: compile, Compiled, gradient, explain, CPUTarget, CUDATarget,
    contract, fold, logsumexp, positions, axis_positions, batched_mul,
    @operation, reference, declared_results, has_pullback, verb_pullback, scalar_partials
@republic using .IR: GenericFold
@republic using .Implementations: Spec, Kernel, Form, pointwise, bytes, register!

include("Operations/Operations.jl")

include("precompile.jl")

end
