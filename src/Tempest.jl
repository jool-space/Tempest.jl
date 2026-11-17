"""
Tempest compiles traced Julia array programs into planned, fused kernels.

A program is Julia array code, and what it computes reaches the compiler as
three things: maps, folds (a scan is a fold that keeps the axis it runs
along, a scatter one that combines each element at a position it computes),
and reads at index formulas. Products, reshapes, permutations, gathers and
scatters are spellings of them, and so is every function a program calls.
`Tempest` names only what Base cannot spell:

- the way in: `compile`, `explain`, `gradient` and `withgradient`, the
  targets, and the implementations of kernel packages Tempest has extensions
  for;
- the derivative of a scalar function, which its code does not carry:
  `scalar_partials`, and a value no gradient flows through: `stopgrad`;
- the laws of a fold's combine: `GenericFold`;
- the log-sum-exp as one fold, where a target that keeps the rounding as
  written would take its spelling in two passes: `logsumexp`.

`Tempest.Compiler` is how a program becomes kernels; its public names are
what packages beside a program need: the step beneath `compile`, what an
extension writes traced methods with, and what kernel packages and targets
read and define. `Compiler.IR` is the vocabulary its levels share and
implementations read, and `Compiler.Implement` the interface kernel packages
implement.

The manual's page [Programs](@ref) says how folds and scans are read, and
where any other function lives.
"""
module Tempest

using Republic: @republic

export compile, explain
public Compiler

include("Compiler/Compiler.jl")

# What programs are written with: the way in, and what Base cannot spell.
@republic using .Compiler: compile, Compiled, Searching, gradient, withgradient, explain, CPUTarget, CUDATarget,
    cuBLASProducts, cuDNNAttention, cuTileGenerator, logsumexp, scalar_partials, stopgrad
@republic using .Compiler.IR: GenericFold

include("precompile.jl")

end
