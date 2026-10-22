"""
Tempest compiles traced Julia array programs into planned, fused kernels.

A program is Julia array code, and what it computes reaches the compiler as
three things: maps, folds, and reads at index formulas. Products, reshapes,
permutations, gathers and scatters to positions it can compute are spellings of
them, and so is every function a program calls. A scatter by index arrays is not
one yet: it runs whole, on stored operands. `Tempest` names only what Base
cannot spell:

- the way in: `compile`, `explain`, `gradient` and the targets;
- folds beyond an operator and its identity, which `reduce` cannot state:
  `fold` and `GenericFold`;
- the derivative of a scalar function, which its code does not carry:
  `scalar_partials`.

`logsumexp` is a fold, a spelling, named here until the log-sum-exp written
out differentiates as it does: its value does not depend on the maximum that
keeps it finite, and the derivative of a spelling with `maximum` goes through
the maximum still.

Any other function, a product by labelled axes or a batched product say, is a
spelling and lives in its own package, as Einops's and NNlib's do; an extension
gives it a method on traced arrays where its body does not trace.

`Tempest.Compiler` is how a program becomes kernels, with the levels beneath
`compile`. `Tempest.IR` is the vocabulary the levels share and implementations
read, and `Tempest.Implement` the interface kernel packages implement.
"""
module Tempest

using Republic: @republic

export compile, explain
public Compiler, IR, Implement

include("IR/IR.jl")
include("Implement/Implement.jl")
include("Compiler/Compiler.jl")

# What programs are written with: the way in, and what Base cannot spell.
@republic using .Compiler: compile, Compiled, gradient, explain, CPUTarget, CUDATarget,
    fold, logsumexp, scalar_partials
@republic using .IR: GenericFold

include("precompile.jl")

end
