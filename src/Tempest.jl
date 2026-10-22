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

`logsumexp` is a fold, a spelling. The log-sum-exp written out with `maximum`
differentiates as it does, and plans as it does where the target permits the
rewrites of sums (see `Compiler.rewrite`); it is named still because where a
target keeps the rounding as written, the spelling is two passes, the maximum
and then the sum, and `logsumexp` one.

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
