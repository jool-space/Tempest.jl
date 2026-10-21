"""
Tempest compiles traced Julia array programs into planned, fused kernels.

A program is Julia array code, and what it computes reaches the compiler as
three things: maps, folds, and reads at index formulas. Products, reshapes,
permutations, gathers and scatters are spellings of them, and so is every
function a program calls. `Tempest` names only what Base cannot spell:

- the way in: `compile`, `explain`, `gradient` and the targets;
- folds beyond an operator and its identity: `fold`, `GenericFold`, and
  `logsumexp`, whose value does not depend on the maximum that keeps it
  finite, which a spelling with `maximum` cannot say;
- the derivative of a scalar function, which its code does not carry:
  `scalar_partials`;
- a product by its axes' labels, which Base spells only through reshapes:
  `contract`.

Any other function, a batched product say, is a spelling and lives in its
own package, as NNlib's do; an extension gives it a method on traced arrays
where its body does not trace.

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

# What models are written with: the way in, and what Base cannot spell. What
# kernel packages are written with: `register!`.
@republic using .Compiler: compile, Compiled, gradient, explain, CPUTarget, CUDATarget,
    contract, fold, logsumexp, scalar_partials
@republic using .IR: GenericFold
@republic using .Implement: register!

include("precompile.jl")

end
