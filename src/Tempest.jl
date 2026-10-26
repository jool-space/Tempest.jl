"""
Tempest compiles traced Julia array programs into planned, fused kernels.

A program is Julia array code, and what it computes reaches the compiler as
three things: maps, folds (a scan is a fold that keeps the axis it runs
along, a scatter one that combines each element at a position it computes),
and reads at index formulas. Products, reshapes, permutations, gathers and
scatters are spellings of them, and so is every function a program calls.
`Tempest` names only what Base cannot spell:

- the way in: `compile`, `explain`, `gradient` and the targets;
- the derivative of a scalar function, which its code does not carry:
  `scalar_partials`, and a value no gradient flows through: `stopgrad`;
- the laws of a fold's combine: `GenericFold`.

A fold is Base's `mapreduce`, its state a tuple if need be, and a scan Base's
`accumulate`; a fold's derivative comes from its combine's, through scans. A
function given to `mapreduce` is taken to associate and commute, as a
reduction groups and interleaves its elements; one given to `accumulate` is
combined in sequence, as Base combines it, from an `init` that need be no
identity. A `GenericFold`, given to `reduce` or `accumulate` where the
function goes, claims that it associates and commutes, and so a scan by it
may group its elements, as a sum's may.

`logsumexp` is a fold, a spelling. The log-sum-exp written out with `maximum`
differentiates as it does, and plans as it does where the target permits the
rewrites of sums (see `Compiler.rewrite`); it is named still because where a
target keeps the rounding as written, the spelling is two passes, the maximum
and then the sum, and `logsumexp` one.

Any other function, a product by labelled axes or a batched product say, is a
spelling and lives in its own package, as Einops's and NNlib's do; an extension
gives it a method on traced arrays where its body does not trace.

`Tempest.Compiler` is how a program becomes kernels; its public names are
what packages beside a program need: the step beneath `compile`, what an
extension writes traced methods with, and what kernel packages and targets
read and define. `Compiler.IR` is the vocabulary its levels share and
implementations read, and `Compiler.Implement` the interface kernel packages
implement.
"""
module Tempest

using Republic: @republic

export compile, explain
public Compiler

include("Compiler/Compiler.jl")

# What programs are written with: the way in, and what Base cannot spell.
@republic using .Compiler: compile, Compiled, gradient, explain, CPUTarget, CUDATarget,
    logsumexp, scalar_partials, stopgrad
@republic using .Compiler.IR: GenericFold

include("precompile.jl")

end
