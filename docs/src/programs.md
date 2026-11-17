# Programs

A program is Julia array code, and what it computes reaches the compiler as
three things: maps, folds (a scan is a fold that keeps the axis it runs
along, a scatter one that combines each element at a position it computes),
and reads at index formulas. Products, reshapes, permutations, gathers and
scatters are spellings of them, and so is every function a program calls.

## What Tempest names

`Tempest` names only what Base cannot spell:

- the way in: [`compile`](@ref), [`explain`](@ref),
  [`gradient`](@ref Tempest.gradient) and
  [`withgradient`](@ref Tempest.withgradient), the targets, and the
  implementations of kernel packages Tempest has extensions for;
- the derivative of a scalar function, which its code does not carry:
  [`scalar_partials`](@ref Tempest.scalar_partials), and a value no gradient
  flows through: [`stopgrad`](@ref Tempest.stopgrad);
- the laws of a fold's combine: [`GenericFold`](@ref Tempest.GenericFold).

## Folds and scans

A fold is Base's `mapreduce`, its state a tuple if need be, and a scan Base's
`accumulate`; a fold's derivative comes from its combine's, through scans.

A function given to `mapreduce` is taken to associate and commute, as a
reduction groups and interleaves its elements; one given to `accumulate` is
combined in sequence, as Base combines it, from an `init` that need be no
identity.

A `GenericFold`, given to `reduce` or `accumulate` where the function goes,
claims that it associates and commutes, and so a scan by it may group its
elements, as a sum's may.

## `logsumexp`

[`logsumexp`](@ref Tempest.logsumexp) is a fold, a spelling. The log-sum-exp
written out with `maximum` differentiates as it does, and plans as it does
where the target permits the rewrites of sums (see `Compiler.rewrite`); it
is named still because where a target keeps the rounding as written, the
spelling is two passes, the maximum and then the sum, and `logsumexp` one.

## Other functions

Any other function, a product by labelled axes or a batched product say, is
a spelling and lives in its own package, as Einops's and NNlib's do; an
extension gives it a method on traced arrays where its body does not trace.
