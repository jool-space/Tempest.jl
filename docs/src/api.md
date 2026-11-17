# Tempest

```@docs
Tempest
```

## Compiling a function

```@docs
compile
Tempest.Compiled
wait(::Tempest.Compiled)
Tempest.Searching
wait(::Tempest.Searching)
Tempest.gradient
Tempest.withgradient
explain
Tempest.CPUTarget
Tempest.CUDATarget
Tempest.cuBLASProducts
Tempest.cuDNNAttention
Tempest.cuTileGenerator
```

## Writing programs

Programs are written with Base: broadcasts, folds (`mapreduce` with a tuple
state among them), scans (`accumulate`, `cumsum`), products, indexing and
ranges (a range broadcast against a traced array is the index itself, a
formula rather than an array). Beyond Base, programs can call `logsumexp`,
and claim with a `GenericFold` the laws of a combine, which Base cannot spell:

```@docs
Tempest.logsumexp
Tempest.GenericFold
```

Any other function a program calls, a batched product say, is written in
these and lives in its own package. Where its body does not trace, an
extension gives it a method on traced arrays, as NNlibExt and EinopsExt do,
with these:

```@docs
Tempest.Compiler.TracedArray
Tempest.Compiler.Traced
Tempest.Compiler.record_map
Tempest.Compiler.contract
Tempest.Compiler.differentiating
```

## Derivative rules

```@docs
Tempest.scalar_partials
Tempest.Compiler.stated_derivatives
Tempest.stopgrad
```
