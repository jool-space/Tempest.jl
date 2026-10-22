# API

```@docs
Tempest
```

## Compiling a function

```@docs
compile
Tempest.Compiled
Tempest.gradient
explain
Tempest.CPUTarget
Tempest.CUDATarget
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
Tempest.Compiler.contract
```

## Derivative rules

```@docs
Tempest.scalar_partials
```

## Tempest.Compiler

```@docs
Tempest.Compiler
Tempest.Compiler.trace
Tempest.Compiler.prepare
Tempest.Compiler.bind!
Tempest.Compiler.execute!
Tempest.Compiler.capture!
Tempest.Compiler.replay!
Tempest.Compiler.outputs
Tempest.Compiler.Plan
Tempest.Compiler.group_views
```

### Adding a target

```@docs
Tempest.Compiler.Target
Tempest.Compiler.target_for
Tempest.Compiler.packages
Tempest.Compiler.numerics
Tempest.Compiler.permits_rewrites
Tempest.Compiler.product_class
Tempest.Compiler.allocate
Tempest.Compiler.upload
Tempest.Compiler.bindable
Tempest.Compiler.task_state
Tempest.Compiler.floor_launch
Tempest.Compiler.ScalarKernel
Tempest.Compiler.accumulate!
Tempest.Compiler.capture
Tempest.Compiler.replay
Tempest.Compiler.release
Tempest.Compiler.device_name
Tempest.Compiler.measurable
Tempest.Compiler.time_launch
Tempest.Compiler.calibrate
```

### Building kernels

```@docs
Tempest.Compiler.content_key
Tempest.Compiler.cached_kernel
Tempest.Compiler.store_kernel
Tempest.Compiler.store_failure
Tempest.Compiler.uncached_kernel
Tempest.Compiler.generated
Tempest.Compiler.concurrently
```

## Tempest.IR

```@docs
Tempest.IR
```

### Index formulas

```@docs
Tempest.IR.Domain
Tempest.IR.Var
Tempest.IR.Affine
Tempest.IR.Div
Tempest.IR.Mod
Tempest.IR.Indirect
Tempest.IR.is_affine
Tempest.IR.indirect_values
Tempest.IR.substitute
```

### Layout

```@docs
Tempest.IR.storage_length
Tempest.IR.address
Tempest.IR.group_stride
```

### Folds

```@docs
Tempest.IR.Relative
Tempest.IR.Exponential
Tempest.IR.Square
Tempest.IR.Statistic
Tempest.IR.Sum
Tempest.IR.Composed
Tempest.IR.LOGSUMEXP
Tempest.IR.InSequence
Tempest.IR.Overwrite
Tempest.IR.fold_reduce
Tempest.IR.product_factors
```

### Terms

```@docs
Tempest.IR.Load
Tempest.IR.Ix
Tempest.IR.Stat
Tempest.IR.Recomputed
Tempest.IR.Cast
Tempest.IR.mentions
```

### Masks

```@docs
Tempest.IR.comparison_mask
Tempest.IR.held_formula
Tempest.IR.masked_result
```

## Tempest.Implement

```@docs
Tempest.Implement
Tempest.Implement.Implementation
Tempest.Implement.register!
Tempest.Implement.GroupView
Tempest.Implement.ValueView
Tempest.Implement.members
Tempest.Implement.product_axes
Tempest.Implement.claims
Tempest.Implement.variants
Tempest.Implement.estimate
Tempest.Implement.Work
Tempest.Implement.build_claim
Tempest.Implement.candidates
Tempest.Implement.record_name
Tempest.Implement.recomputes_folds
Tempest.Implement.in_place
```
