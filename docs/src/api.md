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
Tempest.stopgrad
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
Tempest.Compiler.floor_split
Tempest.Compiler.Split
Tempest.Compiler.merge_lanes
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
Tempest.Compiler.cached_failure
Tempest.Compiler.dependencies
Tempest.Compiler.uncached_kernel
Tempest.Compiler.generated
Tempest.Compiler.concurrently
```

## Tempest.Compiler.IR

```@docs
Tempest.Compiler.IR
```

### Index formulas

```@docs
Tempest.Compiler.IR.Domain
Tempest.Compiler.IR.Var
Tempest.Compiler.IR.Affine
Tempest.Compiler.IR.Div
Tempest.Compiler.IR.Mod
Tempest.Compiler.IR.Indirect
Tempest.Compiler.IR.is_affine
Tempest.Compiler.IR.indirect_values
Tempest.Compiler.IR.substitute
```

### Layout

```@docs
Tempest.Compiler.IR.storage_length
Tempest.Compiler.IR.address
Tempest.Compiler.IR.group_stride
```

### Folds

```@docs
Tempest.Compiler.IR.Relative
Tempest.Compiler.IR.Exponential
Tempest.Compiler.IR.Square
Tempest.Compiler.IR.Statistic
Tempest.Compiler.IR.Sum
Tempest.Compiler.IR.Composed
Tempest.Compiler.IR.LOGSUMEXP
Tempest.Compiler.IR.InSequence
Tempest.Compiler.IR.Overwrite
Tempest.Compiler.IR.fold_reduce
Tempest.Compiler.IR.product_factors
```

### Terms

```@docs
Tempest.Compiler.IR.Load
Tempest.Compiler.IR.Ix
Tempest.Compiler.IR.Stat
Tempest.Compiler.IR.Recomputed
Tempest.Compiler.IR.Cast
Tempest.Compiler.IR.mentions
```

### Masks

```@docs
Tempest.Compiler.IR.comparison_mask
Tempest.Compiler.IR.held_formula
Tempest.Compiler.IR.masked_result
```

## Tempest.Compiler.Implement

```@docs
Tempest.Compiler.Implement
Tempest.Compiler.Implement.Implementation
Tempest.Compiler.Implement.register!
Tempest.Compiler.Implement.GroupView
Tempest.Compiler.Implement.ValueView
Tempest.Compiler.Implement.members
Tempest.Compiler.Implement.product_axes
Tempest.Compiler.Implement.claims
Tempest.Compiler.Implement.variants
Tempest.Compiler.Implement.estimate
Tempest.Compiler.Implement.Work
Tempest.Compiler.Implement.build_claim
Tempest.Compiler.Implement.candidates
Tempest.Compiler.Implement.record_name
Tempest.Compiler.Implement.recomputes_folds
Tempest.Compiler.Implement.in_place
Tempest.Compiler.Implement.lowers
Tempest.Compiler.Implement.compiling
```
