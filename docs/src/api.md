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

Beyond Base's broadcasts, folds, products, indexing and ranges (a range broadcast
against a traced array is the index itself, a formula rather than an array),
programs can call:

```@docs
Tempest.fold
Tempest.GenericFold
Tempest.logsumexp
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
Tempest.Compiler.differentiate
Tempest.Compiler.interpret
Tempest.Compiler.prepare
Tempest.Compiler.bind!
Tempest.Compiler.execute!
Tempest.Compiler.capture!
Tempest.Compiler.replay!
Tempest.Compiler.outputs
Tempest.Compiler.lower
Tempest.Compiler.plan
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
