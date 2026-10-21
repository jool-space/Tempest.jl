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
Tempest.contract
Tempest.fold
Tempest.GenericFold
Tempest.logsumexp
Tempest.batched_mul
```

A package that adds functions programs call writes them on traced arrays, in
these primitives, as NNlibExt does:

```@docs
Tempest.Compiler.TracedArray
Tempest.Compiler.permuted
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
Tempest.Compiler.group_view
```

### Adding a target

```@docs
Tempest.Compiler.Target
Tempest.Compiler.target_for
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
Tempest.Implement.estimate
Tempest.Implement.Work
Tempest.Implement.build_claim
Tempest.Implement.candidates
Tempest.Implement.record_name
Tempest.Implement.recomputes_folds
Tempest.Implement.in_place
```
