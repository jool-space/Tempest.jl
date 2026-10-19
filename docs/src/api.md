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

Beyond Base's broadcasts, folds, products and indexing, programs can call:

```@docs
Tempest.contract
Tempest.fold
Tempest.GenericFold
Tempest.logsumexp
Tempest.positions
Tempest.axis_positions
Tempest.batched_mul
```

## Writing operations

```@docs
@operation
Tempest.reference
Tempest.declared_results
Tempest.verb_pullback
Tempest.has_pullback
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
Tempest.Compiler.group_view
```

## Tempest.IR

```@docs
Tempest.IR
```

## Tempest.Implementations

```@docs
Tempest.Implementations
Tempest.Implementations.Implementation
Tempest.Implementations.register!
Tempest.Implementations.Spec
Tempest.Implementations.bytes
Tempest.Implementations.Kernel
Tempest.Implementations.Form
Tempest.Implementations.pointwise
Tempest.Implementations.GroupView
Tempest.Implementations.ValueView
Tempest.Implementations.CallResult
Tempest.Implementations.members
Tempest.Implementations.product_axes
Tempest.Implementations.claims
Tempest.Implementations.estimate
Tempest.Implementations.Work
Tempest.Implementations.build_claim
Tempest.Implementations.candidates
Tempest.Implementations.record_name
Tempest.Implementations.recomputes_folds
Tempest.Implementations.in_place
Tempest.Implementations.claims_call
Tempest.Implementations.epilogue_results
Tempest.Implementations.build_call
```

## Tempest.Operations

```@docs
Tempest.Operations
Tempest.Operations.attention
Tempest.Operations.dropout
```
