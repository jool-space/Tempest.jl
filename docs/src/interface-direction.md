# Interface direction

What Tempest's user-facing interface would contain if the compiler behind it
were perfect, and what that says about the interface we have. Written
15 October 2026, after the implementation registry and proposed regions landed.
The [compiler direction](compiler-direction.md) page covers the inside of the
compiler; this page covers its boundary with the caller.

## The test

A perfect compiler can work out anything the program implies. What remains in
its interface is what only the caller knows. Anything that steers the compiler
or works around a backend would disappear.

The test is useful because Tempest's compiler is not perfect, and every knob we
add for that reason is easy to mistake for interface. Applying it sorts the
current surface into a small semantic core and a larger layer that exists
because the compiler is imperfect. Both layers are legitimate. The second
should be reachable without being in the way, and it should never be the only
place a semantic choice is recorded.

## What stays

| What only the caller knows | Where Tempest records it today |
|:--|:--|
| What to compute: a pure function of arrays, element types included | `trace`, broadcasting, views, verbs |
| Which transformation of it | `differentiate(wrt)`, `value_and_gradient`, `with_update` |
| The numerical contract | spread across `CUDATarget`, `CuDNNKernels`, fold contracts and comments (see below) |
| Ownership and lifetime | `donate`, `bind!`, `bind_argument!`, borrowed outputs |
| What is fixed and what varies per call | implied by type: float arguments are traced; constants are captured by the traced function |
| Resources and placement | `CUDATarget(device, blas_workspace_bytes)`, the current stream |
| Facts the compiler cannot prove | custom reverse rules, `Fold` declarations, seeds |

**What to compute.** Element types are part of the meaning: a Float16 value
rounds where it rounds, and eliminating a buffer must not eliminate a cast.

**Which transformation.** The compiler cannot guess which derivative is wanted,
or how state should be updated from it.

**The numerical contract.** Floating point is not associative, so even a
perfect compiler needs permission for anything that changes the bits:
reordering a fold, TF32 products, the accumulation type, approximate
transcendental functions, casting Float32 operands to a half type for a
library that only accepts those, and whether repeated runs must agree bitwise.
A `Fold` belongs here too: it declares algebraic structure ("treat this as
associative with this identity"), which is permission, not a hint. Some verbs
stay for the same reason. `logsumexp` and `attention` promise a numerically
stable result that their naive definitions do not.

**Ownership and lifetime.** Which buffers persist across calls (parameters,
optimizer state, a KV cache), which inputs may be overwritten, and how long the
caller holds results. These are the caller's intentions and cannot be inferred
from one call.

**What is fixed and what varies.** Which values are compile-time constants and
which change per call: the decode position, a step counter, a learning rate.
This defines the family of programs being compiled.

**Resources and placement.** Which device, a memory budget, and when results
must be visible to other work on the device.

**Facts the compiler cannot prove.** Custom derivatives that intentionally
differ from the mathematical one (straight-through estimators, stopped
gradients), and seeds for randomness.

## What goes away

Each of these exists today because the compiler cannot yet decide it alone.

- **Choosing implementations.** `CuTileKernels`, `CuDNNKernels`, `CuBLASGemm`
  and the other implementation types, the `implementations` keyword of
  `prepare`, `activate`/`deactivate`, ranks.
- **Steering schedules.** Tile sizes, occupancy hints, cuDNN's cost numbers and
  `l2_fraction`, `fold_resident`, the `inplace`, `grouped` and `reuse` keywords
  of `prepare`.
- **Launch mechanics.** `capture!` and `replay!`, and the split between
  `prepare` and `prepare_forward`: a forward-only program is a program with no
  gradient outputs.
- **Verbs that exist only for speed.** `batched_mul`, or attention used as a
  performance vehicle rather than as a numerical promise.
- **Asserted facts that could be inferred.** `recomputable` and
  `scalar_partials` are declared by users because Julia cannot tell Tempest
  that a function is pure or what its derivative is. A compiler that could see
  through the function would not need them.
- **Reports.** `explain` and `profile` become curiosities, though they are
  still worth keeping for trust.

So nearly everything built in October (proposals, the registry, cost models,
cuDNN's chunking) is the imperfect-compiler layer. That is expected. It should
stay out of the semantic core.

## The pure interface

Roughly:

```julia
step = compile(f, examples; static = ..., donate = ..., numerics = policy, target)
y = step(x)
g = differentiate(f; wrt = ...)   # a program transformation
```

plus declarations: folds, custom derivative rules, numerics-promising verbs,
and randomness. This is essentially JAX: `jit` with static and donated
arguments, `grad`, `custom_vjp`, random keys and a precision setting. That is
no coincidence, since JAX was designed assuming XLA would be the perfect
compiler. What JAX added later (Pallas, rematerialization policies, custom
partitioning) is its imperfect-compiler layer.

Tempest's version is close. `prepare` is `compile`; `differentiate` and
`with_update` are the transformations; `donate` is there. The differences are
the numerical contract, the fixed/varying split, and how much of the steering
layer is in the main path.

## What changes in Tempest

In proposed order. None of this is implemented.

### 1. One numerics policy

The numerical contract is the one real gap, because it is semantics that
currently lives in backend settings:

| Choice | Where it lives today |
|:--|:--|
| TF32 products | `CUDATarget(matmul_precision = :tf32)` |
| Float32 operands cast to a half type for cuDNN attention | `CuDNNKernels(float32_cast = ...)` |
| Half-precision folds accumulate in Float32 | `fold_accumulator`, in the fold contracts |
| Bitwise determinism | not stated; implied by which kernels are chosen |
| Approximations | comments: cuTile's `tanh` choice, the `log1p`/`expm1` exclusion |

A single policy object would make these one semantic choice that every
implementation must honour, that `support` declines on when it cannot, and
that `explain` reports. A sketch:

```julia
Tempest.numerics(; products = :tf32, half_operands = Core.BFloat16) do
    ex = prepare(step, params, batch; donate = (1,))
end
```

Following the registry's precedent, the policy would be a `ScopedValue` read
once by `prepare` and stored in the executable, so a prepared step keeps the
policy it was prepared under, and an explicit `prepare(...; numerics = ...)`
would take precedence over the scope. Requests carry it to implementations.
Accumulation would start as a field with one allowed value (Float32 for half
types), so the rule is visible and the type has room for the Ampere Float16
case if a workload ever wants it.

Open questions for that design:

- **Program-wide or per operation.** A policy read at `prepare` covers the
  whole program. TF32 for attention only would need the policy recorded on
  nodes at trace time instead. Start program-wide; per-node scoping can be
  added later because trace-time scopes nest inside it.
- **Determinism.** It needs an audit of which schedules reorder: split
  reductions, atomics in scatter-add pullbacks, and library heuristics.
- **Approximations.** Whether "approximate" is a policy level or a property of
  each implementation that `explain` reports. The `log1p`/`expm1` work in cuTile
  argues for the second: the goal there is under one ulp, not a switch.

### 2. Make fixed versus varying explicit

Today the split is decided by type. Float arrays and float scalars are traced;
integer arrays are traced as index data; integer scalar arguments are rejected,
so a constant has to be captured by the traced function. Decoding needs a
varying integer position. The proposal is to accept integer scalar arguments as
`TracedScalar{Int32}` with dynamic-offset operations (dynamic slice and dynamic
update), rather than adding a `TracedInteger <: Integer` that would invite
traced control flow. A `static` keyword for float values that should be
constants would complete it. This belongs with the Vallmo decode port.

### 3. Let `prepare` own launch mechanics

- Merge `prepare_forward` into `prepare`. Forward-only is `prepare(f, args...)`
  on a program without gradients; gradients are requested by `differentiate`
  or `wrt`. This changes the current default, where `prepare(f, args...)`
  differentiates with respect to every argument, so it needs a deprecation.
- Let `prepare` decide whether to capture a CUDA graph. `capture!` stays as an
  explicit request; the default becomes capturing when the executable's storage
  is stable, which preparation already knows.

### 4. Keep implementation choice as the escape hatch

`activate(module)` is the right level for the escape hatch: it names a
package, not a type. The implementation types do not need to be exported to
model authors; they can be public for implementation authors. The
`implementations`, `inplace`, `grouped` and `reuse` keywords of `prepare` can
move out of the documented main path.

### 5. A public surface for implementation authors

The out-of-tree test package (`test/packages/MyLib`) implements a named-call
implementation using only public names. Region-level implementations cannot:
they need `Elementwise`, `Read`, `ScalarCall` and the graph operations, which
are not public. cuDNNExt uses 13 non-public names, mostly IR node types
(`MatMul`, `Pointwise`, `FoldReduction`, `Cast`, ...).

That is the same line as above. Named calls (verbs with a reference) are a
stable public surface. Regions expose IR that is still changing. Keep regions
non-public until the IR settles, and say so in the implementation guide, rather
than making node types public piecemeal.

## Already right

- `trace` is pure semantics: no backend appears in a traced program.
- Ownership is explicit: `donate`, `with_update`, `bind_argument!`.
- `Fold` is an algebraic declaration, not a schedule hint.
- Verbs are defined by their reference result, with implementations as
  optional realizations.
- Implementation choice is already ordered by precedence (explicit, scoped,
  registry) and reported by `explain`, so moving it out of the main path loses
  nothing.
