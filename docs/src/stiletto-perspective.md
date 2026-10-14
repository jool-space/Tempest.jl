# What Stiletto adds to the picture

!!! note
    The Stiletto extension discussed here was removed on 14 October 2026; see
    [cuDNN](implementations.md#cuDNN) for why and what replaces it.

21 September 2026. Inspected adjacent Stiletto commit
`46e304c62c63b714b43e51361bf904b6d576ff60`, its tests, and Tempest's current
Stiletto extension. The installed Stiletto 0.1.0 source used for the probe
matched the adjacent checkout's `src/` directory. No Stiletto files were changed.

## The architectural distinction

Stiletto demonstrates a useful frontend and a restrictive execution boundary.
Its [compiler](https://github.com/jool-space/Stiletto.jl/blob/46e304c62c63b714b43e51361bf904b6d576ff60/src/compiler/compile.jl#L80)
constructs one cuDNN graph and calls `build!`. If no engine accepts that graph,
compilation fails. There is no region partitioner, alternative kernel generator,
or decomposed schedule to preserve execution of an otherwise valid formula.

This agrees with the [cuDNN graph contract](https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html):
the graph is commonly a partition of a larger network, and engine availability
is defined over supported graph patterns. An operation's existence does not
establish support for arbitrary composition with other operations. An accepted
cuDNN plan can represent a kernel or a small set of kernels; this investigation
did not count physical launches.

Tempest already provides the surrounding execution planner, but its current
Stiletto bridge offers too little graph context. `ext/StilettoExt/StilettoExt.jl`
accepts only Float32 `Elementwise`, `FullReduction`, and `AxisReductionRegion`
requests, with restricted scalar functions and dense identity mappings. It does
not offer GEMM-plus-epilogue regions, named norms, SDPA, or the newer multi-output
`ReductionRegion` to Stiletto. Existing tests explicitly select separate cuBLAS
and Stiletto steps. Enlarging candidate regions is required to expose vendor
fusion opportunities that the present bridge cannot even ask about.

## What is worth retaining

- **Deferred declaration.** Stiletto records a trace before declaring cuDNN
  tensors. Rank promotion and layouts are chosen with knowledge of their uses.
  Uniform rank is a vendor lowering policy; the useful general idea is to delay
  physical representation decisions until consumers are known.
- **Destination bindings.** Allocating and mutating calls share computation;
  destinations determine output storage. Bindings and workspace can be separated
  from execution, which Tempest already uses to construct stable variant packs
  outside CUDA capture.
- **Runtime inputs and method-aware caching.** Scalars rebind by value. Argument
  signatures include type, shape, strides, and view alignment. Stiletto's
  [trace epoch](https://github.com/jool-space/Stiletto.jl/blob/46e304c62c63b714b43e51361bf904b6d576ff60/src/compiler/cache.jl#L57)
  uses CompilerCaching/native CodeInstance identity, with a world-counter
  fallback. This is useful precedent for caching a Julia trace; device/compiler
  artifacts and concrete bound executions need their own contracts and keys.
- **Logical values with multiple storage components.** The
  [Microscaling extension](https://github.com/jool-space/Stiletto.jl/blob/46e304c62c63b714b43e51361bf904b6d576ff60/ext/MicroscalingExt.jl)
  declares and binds elements plus scales for one array. Stores into blockscaled
  destinations become quantization with multiple physical outputs. A future
  Tempest representation interface should preserve that distinction and make
  its rounding semantics explicit.
- **Attention forward/backward pairing.** Forward can produce Float32 LSE
  statistics; backward consumes those statistics and the forward result. This
  matches the purpose of Tempest's implementation-specific saved tensors.
  Stiletto already supplies a concrete vendor implementation to consider there.

## Where backend constraints became language constraints

Some restrictions occur before cuDNN engine selection:

- A pointwise node stores a cuDNN enum. A Julia scalar function with no enum
  mapping or explicit expansion fails during tracing.
- Numeric literals and scalar inputs become Float32. A full `sum` remains an
  all-singleton array. These are frontend policies that a backend-neutral graph
  should not inherit automatically.
- Computed values cannot be reshaped or viewed. A computed permutation is
  restricted to the output boundary and cannot feed another operation.
- Named normalization rejects computed scale/bias inputs, even a simple
  `scale .+ 1`. Its current lowering uses inference mode and does not expose
  normalization's saved statistics or a corresponding backward operation.
- Attention backward controls its operand declarations: operands must be trace
  inputs, cannot have competing uses, and require a rank-four graph.

These limitations are visible in
[layout tracing](https://github.com/jool-space/Stiletto.jl/blob/46e304c62c63b714b43e51361bf904b6d576ff60/src/language/operations.jl),
[scalar tracing](https://github.com/jool-space/Stiletto.jl/blob/46e304c62c63b714b43e51361bf904b6d576ff60/src/language/broadcast.jl),
and [normalization](https://github.com/jool-space/Stiletto.jl/blob/46e304c62c63b714b43e51361bf904b6d576ff60/src/library/norms.jl).
They explain why adding another emitter underneath the existing Stiletto trace
would not by itself recover a fully general semantic graph.

The named norm methods on `AbstractArray` are also not portable CPU reference
implementations: they allocate with `similar` and call Stiletto's JIT/cuDNN path.
Their independent mathematical references live in the tests. Supporting those
verbs inside Tempest requires explicit semantic/reference integration; the
abstract argument annotation alone does not provide it.

## Interpreting the tests

The main suite requires ordinary RMSNorm/LayerNorm and several attention
forward/backward cases to execute correctly. However, the entire
[composition corpus](https://github.com/jool-space/Stiletto.jl/blob/46e304c62c63b714b43e51361bf904b6d576ff60/test/corpus.jl#L7)
marks `UnsupportedGraphError` cases skipped. Its softmax, normalization epilogue,
mean-centering, and back-to-back GEMM examples establish useful probes, not
unconditional coverage. Other exceptions propagate.

There is a significant exception to any claim that cuDNN cannot consume
reduction results. The
[MXFP8 attention recipe](https://github.com/jool-space/Stiletto.jl/blob/46e304c62c63b714b43e51361bf904b6d576ff60/test/microscaling.jl#L374)
explicitly constructs max/subtract/exp/sum/divide between two GEMMs and returns
statistics and AMAX. Its support expectation depends on compute capability 10
and cuDNN >= 9.21. That is a recognized attention region, not evidence of generic
reduction scheduling. Whole-region structure, outputs, layout, and architecture
all matter.

## Local observations

`examples/research/stiletto_support_probe.jl` records acceptance, rejection, and
numerical comparisons explicitly rather than treating unavailable engines as
passing tests. It uses Float32 inputs of shape `(64, 32)` on GB10, cuDNN library
9.24.0, Julia cuDNN package 6.3.0, and Stiletto 0.1.0. The scalar-math comparison
uses elementwise `rtol = 3e-3`, `atol = 3e-4`, with finiteness checks. These are
bounded observations, not a support guarantee across devices or shapes.

| Computation | Result |
| --- | --- |
| Pointwise tanh | Executed and matched reference |
| Named RMSNorm | Executed and matched reference |
| Residual addition feeding named RMSNorm | Executed and matched reference |
| Named residual RMSNorm returning output and residual | Both outputs matched reference |
| Tanh consuming named RMSNorm | Executed and matched reference |
| RMSNorm written as primitive reductions | No supported engine |
| Primitive residual RMSNorm returning residual and statistic | No supported engine |
| Stable softmax from max/exp/sum | No supported engine |
| Named RMSNorm with computed scale | Frontend rejected |
| Unregistered scalar closure | Frontend rejected |

The GEMM/tanh/terminal-sum case also compiled and executed, but its maximum
absolute difference from the CPU Float32 reference was `0.004260063`; it did not
meet the probe's elementwise tolerance. The probe reports that mismatch. We have
not attributed it to a particular numerical mode or treated it as a compiler
correctness defect. Engine acceptance and numerical policy must remain separate
questions.

Run with:

```sh
julia --project=test examples/research/stiletto_support_probe.jl
```

## Consequence for Tempest

Stiletto strengthens the case for a compiler that preserves a general formula
and can try different region boundaries and implementations. Vendor recognition
can remain valuable within that compiler. Named RMSNorm and attention are
useful candidates; unfamiliar reductions need the native cuTile path or another
region compiler. A declined candidate should leave a correct decomposed plan
available, with its rejection visible in `explain`.

The next vertical slice should prioritize NVIDIA TensorIR as the region compiler.
The Stiletto comparison is architectural evidence, not a proposal to complete or
perfect the cuDNN backend. TensorIR already performs general graph-to-tile
lowering and has executed actual Tempest exports; it is a stronger candidate
foundation for kernel compilation, although its current release still has
measured coverage gaps.

The immediate architectural work is a graph-region candidate interface that can
expose contractions, pointwise producers/consumers, and multiple escaping outputs
to TensorIR. Our current scalar-region bridge is too narrow.
The same region must carry dtype/accumulation policy and layout constraints,
and eventually distinguish logical values from physical storage components.
Candidate selection must account for forward/backward saved values as well as
standalone forward speed. Reuse the existing decomposed execution path as the
fallback; a competing general native cuTile scheduler is not a prerequisite.
The phase-analysis probe remains useful as an independent reference and a way
to understand external compiler failures.

Stiletto is a useful vendor provider and a source of integration mechanisms.
Tempest should own the broader semantics, partitioning, scheduling, and training
storage. The [TensorIR experiments](scheduler-investigation.md) reinforce the
same lesson: a more general lowering backend still has coverage gaps, so it
must remain one candidate behind that boundary.
