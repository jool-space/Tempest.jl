# Decision log

Dated review decisions, newest first. Each entry records priorities at that
date, not promises that every proposal survives implementation and measurement;
"next step" language in older entries describes the state at that time.

## Workload pass — 23 September 2026

Actions taken, each measured before it was kept:

- The generated `:cuda` kernels were the performance floor at about 1% of
  copy bandwidth because they spilled index metadata to local memory. Reads
  are resolved at build time; the fallback is now within 2–6× of fused regions.
- The cuTile streamed tuple-entry failure was diagnosed as an `AssumeOp`
  cached across regions, and fixed on a local cuTile branch. The tuple entry
  turned out to be slower for streamed kernels, so the entry split stays. See
  [cuTile issues](cutile-issues.md).
- `batched_mul` and `logsumexp` became built-in verbs, and `examples/gpt.jl` the
  workload gate. Its profile drove packed resident columns, split reductions,
  32-bit device layout maps, a TMA exclusion for split sweeps, an opt-in TF32
  policy, and `Tempest.profile`.

- Integer arrays became traced index arguments, with column gathers,
  `take_rows`, and deterministic accumulating pullbacks. The GPT example
  dropped its one-hot inputs.

- Counter-based `Tempest.dropout` made randomness functional: a hash of seed,
  stream and index, identical across targets, with the seed as ordinary state.

- `Tempest.attention` became a verb whose TF32 cuTile provider fuses forward
  and backward (FlashAttention-2 style, deterministic), in the model's
  heads layout with grouped key/value heads. GPT-2-small went from 1138 to
  779 ms per step at context 1024 and from 417 to 362 ms at context 256.

Open after this pass: horizontal fusion of products that share an input,
cold preparation time, and symbolic shapes.

## Rename to Tempest — 22 September 2026

The package was renamed from Deductor to Tempest before it had a remote, so no
compatibility alias was kept. Earlier entries use the new name throughout; the
[design notes](design-notes.md) record the reasoning.

## Consolidation — 22 September 2026

Actions taken on the review below, in order:

- The feature work of 20–21 September was committed before any reorganization.
- The older whole-column reduction schedule was timed against the graph
  provider's streamed schedule before deletion. It was four to eight times
  faster on wide matrices. The graph provider gained a resident column schedule
  and a tuple kernel entry, after which it matched or beat the older path at
  every shape; the [reduction-domain report](reduction-domains.md) records the
  measurements and the cuTile 1.0 register-allocation finding behind them.
- The scalar-binding reduction path, its `fuse_reductions` flag, separate
  selector, CPU/cuTile lowerings, tests, example, and proposal page were removed.
  Its captured-Adam regression runs through graph regions as
  `test/cuda_graph_training.jl`.
- Exported names were reduced to the model-author API plus the implementation
  types users pass to `prepare`. Provider and verb APIs are `public` and
  documented under qualification.
- `programs.jl` merged into `graph_ir.jl` with `Node` and `TensorProgram`;
  `with_update` moved next to differentiation; `workspaces.jl` merged into
  `implementations.jl`; `grouping.jl` became `selection.jl`.
- Documentation navigation now separates the guide, compiler pages, and dated
  reports, with the home page stating the current architecture.

Open after this pass: pointwise-only graph regions; the generic CUDA scalar
kernels, which measured under 10 GB/s on a 4 MB elementwise map and are the
baseline every earlier speedup was measured against; and the streamed cuTile
kernels' fixed-arity entry, which waits on an upstream bytecode fix.

## Package review — 22 September 2026

This is a source/contract review, not an implemented reorganization or a fresh
compatibility certification. Core currently has 24 Julia files and 4,541 lines.
There are 47 exported names; a core-only load confirms each has a docstring and
loads no CUDA, cuTile, cuBLAS, Stiletto, or ChainRules modules. The file count is
reasonable; overlapping mechanisms and scattered definitions are the larger issue.

Recommended consolidation:

- Retire the older `ReductionRequest` selection path and `fuse_reductions` flag
  once useful CPU/cuTile schedules and their supported behavior are carried into
  the graph-region path. Its two core files and cuTile lowering total 534 lines.
  Preserve numerical, layout, alias, escaping-output, and replay regressions.
  Scalar expressions remain useful lowering data; grouping independent updates
  remains a distinct physical execution decision.
- Co-locate operation, node, and program definitions currently spread across
  `graph_ir.jl`, `tensor.jl`, and `programs.jl`. Keep symbolic-array behavior in the
  frontend; put update composition with the differentiation/composition code.
- Put the 56-line workspace allocator/binder alongside `ImplementationPlan` and
  `WorkspaceSpec` in `implementations.jl`. The contract and its realization should
  be readable together.
- Give implementation selection an honest home. `grouping.jl` currently also owns
  the complete `select_schedule` loop. Separate that policy from group legality
  through naming and function placement, without creating a generic pass framework.
- Keep the frontend/broadcast, layout, AD, reference interpreter, state, and
  prepared-execution boundaries. In particular, reference execution should remain
  independent of scheduled CPU execution. The three cuTile graph files separate
  analysis, code emission, and resource construction and earn that separation.

Public surface and scope:

- Distinguish ordinary execution APIs from documented, qualified provider-author
  APIs. Exporting every request, provider, and construction helper gives them all
  equal prominence. Compiler internals remain provisional.
- Reduce overlapping policy controls during consolidation. `CUDATarget.kernels`
  and explicit implementation candidates overlap; `fuse_regions` and
  `fuse_reductions` expose successive planner designs. Keep useful diagnostic
  comparisons without making experimental paths permanent product concepts.
- Keep state donation, independent-array grouping, semantic calls, and paired
  reverse rules: they serve whole-step training and custom optimized kernels.
  Keep Adam formulas, model definitions, and external-compiler experiments out of
  core. No universal scheduler framework, arbitrary Julia effects/control flow,
  optimizer library, or TensorIR dependency is justified by this review.
- Make one current architecture/roadmap the entry point. Earlier experiments and
  measured results remain available as dated reports. The older sections below
  record history; phrases such as "next experiment" there are not current plans.

Quality assessment: dependency isolation, numerical/capture testing, ownership
contracts, meaningful provider rejection, documentation, and CI configuration are
strong foundations. API focus, selection consolidation, and current-versus-history
navigation still need work before the package feels settled. GPU CI is manually
triggered; this review did not run a clean-environment or Julia-version matrix.
Passing local tests alone is not evidence for those portability claims.

Do consolidation as bounded changes with existing regression coverage before
adding another schedule family. Preserve the three active priorities in
[compiler direction](compiler-direction.md).

## Review of 6 September 2026 and its implementation record

### Keep

Keep the typed scalar expression tree, Base broadcasting, explicit scalar/array
identity, layout composition, deterministic scatter, owned capture resources,
independent numerical checks, and `explain`. Host compiler metadata may remain
heterogeneous; concrete types matter particularly in generated kernel paths.

## Decisions

- **One implementation protocol is the first priority.** Native and optional
  implementations answer the same support/construction question. Put them on one
  ordered candidate list and expose rejection reasons uniformly. Preserve shared
  temporary workspace; unification must not allocate private scratch per region.
  This can be done before a full operation-type rewrite.
- **Typed operation payloads are next.** Use types where they express contracts,
  prevent invalid payload combinations, and permit useful extension methods.
  Replacing every branch mechanically is not the objective. New operations still
  need mathematical, differentiation, layout, and lowering behavior. Keep layout
  transformations' distinct semantics until a common representation proves useful.
- **Readable Julia takes precedence over formatter transformations.** Disable
  forced expansion of short methods, short-circuit expressions, and ternaries.
  Keep explicit blocks where they help a reader follow control flow.
- **Names and execution convenience are secondary.** `Executable` is a clearer
  name than `Prepared`. A callable executable could bind and run, but retain the
  explicit execution/replay methods for inspection and timing. A name shared with
  DifferentiationInterface alone is not a reason to rename an API; integration
  requires evaluating its actual contracts.
- **Try ChainRules as a bounded experiment.** Establish symbolic compatibility,
  including projections, thunks, nondifferentiable arguments, and branching rules,
  before depending on it. Keep custom scalar-rule errors visible. Do not continue
  expanding an independent derivative table without evaluating reuse.
- **Captured Adam with donation is the next state experiment.** Begin with a
  separately traced update spliced after the differentiated loss. Structured
  inputs and selected gradients support that experiment. Donation needs last-read
  constraints, tied-parameter semantics, warmup preservation, and defined state
  advancement across replay. An input/output alias annotation is insufficient.

## Already addressed

The rewrite separated compiler stages, made semantic calls immutable, made
built-in derivatives visible as graph arithmetic, unified result validation, and
preserved unexpected extension errors. Follow-up changes removed layout cache
invalidation using dependency order, moved shared test helpers into `test/common.jl`,
added missing API docstrings, built the documentation pages, shortened the README,
added an examples environment, and set the prototype version to `0.1.0-DEV`.

Keep the test-side CUDA graph-node inspection independent of the implementation:
it checks capture behavior without trusting the helper being tested.

## Unified selection implementation

Native and optional candidates now share support/build selection, with user
candidates preceding target defaults. `ImplementationPlan` defers workspace
binding until common slots have been assigned. CUDA/cuTile reductions, cuBLAS,
and Stiletto use this pool. Existing BuiltImplementation wrappers still work.
Validation passed with the full default CPU/GPU suite, the CUDA-only provider
configuration, the extension load-order test, and the documentation build. Tests
exercise shared slots across cuBLAS and Stiletto, repeated replay with rebound
inputs, distinct simultaneous workspace requests, and construction rejection.

## Typed operation and region contracts

Operations now own their payloads. Interpretation, reverse recipes, layout
transforms, fusion boundaries, and region construction use operation methods.
`Node` retains an abstract operation field intentionally: the tape is host
compiler metadata. View, slice, reshape, transpose, and permutation keep their
individual semantics. Native storage presentations still use a small internal
layout classification; that is distinct from operation dispatch.

`Step{R}` owns a typed region, and `RegionRequest{R}` preserves it for implementation
dispatch. Reductions and scatter carry their own layout/selection metadata;
`Step.kind`, `Step.expression`, and `RegionRequest.layout` are removed. CPU, CUDA,
cuTile, and Stiletto lowering dispatch on region types. An external-operation test
checks that mathematical and scheduling methods suffice without editing a central
operation table, and that missing lowering fails at the relevant method.

This changes provisional compiler inspection APIs. The array frontend, semantic
verb contract, implementation support/build entry points, workspace ownership,
and execution lifecycle remain intact. Captured optimizer state remains a
separate experiment.

## Scalar-rule reuse decision

The scalar-rule experiment demonstrated that public ChainRules `rrule`s can be
traced on scalar placeholders and lifted over arrays, with saved-output binding,
projection, and deferred cotangents. This is now an optional ChainRulesCore
extension. Existing Tempest graph recipes and user scalar rules remain first;
new shared rules expand into ordinary arithmetic before region planning.

Missing scalar primitives and symbolic predicates are different limitations.
The former were added as forward operations; the latter still fail explicitly.
`ZeroTangent` is supported, while nondifferentiable data inputs and captured
function tangents require a separate contract. Unexpected rule errors propagate.
We did not adopt the experimental `derivatives_given_output` helper or make
DiffRules a competing fallback: its kink conventions differ, and some formulas
also branch. The retained comparison and full limits are described in
[Scalar derivative rules](scalar-rules.md).

The next vertical slice is structured inputs, selected gradients, and an Adam
update with donation, including preservation of state during warmup.

## Preparation cost check

A small CPU comparison against the working-tree snapshot before typed contracts
used `sum(abs2, tanh.(w * x))`, Float32 inputs `x::32×16` and `w::24×32`, and one
BLAS thread. After five warmups, 51 samples measured preparation from an already
differentiated program; execution samples averaged 200 launches each.

| Measurement | Before | Typed contracts |
| --- | ---: | ---: |
| Median warmed preparation | 38.4 μs | 41.5 μs |
| Preparation heap allocation | 54,048 bytes | 58,704 bytes |
| Median execution | 13.9 μs | 13.7 μs |
| Scheduled regions | 6 | 6 |
| Value buffer storage | 8,196 bytes | 8,196 bytes |

This single small case measures some host cost from the new representation;
it does not establish a kernel speedup or characterize cold compilation and
larger graphs. The representation change preserves this schedule and its storage.

## Sequence and evidence

1. Unify implementation selection and workspace planning. Check candidate order,
   support/build rejection, shared-workspace lifetimes, and both CUDA provider
   configurations. Existing custom wrappers should continue to work.
2. Introduce typed operation and region contracts incrementally on a green suite.
   Assess readability and preparation costs rather than counting removed branches.
3. Run the scalar-rule reuse experiment; discard it if the integration is brittle.
4. Build a captured Adam step on a small model, with structured parameters and
   selected gradients. Compare state transitions with an independent update,
   ensure warmup leaves state unchanged, and verify repeated replay.

Correctness tests do not establish a performance improvement. Report preparation,
steady-state execution, and memory costs separately when evaluating these changes.

## First state slice

Tuple/NamedTuple inputs and `differentiate(...; wrt=(...))` now preserve selected
argument structure. `with_update` appends a functional callback to the explicit
backward graph. This follows the review's first option: compose the update after
AD, without making a nested gradient call a traceable operation.

`prepare(...; donate=(1, 2))` currently realizes state transfer through staged
commits into owned input buffers. All old-state reads finish first, and returned
aliases are snapshotted before simultaneous commits. This deliberately leaves
kernel inputs and outputs distinct: claiming direct donation would require both
a liveness/access proof and an implementation aliasing contract. That optimization
is still outstanding and should use the existing support/build selection protocol.

`bind_argument!` changes a batch without resetting state. Capture and recapture
save and restore current donated values, including failure during warmup. CPU and
CUDA tests compare multiple Adam transitions with independent eager model
derivatives, cover changed batches and alias swaps, and check stable addresses.
The runnable Adam example prints timing and the actual staged-copy plan.

### Adam measurement

On GB10, `examples/adam.jl` uses Float32 layers 32 → 64 → 16 with batch size 48.
The combined graph has 26 regions and captures 45 graph nodes, including 32
kernels and 13 state commits. Five changed-batch transitions are checked against
the allocating reference before timing. Each timing resets the same initial
state, runs 100 steps, and synchronizes; binding and capture are excluded.

After explicit warmup, the best of five batches measured 215 μs per uncaptured
step and 67 μs per captured step. The first prepare call in that process took
3.41 seconds (tracing/differentiation excluded; compiler caches may already exist).
An initial single timing batch overstated captured time, so the example now warms
both paths and prints its sampling method. These are lower-bound observations of
one small model on a shared machine, not a general training-stack comparison.
Direct in-place update planning remains a measurable next optimization.


## Direct donation and independent array groups

The next experiment keeps Adam as ordinary Julia and makes two compiler decisions
explicit. Same-lane reads plus old-input lifetimes permit direct assignment to
owned state; `InPlaceRequest` requires implementation opt-in. Otherwise state
remains staged. Capture restoration now covers both kinds of donated buffers.

Independent neighboring elementwise regions can form a `GroupedRequest`, with
physical alias checks after storage assignment. CUDA uses fixed-type descriptor
tables and bounded groups; metadata size grows with the arrays, kernel argument
count does not. Rejected groups fall back to their members. `explain` maps launches
back to regions and reports direct/staged state writes.

The [many-array report](many-array-updates.md) records the 100/1,000-array comparison,
including descriptor storage and the unresolved cold-compilation cost. Native
kernel aliasing remains an explicit contract, not an assumption imposed on custom
or vendor implementations.


## Kernel composition experiment (20 September 2026)

The next bounded question is whether an independent kernel library can keep its
concrete CUDA wrappers while participating in an unchanged model's training graph.
`ReverseRule` now retains that opportunity after differentiation: a provider may
replace a public call and its pullback together, introduce private saved tensors,
and rely on the existing storage planner for their lifetimes. Construction
rejection discards the pair and retries from the original program.

The first exercise is residual RMSNorm with two normalization layers and Adam.
A small deterministic CUDA library provides the regular regression target. The
adapter to the existing archived tile-kernels implementation remains blocked by
its cuTile compilation failures; the standalone reproducer removes Tempest from
that failure path. See [Implementation reverse rules](reverse-rules.md).

This does not establish an optimized attention integration or a nanoGPT speedup.
The next acceptance gate is a working existing kernel pair, followed by attention
and a matched workload comparison. Keep integer/RNG saved state and numerical
policy explicit when extending the current floating-tensor contract. Broadening
operator coverage or introducing another compiler layer is not the next step.


## Compiler direction after the research experiments (21 September 2026)

The [TensorIR experiment](tensor-ir-training.md) established a working external
region compiler through saved forward values, backward, and captured replay.
TensorIR and cuTile.jl share a lower target but have no direct frontend fusion
interface. That distinction changes the next priority: retain Tempest's semantic
graph and develop region selection before committing to a compiler dependency.

The [compiler direction note](compiler-direction.md) records the resulting
contracts and experiments. A CPU-only boundary probe exercises graph contraction,
multiple outputs, logical aliases, and functional state. Automatic extraction
and a [native cuTile schedule](cutile-sweeps.md) now cover dependent reductions,
custom Julia callables, irregular and long columns, and captured replay.

Candidate selection is now integrated before physical storage planning through
the existing implementation protocol. [Graph regions](graph-regions.md) retain
original operations and escaping values; the cuTile extension supplies a native
streaming schedule. Recomputation requires an explicit scalar-function contract,
and late rejection retries smaller candidates before ordinary decomposition.
Aliases, saved backward values, runtime scalars, and staged state donation are
covered by production-path tests.

The subsequent integration review moved coverage and measurement ahead of another
schedule family. Prefix-only fallback fragmented otherwise supported normalization
around an undeclared activation; support-guided growth now keeps that body
connected. Second-axis and full sums, plus intermediate Float64 arithmetic in the
MLP backward, remain actual workload barriers.

Bring a small transformer diagnostic forward, widen reduction domains, and set
up a matched cuTile/Stiletto/TensorIR reference before comparing schedules.
Historical ratios against the untuned decomposition are not evidence of
competitive throughput. The [revised compiler priorities](compiler-direction.md#Integration-review-and-revised-priorities)
record the findings, remaining greedy-search limits, precision issue, and
consolidation of the three existing fusion paths.
