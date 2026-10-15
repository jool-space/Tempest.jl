# Decision log

!!! note "Historical report"
    Written between 6 September and 15 October 2026, entry by entry. It records the state, measurements and plans of that
    date; the current design is described in [Architecture](architecture.md),
    whose open directions replace the plans here.

Dated review decisions, newest first, in detail; [History](history.md) is the
short version. Each entry records priorities at its date, not promises that
every proposal survived implementation and measurement.

## Occupancy hints remeasured — 15 October 2026

Tempest's two cuTile occupancy hints were added when, without them, `tileiras`
gave its kernels 255 registers and they ran several times slower. cuTile main
(`d2fafa3`) made the token-order probe fast, so both were measured again on
GB10. Each hint value was compiled in one process and timed in alternation
from CUDA graphs. Identical kernels agreed within 1% on DRAM-bound shapes but
differed by up to 25% on L2-sized softmax and LayerNorm, so only differences
seen in both passes count.

- **Scalar-region kernels keep 8.** Without a hint the reduction kernels still
  get 255 registers and run up to 8 times slower (column sums over
  256 × 32768: 1164 against 155 μs); 16 matches 8. Pointwise kernels use 16 registers
  either way.
- **Resident graph kernels choose by tile size.** No hint is still up to 5
  times slower for tiles up to 2048 rows. Tiles of at most 512 values now
  compile for 8 instead of 4: row sums and logsumexps ran 30-40% faster
  (256 × 32768 sum: 178 to 120 μs), while DRAM-bound normalizations lost up to
  5%. 1024- and 2048-row tiles keep 4: at 8 some kernels kept 255 registers.
  4096-row tiles drop the hint: every kernel ran 2-5% faster without it.
- **End to end the change is neutral.** The residual RMSNorm benchmark moved
  within noise cold and gained warm at small shapes (256 × 4096: 29 to 23-26 μs).
  The GPT training step, timed as captured replays, took 2.15 ms and 45.9 ms per
  step at the default and an enlarged shape under both policies. Dropping
  either hint cost it at most 2%, because matrix products dominate.

## An implementation registry; device rates; layout copies — 15 October 2026

**Registry.** Default candidates come from a registry keyed by module:
`Tempest.register!(mod, constructor; target, rank, active)` from a package's
or extension's `__init__`, `Tempest.activate(mod; kwargs...)` or an instance,
`Tempest.deactivate(mod)`, and do-block forms of both that change a
`ScopedValue` for the function's extent. Decisions:

- Candidates are ordered by declared rank, then name, never by load order; a
  fresh-process test loads the extensions in two orders and compares.
- Baseline kernels belong to the target and cannot be deactivated. cuBLAS
  and cuTile register active, cuDNN inactive. `deactivate(cuTile)` replaces
  `CUDATarget(kernels = :cuda)`, which is removed rather than deprecated:
  `:cutile` had no remaining meaning and `:cuda` was a special case of the
  general mechanism.
- Explicit `prepare(...; implementations)` precedes a scope, which overrides
  the registry. Candidates are resolved once per `prepare`; `explain` labels
  each with its source.
- A plain `Dict` behind a lock, no `Val` dispatch. Proposals stay their own
  phase before growth-based regions.
- If cuTile moves into CUDA.jl, the registry makes that a change of defaults.

**Device rates.** `CuDNNKernels`'s product rate and bandwidth default to the
device's, derived from attributes and a compute-capability table, or measured
once per device and process for an unknown architecture. On GB10 the derived
107 TFLOP/s and 232 GB/s left every benchmark decision unchanged.

**Layout copies around attention** were measured before deciding whether
implementations should declare accepted strided layouts for the planner to
absorb permutes, slices and reshapes
([measurements](implementations.md#Layout-copies-around-attention)). Heads
split by reshape, as in the GPT example, cost nothing. Packed-projection slices
cost 6–7% of a training step and 27–28% of a forward; row-form permutes 14–17%
and 37–40%. Deferred: the gain is real only for code written in other
layouts, the slice case is the one strides would clearly absorb, and the
row-form permute is likely not unit-stride for cuDNN anyway.

## Why cuDNN's fused products lose, and the rule that replaced 4096 — 15 October 2026

The first cuDNN consumer declined products whose right operand exceeded three
quarters of L2 or whose inner dimension exceeded 4096, from measurements alone.
Kernel traces with `nsys` (performance counters are not available to this
user, so no `ncu`) found two causes, recorded in
[proposed regions](proposed-regions.md#Where-fusion-pays):

- **Tile order.** The runtime-compiled fused kernels walk output tiles along
  columns and reread the whole right operand for each row of tiles; beyond L2
  it streams from memory once per row. Split-K and stream-K knobs and other
  kernel configurations did not fix the order; launching the plan over column
  chunks of the operand did (1024 × 2048 × 8192: 1216 to 475 μs).
- **Product throughput.** Without the L2 effect the fused kernels ran at about
  0.9 of cuBLAS's rate, less with a mostly empty last wave of output tiles, so
  fusion pays only while the epilogue traffic it saves exceeds that loss.

The 4096 limit is gone. `CuDNNKernels` launches a region in column chunks when
the right operand exceeds `l2_fraction` of L2, and declines a product when an
estimate of its fused time, from cuBLAS's rate, the fused efficiency, wave
occupancy, copy bandwidth and launch cost, exceeds the default plan's. All are
keywords with GB10 defaults. Other kernel configurations shortened the rows of
tiles (709 μs) but kept the column-first order and lost to the chunked launch,
so the heuristic's first configuration stays. Choosing one by timing stays
rejected for determinism.

## Float32 accumulation for half-precision folds — 15 October 2026

A Float16 sum accumulated in Float16 on the CPU and CUDA kernels and in the
reference interpreter (Base's `sum` rounds every partial sum), in Float32 in
cuDNN, so the result depended on the backend; adding 0.1 65536 times stalls at
256 in Float16. Folds now declare the type they accumulate in,
`fold_accumulator(fold, T)`: Float32 for Float16 and BFloat16 sums, products
and logsumexps; the element type for maxima and minima, which are exact, and
for `GenericFold`, whose `init` chooses its state. Elements convert to it before
lifting and the observed value rounds once to the element type.

Planning applies it once: a reduction region's expression converts its
contributions and its identity state is the accumulator's, so CPU, CUDA and
cuTile scalar kernels lift and combine in it without knowing the rule. Sum
kernels take their accumulator from the region's state, as do the split-sum
partial workspaces. The interpreter and the logsumexp reference convert before
folding; cuTile graph regions are Float32 throughout; cuDNN already summed in
Float32. The `GraphRequest` and `ProposedRequest` contracts state the rule.

Half-precision sums of 32 MiB on GB10 took the same time before and after
(CUDA 141-223 μs, cuTile 122-224 μs depending on the axis): the kernels are
memory-bound. A BFloat16 column sum on the CUDA warp kernel, which threw a
device exception when it accumulated and shuffled BFloat16, now runs.

## Implementation-proposed regions — 14 October 2026

Tempest partitioned programs itself, and no implementation could fuse across a
matrix product: graph regions stop at contractions, and `support` can only
accept or decline what it is offered. Implementations now also propose regions
(`propose_regions`), in the style of XLA's and TVM's partitioners. The design
and its answers are in [proposed regions](proposed-regions.md). Decisions:

- **Proposals before growth.** Proposals are made on the prepared program, after
  differentiation, and claim their nodes before growth-based graph regions
  form around them. Rejected and failed proposals keep the default partition.
- **Priority, then benefit.** Candidate order ranks implementations. Within
  one, proposals rank by the bytes they keep out of memory against the default
  partition, computed by Tempest from fusion roots and escaping values. Plain
  size ranked a forward epilogue that swallowed the activation derivative above
  the backward product's epilogue; benefit ranks them the other way.
- **Feasibility in support.** A decline in `support` costs nothing; a
  construction failure costs a replan. cuDNN builds its graph (without buffers)
  in `support`, and all construction failures of one pass are excluded before a
  single replan.
- **Deterministic, measured policy.** cuDNN declines products where its fused
  engines were slower on GB10 (right operand beyond three quarters of L2, inner
  dimension above 4096). Selection by timing was rejected because the same
  program must give the same partition.
- **Shared numbering.** `standard_numbering` moved into Tempest; cuTile kernels
  and cuDNN execution plans are both shared between equivalent regions.
- **Half-precision cuBLAS.** The default plan had no Float16 or BFloat16
  product, so there was no baseline; `CuBLASGemm` now uses `cublasGemmEx` with
  Float32 accumulation.

First consumer: `CuDNNKernels` fuses Float16 and BFloat16 products with their
bias, activation, activation derivative and bias gradient. On an MLP block it
was 1.3 to 2.3 times faster than the default plan forward and 1.2 to 1.8 times
in training. Float32 cuDNN products round to TF32 without saying so, and are
proposed only under `matmul_precision = :tf32`.

## Pruning — 14 October 2026

Research code that no current page or test depends on was removed. The git tag
`research-2026-09` keeps the last commit that contains it.

- **TensorIR experiment** (`examples/research/tensor_ir_*`, a C bridge and an
  MLIR fixture). On 21 September a residual RMSNorm training step compiled by
  NVIDIA TensorIR v0.1.4 ran inside Tempest's semantic calls, saved-state reverse
  rules, storage planning and capture: 345 integration assertions, and replay
  7-11x faster than the generic CUDA kernels of the time. It established that an
  external graph compiler can supply a call's realization, but not how to select
  regions, and it needs a separate tensor-ir build. Tempest's direction is a
  composition layer without a compiler binary dependency, and region selection
  is now graph regions. The report was `docs/src/tensor-ir-training.md`.
- **Graph-region probes and native cuTile sweeps** (`graph_regions.jl`,
  `graph_candidates.jl`, `reduction_scopes.jl`, `native_region_baseline.jl`,
  `cutile_sweeps*.jl` and their test scripts). The CPU probes found region
  boundaries and reduction phases on ordinary traced graphs; the sweeps compiled
  a bounded family of them to one cuTile kernel, 3-20x faster than the
  decomposed schedule of 21 September. Both became production code:
  `src/compiler/graph_regions.jl` selects regions and `ext/cuTileExt/graph/`
  schedules and compiles them, with their own tests. The reports were
  `docs/src/cutile-sweeps.md` and `docs/src/scheduler-investigation.md`.
- **Dated reports in docs/src.** The first-experiment, array-frontend and design
  notes predate the current compiler and are superseded by the guides; the
  compiler landscape and scheduler investigation surveys (21 September) fed the
  compiler direction page (since folded into [Architecture](architecture.md)) page; the Stiletto perspective
  described an extension removed on 14 October in favor of cuDNN. The dated
  sections of the compiler direction page moved into this log, next to its
  21 September entry.

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
design notes (archived with the tag `research-2026-09`) record the reasoning.

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
compiler direction page (since folded into [Architecture](architecture.md)).

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


## Compiler direction: experiments and integration — 21–23 September 2026

Moved from the compiler direction page (since folded into [Architecture](architecture.md)) page, which keeps the
principles. Paths under `examples/research` are archived with the tag
`research-2026-09`.

### Boundary probe completed in this iteration

`examples/research/graph_regions.jl` (since archived) referenced the existing graph directly. It
collects inputs and escaping outputs, preserves logical mappings, keeps all
results of a semantic call in one execution unit, and verifies that collapsing
a candidate leaves an acyclic dependency graph.

Its CPU interpreter exposes only the declared inputs to the selected unit and
exports only the declared outputs. This catches omitted dependencies and escapes.
It evaluates the existing operations with ordinary arrays; it does not generate
a GPU kernel or demonstrate fewer allocations.

The accompanying tests cover unfamiliar formulas, a custom scalar callable,
orthogonal reductions, nonadjacent but independent nodes, a rejected path that
leaves and re-enters a candidate, outside consumers, escaping views with internal
roots, atomic multi-result calls inside and outside the candidate, mixed casts,
runtime scalars, duplicate result leaves, and backward plus a functional update.
The probe passed **209 CPU assertions** on Julia 1.13. It establishes structural
boundary accounting and reference execution, not automatic candidate selection,
physical allocation correctness, or a general fusion schedule.

Run without CUDA or TensorIR dependencies:

```sh
julia --project=. examples/research/graph_regions_tests.jl
```

### Native region experiment completed

The follow-up now selects structural candidates automatically and lowers the
accepted matrix family to a native cuTile kernel. Its schedule comes from
reduction dependencies, including several dependent or independent sums. The
custom scalar callable remains a Julia callable inside the kernel. Irregular
259-row and long 8193-row cases pass capture and changed-input replay checks.

The native cuTile sweeps report (archived with the tag `research-2026-09`)
recorded the experiment, measurements, and restrictions. This established that the semantic graph could support both
external compilation and native Julia scheduling.

### Production integration

`prepare` and `prepare_forward` now select semantic `GraphRegion`s before layout
and storage planning. `GraphRequest` uses the existing support/build protocol;
loading cuTile supplies a native streaming implementation. There is no second
provider registry or dependency on the research driver.

The selector tries connected candidates with up to 64 members. After a complete
candidate is declined, growth consults provider support at each addition and
keeps declined neighbors outside the current attempt. Construction rejection excludes
the exact member set and replans. Alias roots, saved backward values, and staged
state commits participate in ordinary lifetime planning. `explain` prints the
selected boundary and rejection reasons. Custom scalar callables require explicit
`recomputable` permission for this schedule.

Validation on Julia 1.13 / GB10 passed all 4,343 CPU/CUDA assertions, including
306 production cuTile graph checks. The normalized-MLP example also passed
captured replay with changed scalar inputs and CPU gradient comparisons.
The forward fixtures at 259 and 8193 rows each captured one kernel.

See [Graph regions](graph-regions.md) for usage and limits. This completes the
boundary integration, not schedule search or a full training performance claim.
The integration review changed the next priority. The existing MLP exposes
fragmentation from unsupported neighbors, second-axis and full reductions, and
mixed intermediate precision in backward. A second leading-axis schedule would
leave those coverage gaps in place. Bring a small transformer training harness
forward as a diagnostic workload, and widen reduction coverage before schedule
competition. Physical ownership and saved backward values must remain visible
through those changes. See the review and revised sequence below.

### Experiments that discriminate between the options

1. **Automatic regions on the existing graph.** Grow bounded candidates from
   ordinary traced expressions, preserve escapes through backward, and compare
   automatic boundaries against the explicit TensorIR fixture. Exercise a
   deliberate rejection and a smaller valid alternative. Use the existing
   native decomposition as the control.
2. **A native cuTile schedule beyond the current family.** Generate whole-column
   and streamed schedules for centered LayerNorm with a custom Julia functor,
   a shared residual, and returned statistics. Use 259 and 8193 rows, an explicit
   purity declaration, and a functor containing supported `ifelse` computation
   without adding a primitive export rule. Compare aligned controls with TensorIR
   where accepted. Measure compilation, finite numerical agreement, launches,
   storage, and captured replay separately.
3. **A training workload that needs both generated and authored kernels.** Add
   an independently validated attention forward/backward pair to a small
   transformer step. Check complete state transitions and retained memory before
   interpreting speed results. This is a subsequent workload gate, not evidence
   supplied by the normalization microbenchmarks.

Keep the first two experiments bounded and avoid a production IR rewrite until
they expose a requirement the present graph cannot express. Do not build a
universal autotuner, a TensorIR/cuTile fragment linker, or an e-graph system as a
prerequisite. The immediate question is whether one truthful region contract can
support both general Julia tile schedules and external graph compilation.


### Integration review and revised priorities

The review correctly identified a flaw in prefix-only fallback. An undeclared
activation before centered normalization split its supported body into a bare
sum and an 11-node region. The same activation after normalization preserved all
12 supported nodes. Support-guided growth now retains the complete 12-node
normalization in either position. It also preserves supported epilogues when
neighboring second-axis or full reductions are declined.

This does not remove all order dependence. Growth is still greedy. A provider
can accept a complete formula while rejecting fragments, which is why the
complete structural candidate is still tested first. Private literals must be
considered with their consumers to avoid artificial scalar boundaries. Late
construction rejection remains keyed to the exact member set.

The 111-node MLP originally generated 53 rejection notes. Its final input-gradient
fusion has another barrier besides reduction axes: integer division in the
reverse recipe creates a Float64 reciprocal and multiply, followed by a Float32
cast. The cuTile region provider rejects that intermediate precision. Changing
region growth cannot safely erase the conversion; numeric policy and the reverse
recipe need a separate decision and validation.

On that same MLP, the growth change enlarges the backward region from seven to
nine members. It still has 24 scheduled launches, 35 captured kernels, and
435,864 bytes of value storage. Capture/replay and CPU gradient comparisons pass.
This improves fusion coverage without establishing an MLP performance gain.
The revised full CPU/CUDA suite passed 4,426 assertions, and the docs build passed.
Metadata support queries increase from 46 to 50 while rejection notes fall from
53 to 49; checking support during growth is not itself a query-count optimization.

Earlier timing ratios compare with Tempest's untuned decomposition. They do not
establish competitive normalization performance. The unexplained 256/259-row
timing reversal and the roughly 19-second initial kernel compilation remain
open issues. A tuned implementation is a comparison point, not a proven
performance bound.

Revised work order:

1. Fix support-guided growth and preserve legality, constants, complete-formula
   support, and construction fallback in regression tests.
2. Use the existing MLP and a small transformer step as coverage diagnostics now.
   Include ordinary forward/backward/update code and record unsupported domains,
   precision conversions, saved activation bytes, and preparation time.
3. Add second-axis and full reductions, then rank-three domains. Start with
   separate valid regions; fusing orthogonal reductions can require a different
   block decomposition or multiple launches.
4. Establish matched benchmarks before comparing schedule families: the same
   RMSNorm formula, dtype, epsilon, outputs, and capture conditions across cuTile,
   Stiletto's named RMSNorm, and TensorIR where supported. Keep the custom-callable
   fixture as a separate expressivity experiment. Report preparation separately,
   repeat warm measurements in alternating order, count physical kernels, and
   compare traffic with a device bandwidth measurement.
5. Compare retention, streaming, and staged schedules using those workloads and
   measurements. Include compilation cost and values moved earlier across the
   forward/backward boundary in the tradeoff.

The planner has scalar fusion for the decomposed schedule and graph regions.
The earlier scalar-binding reduction selector was retired on 22 September 2026
after the graph provider's resident schedule matched it. Scalar expressions
remain useful as a kernel representation, but region-boundary policy should
converge on one path; pointwise-only regions belong in that consolidation.

Escaping intermediates are currently recomputed between sweeps and stored at the
end. Storing an already required output in an earlier sweep may avoid expensive
recomputation, but subsequent loads add memory traffic and ordering requirements.
Treat that as a scheduling alternative, not a free transformation. Similarly,
early computation of backward-only values is legal but can extend activation
lifetimes; correctness checks alone do not establish a profitable placement.

### Broader reduction domains

The production cuTile graph provider now handles second-axis and full reductions
in addition to leading-axis sums, without recognizing named normalization
formulas. The same dependency analysis produces phases for all three schedules.
Axis reductions stay within a block; full reductions declare partial-sum and
statistic workspace through the existing `ImplementationPlan` interface.

The normalized MLP now selects six graph regions and uses 20 scheduled calls,
33 captured kernels, and 435,476 value bytes. Its CPU/GPU loss and gradient checks
pass under capture and scalar rebinding. The preceding plan used 24 calls,
35 kernels, and 435,864 value bytes. This improves backward coverage but does not
establish end-to-end training performance.

`explain` now reports reduction axes, tile shape, dependency phases, and physical
kernel counts. The implementation is split between schedule analysis, tile code
emission, and runtime construction. Mixed axes still form separate regions;
rank-three and mixed-precision graph domains remain unsupported.

The [reduction-domain report](reduction-domains.md) records the independent `sum!`
comparison and its limitations. It is a focused reduction baseline; matched
RMSNorm comparisons with Stiletto and TensorIR, a trustworthy bandwidth reference,
and the small-transformer workload remain outstanding. Before expanding search,
use those measurements to choose schedules and avoid profitable-looking fusion
that only reduces the number of scheduled calls.

The first independent reduction measurements reinforce that distinction. On
512×1024 Float32 inputs, the current full-reduction schedule takes roughly
113 microseconds versus 39 for GPUArrays `sum!`, including the latter's captured
temporary allocation. Second-axis sums are much closer, around 75 versus 73
microseconds. Use this evidence to improve block/tile policy and cost selection
before claiming competitive fusion performance. Full validation covers 4,970
assertions across the suite and final targeted rerun, plus MLP replay and docs.

## Compiler direction after the research experiments (21 September 2026)

The TensorIR experiment established a working external
region compiler through saved forward values, backward, and captured replay.
TensorIR and cuTile.jl share a lower target but have no direct frontend fusion
interface. That distinction changes the next priority: retain Tempest's semantic
graph and develop region selection before committing to a compiler dependency.

The compiler direction note (since folded into [Architecture](architecture.md)) records the resulting
contracts and experiments. A CPU-only boundary probe exercises graph contraction,
multiple outputs, logical aliases, and functional state. Automatic extraction
and a native cuTile schedule now cover dependent reductions,
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
competitive throughput. The [revised compiler priorities](#Integration-review-and-revised-priorities)
record the findings, remaining greedy-search limits, precision issue, and
consolidation of the three existing fusion paths.
