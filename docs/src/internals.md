# Reading and changing Tempest

Start at `prepare` in `src/runtime/preparation.jl`. It validates the input contract,
resolves semantic calls, and prepares a schedule. If an opaque implementation
passes its metadata check but cannot be built, preparation expands that call's
reference and replans. The original trace remains available for inspection.

## Source map

`src/Tempest.jl` lists every file in load order. The directories group the
pipeline: `ir/` (graph data and its meaning), `frontend/` (tracing), `verbs/`
(semantic calls), `transforms/`, `implementations/` (the provider protocol),
`compiler/` (from a trace to a storage plan) and `runtime/` (preparing and running).

| Responsibility | Files | Main entry points |
| --- | --- | --- |
| Index mappings | `src/ir/layout.jl` | `native_layout`, `map_index` |
| Graph IR | `src/ir/graph.jl` | `Operation` payload types, `Node`, `TensorProgram`, `TrainingProgram`, `ForwardProgram`, `live_nodes` |
| Schedule IR | `src/ir/schedule.jl` | scalar expressions (`Read`, `ScalarCall`), regions, `Step`, `StoragePlan`, `needs_indexing` |
| Independent graph execution | `src/ir/interpret.jl` | `interpret`, `evaluate`, `value_and_gradient` |
| Symbolic array frontend | `src/frontend/tensor.jl` | `TracedArray`, `TracedScalar`, `trace`, `validate` |
| Array frontend | `src/frontend/broadcast.jl`, `src/frontend/operations.jl` | `pointwise`, `trace_broadcast`, Base methods |
| Folds | `src/frontend/folds.jl` | `Fold`, `fold`, `fold_pullback`, `fold_resident` |
| Semantic contracts | `src/verbs/contract.jl` | `record_call`, `check_verb_result` |
| Built-in verbs | `src/verbs/builtin.jl` | column gathers, `take_rows`, `dropout`, `attention`; `batched_mul` (a primitive product) and `logsumexp` (a fold) |
| Derivatives | `src/transforms/reverse.jl` | `differentiate`, `pullback`, `local_partials`, `call_pullback`, `with_update` |
| Placement | `src/implementations/targets.jl` | `CPUTarget`, `CUDATarget`, `infer_target` |
| Implementation contract | `src/implementations/protocol.jl` | `support`, `build_implementation`, `choose_implementation`, `implementation_notes` |
| Implementation registry | `src/implementations/registry.jl` | `register!`, `activate`, `deactivate`, `default_candidates`, `baseline_implementations` |
| Implementation reverse rules | `src/implementations/reverse_rules.jl` | `reverse_rule`, `prepare_reverse_rules` |
| CPU kernels | `src/implementations/cpu.jl` | `scalar_eval`, `build_implementation` |
| Reference decomposition | `src/compiler/lowering.jl` | `resolve_semantics`, `lower_call!`, `lower_graph!` |
| Exact canonical forms | `src/compiler/canonical.jl` | `canonical_program`, `sink_reshapes`, `collapse_aliases` |
| Semantic graph regions | `src/compiler/graph_regions.jl` | `GraphRequest`, `select_graph_regions`, `region_boundary`, `standard_numbering` |
| Proposed regions | `src/compiler/proposals.jl` | `propose_regions`, `ProposedRequest`, `select_proposed_regions`, `proposal_benefit` |
| Region and storage planning | `src/compiler/planning.jl` | `plan`, `explain` |
| Implementation selection and grouping | `src/compiler/selection.jl` | `select_schedule`, `GroupedRequest` |
| Runtime hooks | `src/runtime/hooks.jl` | `make_runtime`, `capture_runtime!`, `replay_runtime!` |
| Functional state | `src/runtime/state.jl` | `validate_donation`, `prepare_state!`, `commit_state!` |
| Execution lifecycle | `src/runtime/execution.jl` | `Prepared`, `bind!`, `execute!`, `capture!`, `replay!`, `profile` |
| Concrete preparation | `src/runtime/preparation.jl` | `prepare`, `prepare_schedule`, `allocate_storage` |
| Shared scalar rules | `ext/ChainRulesCoreExt.jl` | `trace_scalar_rule`, `lift_partials` |
| Device and vendor integration | `ext/` | CUDA (with `device_rates` in `device.jl`), cuTile and cuDNN extensions, each registering from `__init__` |

## Folds

Every reduction is a `FoldReduction` (`src/frontend/folds.jl`): a fold's identity state,
element lift, associative and commutative combine, and observer, applied over
`axes`. `SumReduction` is the alias for `FoldReduction{SumFold}`; backends keep
their dedicated sum kernels behind it. The state is a tuple inside kernels
only: the observer is part of the node, so a fold node is one ordinary value.
Fold functions are scalar Julia functions; tile backends map them over lanes.

The array frontend records folds for `sum`, `prod`, `maximum` and `minimum`
(`SumFold`, `ProdFold`, `MaxFold`, `MinFold`), and `reduce`/`mapreduce` map `+`,
`*`, `max` and `min` to them, combining a given `init` with the result once.
Another operator needs `init`, its identity, and becomes a one-value
`GenericFold`. That fold is zero-size when the operator is a singleton and `init`
is zero, one, `typemin` or `typemax` of the element type, which it rebuilds from
the type; another `init` is captured, and the fold runs on CUDA kernels.

Each fold accumulates in `fold_accumulator(fold, T)`: Float32 for Float16 and
BFloat16 sums, products and logsumexps, the element type otherwise. Planning
puts the conversion into a reduction region's expression and its identity
state, so scalar kernels lift and combine in that type, and the store rounds
once; the interpreter converts before folding.

Forward execution is general. The reference interpreter and CPU kernels fold
with the fold's functions; CUDA runs a deterministic warp and block fold for
any bits state; cuTile graph regions handle zero-size folds with Float32 or
Int32 state, on every schedule, and decline folds that carry data. Partial
buffers of split and full reductions hold unobserved states, one column per
component. A fold may add a lowering for slices held in registers
(`fold_resident`); logsumexp uses it as a maximum then a shifted sum.

Derivatives are a rule table: `fold_pullback` methods for sum, product,
maximum/minimum and logsumexp, user methods for their own fold types, or a `pullback` passed to
`GenericFold`. A fold without a rule runs forward and fails at `differentiate`.
The product rule counts each slice's zeros and multiplies its nonzero elements,
so it needs no division by an element that may be zero.

## Graph invariants

Node IDs are positions in a graph's node vector. Inputs precede their consumers.
Logical values have shapes, element types, and scalar/array identity; they do not
own tensor storage. A scalar and a zero-dimensional array are different values
even though both eventually use one-element buffers.

`Node.op` is an `Operation`. Payloads belong to operations: `Pointwise.f`,
`Constant.value`, `FoldReduction.fold` and `.axes`, and the selections or permutations carried
by layout operations. `SemanticCall` retains typed callable, keyword, and result
structures. The tape remains heterogeneous host data.

`evaluate` interprets an operation; `pullback` emits its operand cotangents;
`schedule_region` lowers it to executable work. Layout operations additionally
provide `transform_layout`. Unknown operations are fusion boundaries and fail at
the missing lowering method. Native-input requirements and fusion policy have
operation methods. `test/ir/operations.jl` demonstrates an external operation with
independent interpretation, differentiation, and lowering into an existing scalar
region. These compiler methods are provisional; kernel authors should normally
use semantic verbs and implementation candidates.

A semantic call has one node per output leaf. Those nodes share one immutable
`SemanticCall`, including its complete result tree. Liveness retains all sibling
outputs when any output is needed. Lowering and reference execution process the
call once; differentiation waits until the reverse walk reaches its first output,
so all output cotangents have arrived. The `origin` token identifies the same call
across graph copies and preparation retries.

Built-in derivative recipes emit ordinary tensor operations. Where a recipe uses
a forward output, that dependency is visible to liveness and storage planning.
Custom `scalar_partials` remain opaque scalar callables. A backend supporting
only known operations may reject those callables; GPU compilability alone does
not establish vendor support.

## Planning decisions

Before planning, `canonical_program` resolves semantic calls and then removes
layout operations that move nothing: reshapes (and vector transposes) sink
through single-use pointwise work to the value they present, and a layout
chain whose composed map is the identity onto an ancestor of the same shape
is replaced by that ancestor (`src/compiler/canonical.jl`). The frontend
already applies the trace-time identities: a permutation of a permutation
composes, a matrix permutation is its transpose, a full reduction reads
through reshapes, and a reduction over a permuted array folds the source.

Preparation first collects the regions implementations propose
(`propose_regions`) and selects a non-overlapping, acyclic subset in candidate
order, then by the memory traffic each saves against the default partition
(`proposal_benefit`). It then offers connected semantic graph candidates
through `GraphRequest`, around the claimed proposals. Selection retains
original value IDs and checks that contracting all selected regions leaves an
acyclic graph. Providers see shapes and types before allocation. The boundary
requires dense logical arrays, except that a proposed region's input read only
by matrix products keeps its native transpose; layouts outside the selected
region preserve their own alias relationships.

`plan` then exposes these stages:

1. `plan_layouts` collects consumer requirements, then decides each layout and
   materialization in dependency order without cache invalidation.
2. `fusion_roots` identifies values that stop scalar inlining.
3. `schedule_steps` inserts selected graph regions, builds scalar expression trees
   and opaque/GEMM steps for remaining work, and orders their dependencies.
4. `storage_lifetimes` extends root lifetimes through every consumer and output alias.
5. `assign_slots` reuses matching type/size slots only after their last use.

Generated kernels can read arbitrary supported mappings. GEMM requires a native
dense or transpose presentation, and a batched GEMM dense operands; opaque calls currently require dense inputs.
Materializations at those boundaries are explicit schedule steps. A returned
alias keeps its storage root alive past the last launch. Sibling call outputs
cannot reuse one another's slots because they are written at the same step.

A proposal that its implementation declines in `support` leaves the next one in
order, often a nested alternative, eligible in the same pass; one that fails
construction is excluded like a graph region. See
[proposed regions](proposed-regions.md).

Graph candidates grow from reductions, bounded to 64 members. A complete structural
candidate is tried first. On metadata rejection, growth consults provider support
at each addition and leaves declined neighbors outside that attempt. Private
literals enter together with their consumers. If all constructions reject a
selected graph region, preparation excludes that exact member set and replans,
once for all regions that failed in the pass; smaller overlapping regions
remain eligible. Rewriting the semantic graph discards these exclusions
because node IDs belong to one graph snapshot. Selection is deterministic and
greedy, without a cost model. `fuse_regions=false` bypasses it, proposals included.

The cuTile graph provider (`ext/cuTileExt/graph/`) separates domain and phase
analysis (`schedule.jl`), generated tile code (`codegen.jl`), the generated kernel
entry points (`kernels.jl`), and construction (`build.jl`). A schedule is an execution representation of a selected
region: the original tensor operations and their value identities remain in the
program. Leading-axis reductions keep a column resident in one tile when it has
at most 4096 rows, packing several columns of at most 128 rows into one tile;
longer columns stream bounded tiles. Second-axis reductions stream within a
block. A single-axis reduction with too few kept tiles splits its reduced axis
across blocks, and full reductions likewise use staged partial sums with shared
workspace. Region values are numbered in a standard form (`standard_numbering`:
a walk from the outputs, with commutative binary operands ordered by
structure), so equivalent regions share one compiled kernel; cuDNN keys its
execution plans for proposed regions on the same numbering. It retains actual Julia callables and
requires `recomputable` permission before reevaluating their producers. See [graph regions](graph-regions.md) and
[the implementation guide](implementations.md).

The generated CUDA kernels (`ext/CUDAExt/indexing.jl`) resolve every read of a
region against its iteration shape at build time. Dense reads use the linear
index, affine and broadcast reads become strides over coordinates computed once
per element with 32-bit division, and gathers keep their selection behind a
strided logical index. Per-dimension helpers recurse over tuples, so index
metadata stays in registers. `ext/CUDAExt/verbs.jl` implements the built-in
verbs, `ext/CUDAExt/kernels.jl` builds the generated scalar kernels, and
`ext/CUDAExt/runtime.jl` owns device resources, capture and replay.

## Preparation and ownership

Preparation resolves the candidate list once, from the explicit
implementations, the scoped registry changes and the registry, and records each
candidate's source and `implementation_notes` in the executable. It allocates
input copies and planned slots, resolves aliases, and binds
candidate implementations to their final arrays. Candidates are tried in order.
`Supported()` means eligible for construction. `Unsupported(reason)` from either
support or construction permits fallback; unexpected exceptions propagate.
User candidates and target defaults go through the same support/build protocol.
Selected ImplementationPlans declare temporary workspace. Requests within a
launch receive distinct slots; slots may be reused by later launches. Binding
happens after capacities are final and cannot request fallback.

`Prepared` retains input buffers, slot buffers, aliases, scratch, and launch
objects. `BuiltImplementation` retains its vendor plans and other resource owners.
Construct those resources during preparation. Capture may warm execution, but a
launch must not provision workspace or construct vendor descriptors.

`bind!` changes input contents without changing addresses. Results are borrowed,
can alias other results or inputs, and are overwritten by later execution.
Executables are not safe for concurrent use. CUDA checks the original device and
context before binding, execution, capture, and replay.

The host compiler intentionally handles heterogeneous metadata and callable
objects. Type stability matters in generated scalar expressions and kernel paths;
parameterizing every compiler data structure would obscure these passes without
establishing faster kernels.

## Checking changes

Run `julia --project=. -e 'using Pkg; Pkg.test()'`, or `julia --project=test
test/runtests.jl [name prefixes...]` to run a subset. Every file below `test/`
is one test, run in parallel by ParallelTestRunner; `runtests.jl` only selects
files and sets up workers. Test directories mirror `src/`: `compiler/planning.jl`
checks explicit derivatives, preparation, and storage reuse; `ir/layout.jl`
checks mappings and pullbacks; `verbs/contract.jl` checks semantic contracts and
construction fallback. `cuda/` holds their CUDA counterparts, which check
execution and replay, and `cutile/` the tests that need cuTile. Helpers that a CPU
test shares with its GPU counterpart live in `test/fixtures/`.

Every worker loads the same packages: with a functional GPU, CUDACore, cuBLAS,
cuDNN, and cuTile unless `TEMPEST_TEST_CUTILE=false`. What core use loads, and
loading a provider after preparation, are checked in fresh processes
(`test/extensions/loading.jl` runs the scripts in `test/isolated/`).

For derivative changes, compare with finite differences, not only the interpreter
that executes the same generated derivative graph. For scheduling changes, compare
reuse on/off and check retained aliases. For implementation changes, exercise both
ordinary execution and repeated replay with rebound inputs.

See [review decisions](review-decisions.md) for the agreed priorities and experiments.

## State commits

`TensorProgram.input_structure` maps positional argument trees to flat argument
IDs. `TrainingProgram.structure` preserves the selected gradient trees.
`with_update` traces a callback on those values in a copied graph and returns a
forward program for the combined step.

State donation is planned against old-input lifetimes and scalar read mappings.
A direct output has no temporary slot and shares its owned input buffer.
Implementation selection receives `InPlaceRequest`; rejection replans that input
with staged donation. Graph regions currently use staged donation: their reads
do not inherit an elementwise same-lane overwrite contract. All donated buffers,
including direct destinations, are saved and restored around capture.

`StateBindings` owns remaining commit sources, destinations, and old-state alias
snapshots. Returned old-state roots stay live and therefore prevent direct writes.
Snapshots finish before commits, preserving swaps and observations.

`src/compiler/selection.jl` chooses implementations and groups neighboring compatible elementwise regions after physical
storage is assigned. Both output/output and output/input overlap checks are
required: logical independence alone does not exclude hazards caused by temporary
slot reuse. `GroupedRequest` uses ordinary support/build selection, with fallback
to its members. `Prepared.groups` maps physical launches back to logical regions;
provider and workspace reports remain indexed by region.
