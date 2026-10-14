# Reading and changing Tempest

Start at `prepare` in `src/preparation.jl`. It validates the input contract,
resolves semantic calls, and prepares a schedule. If an opaque implementation
passes its metadata check but cannot be built, preparation expands that call's
reference and replans. The original trace remains available for inspection.

## Source map

| Responsibility | Files | Main entry points |
| --- | --- | --- |
| Graph IR | `src/graph_ir.jl` | `Operation` payload types, `Node`, `TensorProgram`, `TrainingProgram`, `ForwardProgram`, `live_nodes` |
| Symbolic array frontend | `src/tensor.jl` | `TracedArray`, `TracedScalar`, `trace`, `validate` |
| Array frontend | `src/broadcast.jl`, `src/operations.jl` | `pointwise`, `trace_broadcast`, Base methods |
| Index mappings | `src/layout.jl` | `native_layout`, `map_index` |
| Derivatives | `src/reverse.jl` | `differentiate`, `pullback`, `local_partials`, `call_pullback` |
| Semantic contracts | `src/verbs.jl` | `record_call`, `check_verb_result` |
| Built-in verbs | `src/primitives.jl` | `batched_mul`, `logsumexp`, column gathers, `take_rows`, `dropout` |
| Reference decomposition | `src/semantic_lowering.jl` | `resolve_semantics`, `lower_call!`, `lower_graph!` |
| Semantic graph regions | `src/graph_regions.jl` | `GraphRequest`, `select_graph_regions`, `region_boundary` |
| Region and storage planning | `src/planning.jl` | `plan` |
| Implementation contract | `src/implementations.jl` | `support`, `build_implementation`, `choose_implementation` |
| Placement and runtime hooks | `src/runtime.jl` | `infer_target`, `make_runtime` |
| Concrete preparation | `src/preparation.jl` | `prepare_schedule`, `allocate_storage` |
| Functional state | `src/state.jl` | `validate_donation`, `prepare_state!`, `commit_state!` |
| Execution lifecycle | `src/execution.jl` | `Prepared`, `bind!`, `execute!`, `capture!`, `replay!`, `profile` |
| CPU kernels | `src/cpu.jl` | `scalar_eval`, `build_implementation` |
| Independent graph execution | `src/reference.jl` | `interpret`, `value_and_gradient` |
| Shared scalar rules | `ext/ChainRulesCoreExt.jl` | `trace_scalar_rule`, `lift_partials` |
| Device and vendor integration | `ext/` | CUDA and cuTile extensions |

## Folds

Every reduction is a `FoldReduction` (`src/folds.jl`): a fold's identity state,
element lift, associative and commutative combine, and observer, applied over
`axes`. `SumReduction` is the alias for `FoldReduction{SumFold}`; backends keep
their dedicated sum kernels behind it. The state is a tuple inside kernels
only: the observer is part of the node, so a fold node is one ordinary value.
Fold functions are scalar Julia functions; tile backends map them over lanes.

Forward execution is general. The reference interpreter and CPU kernels fold
with the fold's functions; CUDA runs a deterministic warp and block fold for
any bits state; cuTile graph regions handle zero-size folds with Float32 or
Int32 state, on every schedule, and decline folds that carry data. Partial
buffers of split and full reductions hold unobserved states, one column per
component. A fold may add a lowering for slices held in registers
(`fold_resident`); logsumexp uses it as a maximum then a shifted sum.

Derivatives are a rule table: `fold_pullback` methods for sum, maximum/minimum
and logsumexp, user methods for their own fold types, or a `pullback` passed to
`GenericFold`. A fold without a rule runs forward and fails at `differentiate`.

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
operation methods. `test/graph/ir.jl` demonstrates an external operation with
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

Preparation first offers connected semantic graph candidates through
`GraphRequest`. Selection retains original value IDs and checks that contracting
all selected regions leaves an acyclic graph. Providers see shapes and types
before allocation. The current boundary requires dense logical arrays; layouts
outside the selected region preserve their own alias relationships.

`plan` then exposes these stages:

1. `plan_layouts` collects consumer requirements, then decides each layout and
   materialization in dependency order without cache invalidation.
2. `fusion_roots` identifies values that stop scalar inlining.
3. `schedule_steps` inserts selected graph regions, builds scalar expression trees
   and opaque/GEMM steps for remaining work, and orders their dependencies.
4. `storage_lifetimes` extends root lifetimes through every consumer and output alias.
5. `assign_slots` reuses matching type/size slots only after their last use.

Generated kernels can read arbitrary supported mappings. GEMM requires a native
dense or transpose presentation; opaque calls currently require dense inputs.
Materializations at those boundaries are explicit schedule steps. A returned
alias keeps its storage root alive past the last launch. Sibling call outputs
cannot reuse one another's slots because they are written at the same step.

Graph candidates grow from reductions, bounded to 64 members. A complete structural
candidate is tried first. On metadata rejection, growth consults provider support
at each addition and leaves declined neighbors outside that attempt. Private
literals enter together with their consumers. If all constructions reject a
selected graph region, preparation excludes that exact member set and replans; smaller overlapping
regions remain eligible. Rewriting the semantic graph discards these exclusions
because node IDs belong to one graph snapshot. Selection is deterministic and
greedy, without a cost model. `fuse_regions=false` bypasses it.

The cuTile graph provider separates domain and phase analysis
(`graph_schedule.jl`), generated tile code (`graph_codegen.jl`), and construction
(`graph_regions.jl`). A schedule is an execution representation of a selected
region: the original tensor operations and their value identities remain in the
program. Leading-axis reductions keep a column resident in one tile when it has
at most 4096 rows, packing several columns of at most 128 rows into one tile;
longer columns stream bounded tiles. Second-axis reductions stream within a
block. A single-axis reduction with too few kept tiles splits its reduced axis
across blocks, and full reductions likewise use staged partial sums with shared
workspace. Region values are numbered in a standard form (a walk from the
outputs, with commutative binary operands ordered by structure), so equivalent
regions share one compiled kernel. It retains actual Julia callables and
requires `recomputable` permission before reevaluating their producers. See [graph regions](graph-regions.md) and
[the implementation guide](implementations.md).

The generated CUDA kernels (`ext/CUDAExt/indexing.jl`) resolve every read of a
region against its iteration shape at build time. Dense reads use the linear
index, affine and broadcast reads become strides over coordinates computed once
per element with 32-bit division, and gathers keep their selection behind a
strided logical index. Per-dimension helpers recurse over tuples, so index
metadata stays in registers. `ext/CUDAExt/primitives.jl` implements the built-in
verbs.

## Preparation and ownership

Preparation allocates input copies and planned slots, resolves aliases, and binds
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

Run `julia --project=. -e 'using Pkg; Pkg.test()'`. The test environment contains the
optional GPU packages; GPU execution is gated on `CUDACore.functional()`.
`test/planning.jl` checks explicit derivatives, preparation, and storage reuse;
`test/layout.jl` checks mappings and pullbacks; `test/verbs.jl` checks semantic
contracts and construction fallback. Their CUDA counterparts check execution and
replay.

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

`src/selection.jl` chooses implementations and groups neighboring compatible elementwise regions after physical
storage is assigned. Both output/output and output/input overlap checks are
required: logical independence alone does not exclude hazards caused by temporary
slot reuse. `GroupedRequest` uses ordinary support/build selection, with fallback
to its members. `Prepared.groups` maps physical launches back to logical regions;
provider and workspace reports remain indexed by region.
