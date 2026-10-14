# Compiler direction after the backend experiments

21 September 2026. This note records the architectural conclusions from the
compiler survey, the TensorIR execution experiment, and a small graph-boundary
probe. The final section records the resulting production integration. No
external compiler dependency was added.

## Current priorities — 23 September 2026

The workload proof exists. The [transformer training step](transformer-step.md)
trains a grouped-query-attention GPT as one captured CUDA graph. It matches its
reference implementation's loss to seven digits and runs faster than the same
model under Reactant/XLA on the same GPU. A clock-matched profile attributes
the lead to two things. The first is replaying the step's graph with no
per-step host work. It matters at the 0.8-million-parameter size, where XLA
rewrites its graph's kernel parameters every step and its GPU idles 62% of the
step (1.7× in full precision; XLA without graphs was not remeasured). Fused attention matters at GPT-2-small size (1.35× at context 256
with TF32). Without fused attention, GPT-2-small is even with XLA. XLA's
generated pointwise, reduction and FP32 GEMM kernels are as fast as or faster
than Tempest's. What the workload exposed, in order of cost:

1. **Matrix products** are half the step. Horizontal fusion of products that
   share an input (Q, K, V; gate, up) is the next graph-level change, but only
   for products of matching shape: grouping mixed shapes in one cuBLAS call
   made the step 44–53% slower. TF32 showed no training-quality cost over three
   seeds.
2. **Cold preparation** takes about 70 s, mostly serial kernel compilation;
   warm preparation takes 1 s.
3. **Schedule quality** continues against the memory-traffic bound. The
   [measurements](reduction-domains.md#Packed-columns-and-split-reductions,-23-September-2026)
   place normalization regions at 46–87% of copy bandwidth after column
   packing and split reductions.

## Keep Tempest's graph as the source of meaning

Tempest already has a tensor IR. A node records a typed operation, operand value
IDs, concrete dimensions, an element type, and scalar/array identity. Layout
operations preserve coordinate transformations. Differentiation and functional
optimizer updates emit the same graph vocabulary. Semantic calls preserve a
function's identity and can supply reference decompositions and paired reverse
rules.

This is the representation to strengthen. NVIDIA TensorIR can consume a
supported projection of it. A native cuTile provider can compile a different
projection containing actual Julia callables. Library and custom kernels can
realize semantic calls or supported larger regions. None of those providers
should determine which computations the frontend can express.

MatMul should retain its identity while optionally exposing an indexed
contraction. A named normalization or attention verb can retain both a reference
and a specialized implementation. Decomposition is a compiler choice; expanding
everything immediately would require rediscovering useful identities and can
obscure numerical differences between implementations.

The scope remains a traced, concretely shaped, functional array program. This
is not yet an IR for arbitrary Julia control flow, mutation, or effects. Those
require explicit designs if a real workload needs them.

## Shared Tile IR is infrastructure, not frontend interoperability

The inspected cuTile.jl and NVIDIA TensorIR paths independently produce CUDA
Tile IR. Their compiled kernels can share a CUDA context, arrays, streams, and
capture. There is no implemented interface for handing a Julia tile function to
TensorIR as an in-kernel epilogue, or calling a TensorIR graph as a cuTile tile
function.

A fragment interface would have to specify tile shapes, captures, indexing,
effects, synchronization, and compatible IR versions. Sharing a lower compiler
does not supply that contract. It is a possible research direction, but it is
not a prerequisite for building useful fusion in Tempest.

TensorIR's source is available and its lowering can be inspected. The opacity
in our current adapter is the granularity of the contract: it hands off a graph
and receives a launchable program. We should distinguish that limitation from
source availability. Build and binary packaging can be streamlined separately
from deciding whether its semantics and schedules fit a workload.

## Region formation was the immediate architectural gap

Before the integration below, the planner decided materialization, identified
scalar fusion roots, and constructed scalar expression trees before ordinary
provider selection.
The reduction-fusion pass then recovered shared bindings from those trees.
That worked for its deliberately small scope, but it means a more general
provider encounters graph decisions already made on its behalf.

Select candidate regions on the semantic graph first. Keep their original value
identities and derive scalar or tile expressions only for the chosen lowering.
This avoids duplicating the tensor operation vocabulary into a second general IR.

The candidate needs:

- A particular graph snapshot and a set of included operations.
- Ordered external inputs and every result used outside the region, including
  forward values retained only for backward.
- Shapes, element types, scalar/array identity, and logical index mappings.
- The relationship between escaping views and their logical storage roots.
- The numerical and mutation permissions a realization must obey.

A candidate belongs to its graph snapshot. Rewriting the graph requires
remapping or discarding the candidate; positional node IDs are not persistent
identities across arbitrary rewrites.

Four questions must remain distinct:

| Question | What establishes it |
|:--|:--|
| Can this subgraph become one scheduling unit? | Dependencies remain acyclic; all inputs and escaping outputs are accounted for. |
| Can an implementation realize its boundary? | Operations, layouts, types, numerical behavior, effects, and resource requirements are supported. |
| What execution schedule should it use? | Tile domains, reduction passes, retention, recomputation, and internal storage. |
| Is that schedule worth selecting? | Preparation cost, execution measurements or estimates, and surrounding graph costs. |

A region need not be one physical kernel. A staged reduction or a specialized
library implementation can need several launches. Conversely, independent
regions can later share one launch through the existing grouped-update path.

## Let providers supply schedules within a common boundary

The direct Julia/cuTile route is closer than a new compiler from scratch.
`ScalarCall` already preserves the actual Julia callable; the native reduction
lowering invokes that callable on tile operands. The narrow part is its current
whole-column, one-sum schedule and read restrictions.

A native provider can derive reduction completion phases from the graph, choose
whole-column retention or bounded-tile sweeps, and generate ordinary Julia loops,
loads, tile broadcasts, and reductions. cuTile remains responsible for compiling
the Julia functions and lowering tile operations to hardware. A selected
`ReductionProgram` or phase plan is a derived scheduling representation; it does
not have to become the input language of every other provider.

TensorIR can independently select schedules for the graph portion it accepts.
Its Julia adapter should expose its rejection reasons and retain explicit
boundary outputs. The successful RMSNorm integration establishes a usable
execution boundary, not a reason to replace Tempest's graph with TensorIR's
operation set.

The broader references remain useful at different points:

- **nvFuser:** propose graph segments, ask whether they can be scheduled, and
  preserve graph legality while changing boundaries.
- **Halide:** choose computation placement and storage lifetime independently.
  A shared value can be retained, recomputed, or materialized.
- **Helion:** derive and tune schedules within a selected region, including
  reduction loops and tile geometry.
- **TensorIR:** a concrete external lowering to compare with, use where helpful,
  and inspect for domain analysis, tiling, and reduction lowering.

These are complementary lessons. They do not require adding four dependencies.
Source findings and links are recorded in the [compiler landscape](compiler-landscape.md)
and [scheduler investigation](scheduler-investigation.md).

## Julia expressivity requires explicit dependencies and permissions

There are three different abilities:

1. Compile a scalar Julia callable inside a generated kernel.
2. Expose that callable's mathematical body to graph transformations.
3. Prove that repeating or moving its evaluation preserves behavior.

The first does not imply the other two. Symbolically tracing a callable can help
an optional exporter, but traced comparisons and control flow are more limited
than device compilation. Retain the callable when expansion is unavailable;
an external compiler's primitive catalog should not become a universal whitelist.

Recomputation needs its own permission. A whole-column schedule may retain a
value from a custom function; a streamed schedule may need to recompute it or
materialize it once. Device compilability and `isbits` are not purity proofs.
The initial streaming experiment should use known pure primitives plus an
explicitly declared pure fixture. A general effect-inference system is not
required to test that schedule.

Similarly, cuTile can flatten nested structures containing arrays, but that only
solves the kernel argument representation. Tempest still needs those arrays as
explicit dependencies for binding, gradients, lifetime, aliasing, and donation.
A later capture-lifting facility could turn supported captured arrays into graph
operands. Hiding pointers inside static callable metadata is not that facility.

## Logical outputs and physical ownership are different contracts

Consider a region computing `h = sin.(x)` and returning only `view(h, 1:2, :)`.
The logical output is the view. Its map refers to a root produced inside the
region. A CPU `SubArray` retains its parent automatically; a GPU execution plan
must explicitly retain the required storage or materialize the escaping view.
It cannot infer a complete allocation contract from the output value IDs alone.

If both the root and a view escape, their relationship must also survive any
permitted materialization choice. Donation requires checking all old-value uses,
including through aliases. A fused reduction reading many lanes cannot inherit
the same-lane overwrite permission of an elementwise kernel.

Saved values needed by backward are ordinary persistent graph results. Temporary
`WorkspaceSpec` storage can be reused after a launch and is not a place to hide
those values. Provider-private forward/backward state remains governed by the
existing paired reverse-rule contract.

## Evolve preparation without creating a parallel provider system

Keep `support` and `build_implementation` as the shared provider protocol. The
current construction API accepts final arrays; it is adequate for existing
wrappers but constrains decisions that should precede allocation.

The intended ordering is:

1. Resolve the chosen semantic graph and derive logical mappings.
2. Form legal candidate boundaries; ask providers about requirements and support.
3. Provisionally choose a partition and implementation plans.
4. Finalize required materializations, output ownership, lifetimes, donation,
   and value/workspace slots.
5. Allocate and bind the selected plans; then warm and capture.

Extend the existing deferred-plan idea incrementally. Concrete kernel wrappers
can keep a late-construction adapter. Begin with a conservative canonical layout
contract rather than general layout negotiation. A deliberate late rejection
can still trigger bounded replanning.

Record rejection against the candidate and implementation. Failure of
`{a,b,c}` must not disqualify `{a,b}`. Unexpected compiler defects remain visible.
Later, a supported isolated opaque kernel may compete against its expanded
reference fused with neighbors; do not add that alternative-search problem to
the first region selector.

Keep artifact compilation separate from executable binding. A reusable binary
and a captured execution with concrete buffer addresses have different lifetimes
and cache keys. Changing input contents during replay should affect neither.
Compilation caching is a measured follow-on concern, not a reason to put device
addresses or mutable data into semantic identities.

## Numerical choices remain visible

Eliminating a buffer must not silently eliminate a logical dtype conversion.
Reduction association, accumulation type, NaN behavior, and deterministic
execution need a stated policy. A variance rewrite or changed accumulation
precision is a different numerical choice from grouping existing operations.

For padded reductions, apply the validity mask to the computed contribution:
loading zero and then evaluating `exp` adds one in an invalid lane. This is a
correctness requirement for a schedule, independent of whether the backend uses
cuTile or an external compiler.

The current graph carries useful dtype and cast information, but the complete
numerical policy is still informal. Settle it through concrete mixed-precision
and reduction cases before inventing a broad public policy hierarchy.

## Boundary probe completed in this iteration

`examples/research/graph_regions.jl` references the existing graph directly. It
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

## Native region experiment completed

The follow-up now selects structural candidates automatically and lowers the
accepted matrix family to a native cuTile kernel. Its schedule comes from
reduction dependencies, including several dependent or independent sums. The
custom scalar callable remains a Julia callable inside the kernel. Irregular
259-row and long 8193-row cases pass capture and changed-input replay checks.

See [Native cuTile sweeps](cutile-sweeps.md) for the experiment, measurements,
and restrictions. This established that the semantic graph could support both
external compilation and native Julia scheduling.

## Production integration

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

## Experiments that discriminate between the options

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


## Integration review and revised priorities

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

## Broader reduction domains

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
