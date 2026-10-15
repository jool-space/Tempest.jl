# Outside review

An independent review of Tempest's compiler design, written 15 October 2026
by a reviewer who had not seen the project before, after reading the
documentation and source, running the examples and the test suite, and
measuring on the project's GB10. It covers the architecture, the spelling
invariance audit of the same day ([spelling invariance](spelling-invariance.md)),
how the design compares with XLA, MLIR linalg and IREE, TVM, nvFuser,
TorchInductor and tinygrad, whether the IR should grow or move to MLIR, what
attention epilogues would take, how the project tunes and tests, and a ranked
list of recommendations. The last section records the changes made on the
`outside-review` branch in response, with measurements.

Times are device microseconds per execution on GB10, warm, the median of seven
samples of a CUDA graph of 50 executions, Float32 unless stated. Run-to-run
variation is up to 10%; differences below that are not claimed.

## 1. Architecture

### What is sound

The core choices hold up. A concretely shaped, functional graph with typed
operations, real Julia callables in pointwise nodes, folds declared by their
algebra rather than their schedule, and layout operations that compose into
index maps is a good IR for the stated scope: a traced training step replayed
as a graph. Three decisions in particular are right and are not common in
projects of this size:

- **Regions are selected on the semantic graph before storage is planned**,
  and a region keeps original value IDs. The planner's four questions (is the
  subgraph one unit, can an implementation realize its boundary, which
  schedule, is it worth it) are stated and kept apart in the code.
- **Implementation-proposed regions** reverse the direction for libraries:
  cuDNN proposes product-plus-epilogue sets, Tempest computes the boundary
  and decides. This is the bring-your-own-codegen partitioner of XLA and TVM,
  and it is the right mechanism for the user's question about named calls
  taking epilogues.
- **Ownership is explicit** (donation, borrowed outputs, stable addresses,
  captured replay), and the fold contract (accumulate in `fold_accumulator`,
  round once, association free) is written down and tested across providers.

The reference interpreter, the verb contract (`output_spec`, `reference`,
`verb_pullback`), finite-difference tests and the plan-shape tests give the
compiler a correctness floor that many larger projects lack.

### Where order decides what a cost model should

The pipeline is a sequence of greedy decisions, each made with no view of the
alternatives:

1. A semantic call stays opaque if any implementation supports it alone
   (`resolve_semantics`), before anything sees its neighbours. The audit says
   so; after its changes this applies to attention and user verbs.
2. Proposals are ranked by candidate order first and by bytes saved second.
   Two implementations' proposals are never compared; the first candidate's
   worst proposal beats the second's best.
3. Graph regions grow greedily from each reduction in node order, bounded to
   64 members, and stop at any node that is not a pointwise operation, a
   constant or a fold.
4. The default partition then inlines what is left, with shared pointwise
   values materialized.
5. Elementwise steps are grouped before individual selection.

None of this compares a chosen plan with the plan it displaces. The clearest
consequence found in this review is not any of the audit's spelling cases:
**the cuTile full-reduction graph region was four to seven times slower than
the CUDA kernels it replaced, on every full reduction, in the default
configuration.** `sum(a .* b)` over 2²⁰ values took 61 μs on cuTile against
8.7 μs with cuTile deactivated; `sum(abs2, tanh.(x))` over 1024 × 4096 ran at
67 GB/s on a device whose copy bandwidth is 234 GB/s. The cause was mundane
(the partial-sum kernels compiled without the occupancy hint every other
cuTile kernel in the project has; fixed below), but the point is structural:
coverage was treated as profitability. A region that a provider *can* build is
built, and nothing checks that it should be. The audit attributed the 61 μs to
"the full-reduction schedule's staged partial sums" from the plan shape alone;
a comparison with the baseline would have shown a 7× loss, which no schedule
explains.

The cuDNN proposer is the one place with a cost model, and it lives inside the
provider, with constants measured on this device. That is the right place for
a library's own estimate, but the partitioner needs at least one comparison of
its own: a selected graph region against the default partition's kernels for
the same members. Section 7 says how.

### What is structurally missing

**Regions cannot read through index maps.** The planner computes an index map
for every value (`plan_layouts`) and the generated CUDA kernels read any map,
but `GraphRequest` and `ProposedRequest` promise dense boundary buffers. Every
mapped input of a region is a copy, and growth stops at any layout node. This
is the real "layout divergence" of the audit, and it is a property of the
region contract, not of a provider. It also makes every graph pass special-case
layout nodes (growth, proposals, fusion roots, cuDNN's epilogue walk, the
alias pass added below), where systems that represent views as reads with
index expressions (TorchInductor, tinygrad, linalg's indexing maps) have no
such case.

**Regions are seeded only at reductions.** A pointwise chain with two consumers
is never a graph region; it is an `Elementwise` step with one output, and a
shared pointwise value is a fusion root unless a reduction region happens to
absorb it. The transformer report records that seeding at shared pointwise
values made the step slower because of the dense boundary, which is the
previous point again.

**Two lowering vocabularies.** The compiler direction warns against
"duplicating the tensor operation vocabulary into a second general IR", yet
the schedule IR (`Read`, `ScalarCall`, `Elementwise`, `FullReduction`,
`AxisReductionRegion`) is exactly that, consumed by the CPU, CUDA and cuTile
scalar providers, while the graph-region providers consume the semantic graph
directly. cuTileExt therefore contains two code generators (`scalar.jl` and
`graph/codegen.jl`) with different coverage rules (`needs_indexing`,
`tile_expression_support`, `graph_value_support`). The scalar path is older and
should shrink to the cases the graph path cannot take: mapped reads, scatters,
integer values. Until it does, every provider feature is implemented twice or
diverges.

**Primitive versus call.** The audit's rule is honest: a primitive is an
operation every target can realize, a call has an optional realization and a
reference. That is a rule about providers, not about meaning, and it leaks:
whether `logsumexp` fuses depended on which kernels existed. The rule is
acceptable as a pragmatic default; the unsound part is that a claimed call is
never compared with its reference. Since proposals can now claim calls, the
remaining asymmetry between `*` and `attention` is only that nobody has
written the proposer.

### What is overbuilt

The provider protocol has six request kinds (`RegionRequest{R}` over five
region types, `InPlaceRequest`, `GroupedRequest`, `CallRequest`,
`GraphRequest`, `ProposedRequest`) plus reverse rules and workspace plans. For
one maintainer and three providers this is a lot of surface, and the
interface-direction page already concedes that region-level implementations
are not writable against public names. The documentation (4,800 lines for
10,000 lines of source) is thorough but layered: dated reports whose "next
step" language contradicts later pages, and three pages (compiler direction,
interface direction, spelling invariance) that each restate the architecture.
A reader cannot tell which statements are current without reading all of them.

## 2. Spelling invariance and fusion robustness

### The audit's classification

I agree with the central move: verbs whose meaning is a fold, a product or a
pointwise function should lower to those primitives at trace time
(`logsumexp`, `batched_mul`, `dropout`, `causal_fill`), and proposals should be
able to claim calls. The recomputation whitelist was a real gate and listing
Base's pure functions is the right fix. The decision not to expand matrix
products into a broadcast and a sum is also right for a project whose products
run on cuBLAS and cuDNN.

I disagree with four things.

**The spelling matrix was too narrow.** It tested recipes the project
documents, not idioms Julia users write. The commonest layout idioms in
Julia array code are `dropdims(sum(x; dims = 1); dims = 1)` followed by
arithmetic and a transpose back, `vec(h)` under a full reduction, and
`reshape(h, d, :)`. None was in the matrix. Measured before this branch:

| computation | spelling | plan | μs |
|:--|:--|:--|--:|
| RMSNorm, 1024 × 4096 | `(1, n)` statistics | 1 cuTile region | 138 |
| | `dropdims(…; dims = 1)` then `'` | region, cuTile pointwise, CUDA pointwise | 198 |
| | `vec(…)` then `'` | the same | 191 |
| its training step | `(1, n)` / `dropdims` | 4 / 8 steps | 870 / 965 |
| `sum(abs2, ·)`, 1024 × 4096 | `tanh.(x)` | 1 region | 251 |
| | `vec(tanh.(x))` | pointwise, region | 374 |
| | `reshape(tanh.(x), 4096, :)` | the same | 383 |

These are 40–50% penalties for spellings that differ by a free reshape.

**The diagnosis of the dot product was wrong**, as described above. "61 against
10 μs" was recorded as a schedule limitation; it was a missing compiler hint,
and the same hint was missing from every full reduction and every split
second-axis reduction in the provider.

**"Layout operations end graph regions" was filed under provider coverage.**
It is the region contract (dense boundaries) and the graph passes (stopping at
layout nodes), and it affects cuDNN's proposals in the same way. The audit's
"most rows are coverage of one provider" undercounts the structural share.

**The exact-canonicalization subset is larger than "worth a pass".** The audit
named two rewrites and deferred them. Composing permutations, treating a
matrix permutation as a transpose, reading full reductions through reshapes,
folding the source of a permuted reduction, sinking reshapes through pointwise
work and collapsing identity alias chains are all exact, all cheap at trace or
prepare time, and together they remove the whole layout class above and the
permuted softmax (367 → one region). Implemented below.

### What the remaining structural problem is

After this branch, the divergences that remain are, in order of importance:

1. **Dense region boundaries.** Any mapped input (a transpose read by a
   reduction, a strided slice, a gather) is copied before a graph region, and
   a reshape between pointwise values of different shapes that cannot sink
   (because the value has other users, or the consumer reduces over its own
   axes) splits the region.
2. **No profitability comparison.** A graph region is built whenever it can
   be. Rank-N collapse (below) exposed this at once: a last-axis softmax of a
   rank-3 array collapsed to a 262144 × 16 domain where the second-axis
   schedule's tile did no work per block, and lost to the CUDA kernels until
   its tile was retuned.
3. **Reduction-seeded, single-output regions** for pointwise work.
4. Provider coverage: cuTile's Float32-only graph values and missing integer
   and position support; cuDNN's Base-only activation table; interior
   reduction axes.

Rank and full reductions were provider coverage and are no longer divergences.

### What I would do, in order

1. The exact rewrites (done on this branch).
2. Rank-N domains (done).
3. Fix the tuning regressions the comparison revealed (done for the
   occupancy hint and the second-axis tile).
4. Let graph regions take affine-mapped inputs: cuTile loads with strides,
   CUDA already does; `dense_region_input` asks the provider instead of
   assuming. This removes the copy class and lets growth cross transposes.
5. A profitability check for graph regions against the default partition.
6. Pointwise-seeded, multi-output regions once 4 makes their boundaries free.
7. A public callable-to-mode hook so cuDNN accepts user activations.

## 3. Prior art

**XLA.** HLO fusion is greedy producer-consumer fusion with a per-pair cost
analysis (`FusionDecision`), multi-output fusion, and an algebraic simplifier
run to a fixed point before fusion. Reshapes that do not move data are
`bitcast`s that fusion sees through. Custom calls are opaque with declared
operand layouts, and layout assignment runs before fusion so copies are
explicit. Take: a simplifier pass of exact rewrites before partitioning (begun
here), bitcast transparency (the alias pass), and a per-decision cost check,
however crude. Do not take: XLA's layout assignment as a global pass; Tempest's
per-value maps are finer and already exist.

**MLIR linalg and IREE.** A named op (`linalg.matmul`) *is* a `linalg.generic`
with fixed indexing maps; passes treat it generically and keep the name for
library lowering. IREE forms dispatch regions by picking roots (reductions,
contractions) and fusing producers and consumers by predicates
(`isFusableWithConsumer`), and it runs explicit passes that bubble
`tensor.expand_shape` up and sink `tensor.collapse_shape` down so reshapes do
not block fusion. Tempest's regions are dispatch regions; its `IndexMap` on a
read is an indexing map; the difference is that IREE's regions accept the
map and Tempest's require a dense buffer. The reshape-sinking pass on this
branch is IREE's `FoldReshapeOpsByExpansion` in miniature. Take: indexing maps
on region inputs. Do not take: a generic op as the product's node (the audit's
conclusion stands for a library-backed product).

**TVM Relax and TensorIR.** Fusion is by operator pattern class (elementwise,
broadcast, injective, communicative reduce, out-elementwise-fusable), with
"out-elementwise-fusable" being exactly a product that takes a pointwise
epilogue. Schedules are then chosen per fused function by MetaSchedule with a
learned cost model and a per-target tuning database. Take: pattern classes as
a uniform fusibility rule (Tempest has them implicitly in `fusion_boundary`,
`graph_operation` and `proposal_operation`; one table would replace three), and
the idea of a per-device tuning record. Do not take: a search-based tuner as a
prerequisite; Tempest's schedules are few and parametric.

**nvFuser.** Segmentation tries to schedule the whole fusion with one of a
small family of schedulers (pointwise, reduction, normalization, matmul),
and when none accepts, merges neighbours greedily while a scheduler still
accepts. Tempest's support-guided growth is this, restricted to reduction
seeds. nvFuser's launch parameters derive from device properties and
heuristics tuned per architecture, which is the same tuning exposure Tempest
has. Take: seed anywhere, not only at reductions, and keep the
"can a scheduler take this?" question (which Tempest already has) as the only
provider-facing question.

**TorchInductor.** Views are index-expression substitutions in a loop-level
IR, so reshape and permute never block fusion; the scheduler fuses by a score
that counts bytes saved (Tempest's `proposal_benefit`), and it keeps GEMM
candidates (cuBLAS, Triton templates with fused epilogues) and picks by
autotuning at compile time. The retained-candidates point matters for the
user's question: Inductor does not decide "matmul-plus-bias is a library call"
by rule; it benchmarks the fused template against the library plus an
epilogue kernel. Take: reads through index expressions; an empirical check
when two realizations of the same members exist. Do not take: Triton templates
as the fusion vehicle; cuTile is the project's.

**tinygrad.** One small `UOp` vocabulary, movement ops folded into a
`ShapeTracker` on every read (so they are free and never block fusion),
canonicalization by a pattern-rewrite engine, kernels formed where values are
realized, and BEAM search per kernel shape with a cache for the constants
nobody wants to hand-tune. Its matmul is a reshape, a multiply and a reduce
that the scheduler fuses into one kernel; nothing is materialized, because the
reduce absorbs the broadcast. The audit's reason for rejecting that form, that
"the broadcast is materialized when nothing does", describes Tempest's
partitioner, not the form. The real reason to keep `MatMul` named is that
cuBLAS and cuDNN exist and beat generated code; that reason holds. What
Tempest should take from tinygrad is not the einsum but the ShapeTracker: index
maps on every read, including region inputs, and a rewrite engine for exact
identities. BEAM is the honest answer to section 6.

**On the transparent indexed op.** The audit rejected it for the product and
I agree for the product. But the question the user asked, why fusion depends
on whether a thing is a named call, has a different answer than the audit's:
it depends on it because the region contract reads dense buffers and the
graph passes stop at non-primitive nodes. A transparent *read* (an index map
the region accepts) is the structure that is missing, and it is strictly
smaller than a transparent contraction. The audit's own text lists it as what
"it lacks" and then concludes nothing is needed.

## 4. IR expressiveness and MLIR

The IR should grow in these directions, roughly in order: index maps on
region inputs (exposing what the planner has); region seeds at any node and
several outputs per region; a small fixed-point rewrite framework for exact
identities (two passes on this branch are its start, written by hand); integer
and position values inside cuTile regions; a numerics policy recorded on
nodes, as the interface-direction page proposes; and, when a decode workload
needs it, dynamic extents in index maps rather than symbolic shapes
everywhere. It should not grow control flow, mutation or a generic
contraction.

MLIR as the core would cost what makes Tempest distinctive. `Pointwise{F}`
holds an arbitrary Julia callable, user folds are Julia functions, scalar
partials are Julia closures, and cuTile compiles Julia tile code. An MLIR core
would have to lower every callable to `arith`/`math` ops (losing user
functions) or wrap them opaquely (losing fusion). Nothing in the current
problems (dense boundaries, greedy order, tuning) is caused by the IR's host
language. As an export, a projection of the known-function subset to StableHLO
or linalg would be a useful *test oracle* and a way to compare against XLA
and IREE on the same graph; the project already did a version of this with
TensorIR and learned what it needed to. Build an exporter when there is a
consumer, not before.

## 5. Attention epilogues

The user asked whether it is trivial to add an epilogue to a cuTile attention
kernel. Mechanically, in Tempest, it now is: `propose_regions(::CuTileKernels,
…)` would match each live `attention_forward` call, add the pointwise users of
`o` (not of `lse`), offer every prefix, check in `support` that each function
is `recomputable` and compiles for cuTile, and in construction specialize the
forward kernel on the epilogue's standard numbering, applying it to
`acc ./ l_i` before `store_head!` with any extra operands loaded as head tiles.
That is on the order of 150 lines, most of them reusing `graph/codegen.jl`'s
value emission. The kernel cache keys on the numbering, as graph regions do.

The honest part of the answer is about payoff. In training, backward reads
`o` (for `delta`), so the boundary analysis makes `o` an escaping output
whether or not the epilogue is fused; the fused kernel then writes `o` and
`f(o)` and saves only one read of `o`. In inference it saves a read and a
write of `o`. And in a transformer the operation after attention is the output
projection, a product, not pointwise work; the pointwise work (RoPE, the head
permutations, scaling) is *before* attention, on `q` and `k`. A proposal can
include producers as well as consumers, so the useful fusion is a prologue:
apply RoPE while loading the `q` and `k` tiles. The protocol supports it today;
it is the kernel that needs writing. Neither is the highest-value item in
section 7.

In a cuTile kernel in isolation, an epilogue is a callable argument and extra
array arguments; cuTile compiles the specialization. The difficulty is not the
kernel but the contract: which values escape, which must be recomputed in
backward, and whether the saved bytes exceed the extra register pressure in
the attention loop. Tempest's boundary computation answers the first two.

## 6. Tuning and evidence

Constants tuned on one GB10 (48 SMs, unified memory, power-capped clocks):
resident row limits, packed tile sizes, split thresholds, streamed tile
heights, occupancy hints, warp-sum limits, cuDNN's fused efficiency and L2
fraction, and the derived rate tables. The project is unusually disciplined
about recording the measurements next to the constants. It is not disciplined
about keeping them true: the occupancy hint was measured for resident kernels
and the measurement note generalized ("streamed kernels take no hint") without
measuring the full-reduction kernels, which then shipped 4–5× slow.

The test suite (6,842 tests, 2.5 minutes) tests correctness, gradients by
finite differences, lifetimes, replay with rebound inputs, provider
equivalence, and plan shapes for documented recipes. It tests nothing about
speed. A plan-shape test passed while the plan it asserted was seven times
slower than the alternative. The benchmarks exist (`examples/benchmarks`) but
are run by hand and their results live in dated reports.

What a project in this position should do:

- **Assert ratios, not plans.** A benchmark file that runs on a GPU and fails
  when a cuTile region takes more than, say, 1.3× the CUDA kernels on the same
  members, or when a memory-bound region falls below a fraction of measured
  copy bandwidth. It need not run in CI; it needs to run before a commit that
  touches a provider, and `git log` should show that it did.
- **Key constants by device.** `device_rates` already does this for cuDNN
  with a table by compute capability and a measured fallback. The cuTile
  constants should follow: a table keyed by compute capability and SM count,
  with the GB10 row as today's values, and a measured fallback that times two
  or three candidates once per device and process and records the choice in
  `explain`. The determinism argument against timing is about plans varying
  run to run; a persisted per-device choice does not vary.
- **Log the clock.** The transformer report says the GB10 runs between 650
  and 950 MHz under load. Every table in the docs that lacks a clock is
  comparable only with itself.
- **Compare against the baseline the project already has.** `deactivate(cuTile)`
  is one line. Every cuTile schedule should have a row against the CUDA
  kernels; the second-axis schedule's unsplit case is at 60% of copy
  bandwidth and ties the CUDA kernels, which nobody had written down.

## 7. Recommendations

Ranked by value against cost. Costs are for one engineer who knows the code.

1. **Region inputs through affine index maps** (medium: one to two weeks).
   Removes the copy before every transposed or sliced region input and lets
   growth cross layout nodes. cuTile loads take strides; CUDA kernels already
   read maps; `dense_region_input` becomes a provider question.
2. **A profitability check for graph regions** (medium: a week). Compare a
   selected region against the default partition's kernels for the same
   members, by a bytes-and-launches model first and by a measured per-device
   table where the model is uncertain. This is the one cost comparison the
   partitioner lacks, and it would have caught both regressions found here.
3. **Performance regression benchmarks with asserted ratios** (small: days).
4. **Pointwise-seeded, multi-output graph regions** (medium-large), after 1.
5. **Device-keyed tuning tables with a measured fallback** (small-medium).
6. **One operation-class table** for `fusion_boundary`, `graph_operation`
   and `proposal_operation` (small), so a new operation is classified once.
7. **cuDNN activation hook** mapping user callables to pointwise modes
   (small). Today a user's `relu(v) = max(v, zero(v))` costs a kernel.
8. **Fold the scalar-region providers into the graph path** (large, incremental):
   stop extending two code generators.
9. **Attention prologue (RoPE) as a cuTile proposal** (medium; payoff in TF32
   training only), before any epilogue.
10. **`Statistics.mean`, `var`, `std`** on traced arrays (tiny). `mean` on a
    traced array currently fails with a scalar-indexing error.
11. **Documentation consolidation** (small, ongoing): one current architecture
    page, with the dated reports as history.
12. **MLIR export**: not now.

## 8. Work done on this branch

Two code commits, each with tests and measurements, on `outside-review` off
`spelling-audit`. The suite passes on each commit.

**Occupancy hints for partial-sum kernels** (`ext/cuTileExt/graph/build.jl`,
commit `0378422`, with the rank-N change).
The kernels of full and split reductions now compile for 8 blocks per SM;
unsplit streamed kernels keep none, since any hint slowed them by 12–40%.
Second-axis unsplit tiles grew from 32 to 128 rows, measured on four shapes.

**Exact canonical forms** (`src/frontend/operations.jl`, `src/frontend/folds.jl`,
`src/compiler/canonical.jl`, commit `b092508`). While tracing: `permutedims` composes with a
permuted operand and spells a matrix permutation as `transpose`; a full
reduction reads through reshapes and vector transposes; a reduction over a
permuted array folds the source and permutes the result. Before planning:
reshapes sink through single-use pointwise work to the value they present,
and layout chains that compose to the identity onto an ancestor of the same
shape are replaced by that ancestor. The trace keeps Base's distinctions (a
transposed vector's transpose is the vector); the prepared program does not
carry nodes that move nothing.

**Rank-N domains for cuTile graph regions** (`ext/cuTileExt/graph/schedule.jl`,
`build.jl`). A fold input of any rank whose reduced axes are all, leading or
trailing collapses into a matrix; every region value must collapse the same
way or the region is declined and the CUDA kernels run it. Equivalent regions
of different rank share one compiled kernel.

Before is the `spelling-audit` branch; after is this branch, measured after
the suite passed on it (commits `0378422` and `b092508`; 6,937 tests).

| computation | before | after | note |
|:--|--:|--:|:--|
| `sum(a .* b)`, 2²⁰ | 60.8 | 14.5 | CUDA kernels 8.7, GEMM spelling 7.3 |
| `sum(abs2, tanh.(x))`, 1024 × 4096 | 256 | 44 | |
| `sum(abs2, vec(tanh.(x)))` | 374 | 45 | one region |
| `sum(abs2, x)`, 4096 × 4096 | 1294 | 284 | 237 GB/s, copy bandwidth |
| full `logsumexp`, 2048 × 2048 | 473 | 211 | |
| full maximum and shifted sum, 1024 × 4096 | 468 | 104 | two phases |
| RMSNorm, 1024 × 4096, `(1, n)` statistics | 138 | 143 | |
| the same, `dropdims` spelling | 198 | 144 | one region |
| the same, `vec` spelling | 191 | 145 | one region |
| its training step, `(1, n)` / `dropdims` | 870 / 965 | 649 / 759 | the gap remains in training (see below) |
| row softmax, 1024 × 4096, `dims = 2` | 210 | 174 | 128-row tiles |
| the same, permuted spelling | 367 | 175 | one region |
| row softmax, 65536 × 64 | 207 | 154 | |
| row softmax, 4096 × 4096 | 933 | 897 | still 150 GB/s |
| split row sums, 128 × 16384 | 22.5 | 8.2 | |
| softmax, rank 3, 1024 × 256 × 16 | 183 | 138 | reshaped rank-2 spelling 141 |
| naive softmax, rank 3 | 328 | 150 | |
| softmax over the last axis, rank 3 | 318 | 154 | CUDA kernels 306 |
| layer norm with weight, rank 3 | CUDA kernels | 144 | |
| attention reference forward, Float32, 64 × 12 × 1024 × 4 | 7961 | 5239 | softmax over 1024 × 49152 in one region |
| the same, causal | 7850 | 6975 | |
| the same, causal, training | 19544 | 20380, 20425 | 4% slower in two runs; the backward's three regions and the extra kernel launches, within the stated variance but consistent |
| GPT example step (128 hidden, 4 layers, full precision), ms | 2.18, 2.17 | 2.19, 2.19 | 397 → 406 steps, 29 → 37 graph regions; unchanged within noise |

The GPT step did not move because its reductions are small and its products
dominate; the rank-N collapse added graph regions without changing the time.
The training step of the composed causal attention is the one measured
regression, 4% in two runs; its backward now runs three cuTile regions
over score-sized arrays where it ran CUDA axis kernels, and the second-axis
and split schedules behind them are the ones section 6 calls under-tuned.

What was not done, and why: the profitability check (recommendation 2) and
mapped region inputs (1) are each a week of work with design decisions the
maintainer should make; the `Statistics` extension is trivial but adds a weak
dependency, which is the maintainer's call; the second-axis schedule remains
at 60% of bandwidth on long rows and deserves its own measurement pass.
