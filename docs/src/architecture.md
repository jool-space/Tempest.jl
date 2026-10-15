# Architecture

This page describes Tempest as it is: what it compiles, the stages a program
goes through, the rules that decide what fuses, and what is open. It is the
current reference; the pages under Reports are dated records, and where they
disagree with this page, this page is right.

## What Tempest is

Tempest compiles a traced, concretely shaped, functional Julia array program
into an explicit backward graph, fused kernel regions and planned storage, and
replays it, usually as one captured CUDA graph per training step. Model code is
ordinary Julia array code: broadcasting, reductions, matrix products, views,
reshapes and a few verbs (`logsumexp`, `attention`, `dropout`, the gathers).
Nothing in it names a backend.

The scope is deliberately narrow. Shapes are concrete; there is no control flow
on traced values, no mutation, no symbolic shapes. Within that scope, Tempest
owns the semantic graph, differentiation, functional state updates, region
boundaries, storage lifetimes and capture. Implementations, built in or from
other packages, realize the regions; the compiler decides which regions exist.

The core package depends only on LinearAlgebra. CUDA (`CUDACore` and
`cuBLAS`), cuTile, cuDNN, ChainRulesCore and Statistics each enable an
extension.

## The pipeline

A program passes through these stages, in this order. The files are under
`src/` unless noted; the [source guide](internals.md) maps them in more detail.

1. **Trace** (`frontend/`). `trace(f, examples...)` runs `f` on `TracedArray`
   and `TracedScalar` stand-ins, which record typed operations: `Pointwise`
   with the actual Julia callable, `FoldReduction` with a fold's algebra,
   `MatMul` and `BatchedMatMul`, layout operations (reshape, transpose,
   permutation, view, slice), `Positions`, constants, and semantic calls.
   Exact identities are applied while tracing: a transpose of a transpose,
   `identity` and conversions to the same type, multiplication by a literal
   one, constant folding, composition of permutations, a full reduction read
   through reshapes, a reduction over a permuted array folding its source. Verbs
   whose meaning is a primitive lower to it here (`logsumexp` is a fold,
   `batched_mul` a product, `dropout` and `causal_fill` pointwise functions of
   positions; Statistics' `mean`, `var` and `std` are sums).
2. **Differentiate** (`transforms/reverse.jl`). `differentiate(program; wrt)`
   emits the backward graph in the same vocabulary, from per-operation
   pullbacks, fold rules (`fold_pullback`), scalar partials (built in,
   `scalar_partials`, or ChainRules rules) and verb pullbacks.
   `with_update` appends a functional optimizer step. The result is one
   `TrainingProgram` or `ForwardProgram` graph.
3. **Canonicalize** (`compiler/lowering.jl`, `compiler/canonical.jl`), inside
   `prepare`. Implementation reverse rules may first replace a forward and
   backward call pair with private calls (cuDNN's attention). Each remaining
   semantic call is offered alone to the candidates: if one supports it, it
   stays a node; otherwise its reference is traced in its place. Then reshapes
   sink through single-use pointwise work to the value they present, and layout
   chains that compose to the identity are replaced by their ancestor.
4. **Regions** (`compiler/proposals.jl`, `compiler/graph_regions.jl`). First,
   implementations propose node sets they can realize (`propose_regions`), such
   as a matrix product with its epilogue. Proposals are ranked by candidate
   order, then by the bytes they keep out of memory, and accepted greedily when
   they do not overlap, keep the partition acyclic and are supported. Then
   graph regions grow from each reduction through pointwise operations and
   constants, up to 64 members, around the accepted proposals; the complete
   candidate is offered first, then support-guided growth. See
   [graph regions](graph-regions.md) and [proposed regions](proposed-regions.md).
5. **Selection by estimate** (`compiler/costs.jl`). A supported region is kept
   only when it is not estimated to be slower than the default kernels it
   displaces; see [below](#Selection-by-estimate).
6. **Default partition and storage** (`compiler/planning.jl`). What no region
   claimed is planned as scalar regions: fusion roots (reductions, products,
   calls, materialized layouts, results, pointwise values used more than once)
   become steps, and everything else is inlined into their scalar expression
   trees, read through index maps. Layouts are decided per value, materializing
   only where a consumer needs a native or dense presentation. Lifetimes,
   slot reuse and state donation (in place where the planner proves same-lane
   reads, staged otherwise) follow.
7. **Implementation selection** (`compiler/selection.jl`). Each step is offered
   to the candidates in order; consecutive independent elementwise steps are
   first offered as one group. A construction rejection of a region excludes
   that member set, of a call expands its reference, of a donation stages it;
   preparation then replans.
8. **Execute and capture** (`runtime/`). `execute!` runs the launches in order;
   `capture!` records them as a CUDA graph and `replay!` launches it. Inputs are
   owned copies (`bind!` updates them in place), outputs are borrowed buffers,
   and every address is fixed at preparation. `explain` prints the plan, every
   decision and its reason; `Tempest.profile` ranks the launches by device time.

## Primitives and calls

A **primitive** is an operation every target can realize: pointwise
functions, folds, products, layout operations, positions. A **call** is an
operation with an optional realization and a reference: attention, the index
verbs (`take_rows`, `put_rows`, `gather_columns`, `scatter_columns`) and user
verbs ([semantic verbs](verbs.md)). A named verb whose meaning is a primitive
lowers to it at trace time and keeps its name only as the Julia function.

A call that some implementation supports alone stays one node; otherwise it
expands into its reference before regions form, and the reference fuses like
any other code. A proposal may contain calls with all their results, so a
library can claim `{attention, the pointwise work on its output}` the way cuDNN
claims `{product, bias, activation}`. Two consequences are accepted for now: a
claimed call is never compared with its expanded reference (the implementation
is responsible for declining where it would lose), and a claim takes all of a
call's results or none.

Products stay named nodes rather than broadcast-and-sum contractions: cuBLAS
and cuDNN realize them, and no provider schedules a fused contraction.

## Implementations and the registry

An implementation is a value of a subtype of `Tempest.Implementation` that
answers `support(implementation, request)` and, when supported,
`build_implementation`, which returns a launch, a plan that asks for shared
workspace, or a reasoned rejection. Requests describe the work and its buffers
without contents: `RegionRequest` (a scheduled scalar region, product or
scatter), `InPlaceRequest`, `GroupedRequest`, `CallRequest` (a semantic call),
`GraphRequest` (a graph region) and `ProposedRequest` (a proposal). The
[implementations guide](implementations.md) documents the protocol.

| Implementation | Package | Rank | Realizes |
| --- | --- | --- | --- |
| `CuDNNKernels` | cuDNN, inactive until activated | 0 | attention forward and backward; proposes half-precision (or TF32) products with their epilogues |
| `CuTileKernels` | cuTile | 10 | graph regions (Float32 reductions with pointwise neighbours), dense scalar regions, attention under TF32 |
| `CUDAKernels` | baseline of `CUDATarget` | 20 | every scalar region, mapped reads, scatters, the index verbs |
| `CuBLASGemm` | cuBLAS | 30 | products and batched products |
| `CPUKernels`, `LinearAlgebraGemm` | baseline of `CPUTarget` | 20, 30 | everything on the CPU |

Candidates are ordered by rank, then by name, never by load order. Candidates
passed to `prepare` come first, a scope's `activate`/`deactivate` overrides the
registry, and each `prepare` resolves its list once. A package registers its
implementation from `__init__` with `Tempest.register!`.

## Selection by estimate

"Can fuse" is not "should fuse". Before a graph region, or a proposal whose
implementation estimates its launches, displaces the default partition's
kernels for the same members, both are estimated and the region is kept unless
it is estimated to be more than 15% slower.

Implementations describe launches, not times: `region_cost(implementation,
request)` returns one `KernelCost(bytes, parallel)` per launch, the bytes it
moves and the bytes its threads or tiles request at once. The target prices
them with `cost_rates(target)`:

```
seconds = Σ launch_seconds + bytes / min(bandwidth, parallel / latency)
```

By Little's law a kernel with `parallel` bytes in flight moves at most
`parallel / latency` bytes per second. That term makes a CUDA kernel with one
warp per output slow on a few long columns, a completion that loops over many
partial sums in one block slow, and a schedule with too few blocks slow
anywhere.

| Input | Value on CUDA | Source |
| --- | --- | --- |
| bandwidth | 232.1 GB/s on GB10 | `device_rates`: memory clock × transfers per clock × bus width × 0.85, from a table by compute capability; measured by a copy on an unknown architecture |
| latency | 0.45 μs | fitted on GB10 to two low-parallelism kernels, which agree |
| launch | 1 μs | spacing of kernels in a captured graph on GB10 |

The default partition's estimate is the sum over its steps whose outputs are
region members, each priced by the first candidate that supports it
(`CUDAKernels` or cuTile's scalar kernels). A region input that the default
partition would inline costs the region a write and a read back, unless an
earlier region writes it anyway. `explain` prints the rates once and a line per
region. For row sums over 128 × 16384, where the CUDA kernel has 32 lanes for
each of 128 rows and cuTile splits the rows across 256 blocks:

```text
semantic region estimates: memory at 232.1 GB/s (derived: 8.53 GT/s × 256 bits × 0.85), 0.45 μs latency (102 KiB in flight to saturate it), 1.0 μs per launch; a region is kept unless estimated more than 15% slower than the default kernels
semantic graph region [2] kept by estimate: 39.6 μs on cutile (3 launches, 8.0 MiB) against 231.4 μs on cuda (1 launch, 8.0 MiB)
```

They measured 7.7 and 77 μs, warm, with the 8 MiB input in L2.

The model prices every byte at memory bandwidth and every read as coalesced,
and it cannot see code generation: a kernel compiled with too many registers,
like the full reductions before their occupancy hint, estimates as well as a
good one. The 15% margin covers that resolution: on the guard rails below,
kept regions estimated at up to 1.13 times the default kernels measured as
fast as them, and the one region estimated at 1.20, the backward softmax of
the composed causal attention, was 4% slower; declining it brought that
training step from 20.2 to 19.3 ms. The CPU target has no rates and keeps
every supported region. cuDNN keeps its own estimate inside `support` (fused
product time against cuBLAS plus an epilogue pass; see
[proposed regions](proposed-regions.md#Where-fusion-pays)).

## Performance checks

Plan-shape tests (`test/cutile/spellings.jl` and others) check that equivalent
spellings reach the same plan. They cannot see speed: a plan-shape test passed
while its plan was seven times slower than the CUDA kernels it displaced.
`test/benchmarks/guard_rails.jl` asserts relative speed instead. For thirty-three
cases (full, row, column, split and packed reductions, Float16 sums, softmax,
logsumexp and norm spellings, rank-3 domains, and training steps of RMSNorm,
softmax, an MLP block and the composed causal attention) it times four plans:
the default, the CUDA kernels', every supported region kept, and none formed.
Each time is the minimum over fifteen launches of a captured graph of a few
milliseconds of executions, with the plans' launches alternating:

- the default plan must take at most 1.25 times the CUDA kernels' plan
  (`Tempest.deactivate(cuTile)`), plus 0.5 μs;
- a forward case larger than twice L2 must move its single-pass bytes at half
  the measured copy bandwidth or more;
- the cost model's choice must be within 10% of the faster of every region kept
  and none formed, in all but one case.

Timing needs the GPU alone, so the file runs only when named:
`julia --project=test test/runtests.jl benchmarks` (about two minutes). Run it
before committing a change to a provider's schedules or to region selection,
and on a new device. Warm cases of a few tens of microseconds vary by up to
25% between runs on GB10, which the limits allow for.
`examples/benchmarks/` holds the longer benchmarks the reports cite.

## Numerics as it stands

There is no numerics policy object yet; these are the rules every
implementation follows:

- **Element types are meaning.** Every pointwise conversion stays explicit, and
  eliminating a buffer never eliminates a rounding.
- **Folds** may associate in any order (the `Fold` contract), and accumulate in
  `fold_accumulator(fold, T)`: Float32 for Float16 and BFloat16 sums, products
  and logsumexps, the element type otherwise; the result rounds once. Every
  schedule fixes its block counts at preparation, so a plan's results are the
  same on every replay; different plans and different providers may differ by
  rounding.
- **Products** of half-precision operands accumulate in Float32. TF32 is used
  only under `CUDATarget(matmul_precision = :tf32)`, and Float32 attention on
  cuDNN only with `CuDNNKernels(float32_cast = ...)`; `explain` names both.
- **Approximations** are excluded rather than chosen: cuTile has no verified
  lowering of `log1p`, `expm1`, `asin`, `acos` or one-argument `atan`, so
  regions with them run on CUDA, as do regions with Float64 literals, which
  cuTile's promotion could narrow.
- **Determinism.** CUDA reductions combine in a fixed order, scatter-add
  pullbacks accumulate in source order, and cuDNN graphs are built with
  deterministic engines.
- Statistics' `mean` over `dims` divides each sum by the slice length, where
  Statistics multiplies by its reciprocal; results can differ by one rounding.

## Interface principles

The caller states what only the caller knows: what to compute (element types
included), which transformation (`differentiate`, `with_update`), ownership
(`donate`, `bind!`, borrowed outputs), facts the compiler cannot prove (custom
reverse rules, `Fold` declarations, `recomputable`, seeds) and resources (the
target). Everything that steers the compiler (choosing implementations, tile
and occupancy constants, the cost model's rates, `fuse_regions`, `inplace`,
`grouped`) exists because the compiler is imperfect, and should stay out of
model code. `activate`/`deactivate` is the escape hatch at the level of
packages; implementation types are for implementation authors.

## Open directions

In rough order of value against cost. This list replaces the "next step"
sections of the dated reports.

1. **Region inputs through affine index maps.** Region boundaries are dense
   buffers, so a transposed, sliced or permuted input of a graph region or
   proposal is copied first, and growth stops at layout operations. The
   planner already has the maps and the CUDA kernels read them; cuTile loads
   would take strides, and `dense_region_input` would ask the provider.
2. **Pointwise-seeded, multi-output graph regions**, after 1: a pointwise chain
   with several consumers is an elementwise step with one output today. Seeding
   at shared pointwise values was tried and lost while inputs were dense.
3. **Second-axis reductions on long rows.** The streamed schedule reads its
   input once per phase and once for its outputs; row softmax over 4096 × 4096
   moves those bytes at copy bandwidth but reaches 67% of the single-pass
   bound. A single pass needs rows resident across a block.
4. **Cost model reach.** Its constants were fitted on GB10; latency and launch
   spacing need checking on other devices, and caches, coalescing and serial
   completions are not modelled. A claimed call is not compared with its
   reference, and cuDNN's estimate is separate from `region_cost`.
5. **Device-keyed tuning.** Tile sizes, occupancy hints and split thresholds
   were measured on GB10's 48 SMs. A table keyed by compute capability and SM
   count, with a measured fallback recorded in `explain`, as `device_rates`
   does for rates.
6. **One operation-class table** for `fusion_boundary`, `graph_operation` and
   `proposal_operation`, so a new operation is classified once.
7. **Library coverage.** A hook mapping user callables to cuDNN pointwise modes
   (a user's `relu` costs a kernel today); cuDNN proposals anchored on
   `BatchedMatMul` and across a layout operation on the product; cuBLASLt
   epilogues; a cuTile product with an epilogue; an exact cover of proposals.
8. **One code generator for cuTile.** Scalar regions (`ext/cuTileExt/scalar.jl`)
   and graph regions (`graph/codegen.jl`) have different coverage rules; the
   scalar path should shrink to what graph regions cannot take: mapped reads,
   scatters, integer values.
9. **cuTile positions and integer boundaries**, so dropout joins graph regions
   and `take_rows` could become a positional select and a sum, fusing
   cross-entropy; a runtime-index read so `gather_columns` inlines.
10. **Attention.** A RoPE prologue for the cuTile kernel as a proposal;
    epilogues through `propose_regions`; recognizing a spelled-out attention;
    the composed attention's score-sized pointwise passes in full precision.
11. **Products.** Horizontal fusion of same-shape products sharing an input
    (Q, K, V; gate, up): grouping mixed shapes was much slower. Float32
    products at small shapes, where cuBLAS picks split-K SIMT kernels.
12. **Interface.** One numerics policy object read by `prepare` and honoured by
    every implementation; integer scalar arguments for varying values such as
    a decode position; `prepare` deciding capture and absorbing
    `prepare_forward`; a public surface for region-level implementations once
    the IR settles.
13. **Preparation time.** Cold preparation is dominated by serial kernel
    compilation (about 70 s for the GPT step); a persistent cache or parallel
    compilation.

Not planned: an MLIR core, which would lose Julia callables in pointwise nodes
and folds; an MLIR or StableHLO export is worth building when something
consumes it, as a test oracle. Symbolic shapes wait for a decode workload.
