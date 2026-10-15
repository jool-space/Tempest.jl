# Semantics, implementations, and support

Tempest's direction is a semantic graph with multiple possible realizations.
A public verb describes results. A kernel implementation writes supplied buffers.
Neither an eager allocating wrapper nor an allocation scope should determine the
compiled graph. Existing Pol/Vallmo launchers can be adapted; Pol.Allocating is
not the required semantic interface.

## Implemented now: scheduled-region realizations

`prepare(...; implementations=(candidate1, candidate2))` tries user candidates
first, then `default_implementations(target)`. Every candidate uses the same
`support` and `build_implementation` methods. CPU defaults are `CPUKernels()` and
`LinearAlgebraGemm()`. CUDA defaults are eligible loaded `CuTileKernels()`, then
`CUDAKernels()` and `CuBLASGemm()`. `CuDNNKernels()` is opt-in
([cuDNN](#cuDNN)). `CUDATarget(kernels=:cuda)` omits cuTile;
`:cutile` requires it to be loaded but still permits CUDA indexing/reduction fallback.

Resolve the candidate list once during preparation. Loading an extension later
does not change an existing executable. If no candidate
supports a scheduled region, preparation reports all rejection reasons. An
unrealized semantic call instead expands its reference and replans.

Provider and verb APIs are public but not exported: refer to them as
`Tempest.support`, or bring them in with `using Tempest: support, Unsupported`.
An implementation extends these methods:

```julia
struct MyImplementation <: Tempest.Implementation end

function Tempest.support(::MyImplementation, request::Tempest.RegionRequest)
    Tempest.Unsupported("explain why this region is unsupported")
end

# For an eligible request, return Supported() above and implement:
# build_implementation(candidate, request, runtime, out, inputs)
#     -> BuiltImplementation(launch, owners, workspace_bytes)
#     or ImplementationPlan(bind, workspace_specs)
#     or Unsupported(reason)
```

`RegionRequest{R}` contains the target, `Step{R}`, output dtype/shape, and
bound-input dtypes/shapes. `request.step.region` describes the work:

| Region | Payload |
| --- | --- |
| `Tempest.Elementwise` | Scalar expression and iteration count |
| `Tempest.FullReduction` | Scalar expression and input count |
| `Tempest.AxisReductionRegion` | Scalar expression, input count, and `AxisReduction` layout |
| `Tempest.ScatterRegion` | Scalar expression, input count, and `Selection` |
| `Tempest.GemmRegion` | Native matrix operands described by the request |

A candidate can specialize `support` on `RegionRequest{<:Tempest.Elementwise}`
instead of inspecting a symbol tag. Reduction axes live in
`request.step.region.layout.axes`; a scatter's mapping lives in
`request.step.region.selection`. There is no separate request layout to reconcile
with the region. Printed region names are diagnostic labels only.

The construction hook receives actual destination/input arrays, so it can check
physical requirements such as alignment and build vendor plans. It must not
specialize on input contents. These are provisional compiler APIs. Semantic calls
use `CallRequest`, documented in [verbs.md](verbs.md); their scheduled
`Tempest.OpaqueCall` retains the complete semantic contract.

`support` is a cheap eligibility check. `is_supported` is its Boolean convenience:
it does **not** guarantee that a vendor engine exists. Construction is the second
check and can return a reasoned rejection. Invalid user programs and unexpected
compiler errors are errors, not reasons to silently try another kernel.

A `BuiltImplementation` retains its launch resources. Its zero-argument launcher
must read only declared inputs, write the supplied output, and allocate no device
storage. Workspace, descriptors, handles, and compiled kernels must be prepared
outside capture. `explain(ex)` shows selected implementations, rejection reasons,
and workspace slots, along with the candidate order. Private persistent resources
remain owned by the realization. Shareable temporary buffers use the common
workspace plan below.

## Semantic graph regions before storage

`GraphRequest` uses the same `support` and `build_implementation` methods for a
selection of original pointwise, constant, and sum nodes. Defining support for
that request type is the participation mechanism; there is no second provider
registry. Selection runs before layout and storage planning and is enabled by
default. `fuse_regions=false` disables it. Without an eligible graph provider,
ordinary scheduled-region implementations remain in use.

`request.region` carries the execution program and original `members`, `inputs`,
and `outputs` ID vectors. `request.inputs` and `request.outputs` are matching
ordered tuples of logical `TensorSpec`s. Construction receives **tuples** of
dense output and input buffers in that order, shaped as those specs; scalar
values use 0D arrays. Inputs are read-only and do not alias outputs. Every output
must be written, including values escaping only to a view or backward operation.
Views, contractions, and opaque calls stay outside this primitive region.

After metadata rejection, growth asks support about individual additions and
leaves declined neighbors outside the current attempt. A complete candidate is
also tried before incremental growth, since support need not hold for its pieces.
Construction may return `Unsupported`, which excludes that exact member set
and retries selection;
shared nodes are not blacklisted. Unexpected exceptions propagate. Graph regions
have no direct in-place contract, so state updates retain staged donation.
Recomputation additionally needs the public `recomputable(f)` promise; device
compilability alone does not establish it. See [graph regions](graph-regions.md)
for an ordinary-code example, current cuTile coverage, and selection limits.

## Regions an implementation proposes

Growth-based graph regions stop at matrix products. An implementation that
fuses across them proposes its own regions instead:
`propose_regions(implementation, request::ProposalRequest)` returns
`RegionProposal`s, sets of value IDs on the prepared program that may include
matrix products. Tempest computes each proposal's boundary, ranks proposals by
candidate order and then by the memory traffic they save against the default
partition, and accepts those that do not overlap, keep the partition acyclic,
and pass `support(implementation, ::ProposedRequest)`. Only the proposing
implementation builds a selected region, with the same tuple contract as
`GraphRequest`, except that an input read only by matrix products may arrive as
a `Transpose`. Declines and construction failures fall back to the default
partition. [Proposed regions](proposed-regions.md) records the design and the
cuDNN consumer.

## Shared workspace

When construction knows its temporary storage requirements, return an
`ImplementationPlan` instead of immediately binding a launcher:

```julia
function Tempest.build_implementation(::MyImplementation, request, runtime, out, inputs)
    specs = (Tempest.WorkspaceSpec(eltype(out), length(out)), Tempest.WorkspaceSpec(UInt8, 4096))
    Tempest.ImplementationPlan(specs) do buffers
        temporary, vendor_workspace = buffers
        # Bind an already-selected kernel/engine to these final buffers here.
        launch = make_launcher(out, inputs, temporary, vendor_workspace)
        Tempest.BuiltImplementation(launch, (out, inputs, buffers), 0)
    end
end
```

Each workspace request is a flat typed buffer with a minimum capacity. Buffers
may be larger than requested. Requests within one launch receive distinct slots;
later launches may reuse them. Initialize all temporary data read by each launch.
Workspace contents cannot be saved for a later launch or exposed as an output.
The executable owns every slot, and bound implementations retain their buffers.

Support and engine selection finish before workspace binding. A binder must
return `BuiltImplementation`; an error there propagates instead of triggering
fallback. Existing wrappers returning `BuiltImplementation` directly still work.
Its `workspace_bytes` counts privately owned device workspace, excluding shared
slots so `explain` does not double-count them.

CUDA/cuTile reductions, cuTile and cuDNN attention, and cuBLAS all use this common pool. Vendor
handles and descriptors retain fixed workspace pointers, installed or bound
outside capture; reuse is safe because the schedule executes sequentially on one
stream. General multi-stream workspace sharing is not implemented.

## cuDNN

`CuDNNKernels()` realizes semantic verbs as cuDNN graphs, built directly with
cuDNN.jl's graph API. Load `CUDACore` and `cuDNN` to enable it. It is opt-in, so
pass it explicitly:

```julia
ex = prepare(loss, args...; implementations = (CuDNNKernels(float32_cast = Core.BFloat16),))
```

At present it covers scaled dot-product attention:

- **Forward-only programs.** It runs the public `attention_forward` call and
  returns `o` and `lse` in the operand dtype and in Tempest's order.
- **Training.** A reverse rule replaces the forward/backward pair with two
  private calls. `sdpa_forward` saves cuDNN's Float32 statistic, packed
  `(1, queries, heads, batch)` as the backward reads it. `sdpa_backward` is
  cuDNN.jl's composite `sdpa_bwd!`, the form the fMHA backward engines match.

cuDNN's statistic is Tempest's `lse`: the natural-log log-sum-exp of the scaled
scores. A forward from either provider therefore pairs with a backward from the
other. Grouped-query attention and causal masks map directly onto cuDNN's
attention.

cuDNN's backward has no input for a log-sum-exp cotangent. `Tempest.attention`
discards `lse`, so its cotangent is the symbolic zero and the private backward
applies. If a program uses `lse`, the rule passes the public
`attention_backward` to the next provider: cuTile under TF32. If no provider
accepts it, the pair is rejected, and the cuDNN forward pairs with the reference
pullback.

### Precision

These results were probed with cuDNN 9.27 (CUDNN_jll 9.27, cuDNN.jl 6.4) on
GB10 (sm_121), for every math mode:

| Operands | Result |
| --- | --- |
| Float32 | no engine, forward or backward |
| Float32 I/O around in-graph casts to Float16/BFloat16 | no engine |
| Float16/BFloat16 inputs with a Float32 output | no engine |
| Float16 or BFloat16 | builds. The forward takes up to 256 features and the backward up to 128. Any sequence length works, with grouped queries and causal masks. |

Graphs are built with `DEFAULT_MATH`, which allows tensor cores but excludes
engines that down-convert inputs, and with `deterministic = true`.
`PEDANTIC_MATH` finds no engine: cuDNN.jl excludes every tensor-core engine
whenever a Float32 tensor, here the statistic, is graph I/O.

Half-precision operands run natively. **Float32 is rejected** unless the
implementation is constructed with `float32_cast = BFloat16` or `Float16`. This
mirrors `CUDATarget(matmul_precision = :tf32)`: rounding is an explicit request,
never a fallback.

With the cast:

- q, k, v and the output cotangent are rounded to the cast type.
- Scores, softmax, the statistic and accumulation stay Float32.
- The probabilities enter the second product in the half type.
- Results return as Float32.

In training, the casts are ordinary graph operations, so they can fuse with the
operations that produce or consume them. In a forward-only call, they are
conversion kernels around the cuDNN launch. `explain` names the rounding, for
example `cuDNN attention forward: Float32 operands rounded to BFloat16`.

Rejections and their reasons:

| Request | Reason shown by `explain` |
| --- | --- |
| Float32 without `float32_cast` | `cuDNN has no Float32 attention engine; ...` |
| Float64 or other types | `cuDNN attention takes Float16 or BFloat16 operands, not Float64` |
| Features not a multiple of 8, or `dv != d` | layout requirement |
| No engine at construction, e.g. backward with 160 features on GB10 | `cuDNN has no engine for attention backward ...`; the training pair is rejected |
| Public `attention_backward` | no log-sum-exp cotangent input |
| Scalar regions | `cuDNN implements semantic verbs, not scalar regions` |

cuDNN is never offered lone reductions or pointwise regions. With cuDNN 9.26 on
GB10, a sum alone, or pointwise work followed by a sum, had no engine. A sum
built only after a matrix product. `CuDNNKernels` defines no `GraphRequest`
method, so it stays out of growth-based graph-region selection. It proposes
regions around matrix products instead: Float16 and BFloat16 products with
their pointwise epilogue and the sums after them, forward and backward. See
[proposed regions](proposed-regions.md#cuDNN:-matrix-products-with-epilogues)
for coverage, precision and measurements.

### Execution and workspace

Each executable runtime owns one cuDNN handle. Plans and variant packs are
built outside capture with the final buffer pointers. A launch sets the stream
and calls `cudnnBackendExecute`, which CUDA graph capture records.

The shared workspace pool provides several buffers:

- the engine workspace;
- the backward's Float32 softmax row sums;
- its Float32 dQ accumulator;
- with grouped heads, per-query-head dK and dV.

The engine writes each of these before reading it. Buffers filled with NaN gave
correct gradients, so shared slots are safe. Forward-only Float32 calls take
their half-precision copies from the same pool.

### Measured against cuTile

`examples/benchmarks/attention.jl` times causal attention on GB10 with
cuDNN 9.27, on 14 October 2026, with no other GPU process running. Each figure
is the median device time per step, in μs, over CUDA graphs of 20 back-to-back
executions. TFLOP/s follows in parentheses, counting 4·b·h·t²·d per forward,
3.5 times that for forward + backward, and halved for the mask. "Cast" runs
Float32 data with `float32_cast`, so its time includes the rounding. "bf16"
runs BFloat16 data, the engines alone.

| pass | d×heads×seq×batch | cuTile TF32 | cuDNN bf16 cast | cuDNN fp16 cast | cuDNN bf16 |
| --- | --- | --- | --- | --- | --- |
| fwd | 64×12×1024×8 | 647 (19.9) | 943 (13.7) | 947 (13.6) | 297 (43.3) |
| fwd | 64×12×2048×4 | 1184 (21.8) | 1041 (24.8) | 1075 (24.0) | 405 (63.6) |
| fwd | 128×12×1024×8 | 2523 (10.2) | 1826 (14.1) | 1823 (14.1) | 500 (51.6) |
| fwd | 128×12×2048×4 | 4730 (10.9) | 1970 (26.2) | 2013 (25.6) | 642 (80.2) |
| fwd+bwd | 64×12×1024×8 | 4035 (11.2) | 3031 (14.9) | 2950 (15.3) | 1768 (25.5) |
| fwd+bwd | 64×12×2048×4 | 6787 (13.3) | 3506 (25.7) | 3513 (25.7) | 2267 (39.8) |
| fwd+bwd | 128×12×1024×8 | 34668 (2.6) | 6125 (14.7) | 6146 (14.7) | 3606 (25.0) |
| fwd+bwd | 128×12×2048×4 | 62061 (2.9) | 6951 (26.0) | 6905 (26.1) | 4433 (40.7) |

The cuDNN engines are 2.2 to 7.4 times faster than cuTile's forward, and 2.3
to 14 times faster than its forward + backward. Two observations follow from
the table without being profiled.

- **Casts cost the most at short sequences.** A forward-only cast converts q,
  k, v and o. At 64×12×1024×8 that is about 150 MB of traffic, which costs more
  than the attention itself on GB10's memory. In training, the casts are graph
  operations that fuse with their neighbors.
- **cuTile's 128-feature backward collapses** to about 3 TFLOP/s.

### Layout copies around attention

Stiletto described permuted, sliced and reshaped inputs to cuDNN as tensor
strides. Tempest's attention implementations take dense operands, so a layout
operation between a producer and the call is a copy step, unless a pointwise
region reads it anyway. `examples/benchmarks/attention_layouts.jl` counts
those steps with `explain` and times them with `Tempest.profile`, at
768 features, 12 heads, 1024 positions and batch 8 (GPT: 2 layers, batch 4),
on GB10 on 15 October 2026. Times are μs per execution; "share" is the layout
copies' part of the summed call times.

| case | pass | data, attention | copies | their time | attention | total | share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT example | fwd | Float32, cuDNN cast / cuTile | 0 | 0 | 596 / 668 | 8083 / 7069 | 0% |
| GPT example | fwd+bwd | Float32, cuDNN cast / cuTile | 0 | 0 | 1700 / 3837 | 25089 / 26190 | 0% |
| heads by reshape | fwd, fwd+bwd | Float16, BFloat16 cuDNN; Float32 cuTile | 0 | 0 | | | 0% |
| rows: `permutedims` split | fwd | Float16 cuDNN | 3 | 510 | 291 | 1335 | 38% |
| rows: `permutedims` split | fwd+bwd | Float16 cuDNN | 4 | 719 | 1600 | 4124 | 17% |
| rows: `permutedims` split | fwd | Float32 cuTile | 3 | 1048 | 651 | 2816 | 37% |
| rows: `permutedims` split | fwd+bwd | Float32 cuTile | 4 | 1312 | 3706 | 9519 | 14% |
| packed `[q; k; v]` slices | fwd | Float16 cuDNN | 3 | 332 | 281 | 1202 | 28% |
| packed `[q; k; v]` slices | fwd+bwd | Float16 cuDNN | 3 | 336 | 1611 | 5060 | 7% |
| packed `[q; k; v]` slices | fwd | Float32 cuTile | 3 | 668 | 662 | 2505 | 27% |
| packed `[q; k; v]` slices | fwd+bwd | Float32 cuTile | 3 | 668 | 3716 | 11334 | 6% |

BFloat16 matched Float16 within 6%. A head split and merge by `reshape`, the
order attention takes, costs nothing, and so does the GPT example, whose
rotary embedding reads its operands in a pointwise region. The GPT training
step has eight other copies, 0.6% of its time: weight gradients reshaped
from a transposed product into three-dimensional weights. A merge by
`permutedims` before a matrix product is a transposed operand and also
free. Copies appear when operands are written as `(tokens, features)` rows
(`permutedims(reshape(x * w, ...))`, about 170 μs per Float16 copy, a
strided transpose) or sliced from one packed projection (about 110 μs, at
copy bandwidth). In training they are 6 to 17% of the step; forward-only,
27 to 38%. The row form's feature axis is not unit-stride, which cuDNN's
attention engines likely require, so strides would absorb the packed slices
but probably not the row permutation.

### Next

Norms are the next cuDNN pattern: `norm_fwd!`/`norm_bwd!` with saved statistics,
once Tempest has a normalization verb. They would follow the attention rule's
shape. Normalization fused into a matrix product would be another proposal
around the product, like the epilogues `CuDNNKernels` proposes now.

## How fusion works today

First, implementations propose regions of their own, such as a matrix product
with its epilogue, and the selected proposals claim their nodes. Then supported
`GraphRequest` candidates can cover connected semantic nodes around reductions,
before layouts and storage are assigned. Their original
operations and escaping outputs remain visible to the selected provider. This
lets a provider compile several reduction phases and their pointwise consumers.

For remaining work, Tempest counts uses and creates materialization boundaries
at matrix products, reductions, scatter, returned values, and certain shared
values/layouts. It recursively inlines pointwise nodes into scalar expression
trees. CPU/CUDA/cuTile providers evaluate those trees without allocating every
intermediate.

Graph selection is a bounded greedy policy. It does not negotiate arbitrary
output layouts or optimize across opaque operations. Construction of ordinary
scheduled regions still happens after storage planning. Only selected
constructions contribute workspace requests, so rejected alternatives consume
no shared workspace.

## Reductions inside graph regions

Reductions with their pointwise producers and consumers are handled by graph
regions; see [Graph regions](graph-regions.md). The earlier scalar-binding
reduction path and its `fuse_reductions` flag were retired once the graph
provider's resident schedule matched its measured performance. The compiler
problems met along the way, and their upstream fixes, are recorded in
[cuTile issues](cutile-issues.md). Metadata acceptance does not prove scalar
compilation; compiler errors propagate.

## What an opaque operation means

Launching a kernel does not reveal its mathematics to Tempest. An opaque
operation needs an explicit semantic definition or reference decomposition, output
metadata and effects; automatic differentiation additionally needs a derivative
recipe. A concrete implementation separately declares supported layouts, types,
numerical policy, saved values, workspace, and launches.

A reference decomposition explains an operation and can provide a verification
oracle. It does not prove that a device kernel implements that reference. Test
forward and backward numerical behavior independently. Likewise, a kernel call
is not fusible just because it has a familiar name: crossing its boundary needs
an explicit decomposition or a registered implementation of a larger region.

## Semantic calls and graph selection

Multi-output semantic calls and forward-only programs use the interfaces in
[verbs.md](verbs.md). A call can select an opaque implementation or expand its
reference before graph-region selection, scalar fusion, and storage assignment.
Construction rejection also permits reference expansion and replanning.
[Implementation-specific reverse rules](reverse-rules.md) can coordinate
forward/backward saved values without changing the public verb.

The graph-region selector now lets candidates cover multiple primitive semantic
nodes. Its bounded scope is described in [graph-regions.md](graph-regions.md);
layout negotiation and a general scheduling cost model remain open. Pol's buffer
descriptions and mutating launchers can serve as adapters without defining the
graph contract. Arbitrary input mutation and state aliases still require
explicit effect tracking.

## In-place and grouped requests

`InPlaceRequest` wraps an elementwise `RegionRequest` and lists aliased input
positions. The planner proves the access pattern and old-value lifetime; the
implementation must opt in through `support`. Its builder may delegate to the
wrapped region after that explicit acceptance. Existing region methods alone do
not authorize aliased execution.

`GroupedRequest.requests` is a vector of independent elementwise requests, each
retaining its own alias contract. `build_implementation` receives vectors of
outputs and input tuples. A group may be rejected by either support or build;
selection then falls back to individual requests. Unexpected errors propagate.
Grouping is attempted before selecting the individual members, so a candidate
supporting only individual regions can be bypassed by a supported grouped
implementation. Set `grouped=false` when individual-region selection is required.
