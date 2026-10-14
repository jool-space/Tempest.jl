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
`CUDAKernels()` and `CuBLASGemm()`. `CUDATarget(kernels=:cuda)` omits cuTile;
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

CUDA/cuTile reductions, attention and cuBLAS all use this common pool. Vendor
handles and descriptors retain fixed workspace pointers, installed or bound
outside capture; reuse is safe because the schedule executes sequentially on one
stream. General multi-stream workspace sharing is not implemented.

## cuDNN

There is no cuDNN implementation yet. An opt-in Stiletto bridge offered cuDNN
the pointwise and sum regions Tempest had already formed; it was removed on
14 October 2026. Probed through `Stiletto.compile` with cuDNN 9.26 on GB10,
pointwise-only graphs build, but a sum alone, or pointwise work followed by a
sum, has no engine. A sum builds only after a matrix product
(`sum(tanh.(w * x); dims = 1)`). The bridge never received matrix products, so it
competed with cuTile for elementwise regions and paid a failed cuDNN build for
every reduction region.

cuDNN is a library of fusable patterns: a matrix product with its epilogue,
normalization, scaled dot-product attention, convolution. A cuDNN implementation
should claim such patterns from the semantic graph and build them directly with
cuDNN.jl's graph API, rather than accept regions formed for scalar fusion. Verbs
such as attention can use cuDNN through `CallRequest` today; matrix-product
epilogues need implementations that can propose the region they cover.

## How fusion works today

First, supported `GraphRequest` candidates can cover connected semantic nodes
around reductions, before layouts and storage are assigned. Their original
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
provider's resident schedule matched its measured performance. Two standalone
compiler probes from that work remain: `examples/cutile_wrapped_reduction.jl`
reproduces cuTile 1.0's loop failure with wrapped kernel arguments, and
`examples/cutile_scalar_derivative.jl` reproduces an FMA type-verification
failure for the polynomial derivative `x -> 1 + 3x*x`. Metadata acceptance does
not prove scalar compilation; compiler errors propagate.

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
