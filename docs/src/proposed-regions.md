# Proposed regions

Tempest partitions a program itself: [graph regions](graph-regions.md) grow from
reductions through pointwise work, and matrix products, layout operations and
semantic calls stay outside as their own steps. An implementation can accept or
decline what it is offered, but nothing it is offered crosses a matrix product.
A library that fuses a product with its epilogue (cuDNN, cuBLASLt, a cuTile
matmul with an epilogue) has no request to accept.

A proposed region reverses the direction, as XLA's and TVM's bring-your-own-
codegen partitioners do. An implementation inspects the prepared program and
proposes the sets of nodes it can realize. Tempest chooses among the
proposals, and everything not chosen keeps the default partition: graph
regions, scalar fusion, cuBLAS products. This page records the design and its
first consumer, cuDNN matrix products with epilogues.

## The protocol

```julia
struct MyLibrary <: Tempest.Implementation end

function Tempest.propose_regions(::MyLibrary, request::Tempest.ProposalRequest)
    # request.program, request.live, request.users, request.target
    [Tempest.RegionProposal(members; label = "matmul → bias, relu") for members in matches(request)]
end

function Tempest.support(::MyLibrary, request::Tempest.ProposedRequest)
    # request.region.members, .inputs, .outputs; request.inputs, request.outputs (TensorSpecs)
    Tempest.Supported()
end

# Tuples of outputs and inputs, in the order of request.outputs and request.inputs.
Tempest.build_implementation(::MyLibrary, request::Tempest.ProposedRequest, runtime, outputs, inputs) = ...
```

Defining `propose_regions` is the participation mechanism, as defining
`support` for `GraphRequest` is for graph regions; there is no registry. A
proposal lists value IDs: pointwise operations, constants, folds and matrix
products. Constants used only by members join automatically. Tempest derives
the boundary: the inputs, and every member that is read outside the proposal,
by backward or as a result, as an output. `support` sees that complete boundary
before storage is planned; construction receives buffers. A proposed region is
built only by the implementation that proposed it.

## Decisions

### When proposals are made

On the prepared program: after differentiation, implementation reverse rules
and semantic resolution, at the start of each planning pass, before growth-based
graph regions, layouts and storage. Forward and backward work are one graph by
then, so a proposal can cover a backward product with the activation derivative
and bias gradient that follow it, and a forward product's proposal knows which
of its values backward reads.

Proposals are made again in every planning pass. They are cheap to recompute,
and node IDs belong to one graph snapshot: a pass that follows a semantic
rollback sees a different graph.

### Conflicts and overlaps

Proposals are ordered by three keys, and the order is total:

1. **Implementation priority**: the candidate order of `prepare`, user
   implementations first. This is the order every other selection uses, and
   the user's explicit choice outranks any estimate.
2. **Benefit**, within one implementation: the bytes the proposal keeps out of
   memory against the default partition. Values the default schedule writes
   (its fusion roots: products, reductions, shared values) count when the
   proposal keeps them internal; values the default would have inlined count
   against it when they escape the proposal. Tempest computes this from the
   program alone, without any cost model of an implementation.
3. Size, then the members' value IDs.

The partitioner walks that order and accepts a proposal when it is not blocked
by an earlier construction failure, overlaps no accepted proposal, keeps the
contracted graph acyclic, and its implementation supports the complete
request. Growth-based graph regions then form around the accepted proposals,
which count as claimed and take part in the acyclicity check.

Benefit decides real cases. In a two-layer MLP, the forward product's epilogue
can grow into the activation derivative `(1 - h) .* (1 + h)`, which only the
backward product's epilogue reads, and both proposals then have seven members.
Claiming the derivative forward would write it to memory where the default
plan inlines it, and leave the backward product unfused. Its benefit is
lower, so the backward proposal wins and the forward one shrinks to product,
bias and activation.

Greedy acceptance is not an optimal cover. A better pair of smaller proposals
can lose to one larger proposal, and benefits of different implementations are
not compared. The second is deliberate: benefit counts bytes, not time, and an
implementation earlier in the candidate list was put there by the user.

### Storage planning and donation

An accepted proposal is an ordinary step whose outputs get value slots, with
lifetimes, reuse and liveness like any other step. Its inputs are dense
buffers, except an input read only by matrix products, which keeps its native
presentation: dense, or a `Transpose` of a dense buffer. The backward products
of a dense layer read `transpose(W)` and `transpose(x)`, and a dense contract
would copy both every step. Temporary storage comes from the shared workspace
pool through `ImplementationPlan`. Proposed regions, like graph regions, have
no in-place contract, so state updates keep staged donation.

### Kernel reuse

`standard_numbering(region)` moved from the cuTile extension into Tempest. It
numbers a region's values in a form that ignores graph IDs and the operand
order of commutative operations. cuTile graph regions use it for one compiled
kernel per equivalent region; cuDNN keys its execution plans on it, so every
layer of the same shape and the same pattern shares one plan. Constants are
bound by value at launch and are not part of the key.

### Values that escape

The boundary is Tempest's, not the proposer's. A forward activation that
backward reads, a cotangent that both a weight gradient and a bias gradient
read, a returned intermediate: each is an output the implementation must
write. `support` sees the outputs and may decline them; benefit charges the
writes that the default would have avoided.

### Failures

- `Unsupported` from `support` declines that proposal; the next one in order,
  often a nested alternative, is considered in the same pass.
- `Unsupported` from construction blocks that exact member set, with its
  reason. All regions that fail in one pass are blocked together and
  preparation replans once; smaller alternatives remain eligible.
- Unexpected exceptions propagate, and a malformed proposal (a dead value, an
  argument, a layout operation, a semantic call) is an error, not a fallback.

An implementation that can check feasibility without buffers should do so in
`support`, where a decline costs nothing. cuDNN builds its graph there; the
plan is cached for construction.

### `explain`

`explain` lists every proposal with its outcome:

```text
Region proposals:
  cuDNN matmul → +, tanh, abs2, sum, *, -, +, *, *, sum dims=(2,) [9, …, 25]: selected as region 2
  cuDNN matmul → -, +, *, *, sum dims=(2,) [30, 31, 32, 33, 34, 35, 38]: selected as region 4
  cuDNN matmul → +, tanh [6, 7, 8]: selected as region 1
  cuDNN matmul → +, tanh, -, +, * [6, 7, 8, 31, 32, 33, 34]: conflict: its alternative [6, 7, 8] ranked higher
  cuDNN matmul → tanh [3, 4]: declined: cuDNN's fused engines were slower than cuBLAS for an inner dimension above 4096
  cuDNN … : fell back after construction: …
```

Selected regions print as `proposed_region [cuDNN] matmul → +, tanh covers [6, 7, 8]`,
and their calls carry the implementation's summary.

### Determinism

Proposals depend only on the program and the target; the sort key is a total
order, duplicate proposals of one implementation are merged, and acceptance is
a deterministic walk. The same program on the same device gives the same
partition. cuDNN's profitability rule reads the device's L2 size, which is a
property of the device. Selection by timing was rejected for this reason.

## cuDNN: matrix products with epilogues

`CuDNNKernels()` proposes each matrix product with Float16 or BFloat16
operands, together with the work that consumes it, grown in value order:

- pointwise operations with a cuDNN mode (`+ - * /`, `max`, `min`, `exp`,
  `log`, `sqrt`, `tanh`, `abs`, `sin`, `cos`, `tan`, `inv`, negation,
  squares and conversions), with operands from the region, from constants, or
  from region inputs of any broadcastable shape;
- pointwise producers of those operands that nothing outside the region reads,
  such as an activation derivative computed from a saved forward value;
- sums over one or both axes of a value in the region, which end their branch.

Every prefix of that growth is a nested alternative. The forward of a dense
layer becomes product, bias and activation; in training, the backward product
`W2' * dy` also takes the activation's derivative and the bias gradient, and
writes the cotangent that the weight gradients read. A lone product is never
proposed: cuBLAS serves it. Neither is a lone sum, which had no engine.

The pointwise mode table and rank lifting are from Stiletto (noted in
`ext/cuDNNExt/matmul.jl`). Each region becomes a cuDNN graph with every tensor
declared at rank 3 or more by trailing singleton dimensions, the same as
Julia's broadcasting of a bias vector.

### Precision

Probed with cuDNN 9.27 on GB10, `DEFAULT_MATH`, deterministic engines:

| Graph | Result |
| --- | --- |
| Float16 or BFloat16 product with bias, tanh, ReLU, GELU; product also written; transposed operand | builds |
| Product followed by a sum over one or both axes, half operands | builds only with a Float32 sum output |
| A lone sum, or pointwise work followed by a sum | no engine |
| Half-precision intermediate tensors | build |
| Float32 product | builds, but rounds operands to TF32 without a numerical note |

Every value is declared in its Tempest element type, so it rounds where
Tempest rounds it: the product rounds to Float16 before the bias is added, as
it does in the default plan. Products accumulate in Float32, as cuBLAS does in
the default plan. Sums of half-precision values accumulate in Float32, are
written to a Float32 workspace, and are rounded once into their output; the
default CUDA kernels accumulate those sums in the element type, so cuDNN's sums
are the more precise. `explain` says `sums written in Float32 and rounded to
Float16`.

Float32 products round to TF32, which matched cuBLAS's TF32 error exactly, so
cuDNN proposes them only under `CUDATarget(matmul_precision = :tf32)`. Float64
is never proposed.

### Profitability

Every fused engine on GB10 is runtime-compiled; the cuBLASLt engine serves a
lone product only. The fused engines were fast while the right operand
(`k × n`) fit in about three quarters of the 24 MiB L2, and two to three times
slower beyond it, regardless of either operand's layout. Single layers
`tanh.(W * x .+ b)` in Float16, with the policy below switched off
(`LAYERS=... examples/benchmarks/matmul_epilogue.jl`), device time per
execution in μs:

| m × k × n | right operand | default | cuDNN | speedup |
| --- | --- | --- | --- | --- |
| 1024 × 1024 × 8192 | 16 MiB | 335.8 | 241.1 | 1.39 |
| 1024 × 1152 × 8192 | 18 MiB | 364.7 | 284.9 | 1.28 |
| 1024 × 1024 × 10240 | 20 MiB | 428.3 | 491.5 | 0.87 |
| 1024 × 2048 × 8192 | 32 MiB | 495.6 | 1293.0 | 0.38 |
| 4096 × 2048 × 2048 | 8 MiB | 493.3 | 342.5 | 1.44 |
| 2048 × 4096 × 1024 | 8 MiB | 187.7 | 177.8 | 1.06 |
| 1024 × 8192 × 1024 | 16 MiB | 202.5 | 237.1 | 0.85 |

`support` therefore declines a product whose right operand exceeds three
quarters of the device's L2, or whose inner dimension exceeds 4096, with that
reason in `explain`. This is a measured policy, not a capability limit, and it
was measured on one device.

### Measured against the default plan

`examples/benchmarks/matmul_epilogue.jl` times the block
`w2 * tanh.(w1 * x .+ b1) .+ b2`, a transformer feed-forward layer with hidden
width `4d` and `n` tokens, and the training step of `sum(abs2, block)`. It ran
on GB10 with cuDNN 9.27 on 14 October 2026, with no other GPU process. Each
figure is the median device time per step in μs, over CUDA graphs of 20
back-to-back executions; TFLOP/s counts the products only. "default" is
Tempest's plan without cuDNN: cuBLAS products with Float32 accumulation and
generated epilogue kernels.

| pass | type | d × n | default | cuDNN | speedup |
| --- | --- | --- | --- | --- | --- |
| fwd | Float16 | 256 × 4096 | 142.7 (30.1) | 63.2 (68.0) | 2.26 |
| fwd | Float16 | 512 × 4096 | 395.3 (43.5) | 231.4 (74.2) | 1.71 |
| fwd | Float16 | 768 × 8192 | 1461.5 (52.9) | 1038.7 (74.4) | 1.41 |
| fwd | Float16 | 1024 × 8192 | 2263.0 (60.7) | 1754.2 (78.3) | 1.29 |
| fwd | BFloat16 | 256 × 4096 | 142.6 (30.1) | 66.9 (64.2) | 2.13 |
| fwd | BFloat16 | 512 × 4096 | 377.2 (45.5) | 233.2 (73.7) | 1.62 |
| fwd | BFloat16 | 768 × 8192 | 1503.1 (51.4) | 1037.1 (74.5) | 1.45 |
| fwd | BFloat16 | 1024 × 8192 | 2266.2 (60.6) | 1796.0 (76.5) | 1.26 |
| fwd+bwd | Float16 | 256 × 4096 | 450.9 (28.6) | 248.2 (51.9) | 1.82 |
| fwd+bwd | Float16 | 512 × 4096 | 1154.7 (44.6) | 688.6 (74.9) | 1.68 |
| fwd+bwd | Float16 | 768 × 8192 | 4519.1 (51.3) | 3166.5 (73.2) | 1.43 |
| fwd+bwd | Float16 | 1024 × 8192 | 6876.0 (60.0) | 5870.8 (70.2) | 1.17 |
| fwd+bwd | BFloat16 | 256 × 4096 | 413.3 (31.2) | 246.7 (52.2) | 1.67 |
| fwd+bwd | BFloat16 | 512 × 4096 | 1151.8 (44.7) | 683.9 (75.4) | 1.68 |
| fwd+bwd | BFloat16 | 768 × 8192 | 4514.2 (51.4) | 3221.9 (72.0) | 1.40 |
| fwd+bwd | BFloat16 | 1024 × 8192 | 6852.6 (60.2) | 5635.0 (73.2) | 1.22 |

With the profitability rule, fusion won at every shape, 1.2 to 2.3 times. At
`d = 768` and `1024` the second product's right operand, the hidden activation,
is 48 or 64 MiB, so that product and its bias stay with cuBLAS and the generated
kernels; the first product, and in training the backward product with the
activation derivative and bias gradient, are fused. Without the rule, the
forward at `d = 1024` ran at 0.61 times the default, and training at 0.86.
Preparation is not slower once warm (10 ms either way); the first cuDNN
preparation compiles the fused engines, about 5 s in this process.

## Not done

- **cuBLASLt epilogues and cuTile products.** cuBLASLt's bias/ReLU/GELU
  epilogues, with their auxiliary output for backward, and a cuTile product
  with a fused epilogue would each be another `propose_regions` method on the
  same protocol.
- **User activations.** cuDNN recognizes Base functions only. A user's
  `swish` or `gelu` stays outside the region even where cuDNN has a mode for
  it; a public hook mapping callables to modes would let it join.
- **Cover optimization.** Selection is greedy over a total order; an exact
  cover, or comparing benefits across implementations, is open.
