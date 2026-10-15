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
  cuDNN matmul → tanh [3, 4]: declined: cuDNN's fused product would take longer than cuBLAS and an epilogue kernel (estimated 265.1 against 179.0 μs; 67% of tile slots used)
  cuDNN … : fell back after construction: …
```

Selected regions print as `proposed_region [cuDNN] matmul → +, tanh covers [6, 7, 8]`,
and their calls carry the implementation's summary.

### Determinism

Proposals depend only on the program and the target; the sort key is a total
order, duplicate proposals of one implementation are merged, and acceptance is
a deterministic walk. The same program on the same device gives the same
partition. cuDNN's chunking and cost estimate read the device's L2 size, SM
count and rates, which are properties of the device. Selection by timing was
rejected for this reason; the one measurement, of an architecture without
derived rates, happens once per device and process, before any plan reads it.

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
the default plan. Sums of half-precision values accumulate in Float32 and
round once, as every Tempest fold does (`fold_accumulator`); no cuDNN engine
wrote a half-precision sum, so the sum is written to a Float32 workspace and
converted into its output.

Float32 products round to TF32, which matched cuBLAS's TF32 error exactly, so
cuDNN proposes them only under `CUDATarget(matmul_precision = :tf32)`. Float64
is never proposed.

### Where fusion pays

Every fused engine on GB10 is runtime-compiled (`cudnn_generated_fort_native`
kernels for sm80 and sm120, without TMA); the cuBLASLt engine serves a lone
product only. Two causes made the fused engines lose to cuBLAS plus Tempest's
epilogue kernels, found with `nsys` kernel traces (GPU performance counters,
and so `ncu`, are not available to this user):

**Tile order.** The fused kernel's grid walks 128 × 128 output tiles along
columns, `columns / 128` by `rows / 128`, and every row of tiles reads the
whole right operand (`k × n`). While it fits in L2 that costs nothing. Beyond
it, the operand streams from memory once per row of tiles: 1024 × 2048 × 8192
(a 32 MiB operand, 8 rows of tiles) took 1216 μs on 128 × 128 tiles, 709 μs on
64 × 256 tiles (4 rows), while cuBLAS's `nvjet_sm121` kernel, with TMA loads
and its own tile order, took 351 μs for the product alone. Splitting K
(`SPLIT_K_SLC` knob) only slowed it, and swapping the operands did not change
the order. Launching the same plan over column chunks of the right operand,
each a pointer offset into the same buffers, keeps the chunk in L2: four chunks
took 475 μs. The default epilogue kernels are single streaming passes, which no
cache reuse would speed up, which is why the default plan does not suffer.

**Product throughput.** With L2 not the limit, the fused kernels reached about
0.9 of cuBLAS's rate on full waves, less when a launch's last wave is mostly
empty: 1024 × 8192 × 1024 has 64 output tiles for 48 SMs. Fusion saves writing
and rereading the product, a fixed cost per output element, and loses a share
of the product's time, which grows with the inner dimension.

`CuDNNKernels` therefore:

- launches a region in equal column chunks when the right operand exceeds
  `l2_fraction` of L2 (default 3/4: at 18 MiB one launch still held, at 20 MiB
  it did not); a region that sums across columns cannot be cut and is declined;
- declines a product whose estimated fused time exceeds the default plan's:
  `T / (e u)` against `T + 2 s m n / bandwidth + launch`, with the product time
  `T = 2 m n k / product_flops`, `e = fused_efficiency`, and `u` the share of the
  launched waves' tile slots that hold a tile. Launches cost 2 μs and the
  efficiency is 0.9, as measured on GB10. The product rate and the bandwidth
  are the device's (`Tempest.device_rates`).

All are `CuDNNKernels` keywords. The two rates default to `nothing`, which
derives them from device attributes: the product rate as SMs × clock × dense
Float16 tensor FLOPs per clock per SM, from a table by compute capability, and
the bandwidth as memory clock × transfers per clock × bus width. Each is
scaled by the share of peak that cuBLAS and a copy reached on GB10: 108 of
119 TFLOP/s (0.9) and 234 of 273 GB/s (0.85). On GB10 that derives 107.0
TFLOP/s and 232.1 GB/s, and every decision in the tables below is unchanged.
An architecture without a table entry is measured instead, once per device
and process: a 256 MiB copy and a 4096³ cuBLAS Float16 product, which on GB10
gave about 98 TFLOP/s and 220 GB/s. The values are fixed per device for the
process, so the plan is deterministic. `explain` prints them under the
candidate list, for example:

```text
  cuDNN cost model: products at 107.0 TFLOP/s (derived: 48 SMs × 2.42 GHz × 1024 FLOP/clock × 0.9), memory at 232.1 GB/s (derived: 8.53 GT/s × 256 bits × 0.85), 2.0 μs per launch, fused efficiency 0.9
```

A number given for either rate overrides it and prints as "given". GeForce
boards of compute capability 7.5 to 8.9 accumulate Float32 at half the table's
rate; give them `product_flops`. On these single layers the estimate decided
right except within about 10% of even. Its misses are 1024 × 2048 × 8192, where
two 16 MiB chunks were slower than four 8 MiB chunks would have been, and
2048 × 4096 × 1024, a 6% win it declines. Single layers `tanh.(W * x .+ b)` in
Float16 (`LAYERS=... examples/benchmarks/matmul_epilogue.jl`), device time per
execution in μs, "unlimited" being one launch never declined:

| m × k × n | right operand | default | cuDNN unlimited | cuDNN | plan | speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 1024 × 1024 × 8192 | 16 MiB | 409.9 | 235.9 | 239.3 | one launch | 1.71 |
| 1024 × 1152 × 8192 | 18 MiB | 362.0 | 333.8 | 360.5 | one launch | 1.00 |
| 1024 × 1024 × 10240 | 20 MiB | 430.2 | 499.2 | 352.7 | 2 chunks | 1.22 |
| 1024 × 2048 × 8192 | 32 MiB | 526.7 | 1255.6 | 565.2 | 2 chunks | 0.93 |
| 1024 × 4096 × 8192 | 64 MiB | 839.7 | 2942.8 | 828.8 | declined | 1.01 |
| 4096 × 1024 × 8192 | 16 MiB | 1442.7 | 980.2 | 1008.5 | one launch | 1.43 |
| 4096 × 2048 × 2048 | 8 MiB | 498.4 | 366.2 | 358.5 | one launch | 1.39 |
| 4096 × 4096 × 4096 | 32 MiB | 1726.6 | 2333.6 | 1551.3 | 2 chunks | 1.11 |
| 2048 × 4096 × 1024 | 8 MiB | 190.8 | 178.6 | 190.7 | declined | 1.00 |
| 1024 × 8192 × 1024 | 16 MiB | 249.7 | 246.6 | 263.0 | declined | 0.95 |
| 512 × 2048 × 4096 | 16 MiB | 114.9 | 107.1 | 108.1 | one launch | 1.06 |
| 512 × 512 × 2048 | 2 MiB | 25.5 | 16.4 | 16.6 | one launch | 1.54 |

Successive runs of the same shape varied by up to 10%.

### Measured against the default plan

`examples/benchmarks/matmul_epilogue.jl` times the block
`w2 * tanh.(w1 * x .+ b1) .+ b2`, a transformer feed-forward layer with hidden
width `4d` and `n` tokens, and the training step of `sum(abs2, block)`. It ran
on GB10 with cuDNN 9.27 on 15 October 2026, with no other GPU process. Each
figure is the median device time per step in μs, over CUDA graphs of 20
back-to-back executions; TFLOP/s counts the products only. "default" is
Tempest's plan without cuDNN: cuBLAS products with Float32 accumulation and
generated epilogue kernels.

| pass | type | d × n | default | cuDNN | speedup |
| --- | --- | --- | --- | --- | --- |
| fwd | Float16 | 256 × 4096 | 143.2 (30.0) | 65.9 (65.1) | 2.17 |
| fwd | Float16 | 512 × 4096 | 375.9 (45.7) | 232.4 (73.9) | 1.62 |
| fwd | Float16 | 768 × 8192 | 1486.6 (52.0) | 970.1 (79.7) | 1.53 |
| fwd | Float16 | 1024 × 8192 | 2252.4 (61.0) | 1736.0 (79.2) | 1.3 |
| fwd | BFloat16 | 256 × 4096 | 143.9 (29.9) | 64.6 (66.4) | 2.23 |
| fwd | BFloat16 | 512 × 4096 | 378.4 (45.4) | 225.5 (76.2) | 1.68 |
| fwd | BFloat16 | 768 × 8192 | 1489.9 (51.9) | 964.2 (80.2) | 1.55 |
| fwd | BFloat16 | 1024 × 8192 | 2256.4 (60.9) | 1777.5 (77.3) | 1.27 |
| fwd+bwd | Float16 | 256 × 4096 | 414.3 (31.1) | 250.8 (51.4) | 1.65 |
| fwd+bwd | Float16 | 512 × 4096 | 1152.5 (44.7) | 700.5 (73.6) | 1.65 |
| fwd+bwd | Float16 | 768 × 8192 | 4502.5 (51.5) | 3202.5 (72.4) | 1.41 |
| fwd+bwd | Float16 | 1024 × 8192 | 6820.9 (60.4) | 5948.3 (69.3) | 1.15 |
| fwd+bwd | BFloat16 | 256 × 4096 | 419.1 (30.7) | 253.0 (50.9) | 1.66 |
| fwd+bwd | BFloat16 | 512 × 4096 | 1144.1 (45.1) | 683.6 (75.4) | 1.67 |
| fwd+bwd | BFloat16 | 768 × 8192 | 4506.4 (51.5) | 3161.7 (73.4) | 1.43 |
| fwd+bwd | BFloat16 | 1024 × 8192 | 6852.1 (60.2) | 5691.9 (72.4) | 1.2 |

Fusion won at every shape, 1.15 to 2.2 times. At `d = 1024` the second
product's right operand, the 64 MiB hidden activation, would need eight chunks
of 64 tiles each, and the estimate declines it; the first product runs as one
launch, and in training the backward product with the activation derivative
and bias gradient is fused too. The first rule, which declined products past
three quarters of L2 or an inner dimension of 4096, gave 1.26 to 2.26 times;
column chunks raised the forward at `d = 768` from 1.41 to 1.53.

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
