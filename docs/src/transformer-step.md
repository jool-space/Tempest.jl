# A transformer training step

23 September 2026. `examples/gpt.jl` prepares and captures one training step
of a character-level GPT as a single CUDA graph. It is the workload the
[compiler direction](compiler-direction.md) named as the proof of the design:
unsupported operations, retained activations, state transitions and
preparation cost all show up in one program.

## The model

The example follows nanoGPT's `model_full.jl` (jool-space), which trains with
Reactant/XLA, Enzyme and Optimisers.jl, layer for layer:

- token embedding with a tied output head, and a final RMSNorm;
- four blocks of RMSNorm, grouped-query attention (4 query heads, 2 key/value
  heads, rotary positions, causal mask), RMSNorm and a SwiGLU feedforward;
- mean cross-entropy, and Adam (Optimisers' `AdamW(3f-3)`, whose default weight
  decay is zero).

Hidden size 128, feedforward 384, context 64, batch 16: 806,400 parameters and
1,024 tokens per step. Two formulations differ from the reference without
changing the mathematics. The causal mask is an additive 0/−∞ matrix. Rotary
embedding uses the rotate-half form, a fixed row permutation, instead of
concatenation. Tokens and targets are runtime `Int32` indices:
`wte[:, tokens]` gathers the embeddings, and `Tempest.take_rows(logits, targets)`
picks the target logits. Both have deterministic accumulating pullbacks.
The speed comparison below was measured with one-hot token and target
matrices, before runtime indices existed; switching removed an embedding
product and 0.9 ms of host batch preparation, and replay went from 6.18 to
6.12 ms.

Built-in verbs carry the operations the array vocabulary could not.
`Tempest.batched_mul` (with transpose flags) computes the per-head products and
`Tempest.logsumexp` a stable softmax, `exp.(s .- logsumexp(s))`, and the
cross-entropy. CPU and CUDA implement both: `LinearAlgebra.mul!` and cuBLAS
strided-batched GEMM, and a shifted loop and a warp-per-column kernel. Their
pullbacks are further batched products and a pointwise expression, so the
softmax and its gradient fuse into ordinary graph regions.
`Tempest.attention(q, k, v; causal, scale)` is that composition as one verb. It
takes the model's `(features, heads, positions, batch)` layout directly, with
grouped key/value heads, so the model needs no head copies.
Under `matmul_precision = :tf32`, a cuTile provider realizes it with fused
FlashAttention-2-style kernels that never materialize the scores: an online
softmax forward that stores the log-sum-exp, and a deterministic backward
(dK and dV per key block, dQ per query block, no atomics). In full precision,
and for shapes the kernels do not take, the verb expands into the composed
form.

## Correctness

With the same random stream the example draws the same initial parameters and
batches as the reference. Its loss matches the reference model's to seven
digits on the first three batches (4.937932 against 4.9379315). The initial
mismatch of 0.3% came from the key/value head assignment: in the reference's
`(g h)` axis the group index varies fastest. Gradients of every parameter kind,
at random coordinates, agree with Float64 central differences to five
significant digits. Over 200 training steps the two loss curves stay within the
spread that XLA's own FP32 and TF32 runs show against each other.

## Speed against Reactant/XLA

The tables in this section were measured without clock control and overstate
some of the differences; the [profiled comparison](#Where-the-difference-comes-from)
below matches clocks and replaces them where they disagree.

Same GB10, same corpus (Reactant's documentation), 200 steps, run back to back.
Both loops build the next batch on the host and read the loss every ten
steps. *Average* is the running mean from step 2, as the reference script prints
it; *steady* covers steps 101–200. XLA's default math uses TF32 for Float32
products; `CUDATarget(matmul_precision = :tf32)` is Tempest's opt-in
equivalent, and full precision is Tempest's default.

| Products | Reactant/XLA average / steady | Tempest average / steady | Steady speedup |
| --- | ---: | ---: | ---: |
| Full FP32 | 8.6 / 8.3 ms | 6.97 / 6.30 ms | 1.3× |
| TF32 | 7.4 / 7.4 ms | 5.10 / 4.38 ms | 1.7× |
| TF32, fused attention (later) | — | 3.91 / 3.80 ms | 1.9× |
| Full FP32, attention verb (later) | — | 6.86 / 6.76 ms | 1.2× |

An earlier session the same night measured XLA at 7.9 and 6.9 ms (average),
with unlocked clocks; Tempest measured 6.99 and 5.12 ms. Even against those,
Tempest is faster by 1.2× and 1.6× on the steady rate. At step 200 the losses
were 3.084 (XLA) and 3.026 (Tempest) in full precision, and 2.964 and 2.955
with TF32. XLA's own runs vary by about 0.08 at that step.

At GPT-2-small scale (hidden 768, 12 layers, 12 heads without grouping,
feedforward 3072, context 256, batch 16: 113 million parameters and 4,096
tokens per step), measured the same night with TF32 products:

| | Reactant/XLA | Tempest, composed attention | Tempest, fused attention |
| --- | ---: | ---: | ---: |
| Step time, context 256, batch 16 | 495 ms | 417 ms | 362 ms |
| Step time, context 1024, batch 8 | 1226 ms | 1138 ms | 779 ms |
| Compile / prepare, then first step (context 256) | 93 s + 135 s | 117 s in total | 79 s in total |
| Loss at step 30, context 256 / 1024 | 3.737 / 3.570 | 3.725 / 3.577 | 3.725 / 3.577 |

Step times are running means over 30 steps. The fused and composed Tempest
runs have identical loss trajectories to three decimals. At context 1024,
the composed step's profile spent about 480 ms of 1135 on attention.
That was 193 ms in batched products, most of the 261 ms of CUDA pointwise
work (passes over 400 MB score tensors), 31 ms in logsumexp and 29 ms in
softmax sums. In isolation, the fused kernels take about 13 ms per layer for
forward and backward together.

At this size matrix products are half of Tempest's step (206 ms), with cuBLAS
workspace size making no difference. Pointwise work runs at copy bandwidth,
apart from two items. The Adam update writes its new moments and reads them
back in a second pass, about 43% more traffic than one fused pass.
The CUDA axis sum over 256-long softmax columns reached 65 GB/s with one block
per column. Columns of at most 1024 elements now take one warp each: 166 GB/s
on the same sum in isolation.

Compilation is comparable: XLA spends 68–77 s compiling plus 12–14 s in its
first step; Tempest spends 72 s tracing, differentiating, preparing and
capturing. A second preparation of the same program in the same session takes
1 s. Renumbering graph-region values locally, so identical regions in every
layer share one compiled kernel, later cut cold preparation from 56 to 49 s. Nearly all of the cold time is Julia and GPU code generation for the ~100
distinct kernels, of which about 13 s is cuTile compilation.

In full precision the verb expands into its composed reference, and the
reference backward recomputes the scores from the saved log-sum-exp instead of
keeping the probabilities. That costs the small model about 7% against the
hand-composed attention it replaced (6.76 against 6.30 ms steady).

**TF32 and training quality.** Under TF32, gradients at the initial parameters
differ from full precision by 6 × 10⁻⁴ (median relative error over the
parameters, 10⁻³ at worst), the same with fused and composed attention.
One 400-step run with nanoGPT's seed averaged a higher TF32 loss from step 210
on (2.73 composed, 2.82 fused, against 2.56). Three further seeds
(`TEMPEST_GPT_SEED=1:3`) showed no penalty; the first run was trajectory
variance.

| Mean loss, steps 210–400 | Seed 1 | Seed 2 | Seed 3 | Mean |
| --- | ---: | ---: | ---: | ---: |
| Full precision | 2.438 | 2.448 | 2.472 | 2.453 |
| TF32, composed attention | 2.447 | 2.496 | 2.383 | 2.442 |
| TF32, fused attention | 2.438 | 2.377 | 2.457 | 2.424 |

## Where the difference comes from

Both steps were traced with Nsight Systems on the afternoon of 23 September
(scripts in `examples/research/xla_comparison`),
minutes apart, with the SM clock logged every 0.5 s. Steps are delimited by
the per-step batch upload; the tables cover the steady steps of each run.
The GB10 was power-capped the whole time: 650–950 MHz under load against its
2418 MHz application clock. The night's runs sat as low as 650 MHz, and that
varied between runs. A GPU-bound step (Tempest) slows with the clock far more
than a host-bound one (XLA at the small size), which is how the tables above
came to overstate some ratios.

**Small model, full FP32** (clock not logged for these two runs):

| Per step | Reactant/XLA | Tempest |
| --- | ---: | ---: |
| Wall time | 8.66 ms | 4.96 ms |
| GPU busy | 3.25 ms | 4.66 ms |
| GPU idle | 5.41 ms | 0.29 ms |
| Kernels launched | 407 | 418 |
| Matrix products | 1.82 ms | 2.75 ms |
| Everything else | 1.72 ms | 1.91 ms |

At this size Tempest wins only through how it replays its graph. XLA's kernels
are about a quarter faster in total, and XLA also runs them as a CUDA graph,
but the GPU idles 62% of its step. Each step, XLA rewrites the parameters of
about 400 kernel nodes (`cuGraphExecKernelNodeSetParams`), because buffer
addresses change between steps. Reactant also waits on the host for the
previous step's donated outputs before it launches, so that host work never
overlaps the GPU. Tempest's buffers are owned and stable, so its replay is one
launch with no updates. An earlier measurement (8 September, clocks unlogged)
found XLA at 4.9 ms per step with graphs disabled, against 6.6 ms with them.
With graphs disabled, XLA may be close to Tempest at this size; that was not
remeasured here. XLA's products at `highest` precision are
Triton-generated GEMM fusions of pure FP32 multiply-adds (the dumped LLVM IR has
1.7 million `fmuladd` calls and no tensor-core instructions). They beat the FP32
SIMT kernels cuBLAS picks for Tempest's shapes. cuBLAS adds 49 split-K reduction
passes per step (0.27 ms).

**GPT-2-small shape, TF32, context 256, batch 16:**

| Per step | Reactant/XLA | Tempest, composed attention | Tempest, fused attention |
| --- | ---: | ---: | ---: |
| Mean SM clock | 942 MHz | 880 MHz (650–942) | 942 MHz |
| Wall time (traced) | 362 ms | 376 ms | 269 ms |
| GPU idle | 24 ms | 5 ms | 4 ms |
| Matrix products | 217 ms | 197 ms | 144 ms |
| Everything else | 122 ms | 174 ms | 121 ms |

At equal clocks the composed-attention step is roughly even with XLA, not 1.19×
faster. Scaling it by the clock ratio gives about 351 ms, though memory-bound
kernels do not scale with the SM clock. Tempest's products are 20 ms cheaper:
cuBLAS picks Blackwell `nvjet` TF32 kernels for some of its shapes, while XLA's
cuBLAS calls use `cutlass_80` kernels. Its other work is 52 ms dearer: pointwise
passes over score-sized tensors, the causal fill (12 ms) and the head copies.
Fused attention is the real gain, 1.35× over XLA at matched clocks, by removing
54 ms of batched products and most of those passes. The context-1024 runs were
not repeated under the profiler.

Summary: Tempest's lead comes from replaying a static graph over stable
buffers, which matters when the step is small, and from a pluggable fused attention kernel, which matters
when it is large. It does not come from better generated code. XLA's pointwise
and reduction fusions, and its FP32 GEMMs, are as good as or better than
Tempest's.

## Where the time goes

`Tempest.profile(ex)` times every call of the schedule. Full precision, 302
calls for 363 regions, sums of per-call medians:

| Work | Time per step | Calls |
| --- | ---: | ---: |
| Matrix products (cuBLAS, FP32 SIMT) | 3.86 ms | 89 |
| Pointwise regions | 1.56 ms | 115 |
| Batched products (attention) | 0.72 ms | 24 |
| cuTile resident column regions (norms) | 0.52 ms | 19 |
| Other cuTile regions | 0.42 ms | 21 |
| Scatter (gather pullbacks) | 0.31 ms | 16 |
| cuTile split row sums (norm-weight gradients) | 0.20 ms | 9 |

Matrix products are half the step. At these sizes cuBLAS's FP32 kernels reach
about 1.7 TFLOP/s, and the GPU's clocks are not locked. TF32 halves them.

## What this workload changed

Profiling the step drove five changes, each measured on the step and on the
normalization fixture (see [reduction domains](reduction-domains.md)):

1. **Generated CUDA kernels** resolved every read through runtime-indexed
   per-dimension loops and an outlined closure, which spilled 704 bytes per
   thread to local memory. Reads are now resolved against the region's shape
   at build time. The decomposed RMSNorm step went from 5611 to 149 μs.
2. **Mapped reads** (the attention head copies: a gather inside a
   permutation) use unsigned 32-bit division: 37 to 20 μs per copy.
3. **Short resident columns** (128-row RMSNorm) pack four columns per block:
   34% to 60% of copy bandwidth.
4. **Row sums over tokens** (norm-weight gradients) ran on four blocks. They
   now split across blocks with a finishing pass: 76 to 23 μs each.
5. **TF32** became an explicit option.

Device time per step (replay only) went from 7.45 ms to about 6.2 ms in full
precision.

## Open

- **FP32 products at small shapes.** cuBLAS picks SIMT kernels with split-K
  reduction passes. XLA's Triton FP32 GEMMs take 1.82 against 2.75 ms per step
  on the small model. A cuTile GEMM provider, or cuBLASLt heuristics with
  split-K disabled, is the obvious experiment.
- **Composed attention's non-product work.** 52 ms more per step than XLA's
  fusions at GPT-2-small size, mostly score-sized pointwise passes and a
  separate causal fill. Fused attention avoids it under TF32; full precision
  still pays it.
- **Clock control.** The GB10 is power-capped at 650–950 MHz under load, and
  runs differ. Comparisons need the clock logged, or runs interleaved.
- **Index accumulation at scale.** The embedding gradient scans every index
  once per table column (in chunks, with warp-ballot compaction). That is
  fine for this vocabulary. For a 50,000-token vocabulary it would want a
  device sort by index.
- **Horizontal fusion of products that share an input** (Q, K and V; gate and
  up). A grouped cuBLAS launch (`cublasSgemmGroupedBatched`) was 22% faster
  than three separate same-shape products in isolation. Grouping every run of
  consecutive independent products in the step, however, made the step much
  slower: 9.5 against 6.2 ms in full precision and 5.4 against 3.7 ms with
  TF32. Mixed shapes, such as a weight gradient with a 1024-long inner
  dimension next to an input gradient with 128, share one kernel configuration
  there. A useful version must group only products of matching shape, which
  first needs a schedule that places Q, K and V together.
- **Multi-output pointwise fusion.** Seeding cuTile graph regions at pointwise
  values with several consumers was tried and made the step slower (6.28
  against 6.18 ms). The regions stayed at a shared value and one consumer. Their
  dense-boundary inputs had to be materialized, where the scalar kernels had
  computed them inline, and growth stopped at producers that read layout views.
  Useful multi-output fusion needs region growth that can absorb those
  producers.
- **Cold preparation.** Kernel compilation is serial; a persistent cache or
  parallel compilation would change the 72 s the most.
- **Symbolic shapes** remain unsupported. Dropout exists: the counter-based
  `Tempest.dropout` gives residual dropout with `TEMPEST_GPT_DROPOUT=0.1`
  (6.39 ms/step steady in a short run), and the CUDA-versus-CPU transformer
  test runs with it.
