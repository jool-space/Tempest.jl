# Spelling invariance

!!! note "Historical report"
    Written 15 October 2026, with changes on the `spelling-audit` and `outside-review` branches. It records the state, measurements and plans of that
    date; the current design is described in [Architecture](architecture.md),
    whose open directions replace the plans here.

An audit of whether equivalent spellings of a computation reach equivalent
plans, what made them diverge, and the changes made on 15 October 2026. The
interface direction of the same day (now [interface principles](architecture.md#Interface-principles))
asked which parts of the interface a perfect compiler would keep. Its answer implies this property: what to
compute is the caller's choice, and how it is spelled is not. A spelling that
plans worse is the compiler leaking into the interface.

## The trigger

Softmax down the columns of a Float32 matrix, forward only, on GB10:

| spelling | plan before | 1024×4096 | 16384×512 | 65536×128 |
|:--|:--|--:|--:|--:|
| `e = exp.(x .- maximum(x; dims=1)); e ./ sum(e; dims=1)` | 1 cuTile graph region | 145 | 306 | 337 |
| `exp.(x .- fold(LogSumExpFold(), x; dims=1))` | 1 cuTile graph region | 145 | 305 | 370 |
| `exp.(x .- logsumexp(x; dims=1))`, the documented recipe | CUDA logsumexp call, CUDA pointwise | 183 | 702 | 1931 |

Times are μs, warm, the median of seven samples of a CUDA graph of 50
executions. Run-to-run variation is up to 10%.

`logsumexp` recorded a semantic call. Calls are resolved before any region
forms: if an implementation supports the call alone, it stays opaque, and its
reference, which is the fold, never meets its neighbours. The CUDA warp-per-column
kernel supported it. The pointwise consumer then ran alone, outside cuTile,
whose scalar regions decline broadcast reads. The same verb along the second
axis, which the CUDA kernel declines, expanded to the fold and fused, so
`logsumexp(x; dims=2)` was fast and `logsumexp(x; dims=1)` slow.

## How a program becomes regions

In order, in `prepare` (`src/runtime/preparation.jl`):

1. **Implementation reverse rules** (`prepare_reverse_rules`) may replace a
   forward/backward call pair with private calls, such as cuDNN's attention
   pair.
2. **Semantic resolution** (`resolve_semantics`, `src/compiler/lowering.jl`).
   Each call is offered to the candidates as a `CallRequest`. The first
   `Supported()` keeps it as one opaque node; otherwise its reference is traced
   and spliced in, recursively. This is decided per call, by support alone,
   before anything sees the call's neighbours.
3. **Proposals** (`select_proposed_regions`). Implementations propose node sets
   on the resolved program; Tempest ranks them by candidate order, then by the
   bytes they keep out of memory against the default partition, and accepts
   them greedily if they do not overlap, keep the partition acyclic and pass
   `support`.
4. **Graph regions** (`select_graph_regions`). Regions grow from each
   reduction, in node order, through pointwise operations and constants, up to
   64 members, around claimed proposals. A complete candidate is tried first,
   then support-guided growth.
5. **Default partition** (`plan`). Fusion roots are materialized values,
   reductions, products, calls, layout boundaries, results, and pointwise
   values used more than once. Everything else inlines into scalar
   expressions.
6. **Construction** (`select_schedule`). Candidates are tried in order;
   independent elementwise steps are grouped first. A construction rejection of
   a call expands its reference and replans; of a region, blocks that member
   set and replans.

Where the order of these decisions, not a comparison, decided the outcome:

- A call that any implementation supported was never compared with its
  reference fused into its neighbours (step 2). The trigger above.
- Proposals could not contain calls, so a named call could not be claimed with
  its epilogue the way a product is (step 3).
- Growth in step 4 is greedy from seeds in node order; the documentation
  already records it.
- Grouping in step 6 precedes individual selection; documented.

Other gates that made spellings diverge, without being decisions about order:

- **Recomputation permission.** `recomputable` listed twelve Base functions.
  `x .^ -0.5f0`, `max(init, y)` and the broadcast `identity` that a sum's
  pullback emits all left graph regions.
- **cuTile scalar regions** decline mapped or broadcast reads. A pointwise step
  cut off from a reduction by a call ran on the CUDA kernels.
- **cuTile graph regions** accept Float32 values of rank at most two, and no
  layout operations, matrix products or calls.
- **cuDNN epilogue proposals** anchor on `MatMul`, grow through Base functions
  with a cuDNN mode, and stop at layout operations.

## The spelling matrix

Each row is a set of spellings with the same mathematical result. Times are as
above; "regions" counts scheduled steps and, after the slash, kernels. Float32
on cuTile unless stated.

### Divergences found, and their state after the changes

| computation | spellings | before | after |
|:--|:--|:--|:--|
| softmax, 16384×512 | naive / fold / `logsumexp` | 306 / 305 / 702 μs; 1 / 1 / 2 regions | 311 / 299 / 305 μs; 1 region each |
| softmax, 65536×128 | naive / fold / `logsumexp` | 337 / 370 / 1931 μs | 352 / 363 / 362 μs |
| log-softmax, 65536×128 | naive / `logsumexp` | 368 / 1931 μs | 364 / 362 μs |
| `logsumexp`, 65536×128 | manual / fold / verb | 133 / 121 / 1678 μs | 134 / 124 / 130 μs |
| softmax, no cuTile, 65536×128 | fold / `logsumexp` | 1140 / 1933 μs | 1141 / 1142 μs |
| softmax gradient, 1024×4096 | naive / fold / `logsumexp` | 1267 / 648 / 1086 μs; 4 / 2 / 3 regions | 660 / 661 / 660 μs; 2 regions each |
| cross-entropy training | `logsumexp` / fold | 374 / 295 μs | 286 / 302 μs |
| RMSNorm, 1024×4096 | `inv.(sqrt.(…))` / `(…) .^ -0.5f0` | 143 / 196 μs; 1 / 2 regions | 134 / 141 μs; 1 region |
| column maximum | `maximum` / `reduce(max; init = -Inf32)` | 1 / 2 regions | 1 region |
| Float16 MLP block | `*` / `batched_mul` with batch 1 | 223 μs / error | 236 / 379 μs (see below) |
| dropout between pointwise work, 1024×4096 | with / without dropout | 385 / 141 μs; 3 / 1 regions | 139 / 136 μs; 1 region |
| causal attention without a fused kernel, 64×12×1024×4 | forward; forward and backward | 9.3 ms; 24.2 ms | 7.9 ms; 19.6 ms |

### Spellings that already agreed

| computation | spellings | plan |
|:--|:--|:--|
| `A' * B` | `transpose`, `'`, `permutedims`, `permutedims(A, (2, 1))`, `transpose(B' * A)` | one GEMM, 174–176 μs |
| fold spellings | `prod`, `reduce(*)`, `mapreduce(identity, *)`, `fold(ProdFold())`; `maximum`, `reduce(max)` | one graph region |
| `sum(abs2, x)` | `sum(abs2, x)`, `mapreduce(abs2, +, x)`, `sum(x .* x)` | one region of three kernels |
| Float16 MLP with cuDNN | `w2 * tanh.(w1 * x .+ b1) .+ b2`, bias first, step by step with `map`, row-major `tanh.(x * w1' .+ b1') * w2' .+ b2'` | two cuDNN regions, 216–249 μs |
| product + bias + `tanh` | `batched_mul` with batch 1 / reshaped `*`, Float32 | GEMM and one pointwise, 12 μs |
| RMSNorm | `sum(abs2, x)`, `sum(x .^ 2)`, `mapreduce(abs2, +, x)` | one graph region |

### Divergences that remain

| computation | spellings | plan and time | why |
|:--|:--|:--|:--|
| mean-based norms | `Statistics.mean(x; dims)` / `sum(x; dims) ./ n` | error; now one graph region each | the Statistics extension spells `mean`, `var` and `std` with sums |
| dot product, 2²⁰ | `sum(a .* b)` / `reshape(a, 1, :) * reshape(b, :, 1)` | 61 / 10 μs; now 14 / 7 | not the staged partial sums: their kernels compiled without an occupancy hint ([outside review](outside-review.md)) |
| softmax over a 3-D array | rank 3 / reshaped to rank 2 | CUDA kernels / one graph region; now one graph region each, 145 / 136 μs | was: cuTile graph regions are rank ≤ 2; leading and trailing axes now collapse |
| softmax of a permuted matrix | `logsumexp(x; dims=2)` / `permutedims(logsumexp(permutedims(x)))` | 155 / 358 μs; now one graph region each | was: layout operations end graph regions; a fold over a permutation now folds the source |
| RMSNorm with dropped statistics | `(1, n)` statistics / `dropdims(…; dims = 1)` then `'` / `vec(…)` then `'` | 138 / 198 / 191 μs; now one graph region each | was: the reshape and the transpose back split the region; reshapes now sink to their source and exact aliases collapse |
| sum of a flattened value | `sum(abs2, tanh.(x))` / `sum(abs2, vec(tanh.(x)))` | 251 / 374 μs; now 48 each | as above, with the occupancy hint |
| Float16 MLP with cuDNN | `*` / doubly transposed products | 236 / 577 μs | cuDNN proposals stop at a layout operation on the product |
| Float16 MLP with cuDNN | `*` / `batched_mul` | 236 / 379 μs; training 736 / 1126 μs | cuDNN proposals anchor on `MatMul` only |
| Float16 MLP with cuDNN | `max.(h, 0)` / user `relu(v) = max(v, zero(v))` | 2 / 3 regions | cuDNN recognizes Base functions only |
| softmax without cuTile | naive / fold, 65536×128 | 1491 / 1141 μs; 4 / 2 regions | the CUDA kernels have no multi-reduction regions; the naive spelling has two reductions |
| attention under TF32 | `attention` / spelled with `batched_mul` and `logsumexp` | 37 / 161 μs | nothing recognizes a spelled-out attention |
| attention training without a fused kernel | `attention` / spelled out | 562 / 501 μs | the reference backward recomputes the scores; differentiating the spelled form stores them |
| dropout in training | dropout inside a cuTile graph region | outside | cuTile kernels do not compute positions or take integer boundaries |

The last group is not one problem. Most rows are coverage of one provider
(cuTile's rank and type limits, cuDNN's anchors and modes, the full-reduction
schedule); two are missing vocabulary (`mean`, recognizing attention); one is
inherent to the fallback provider. None needs a change to how calls and
regions interact.

## How deep the problem is

Ten built-in verbs recorded semantic calls. Classified by what they are:

| verb | class | now |
|:--|:--|:--|
| `logsumexp(x; dims)` | a fold | `fold(LogSumExpFold(), x; dims)`; the CUDA kernel and the CPU call are removed |
| `batched_mul` | a primitive product, like `*` | a `BatchedMatMul` node, scheduled as a batched GEMM |
| `dropout` | pointwise, given each element's position | a pointwise function of the element, the seed and a `Positions` node |
| `causal_fill` | pointwise, given each element's position | a pointwise function of the element and its position |
| `take_rows`, `put_rows` | pointwise and a fold, given positions; see below | calls |
| `gather_columns`, `scatter_columns` | runtime-index reads and scatters | calls |
| `attention_forward`, `attention_backward` | genuine semantic units with a numerical promise | calls |

So most of the problem was a handful of verbs that should not have been calls.
Four of them were operations Tempest's vocabulary could already express, once
it had positions. The structural part is narrower: a named call was opaque to
proposals, and whether a call stays whole was decided before region
formation, by support. After the changes, the calls that remain are the four
index verbs, whose references are host loops that could not fuse anyway, and
attention, whose fused kernels beat the reference wherever they apply: 4.4
times forward and 2 times in training against the spelled-out form above, at
512 positions, and more at longer sequences.

`take_rows` and `put_rows` could be a select by position followed by a sum over
rows, which would fuse cross-entropy's `logsumexp(x) .- take_rows(x, t)` into
one region over `x`. It would read every element where the gather reads one
per column, and it only pays inside a cuTile graph region, which does not take
integer inputs yet. They stay calls until it does.

## What is a primitive and what is a call

`*` is a named operation, so why can a library fuse it with a bias while
`attention` could not take an epilogue? There was no principled reason. The
difference that mattered was only this: every target has a realization of a
product (cuBLAS, LinearAlgebra), so a product never needs to fall back, and it
became a primitive node; attention has optional realizations and a reference to
fall back on, so it became a call, and calls were invisible to proposals.

The rule after these changes:

- **Primitives** are operations every target can realize: pointwise
  functions, folds, products (`MatMul`, `BatchedMatMul`), layout operations,
  positions. A named verb whose meaning is one of these lowers to it at trace
  time and keeps its name only as the Julia function (`logsumexp`,
  `batched_mul`, `dropout`).
- **Calls** are operations with an optional realization and a reference:
  attention, the index verbs, and user verbs. A call is a graph node like a
  product. If an implementation supports it alone, it stays a node; otherwise it
  expands into its reference before regions form, and the reference fuses like
  any other code.
- **Claims are uniform.** A proposal may contain products, folds, pointwise
  work and calls. A library claims `{product, bias, activation}` and
  `{attention, the pointwise work on its output}` through the same
  `propose_regions`, ranking and boundary computation.

### Attention with an epilogue

A cuTile or cuDNN attention kernel that applies pointwise work to its output
defines `propose_regions`: for each live `attention_forward` call, the call's
result nodes and the pointwise users of `o`, every prefix as a nested
alternative, as cuDNN's product epilogues are proposed
(`ext/cuDNNExt/matmul.jl`). The partitioner adds the call's other result
(`lse`) to the proposal, computes the boundary (`lse` escapes when backward
reads it), ranks it by the bytes it keeps out of memory, and offers the
implementation a `ProposedRequest`. `support` can decline an epilogue it cannot
generate; construction receives every input and escaping output. In training,
the proposal sees the private forward call left by the reverse rule. A
declined proposal leaves the call as it was. `test/compiler/proposals.jl`
claims a two-result verb with its epilogue this way, forward and in training.
Neither attention kernel generates an epilogue yet.

The call must still survive semantic resolution on its own: some
implementation must support it alone. An implementation that could realize
attention only together with an epilogue would have to support the lone call
too. Both current attention implementations do.

### What this costs

- **A claimed call is not compared with its reference.** If an implementation
  supports a call, the call stays whole even where the expanded reference would
  fuse into a cheaper plan. The implementation is responsible for declining
  when it would lose, as cuDNN's product proposals decline by a cost estimate.
  This is the order dependence that caused the trigger; after the changes it
  applies to attention, where the kernel wins, and to user verbs.
- **A claim is all or nothing.** A proposal containing a call contains all of
  its results; the call's internals cannot be split across regions.
- **Proposers see internal IR.** Matching a call reads `node.op.call.f` and
  `.kwargs`; matching a product reads `MatMul`. Neither is public (see
  [Architecture](architecture.md#Open-directions)).
- **`logsumexp` is no longer a call an outside package can replace.** A library
  logsumexp kernel would have to be a graph-region or proposal implementation.
  `test/packages/MyLib` now implements `take_rows`.

## Alternatives considered

**A transparent generic indexed operation.** tinygrad spells a product as
reshape, broadcast multiply and sum, so one vocabulary covers pointwise work,
movement and reductions and fusion sees everything. The September comparison
concluded that a `MatMul` should expose an indexed contraction to a scheduler
while keeping its identity for cuBLAS and custom kernels, and that fusion
needs a more general domain than reduction-seeded regions. Tempest's
vocabulary is already most of such an operation: pointwise nodes carry a
scalar Julia body, folds are indexed reductions with an algebraic declaration,
layout operations are index maps, and positions now give a body its
coordinates. What it lacks is a contraction that a scheduler can fuse. Making
products transparent now would not help: no provider schedules a fused
contraction, so every library realization would have to rediscover the
product from a broadcast and a sum, and the broadcast is materialized when
nothing does (the old `batched_mul` reference built an m × k × n × batch tensor
and failed in Float16). The product stays a named node, and its indexed form
stays available as a reference for the day a provider (a cuTile product with
fused neighbours) can schedule it. Named operations are then labelled sugar
over that form, which is the same design with the label kept as the node.

**Always expand, and keep the name as a label on the node group.** Every call
would expand before region formation, and proposers would match labelled groups.
The benefit, the reference fusing with neighbours when nobody claims the call,
Tempest already has: an unclaimed call expands. The costs are real: attention's
group contains score-sized intermediates that the default partition must not
see unclaimed; graph regions could absorb part of a group before a proposer
looks, which is another order dependence; and reverse rules and paired
forward/backward saved state work on calls, not groups.

**A claimed call competing with its fused reference by cost.** The compiler
direction already deferred this. It needs a time comparison between an opaque
kernel and a fused region, where Tempest's selection compares bytes and is
deterministic. After the changes no built-in call has a case for it: attention's
kernels win, and the index verbs' references cannot fuse.
Reconsider it when a user verb needs it.

**Canonicalization to one spelling.** tinygrad reaches spelling invariance by
rewriting equivalent forms to one. Tempest already canonicalizes exact
identities while tracing: a transpose of a transpose, `identity` and conversions
to the same type, multiplication by a literal one, constant folding, `reduce`
and `mapreduce` with `+ * max min` to folds. The divergences above are mostly
not of that kind. Naive softmax and the logsumexp fold round differently, and
rewriting one into the other is a numerical choice the caller has not
permitted ([interface principles](architecture.md#Interface-principles)). The
exact rewrites that remain are worth a pass: `transpose(A * B)` to
`transpose(B) * transpose(A)`, which would give the doubly transposed MLP the
cuDNN plan, and a permuted reduction to a reduction over the permuted axis.

## Changes on the `spelling-audit` branch

1. **`logsumexp` is its fold**: no call, no CUDA kernel. The docstring's softmax
   and log-softmax recipes now reach one graph region.
2. **Base's pure functions are recomputable**: arithmetic, elementary
   functions, conversions and `identity`.
3. **Proposals can claim semantic calls**, with all of a call's results.
4. **`batched_mul` is a primitive product**: `BatchedMatMul`, a batched GEMM on
   cuBLAS (Float32, Float64 and, with Float32 accumulation, Float16 and
   BFloat16) and LinearAlgebra.
5. **Dropout is pointwise** over a new `Positions` node, the one-based linear
   index of each element, which kernels compute from their iteration index. The
   keep test compares the hash's 24 bits with an integer threshold, which
   selects the same elements as the old Float64 comparison. Kept values are now
   multiplied by `1 / (1 - rate)` rounded to the element type instead of divided
   by `1 - rate`, a difference of at most one rounding.
6. **`causal_fill` is pointwise** over positions.

`test/cutile/spellings.jl` asserts plan shapes for the documented recipes and
the spelling pairs: softmax, log-softmax and logsumexp in every spelling, the
RMSNorm power spelling, `reduce(max; init)`, the softmax gradient, batched
against plain products in Float32 and Float16, dropout between pointwise work,
and the attention reference without calls.

### Benchmarks

The GPT example's captured training step (`examples/gpt.jl`, device time per
replay, median of nine samples of 20) did not change beyond noise:

| configuration | before (ms) | after (ms) |
|:--|--:|--:|
| default (128 hidden, 4 layers) | 2.264 | 2.277 |
| TF32 | 1.810 | 1.818 |
| dropout 0.1 | 2.317, 2.303 | 2.315, 2.318 |
| 768 hidden, 12 heads, 2 layers, T = 256, B = 8 | 30.61 | 30.74 |
| the same, TF32 | 21.16 | 21.19 |
| the same, dropout 0.1 | 31.02, 31.12 | 30.95, 30.85 |

The step's logsumexp is over a vocabulary of about a hundred characters, and
its attention runs on cuTile under TF32 or as the reference otherwise.

## Changes on the `outside-review` branch

Made on 15 October 2026 after the [outside review](outside-review.md), which
found the divergences added to the table above:

1. **Occupancy hints for partial-sum kernels.** The cuTile kernels of full and
   split reductions compiled without a hint and ran 4–5 times slower than the
   CUDA kernels they displaced. See [graph regions](graph-regions.md#Native-cuTile-scope).
2. **Exact canonical forms.** While tracing: a permutation of a permutation
   composes, `permutedims(m, (2, 1))` is `transpose(m)`, a full reduction
   reads through reshapes, and a reduction over a permuted array folds the
   source and permutes the result. Before planning
   (`src/compiler/canonical.jl`): reshapes and vector transposes sink through
   single-use pointwise work to the value they present, and layout chains
   that compose to the identity onto an ancestor of the same shape are
   replaced by that ancestor. `dropdims(s; dims = 1)` followed by vector
   arithmetic and `'`, `vec(h)` under a full reduction, `x[:, :]` and a
   cancelled pair of transposes then plan as their direct spellings.
3. **Rank-N graph domains.** A cuTile graph region collapses the leading or
   trailing reduced axes of a fold input of any rank into one matrix axis and
   the rest into the other. Softmax over a rank-3 array, a layer norm over
   `(hidden, tokens, batch)` and the attention reference's softmax over
   `(keys, queries, heads × batch)` form one region, with the kernel of their
   reshaped spellings. The Float32 attention reference forward at
   64 × 12 × 1024 × 4 went from 7.96 to 5.19 ms.

`test/cutile/spellings.jl` asserts the plan shapes; `test/frontend/arrays.jl`
and `test/compiler/planning.jl` the rewrites and the passes on the CPU.
