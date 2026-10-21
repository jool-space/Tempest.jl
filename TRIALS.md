# Trials

Tempest is built by putting its own code on trial against the best known
answer to each problem, measured on its own terms. This file is the docket.
It says what is on trial and how it will be judged; it never says how
Tempest works, which the code and the tests say, and it never plans past the
next measurement. An entry is written when a trial opens and closed with the
number that decided it. Nothing here has tenure, including the challengers.

Each entry: the problem; what stands trial, by file; the challenger, where it
is read; how it is judged. A trial closes one of three ways: the challenger
wins and replaces the code, the code wins and the entry records the number,
or the challenger does not fit a stated constraint, named.

## Open

**Attention's backward costs fifteen forwards.** At 1024 positions and
TF32 the forward takes 0.19 ms and the training step 2.8 ms, at 2048 0.46
and 10.6 ms (the masked-attention trial's table). The step stores the
scores and their probabilities, since the partition may not recompute a
fold, and the backward is a scores product, a resident softmax gradient
over the stored scores, and three products over the stored probabilities
and their gradient (`src/Compiler/plan/partition.jl`, the ban on folds;
`ext/cuTileExt/build.jl`, the resident kernel). Challenger:
FlashAttention-2's backward (`flash_attn/flash_attn_triton.py`, the
Triton version reads shortest), which keeps only the log-sum-exp and the
output's dot with its gradient, and recomputes the probabilities inside
the sweeps of the three backward products from the queries, the keys and
the log-sum-exp, one pass over keys writing the key and value gradients
and accumulating the query gradient. Tempest's index level has the pieces:
the forward already recomputes the scores inside its sweep, and a fold
read in the backward as `exp(s - lse)` with `lse` stored is a map of a
recomputed product, which the partition could price rather than ban.
Judged by the step's time at 1024 and 2048 positions against the forward's,
which FlashAttention holds near two and a half, and by the bytes the step
stores; the answers agreeing with the interpreter.

**The schedule search's cold cost.** Every tile of every product is
compiled and timed once per device and shape: the GPT step's cold
preparation doubled, 71 to 142 s (the schedules trial's table). Challenger:
the staged form of the same search, output tile first at the default
depth, then the depth of the best, then split-K, which tinygrad's BEAM
approximates with its beam width. Judged by the cold preparation's seconds
against the measured step, which must not lose beyond the noise, on the
GPT step and the guard rails.

## Decided

**Masked attention does twice the work: the mask's formula bounds the
sweep, and is adopted.** On trial was the generator visiting every tile of
a product's sweep where a mask rules whole tiles out
(`ext/cuTileExt/build.jl`). The challenger, FlexAttention's block mask
(`torch/_inductor/kernel/flex_attention.py`): the key blocks a mask rules
out entirely are skipped, found by evaluating the mask on blocks. Here the
mask is a formula, so the planner reads it: a value `ifelse(cond, x,
literal)` with `cond` a conjunction of comparisons between index formulas,
or a map of one that is a known number where the mask fails, as the
probabilities of masked scores are zero, carries its formulas and that
number in the view (`src/Compiler/plan/masks.jl`, `masked`). A fold whose
body is that number's identity where the mask fails, in its own term or in
a stored value it loads, sees nothing of a step whose tile fails every
comparison; interval arithmetic on the formulas over the block's indices
bounds the live steps, an interval of the sweep, and the kernel's loop runs
over it (`ext/cuTileExt/mask.jl`, `live_steps`). Judged on the GB10 at
TF32, 64 features, 8 heads, batch 2, the answers agreeing with the
interpreter at 512 positions:

| case | non-causal | causal | ratio |
|---|---|---|---|
| forward, 1024 positions | 195 μs | 145 μs | 0.74 |
| forward, 2048 positions | 771 μs | 457 μs | 0.59 |
| training step, 1024 positions | 3.14 ms | 2.79 ms | 0.89 |
| training step, 2048 positions | 11.95 ms | 10.57 ms | 0.88 |

The forward approaches half at 2048, where the sixteen key steps a query
block would take become eight and a half; at 1024, eight become four and a
half, the diagonal step and the block's fixed cost weighing more. The step
skips in three of its seven kernels: the probabilities' product with the
values, and the two backward products that read the stored, masked
probabilities or their gradient, each from the formula carried with the
stored value; the kernel computing both query and key gradients in one
sweep sees the mask both ways, and runs the union, every step. What is not
judged, named: the step's other kernels, the scores' product and the
resident softmax and its gradient, still visit every tile, since their
sweeps are not along the masked axis or they are no sweep; and the two
WRONG marks the guard rails give TF32 at a sum of 8192 are the tolerance,
not the mask.

**The generator's schedules: a measured set of product tiles wins and is
adopted; neighbours of the other sketches' constants do not, on this
device.** On trial were the cuTile generator's tiles, splits and occupancy
from six constants measured once on one device and a decision tree over
them (`ext/cuTileExt/cuTileExt.jl`, `default_tiles`;
`ext/cuTileExt/contract.jl`, `product_tiles`). The challenger, tinygrad's
BEAM search (`tinygrad/engine/search.py`), the small version: a fixed set of
options per kernel, each compiled and timed on the device, the fastest
kept. For products, the generator now claims a group under each tile of a
set (four output tiles and two depths per class, two and three for a
product that recomputes a fold, with and without split-K where the grid is
small), and the cover measures every claim of the implementation whose
claim it ranks best, since the estimate cannot rank tiles of one product;
unmeasured, each implementation stands by its first claim, the constants'
tile. The tuning record keeps what was found. Judged on
the GB10 by measured time, the same plans agreeing with the interpreter:

| case | before | after |
|---|---|---|
| GPT step, 4 layers, T=64, B=16, tf32, steady | 1.914 ms | 1.382 ms |
| GPT step, full precision, steady | 2.291 ms | 2.190 ms |
| attention's weighted sum, tf32, one kernel | 77 μs | 22.7 μs |
| guard rails, full precision, 36 cases | 8937 μs | 8957 μs |
| guard rails, tf32 | 6255 μs | 6249 μs |
| GPT cold preparation, full and tf32 | 71 s, 76 s | 142 s, 156 s |

The cold preparation doubles: every tile of every product is compiled and
timed once per device and shape, which the record then remembers. For the
other sketches, neighbours of each constant (the elements of a pointwise or
full tile, what a resident tile packs and its occupancy, a streamed
column's rows per step, a streamed row's columns, a split sum's ways) were
claimed and measured the same way: the guard rails came to 8945 μs against
8937, a few small kernels twenty percent faster and as many twenty percent
slower under measurement, and the one occupancy neighbour measured four
times slower than the constant. Not adopted; the patch is kept with the
session. What was not judged, named: a device the constants were not
measured on, where the neighbours are the only way the constants would be
corrected, and the staged search (output tile first, then depth, then
split) that would cut the cold cost by half.

**The constants of the cost model: the fit does not win yet, and is not
adopted.** On trial were the constants a kernel's estimate is made of,
each measured at a few shapes: a package's product line, a memory line, the
seconds per position of each scalar function in each regime, the device's
share from a block count, the overlap of memory and compute
(`src/Compiler/cover/cost.jl`, `seconds`). The challenger, after the
cost model of Halide's Adams autoscheduler (`src/autoschedulers/adams2019`):
the estimate as a sum of its terms with non-negative coefficients fitted to
the tuning record, each measured launch's error relative to its seconds,
drawn toward the prior, with the terms kept beside each measurement. Judged
on the GB10 over the 46 launches the coverage compositions measure, held
out one at a time:

| estimate | mean relative error | median | worst | within 1.5× |
|---|---|---|---|---|
| the constants | 0.359 | 0.238 | 1.43 | 33 of 46 |
| the fit, held out | 0.357 | 0.339 | 1.07 | 35 of 46 |

The worst misses are one kind of kernel: a product of sixteen blocks whose
sweep applies an exponential or a softcap to each stored score, measured at
82 μs where the estimate says 17 at tf32, and at 129 μs where it says 315
at full precision. The same kernel costs about the same in both precisions,
and the per-class table says otherwise: what a function adds to a product
at the calibration shape is not what it costs in a sweep of sixteen blocks.
A term for the serial steps each block takes was offered and given weight
zero, since the same step count is over-priced in one precision and
under-priced in the other. The constraint, named: 46 launches of six
compositions cannot fit nine coefficients, and the misses need a term the
record does not yet discriminate. The entry reopens when the generator's
schedule trial has filled the record with many measured schedules per
group; the fit, the held-out test and the terms file are kept as a patch
with the session, not in the code.

**The cost of a kernel: a measured table of scalar functions, in three
regimes, is adopted.** On trial were estimates that priced a product by its
flops on a fitted line and a kernel's bytes at a bandwidth, and pointwise
work not at all (`src/Compiler/cover/cost.jl`, `seconds`). Where a
product's sweep applies a tanh or a SiLU to each score, the estimate chose
to store the scores, and the measurement said to recompute them. The
challenger, XLA's GPU performance model (`xla/service/gpu/model/`), prices
each elementwise op by a per-op cycle count and combines memory and compute
as the larger plus a twentieth of the smaller. Adopted as a table of seconds
per position for each scalar function, measured on the device on first need
and kept in the rates file: streamed, as what eight applications add to a
fold; swept, as what computing a product's factor by the function adds to
the product; inner, the same for a product that recomputes the fold the
function is applied to. The last two per product class, since what a
function adds to a tf32 product's sweep is a fifth of what it adds to an
FP32 one. The generator reports which functions each kernel applies, at how
many positions, in which regime. Judged by the estimate test on the GB10,
each comparison's two sides estimated against measured, in μs:

| composition, precision | before | after | decision before, after |
|---|---|---|---|
| softcap attention, full | 66/154 vs 358/164 | 337/203 vs 356/163 | stored, wrong; streams |
| SiLU attention (HSTU), full | 66/307 vs 300/135 | a close call, measured | stored, wrong; streams |
| SiLU attention (HSTU), tf32 | 20/67 vs 51/36 | a close call, measured | stored, wrong; streams |
| attention, full | 66/154 vs 97/128 | 76/171 vs 96/127 | streams, both |
| softcap attention, tf32 | measured | 121/34 vs 408/54 | streams, both |
| HSTU step, full, forward | 144/771 vs 120/568 | 163/848 vs 119/559 | stored, both |

What the challenger's model does not see, named: a function's price per
position is one constant per regime and class, measured at one shape at full
occupancy, and the HSTU step's recomputing kernels and the softcap's tf32
kernels run at other shapes and occupancies, where the estimate misses by up
to three times while the decision stands. That is the constants' trial. The
device's division and square root cost twenty times its exponential in the
generated code, which the table now says rather than hides. The table costs
a minute per precision to measure on a cold cache, each function under its
reference product, which a fresh tuning directory, as the test suite uses,
pays at the start.

**Store or recompute across forward and backward: the min-cut partition
wins, and is adopted.** On trial was the cover deciding store against
recompute one proposal at a time, by estimate (`src/Compiler/cover/cover.jl`,
`cover`), which could not express recomputing a map in the backward from a
smaller saved value at all. The challenger, PyTorch's rematerialization
partitioner (`torch/_functorch/partitioners.py`), read in full: a minimum cut
of the joint graph with bytes as the capacity of saving a value, infinite
edges for reads, a source feeding what may not be recomputed, a sink
draining what the backward reads, and recomputation counted as free.
Implemented as `src/Compiler/plan/partition.jl` on the forward-backward
boundary, with Tempest's bytes, folds and opaque work banned, maps
recomputable, a map a backward product reads as a factor banned, and the
cut pinned into the planner; the cover still prices folds. Judged by
measured step time on the GB10, every plan agreeing with the interpreter:

| step | stored, before | stored, after | time, before | time, after |
|---|---|---|---|---|
| SwiGLU block | 62.0 MiB | 38.0 MiB | 1609 μs | 1324 μs |
| norm, SwiGLU, residual | 74.0 MiB | 50.0 MiB | 1552 μs | 1281 μs |
| attention | 10.5 MiB | 10.5 MiB | 212 μs | 210 μs |
| tanh MLP, residual | 28.0 MiB | 28.0 MiB | 654 μs | 660 μs |
| GPT, 4 layers, T=64, B=16, steady | | 4 kernels fewer | 2.450 ms | 2.279 ms |

The first measurement lost by two to six times: the planner's rule that
maps of a recomputed value are recomputed with it, written for folds,
cascaded a recomputed map into a backward product's factor, which cuBLAS
cannot claim and the generator runs at seven times the cost. Limiting the
cascade to folds was the whole difference. What the challenger's objective
does not see, named: bytes stand for time only where a recomputed map lands
in a kernel that absorbs it, which Tempest's products do not; the ban on
product factors is the constraint, and the cost-model trial is where it
would be priced instead of banned.
