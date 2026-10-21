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

**The constants of the cost model.** A kernel's estimate is memory time
against compute time, each from constants measured at a few shapes: a
package's product line from three reference products, a memory line from
two, the seconds per position of each scalar function in each of three
regimes from one shape each, the device's share from a block count, and the
overlap of memory and compute from XLA (`src/Compiler/runtime/decide.jl`,
`seconds`, `fit!`, `fit_applied!`). Where a kernel's shape or occupancy
differs from the reference's, the estimate misses by up to three times (the
skipped cases in `test/decisions.jl`). Challenger: the featurization of
Halide's Adams autoscheduler (`src/autoschedulers/adams2019`), whose cost is
hand-written terms with non-negative coefficients learned from measured
schedules, here fitted to the tuning record Tempest already keeps, as it
grows. Judged by the held-out relative error of the fitted estimate against
the constants', over the record, and by the estimate test holding with no
case skipped.

**The generator's schedules.** The cuTile generator picks tiles, splits and
occupancy from six constants measured on one device and a decision tree over
them (`ext/cuTileExt/cuTileExt.jl`, `default_tiles`; `ext/cuTileExt/contract.jl`,
`product_tiles`). Challengers: tinygrad's BEAM search over per-kernel options
measured on device (`tinygrad/engine/search.py`), the small version, and TVM's
MetaSchedule (`python/tvm/meta_schedule`), the full one. Judged by the
guard-rail cases and the coverage compositions against the constants, on
this device and on one it was not measured on, and by the seconds a cold
preparation adds.

**Masked attention does twice the work.** A causal mask is a formula over
positions, and the generator still visits every tile of the scores
(`ext/cuTileExt/build.jl`). Challenger: FlexAttention's block mask
(`torch/_inductor/kernel/flex_attention.py`), which skips tiles the mask
rules out entirely. Tempest can derive which tiles those are from the mask's
index formula at plan time. Judged by the measured time of causal attention
at 1024 and 2048 positions against non-causal, which should approach half.

## Decided

**The cost of a kernel: a measured table of scalar functions, in three
regimes, is adopted.** On trial were estimates that priced a product by its
flops on a fitted line and a kernel's bytes at a bandwidth, and pointwise
work not at all (`src/Compiler/runtime/decide.jl`, `seconds`). Where a
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
generated code, which the table now says rather than hides.

**Store or recompute across forward and backward: the min-cut partition
wins, and is adopted.** On trial was the cover deciding store against
recompute one proposal at a time, by estimate (`src/Compiler/runtime/decide.jl`,
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
