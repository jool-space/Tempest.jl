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

**The cost of a kernel.** Estimates are lines fitted from two reference
products per package, plus bytes at a measured bandwidth
(`src/Compiler/runtime/decide.jl`, `line!`, `fit!`, `seconds`). They do not
price pointwise work inside a product's sweep, and where that work decides,
the estimate contradicts the measurement (the skipped cases in
`test/decisions.jl`). Challengers: XLA's GPU performance model
(`xla/service/gpu/model/`), which counts elementwise work per element, and
the featurization of Halide's Adams autoscheduler
(`src/autoschedulers/adams2019`), fitted to the tuning record Tempest already
keeps. Judged by the estimate test holding for every composition with no
case skipped, and by no comparison the cover decides by estimate
contradicting its measurement beyond the noise.

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
