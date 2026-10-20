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

**Store or recompute across forward and backward.** The cover decides store
against recompute one proposal at a time, greedily, priced by estimate
(`src/Compiler/runtime/decide.jl`, `cover`). Measured, the training step's
streaming backward is fifteen times slower than storing, and nothing in the
cover sees the forward and the backward as one graph. Challenger: PyTorch's
min-cut rematerialization (`torch/_functorch/partitioners.py`), which cuts the
joint graph by bytes with cheap work allowed to recompute. Judged by the
measured time of the attention and layer-norm training steps at the
coverage suite's sizes and at 2048 positions, and by compile time at eight
identical layers.

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

Nothing yet.
