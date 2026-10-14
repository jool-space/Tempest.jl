# Scheduler investigation

21 September 2026. Follow-up to [the compiler landscape](compiler-landscape.md).
This note separates inspected source, local experiments, and design conclusions.
No external compiler performance results are implied by source inspection.

## The finding that changes the build-versus-reuse question

[NVIDIA TensorIR v0.1.4](https://github.com/NVIDIA/tensor-ir/tree/v0.1.4), commit
`664d385996371db2a8de67b76f82fc89e2470109`, was tagged on September 18. It is an
open C++/MLIR compiler from tensor graphs to CUDA Tile IR, with an optional Python
frontend. This is distinct from TVM's TensorIR. Its early-release status does not
establish production readiness or performance.

The [actual pipeline](https://github.com/NVIDIA/tensor-ir/blob/v0.1.4/lib/Compiler/CudaTile/Pipelines.cpp)
propagates layouts, generates tile candidates, selects tile geometry, creates
buffer boundaries and a GPU grid, forms contraction loops, and lowers the result
to CUDA Tile IR. This is already more than a catalog of named fused operations.
Its graph splitting clones computations for different layout/iteration scopes;
it is not a replacement for training-wide partitioning, saved tensors, or
storage reuse.

The [public C API](https://github.com/NVIDIA/tensor-ir/blob/v0.1.4/include/tensor_ir-c/TensorIR.h)
accepts an MLIR module, compile options, concrete tensor views, and a caller's CUDA
stream. It exposes compilation, initialization, artifact extraction, and
serialization. A Julia adapter could use this interface directly. The Python
DSL is optional, and adopting this backend need not make model code Python-like
or route tracing through Python.

That makes an optional region provider a credible integration experiment.
Tempest would keep its semantic graph, derivatives, kernel pairs, region
selection, output storage, and capture. Eligible subgraphs could be lowered
through this compiler alongside native Julia/cuTile generation and opaque
implementations. The subsequent [training experiment](tensor-ir-training.md)
implements this boundary for explicitly outlined regions, including saved
forward values and captured backward execution. Automatic region selection
remains open.

## What the source supports, and what it does not prove

NVIDIA's [reduction lowering](https://github.com/NVIDIA/tensor-ir/blob/v0.1.4/lib/Transform/TileReductions.cpp)
handles built-in and restricted user-defined tuple reducers, contraction loops,
producer cloning, and post-expression masking. However, `stripMineReduction`
leaves an outer reduction coarse when its producer still contains another
contraction. Sequential reductions over broadcast-back values are therefore
important tests of its scheduling quality.

The [iteration-space tests](https://github.com/NVIDIA/tensor-ir/blob/v0.1.4/test/Conversion/TensorToCudaTile/iterspace_reduce.mlir)
and [multi-output tests](https://github.com/NVIDIA/tensor-ir/blob/v0.1.4/test/Conversion/TensorToCudaTile/multi_output.mlir)
exercise reduction/broadcast composition and normalization-shaped graphs. Most
are lowering/FileCheck tests. The softmax example uses exp/sum without a
stabilizing maximum; a LayerNorm fixture accepts separate x and xx inputs and
omits parts of the complete numerical computation. These fixtures establish
structural capabilities, not numerically validated full training algorithms.

Two runtime functions are explicitly unfinished in
[CudaTileRuntimeKernel.cpp](https://github.com/NVIDIA/tensor-ir/blob/v0.1.4/lib/Runtime/CudaTileRuntimeKernel.cpp#L174):
argument support checking returns success without signature validation, and
workspace accounting returns zero. The advertised interface should not be
mistaken for a completed implementation of every contract. Tempest must retain
its own shape, dtype, layout, ownership, and build-failure checks.

The [compile eligibility gate](https://github.com/NVIDIA/tensor-ir/blob/v0.1.4/lib/Compiler/CudaTile/CudaTileFrontend.cpp)
requires one graph and performs coarse operation/options checks. A positive
answer does not prove that later scheduling and code generation will succeed.
GB10's SM121 is represented in the
[target metadata](https://github.com/NVIDIA/tensor-ir/blob/v0.1.4/include/tensor_ir/Utils/ComputeCapability.h),
but the default target is SM100-family and must be overridden. Source-level
recognition is not evidence of a successful GB10 launch.

## The smallest useful scheduling mechanisms

### Segmentation asks whether a schedule exists

At nvFuser commit `5198f114e54ed87563533d8ff3a47bd4d872e739`,
[`tryMerge`](https://github.com/NVIDIA/Fuser/blob/5198f114e54ed87563533d8ff3a47bd4d872e739/csrc/fusion_segmenter.cpp#L2137)
tentatively combines groups and asks the scheduler registry whether the combined
region is schedulable. Separate graph checks prevent cycles. This is a useful
initial policy before global partition search.

Its [normalization eligibility checks](https://github.com/NVIDIA/Fuser/blob/5198f114e54ed87563533d8ff3a47bd4d872e739/csrc/scheduler/normalization_utils.cpp#L1144)
compare mapped reduction/iteration domains and output coverage. They do not need
a function named softmax. Several reductions over the same mapped axes can share
a scheduling family, while a reduction over another axis requires a different
schedule or another region.

Equal sizes do not prove matching coordinates. The
[domain model](https://github.com/NVIDIA/Fuser/blob/5198f114e54ed87563533d8ff3a47bd4d872e739/csrc/id_model/id_model.h#L50)
distinguishes exact indexing relations from broadcast-compatible scheduling
relations. Tempest should retain its layout maps and establish axis identity
through those maps. A square transpose is an immediate counterexample to
shape-only eligibility.

### Reduction completion determines pass boundaries

Helion commit `7e57e150600877ccf3a5db12e6dd8190004e2091` implements this in
[`ReductionRoller`](https://github.com/pytorch/helion/blob/7e57e150600877ccf3a5db12e6dd8190004e2091/helion/_compiler/roll_reduction.py).
It classifies values by reduction-axis dependence, completes a pass when a
consumer needs its reduction, and reconstructs required non-reduction producers
in later passes. Earlier reductions remain completed values.

For stable softmax, that yields a maximum pass, an exp/sum pass, and an output
pass. A resident schedule instead keeps the complete input tile and intermediate
values. These are two implementations of the same graph. Generic pass formation
does not derive online softmax: Helion's online recurrence is written explicitly
in its [example](https://github.com/pytorch/helion/blob/7e57e150600877ccf3a5db12e6dd8190004e2091/examples/softmax.py#L68).

Helion also has a [TileIR backend](https://github.com/pytorch/helion/blob/7e57e150600877ccf3a5db12e6dd8190004e2091/docs/tileir_backend.md)
through Triton. Its tuning choices change with what the lower compiler controls.
That supports leaving physical tile layout and instruction scheduling to cuTile
while deciding reduction passes, region boundaries, and logical placement above
it.

### Liveness and ownership constrain reuse

nvFuser's [persistent-buffer analysis](https://github.com/NVIDIA/Fuser/blob/5198f114e54ed87563533d8ff3a47bd4d872e739/csrc/scheduler/utils.cpp#L566)
finds values needed across reduction paths and considers reconstructing them
from inputs. Retention also needs compatible thread ownership; a value's graph
lifetime alone does not establish that its consumers can read it from registers.

[Halide's computation/storage distinction](https://halide-lang.org/docs/tutorial/lesson_08_scheduling_2.html)
provides the clean vocabulary: logical value identity, computation placement,
and storage lifetime are separate. For the first Tempest schedule, one block
owning a complete column makes the ownership proof simple. More elaborate
remapping needs an explicit proof or a rejection reason.

### Legality and estimated speed should remain distinct

nvFuser's [persistent runtime checks](https://github.com/NVIDIA/Fuser/blob/5198f114e54ed87563533d8ff3a47bd4d872e739/csrc/scheduler/normalization_inner.cpp#L73)
include resource limits and insufficient independent blocks. Helion's
[reduction heuristic](https://github.com/pytorch/helion/blob/7e57e150600877ccf3a5db12e6dd8190004e2091/helion/_compiler/autotuner_heuristics/triton.py#L4428)
estimates peak live tile storage and available work. Its hardware calibration
should not be copied to GB10.

Tempest should distinguish an invalid mapping, a resource constraint, and a
schedule merely predicted slower than an alternative. Start with a few explicit
candidates and measurements; retain enough metadata to explain wrong predictions.

## cuTile responsibilities and interface

The adjacent checkout at `2f4e6cc1516b4a9892cfc6c315d7376488d48971` already includes
[rank-generic eager mapreduce scheduling](https://github.com/JuliaGPU/cuTile.jl/blob/2f4e6cc1516b4a9892cfc6c315d7376488d48971/src/mapreduce.jl).
That is different from scheduling several dependent reductions and their
consumers. Our experiment environment currently loads registry cuTile 1.0.0;
checkout findings must not be assumed to describe identical installed code.

Public tile loads and reductions suffice to construct reread passes. A stable
broadcast-expression loader would make integration cleaner than depending on
`preprocess` and `_eval_bc`, but its absence is not a fundamental expressiveness
barrier. Such an interface should expose coordinates, logical shape, index maps,
and validity masks without allocating or choosing the enclosing reduction
schedule.

Numerical contracts require explicit attention. Mask padded lanes after
computing the producer expression: a zero load followed by exp contributes one.
The inspected tile maximum defaults to ignoring NaNs, so matching Julia maximum
requires an explicit policy. Preserve casts at semantic bindings even when
changing the reduction association. These are correctness obligations, not cost
heuristics.

The padding concern reproduced on the GB10 with the installed cuTile 1.0.0:
`mapreduce(exp, +, cuTile.Tiled(CuArray(zeros(Float32, 259, 7))); dims = 1)`
returned 512 in each column instead of 259. The aligned eight-row control
returned eight. The installed implementation pads loads with the reduction
identity before applying the map; exp turns those padded zeros into ones.
`examples/research/cutile_padding_probe.jl` preserves the diagnostic. This is an
upstream eager-mapreduce result, not a failure measured in Tempest's native
reduction provider. Dependencies were not modified.

## The executable Tempest probe

`examples/research/reduction_scopes.jl` derives phases from real traced graphs.
It intentionally accepts one rectangular identity-indexed domain and one common
reduction-axis set. Layout operations and orthogonal reductions reject rather
than being inferred safe from dimensions. A research-owned maximum operation
lets stable softmax participate without changing production Base methods or AD.

The experiment executes the plan with one CPU sweep per reduction phase and a
final output sweep. Independent reductions share a sweep; completed reductions
survive while pure pointwise producers are reconstructed. It checks escaping
intermediates, logical liveness, and reconstruction recipes that stop at
completed reductions. Unknown scalar functions are not assumed pure.

The reviewed run passed **50 assertions**, including rank-three tuple-axis
reductions, directional finite differences for the explicit RMSNorm input
gradient, and exp evaluation counts demonstrating softmax recomputation. These
results concern bounded CPU execution, not a GPU schedule or speedup.

| Formula | Reduction completion phases | Total CPU sweeps including output |
| --- | --- | --- |
| RMSNorm with saved residual/statistic | 1 | 2 |
| Centered LayerNorm | 1, 2 | 3 |
| RMSNorm input gradient | 1, 1 | 2 |
| Unrecognized three-reduction composition | 1, 1, 2 | 3 |
| Stable softmax | 1, 2 | 3 |

Full normalization backward also needs a weight-gradient reduction over another
axis. The common-domain schedule deliberately rejects that whole graph; a
region partitioner would separate it or select a different schedule family.

Run it with:

```sh
julia --project=. examples/research/reduction_scopes.jl
```

The companion `examples/research/native_region_baseline.jl` runs these formulas
through the existing production CPU planner, with and without experimental
reduction fusion. On Float32 inputs of shape `(259, 7)` it produced:

| Formula | Regions without reduction fusion | Regions with reduction fusion |
| --- | --- | --- |
| RMSNorm with escaping values | 4 | 1 |
| Centered LayerNorm | 4 | 3 |
| RMSNorm input gradient | 5 | 2 |
| Three-reduction composition | 5 | 4 |

All eight prepared executions matched the eager formulas. These are scheduled
region counts, not GPU launch counts. They show the present boundary: the
semantic graph already contains these formulas, while native fusion remains
limited to the one-sum region family.

## Exporting real graphs to NVIDIA TensorIR

`examples/research/tensor_ir_export.jl` is a bounded Julia exporter from actual
Tempest graphs to NVIDIA TensorIR MLIR. It preserves shared value identities,
column-major boundary strides, leading-axis Julia broadcasting, and flattened
structured outputs. It supports dense Float32 arrays, selected scalar math,
sum, and the experiment's maximum. Unknown functions, layout operations, opaque
calls, mixed precision, and runtime scalars reject explicitly. This is a
research bridge, not a registered backend or a complete Julia runtime adapter.
Its 18 assertions include a singleton-column case where rank padding must not
emit a redundant broadcast.

We built the pinned v0.1.4 optimizer and compiler with their pinned LLVM and
CUDA Tile sources, with Python and the upstream test suite disabled. An earlier
attempt using the installed Julia LLVM 21.1.8 failed in TableGen; the pinned build
succeeded. The pinned LLVM revision was
`57109befac92811d2253109242ca6fa69c961fb2`, and CUDA Tile was
`af2417041cc939b87ef56d92cfdcf61737c5457e`. No project dependencies or system
toolchain were changed. The upstream build options used were:

```sh
cmake -S /path/to/tensor-ir -B /tmp/tensor-ir-build -G 'Unix Makefiles' \
    -DTENSOR_IR_DOWNLOAD_LLVM=ON -DTENSOR_IR_ENABLE_BINDINGS_PYTHON=OFF \
    -DTENSOR_IR_INCLUDE_TESTS=OFF -DCMAKE_BUILD_TYPE=Release \
    -DCUDAToolkit_ROOT=/usr/local/cuda -DLLVM_INCLUDE_TESTS=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_ENABLE_ASSERTIONS=OFF
cmake --build /tmp/tensor-ir-build --target tensor_ir-opt tensor_ir-compiler --parallel 8
```

All four exported graphs parsed. On shape `(256, 7)`, RMSNorm with escaping
residual/statistic outputs, centered LayerNorm, stable softmax, and an unfamiliar
three-reduction expression all lowered automatically to CUDA Tile IR. Shape
`(259, 7)` exposed two distinct failures:

- Returning a projected statistic alongside a full tensor required a tile width
  exactly equal to 259. That conflicts with the power-of-two tile requirement;
  supplying a padded width of 512 was also rejected.
- A single-output stable softmax lowered to IR containing a non-power-of-two
  tile. Later assembly rejected it. Successful conversion and bytecode emission
  did not imply a compilable kernel.

The handwritten long-width probes also showed the nested-reduction limitation:
with width 8193 and reduction chunks of 256, the maximum was streamed but the
outer exp/sum retained the full logical width. Splitting the output grid could
repeat complete-column statistics across multiple output tiles. This is a
concrete scheduling concern, not a measured performance comparison.

Using an existing Tile IR assembler and explicitly targeting `sm_121a`, all
four real Tempest exports at `(256, 7)` compiled to cubins, ran on GB10, and
passed TensorIR's reference verifier. The RMSNorm and unfamiliar composed graph
each returned three outputs. A separate handwritten centered LayerNorm returning
y, mean, and inverse standard deviation also passed. At `(259, 7)`, the
single-output LayerNorm and softmax both failed assembly with non-power-of-two
tiles after successful IR lowering.

| Actual Tempest graph | Width 256: compile, launch, verify | Width 259: final result |
| --- | --- | --- |
| RMSNorm with residual/statistic outputs | Pass | Rejected during tile selection |
| Centered LayerNorm | Pass | Rejected by assembler |
| Stable softmax | Pass | Rejected by assembler |
| Unfamiliar expression with shared/statistic outputs | Pass | Rejected during tile selection |

The verifier evaluates the exported MLIR in C++, separately from its GPU
lowering. These were Float32 random inputs in `[-1, 1]`, seed 1234, with tolerance
`1e-4`; this is not an independent comparison against Julia on the same GPU test
inputs. Source inspection also found no explicit NaN mismatch check in that
verifier; its truncated output printing cannot independently establish that
every GPU value is finite. These passes are therefore smoke checks. The CPU
sweep probe separately compares its Julia graph interpretation and eager formulas. Neither test proves broad numerical, capture, caching, or
performance guarantees. No performance claim is based on the CLI's single timed
iteration.

We then independently loaded the RMSNorm cubin from Julia using CUDACore's
`CuModule`, `CuFunction`, and `cudacall`. This exercised six pointer arguments
(three inputs and three outputs) on ordinary CuArrays. For this artifact the
grid is seven columns and `threads = 1`, matching the Tile IR launch convention;
the cubin carries its worker-thread metadata. This is a fixed artifact ABI
check, not automatic launch-metadata discovery.

`examples/research/tensor_ir_rmsnorm_check.jl` passed **27 assertions** against
eager Julia over random, changed, and zero inputs. Before every launch, each
output was filled with NaN; the checks require every returned value to be finite
and match its reference elementwise at `rtol = 2e-5`, `atol = 2e-6`. This covers
the full output, residual, and inverse-standard-deviation arrays. The largest
absolute errors were `1.43e-6`, zero, and `1.19e-7`, respectively. It closes the
built-in verifier's finiteness gap for this RMSNorm fixture; the other GPU graphs
retain the smoke-check qualification above.

The tested toolchain used Tile IR assembler 13.4.59, CUDA toolkit 13.0.88, driver
580.159.03, and a GB10. Compiling a cubin with the available assembler avoided
relying on the driver's ability to consume newer Tile IR directly. The assembler
was added only to the compiler process's `PATH`.

Generate the actual Tempest graph inputs with:

```sh
julia --project=. examples/research/tensor_ir_export.jl --test
julia --project=. examples/research/tensor_ir_export.jl /tmp/tempest-tensor-ir-inputs
```

The CLI above emits width 259 intentionally. Generate the passing width-256
controls without editing the exporter:

```julia
include("examples/research/tensor_ir_export.jl")
directory = mktempdir()
for (name, program) in pairs(TensorIRExport.examples(; rows = 256))
    source = TensorIRExport.export_mlir(program; name = string(name))
    write(joinpath(directory, "$name.mlir"), source)
end
println(directory)
```

With the pinned compiler built and a compatible `tileiras` on the process path,
run each generated file with:

```sh
tensor_ir-compiler /path/to/softmax.mlir --target-sm=sm_121a \
    --artifact-kind=cubin --verify --seed=1234 --tolerance=0.0001 --iterations=1
```

To reproduce the independent Julia check, generate the width-256 RMSNorm
source above, compile it with `--dump-artifact=/tmp/rmsnorm.h256.cubin`, then run:

```sh
julia --project=examples examples/research/tensor_ir_rmsnorm_check.jl /tmp/rmsnorm.h256.cubin
```

The successful handwritten saved-statistics control is retained in
`examples/research/tensor_ir/layernorm_saved_stats.mlir`.

## Integration decision

The source investigation and execution probes support retaining Tempest's
current semantic and execution architecture. NVIDIA TensorIR is now a concrete
optional region-compiler candidate, with an end-to-end path demonstrated on a
bounded set of actual graphs. The measured coverage gaps make this release
unsuitable as our sole or default reduction backend. A direct Julia/cuTile path
remains valuable for Julia-defined scalar computation, transparent schedules,
and gaps in the external compiler.

Following the [Stiletto comparison](stiletto-perspective.md), the
[TensorIR training experiment](tensor-ir-training.md) now demonstrates an optional
region provider with forward/backward execution and saved outputs. The next slice
is automatic region extraction and a graph-region contract with explicit outputs,
layouts, and numerical policy. The existing decomposed
execution path supplies a fallback. A new general native cuTile scheduler is
not a prerequisite: the phase-analysis prototype can first serve as a reference
and a diagnostic for external scheduling gaps. This keeps TensorIR's lowering
useful without making its current support surface Tempest's semantic vocabulary.
The provider remains research code; production Tempest has no TensorIR dependency.

The production interface should accept a selected graph region with declared inputs,
outputs, layouts, and numerical policy. Compile outside capture, initialize
resources, validate concrete arguments, retain artifact owners, and launch on
the prepared stream. Existing support/build/workspace semantics already fit
much of this contract. Unsupported mathematical operations should have explicit
fallback; unexpected compiler failures should stay visible.

The decisive comparison remains stable softmax, centered LayerNorm, and explicit
normalization backward, including long reductions, few independent columns,
non-power-of-two tails, views, and saved outputs. Compare build success, numerical
behavior, physical launches, storage, compilation time, and steady-state runtime
separately. Neither a positive support query nor a lowering test is a substitute
for that experiment.

The subsequent [Stiletto review](stiletto-perspective.md) identifies a concrete
integration constraint: our current vendor bridge sees only small scalar
regions. Graph-region candidates must expose producer/consumer fusion and
multiple outputs to make the alternative backends useful.
