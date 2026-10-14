# Compilers above tile languages

Research snapshot: 21 September 2026. This survey used official documentation,
source repositories, and research papers. It does not report new benchmarks or
claim compatibility with Tempest's installed dependencies. Development-branch
documentation can describe more than a released package supports.

The question is which compiler should turn ordinary tensor computations into
good compositions of generated, vendor, and user kernels. cuTile supplies a
kernel language and compiler; the missing decisions include region boundaries,
iteration domains, reduction scope, compute placement, and materialization.

## Where kernel languages sit

These are overlapping abstractions, not a strict ranking by capability.

| System | What the author supplies | Relationship to the proposed layer |
| --- | --- | --- |
| [CUDA Tile / cuTile](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html) | A tile program and its block decomposition | Compiles tile operations into their physical execution; a graph compiler must still choose the program and decomposition. |
| [Triton](https://triton-lang.org/main/getting-started/tutorials/gluon/intro.html) | A tile-SPMD kernel with indexing and algorithmic structure | Broadly the same kernel-authoring layer. Gluon exposes more explicit layouts, memory, and asynchronous execution. |
| [TileLang](https://github.com/tile-ai/tilelang) | Tile dataflow with buffers, copies, loops, and scheduling annotations | Also a kernel language; automates layout inference, tensorization, and pipelines. It exposes different scheduling choices from cuTile. |
| [CuTe DSL](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html) | Layouts, thread/data partitioning, copy and matrix-operation primitives, pipelines | More explicit control over hardware execution. Its surrounding ecosystem also contains higher-level interfaces. |
| [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) | CUDA kernels using register/shared tiles and coordination templates | A tile library and reusable kernel infrastructure, not automatic tensor-graph partitioning. |
| [Pallas](https://docs.jax.dev/en/latest/pallas/design/design.html) | A custom kernel with references, grids, and block mappings | Integrates kernel programming with JAX; the linked design is a historical design document, not a current support matrix. |

A callable carrying auxiliary arrays is a useful way to package a generated
epilogue. It does not establish how those arrays are indexed, whether they
require reductions, or when a reduced value becomes available. Those remain
compiler obligations even when the language supports nested closures naturally.

## The intermediate layers are particularly relevant

### Helion: scheduling above a kernel language

[Helion](https://github.com/pytorch/helion) takes tiled loops containing tensor
operations and generates/tunes Triton implementations. It automates indexing,
masks, tile choices, loop ordering, program-ID mapping, and looped reductions.
The tiled body defines one compiled GPU region; the author supplies that
region. A decorated invocation can also execute host-side tensor operations
that launch additional kernels. Its search can be substantial; automatic scheduling
does not imply negligible preparation latency.

The transferable idea is a representation of the computation within a candidate
region that is richer than a handwritten kernel but still constrained enough to
generate useful schedules. Tempest would additionally need to select regions
and connect them to its storage, differentiation, and external-kernel machinery.

### nvFuser: segmentation guided by schedulability

nvFuser's [segmenter](https://github.com/NVIDIA/Fuser/blob/main/csrc/fusion_segmenter.h)
groups expressions and associates groups with schedulers. Its
[scheduler registry](https://github.com/NVIDIA/Fuser/blob/main/csrc/scheduler/registry.cpp)
contains pointwise, reduction, persistent-reduction, transpose, matmul, and other
families. It separates compile-time and runtime eligibility, records rejection
reasons, and can choose among schedulers in priority order.

This is a close practical reference for Tempest: generality can come from
schedule families accepting unfamiliar compatible expressions. It need not mean
either one named pattern per operation or unconstrained algorithm search.

### cuDNN Frontend: more than the older thin-wrapper description

The [cuDNN graph API](https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html)
separates operation graphs from engines, configurations, workspace, and execution
plans. An accepted plan can use one kernel or a small set. Its documented generic
fusion support is bounded: generic epilogues allow at most one reduction at the
exit. That does not establish support for reduction results broadcast back into
further normalization. Specialized engines may support additional patterns.

The [September 9 roadmap](https://github.com/NVIDIA/cudnn-frontend/issues/442)
reports released Frontend v1.27/v1.28 with Python-native graphs and open-source
FROST engines, including GEMM/SDPA, still opt-in in that release account. It
lists **Tensor IR to CUDA Tile lowering for normalization and memory-bound
graphs as future integration work**. Subsequent source investigation found the
public [NVIDIA TensorIR v0.1.4](https://github.com/NVIDIA/tensor-ir/tree/v0.1.4),
tagged September 18. This is an early compiler implementation, distinct from
TVM TensorIR; its availability does not establish released cuDNN integration.
The [source investigation](scheduler-investigation.md) records its concrete
capabilities, incomplete runtime checks, and potential Julia C API integration.

The [current repository](https://github.com/NVIDIA/cudnn-frontend) describes
CuTe-based JIT GEMM engines for supported matmul graphs with chained pointwise
epilogues, alongside backend execution plans. Development defaults and coverage
must not be projected onto the released versions or our installed Julia bridge.

### CUTLASS Operator API: a provider interface with an epilogue compiler

The [custom-kernel tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/operators/tutorials/002_bring_your_own_kernel.html)
separates implementation registration, support/configuration discovery,
compilation, execution, and workspace. This is a strong comparator for
Tempest's implementation protocol. It also suggests a future candidate should
identify a configuration, rather than merely a provider name.

The [epilogue tutorial](https://docs.nvidia.com/cutlass/latest/media/docs/operators/tutorials/001_gemm_with_fused_epilogue.html)
accepts auxiliary tensors/scalars, broadcasts, multiple outputs, and reductions.
An important boundary is that its documented reduction outputs use atomic
accumulation and require initialized destinations. Such a reduction is not
automatically a completed value that a subsequent normalization can consume
within the same kernel. These are observations about current published docs,
not a verified release or Julia interoperability claim.

By comparison, [cuBLASLt](https://docs.nvidia.com/cuda/cublas/index.html#cublasltepilogue-t)
provides enumerated GEMM epilogues and algorithm selection. It is a useful
implementation candidate, not a general epilogue programming language.

## Broader compiler precedents

| System | Relevant design | Lesson for Tempest |
| --- | --- | --- |
| [TorchInductor](https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747) | Graph lowering to indexed loop computations, scheduling, generated kernels, and external/template calls | Keep scalar/index expressions, storage identity, and implementation selection distinct. |
| [TVM Relax / TensorIR](https://tvm.apache.org/docs/arch/) | Graph transformations above structured tensor programs and schedules | Kernel partitioning and scheduling a chosen primitive function are separate problems. |
| [Halide](https://halide-lang.org/docs/tutorial/lesson_08_scheduling_2.html) | Algorithm/schedule separation; independent compute and storage placement | A value's identity, computation location, and storage lifetime need not coincide. |
| [IREE](https://iree.dev/reference/mlir-passes/DispatchCreation/) | Dispatch formation before backend code generation | Make the chosen region and its execution schedule separately inspectable. |
| [XLA:GPU](https://openxla.org/xla/emitters) | Library calls, Triton lowering, and specialized native fusion emitters | A broad semantic graph can coexist with a finite set of scheduling strategies. |
| [tinygrad](https://docs.tinygrad.org/developer/speed/) | Graph scheduling/materialization followed by kernel optimization | Searching implementations of one kernel does not itself optimize global fusion or recomputation. |

Inductor's [April 2026 GEMM article](https://pytorch.org/blog/gemms-torchinductor-cutedsl-backend/)
describes retaining multiple Triton GEMM candidates until epilogue fusion is
benchmarked. CuTe DSL epilogue integration was future work in that article.
The general lesson is useful regardless of frontend language: committing to the
fastest isolated kernel can prevent a faster surrounding region.

TVM's [MetaSchedule](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html)
explores rule-generated schedules through cost prediction and measurements.
This is different from assuming every possible algorithm lies in the search
space. IREE's [tuning interface](https://iree.dev/reference/tuning/) similarly
offers a precedent for externally supplied, inspectable code-generation choices.

Halide is especially useful conceptually: `compute_at` and `store_at` let a
producer's computation and allocation have different scopes. For Tempest, this
would distinguish retaining a residual tile, recomputing it on another pass,
and materializing the full residual for backward. Those are separate decisions
currently compressed into the word "fusion."

## Research beyond local fusion

[Welder](https://www.usenix.org/conference/osdi23/presentation/shi) models tile
dataflow and chooses where connected tiles reuse data in the memory hierarchy.
Its [paper](https://www.usenix.org/system/files/osdi23-shi.pdf) uses traffic-based
costs and decomposed scheduling spaces to balance intra-operator and
inter-operator reuse. This is directly relevant to combining contractions and
reductions without expressing every combination as a named fusion. It is a
research system to learn from, not an established drop-in backend for Tempest.

[Mirage](https://arxiv.org/html/2405.05751v3) searches hierarchical tensor programs
across kernel, block, and thread levels, including algebraic transformations. Its
probabilistic equivalence checking applies to a specified algebraic class;
floating-point numerical behavior needs separate tests. This is a larger search
and verification undertaking than parameterizing a few schedules.

[Mirage Persistent Kernel](https://arxiv.org/html/2512.22219v1) explores a different
extension: scheduling dependent work through a persistent GPU runtime with
finer-grained overlap. Megakernel execution and local fusion are distinct
choices. Reducing launch overhead does not imply every intermediate stays in
registers or shared memory.

## Implications for Tempest

The following is our design interpretation, not a claim from the surveyed
projects and not a committed public API.

Tempest already has typed scalar expressions, layout maps, multi-output
regions, implementation selection, saved-tensor reverse rules, storage
lifetimes, donation, and capture. The experimental cuTile reduction path adds
local sharing across one leading-axis matrix reduction. It does not yet supply
general domain analysis or alternative schedules for dependent reductions.

The most promising scope is a Julia-native region compiler and implementation
planner. The useful combination is automatic region formation, reusable cuTile
schedule families, external kernel composition, and training-aware storage.
This is an implementation opportunity within an active field, not a claim that
no existing compiler addresses those concerns.

Keep these decisions explicit:

1. **Semantics:** indexed reads, shared value identities, reduction axes and
   identities, casts, accumulation types, and allowed reassociation.
2. **Region formation:** external inputs, escaping outputs, dependencies, and
   proposed kernel boundaries.
3. **Scheduling:** tile domains, reduction passes, compute placement, retention,
   recomputation, and required temporary storage.
4. **Realization:** generated cuTile code, a vendor plan, or a user implementation,
   potentially requiring more than one physical kernel.
5. **Selection:** eligibility first, then bounded profitability estimates or
   measurement, with preparation latency and cache behavior included.

These distinctions need not become five public IRs or five new protocols.
Derived scheduling views of the current graph are a reasonable starting point.
cuTile should retain responsibility for lowering tile operations to hardware.

### A discriminating next experiment

Use stable softmax, centered LayerNorm, and normalization backward to require
dependent reductions and broadcasts without matching those operation names.
Expose one whole-domain schedule and one staged or streaming alternative.
Preserve intermediate outputs and dtype conversions, exercise irregular shapes
and mapped inputs, and retain a decomposed fallback.

The success criterion is unfamiliar formulas working through existing domain
analysis and scheduling rules. Correctness, gradient agreement, physical launch
counts, memory traffic, preparation cost, and performance across shapes should
be measured separately. Include cases where fusion loses; the planner must be
able to select that outcome.

Start by studying Halide for placement, nvFuser for eligibility and segmentation,
and Helion for schedule generation. Use Welder to inform cost decisions and
cuDNN/CUTLASS to challenge our implementation interface. Keep Mirage as a guide
to a larger possible search space after a small, testable scheduler exists.

The follow-up [scheduler investigation](scheduler-investigation.md) pins source
revisions and includes an executable scope-analysis experiment.

The [Stiletto perspective](stiletto-perspective.md) examines the adjacent Julia
frontend, its current cuDNN coverage, and what Tempest’s bridge does not expose.

The [TensorIR training experiment](tensor-ir-training.md) subsequently verifies
a Julia-to-TensorIR compiler/runtime adapter with saved forward tensors, explicit
backward regions, and CUDA graph replay. It keeps region selection explicit
while establishing the execution and ownership contract.

The [compiler direction note](compiler-direction.md) reconciles these references
with the execution experiments. It keeps Tempest's graph authoritative and
prioritizes a common region boundary for native Julia/cuTile and external compilers.
