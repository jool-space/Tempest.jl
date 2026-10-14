# Native cuTile sweeps from ordinary graphs

This research experiment automatically selects reduction regions from Tempest's
existing graph and compiles a bounded family to cuTile. It tests two separate
ideas: selecting a region without a named-operation pattern, and deriving
several reduction sweeps from dependencies inside that region. The code lives
in `examples/research/`. A separate production integration now provides this
schedule through ordinary `prepare`; see [Graph regions](graph-regions.md).
The rest of this page records the earlier standalone experiment and its timings.

## From graph to candidate

`GraphCandidates.reduction_candidates(program)` starts at live sum reductions
and grows connected selections along graph edges. Pointwise operations and
constants can join a selection. Matmul, layout operations, semantic calls, and
other operations remain outside it. An outside operation can still produce an
input to a candidate.

Each candidate is a `GraphRegions.RegionBoundary` containing original node IDs,
ordered inputs, all escaping outputs, and logical index mappings. The selector
checks that contracting the proposed regions leaves an acyclic graph. This check
covers the whole proposed partition: two individually legal selections can form
a cycle when both are contracted. Unrelated operations between selected node
IDs do not prevent a region. A dependency path that leaves and re-enters it can.

This is a greedy structural selector, not a backend support check or a cost
model. It does not prove that an arbitrary Julia function is pure. It also does
not promise the largest possible region or select a smaller candidate after a
backend rejects one. These are limitations of the standalone selector; production
selection now consults support while growing alternatives through the implementation protocol.

`GraphCandidates.standalone_program(boundary)` constructs a small `TensorProgram`
whose positional arguments follow `boundary.inputs` and whose result is a tuple
following `boundary.outputs`. It also returns `original_to_local` and
`local_to_original` maps. Body nodes retain the original `Operation` objects,
including scalar callable instances; only operand IDs change. No formula is
retraced or expanded into a separate list of recognized scalar functions.

For a mixed matmul and normalization graph, the selected region can therefore
start with the matmul result as an argument. The selector does not compile that
matmul. The subsequent production integration supplies the partition and binding
needed to execute a complete mixed graph.

## A dependency-derived schedule

`CuTileSweeps.prepare_sweeps(graph, device_inputs...; tile = 256)` analyzes the
standalone graph and builds a separate research executable. Each reduction
completes one phase after its latest input dependency. Independent sums can
share a phase; a sum that consumes an earlier sum's broadcast result belongs to
a later phase. Pointwise expressions do not add a reduction phase.

The generated kernel gives one CUDA block ownership of each column. For every
phase, it traverses the rows in bounded tiles, evaluates the required pointwise
producers, and accumulates every sum in that phase. Completed sums remain
available for subsequent phases. It then writes column statistics and makes a
final row sweep for elementwise outputs when needed.

A per-sweep expression cache shares a producer among consumers in that sweep.
Across sweeps, pure pointwise work is recomputed. The schedule does not retain
an entire column merely because a graph value has several uses. Escaping
intermediates receive output buffers and can be written alongside the main
result. Repeated result leaves refer to the same output buffer.

This is one scheduling strategy. It can trade lower local storage requirements
for additional loads and arithmetic, particularly for expensive scalar
functions. Many columns provide independent blocks; a graph with few columns
may expose little parallelism. There is no autotuner or choice between this
schedule, whole-column retention, and a staged implementation.

The `SweepProgram` passed to cuTile is a derived schedule containing typed
bindings, input and output IDs, and reduction groups. It is not a replacement
for Tempest's semantic graph or an input language other providers must adopt.

## Julia callables and recomputation

Pointwise evaluation invokes the callable stored in the original operation.
The GPU test includes a callable struct carrying nested scalar parameters and
using a scalar conditional inside its body. Region extraction does not need to
recognize that function or symbolically trace its internals. cuTile compiles the
Julia function in the generated kernel.

Recomputation still needs permission. The experiment uses the conservative
`ReductionScopes.pure_scalar` predicate for familiar primitives. A custom
fixture opts in explicitly, for example:

```julia
CuTileSweeps.ReductionScopes.pure_scalar(::MyScalarFunctor) = true
```

This is a research hook, not a settled public effects API. The declaration
promises that repeating evaluation with the same arguments is valid and that
reads come from those arguments and scalar captures. The experiment does not
infer these properties from arbitrary Julia bodies. Device compilability alone
does not establish them.

Captures are separately restricted to immutable scalar data, including nested
scalar structs. Captured arrays, references, and pointers are rejected. Tensor
operands must appear as graph inputs so their binding, lifetime, and aliasing
are visible. The fact that a compiler can flatten a nested argument does not
make hidden tensor reads part of Tempest's dependency graph.

## Padding and numerical behavior

Row counts need not be powers of two. Input loads use zero padding, but the
reduction masks the **computed contribution** before accumulation. Masking only
the load would be wrong for `exp`, which maps a padded zero to one, or for
functions such as `log`, whose padded evaluation can be nonfinite. Output stores
handle the final partial tile separately from the reduction mask.

The generated pointwise work preserves each node's logical result conversion.
The schedule currently uses Float32 sums and a tiled accumulation order; it
does not promise bitwise agreement with Julia's CPU sum or other GPU reduction
trees. Tests compare numerical results with tolerances. Mixed-precision policy,
general reduction operators, and broader numerical transformation permissions
remain separate work. In particular, the CPU maximum fixture in the scope probe
does not imply native GPU softmax support here.

## The cuTile entry-point workaround

The installed cuTile 1.0 path exposed a compiler problem with streamed loads
through tuple or vararg array parameters at the kernel entry. The independent
`cutile_streaming_abi_probe.jl` reproduces the distinction using the same
centering computation with two entry signatures.

The generated schedule uses a fixed-arity outer entry, with one parameter per
array. Inside that entry, the ordinary tuple-valued and generated Julia code
still specializes. Adapters are cached by argument count. This is an ABI
workaround for the observed compiler behavior, not a restriction on the
mathematical graph or evidence that nested arrays cannot be supported.

The diagnostic permits a future compiler to accept the tuple entry. If it does,
that entry must pass the same numerical checks; the test does not require the
compiler defect to remain.

## Preparation, execution, and validation

Preparation owns input copies and output arrays, constructs the constant
schedule, and compiles the kernel. `bind_inputs!` replaces values in the stable
input buffers. `launch!` launches on the current stream and returns borrowed
outputs. The experiment allocates no explicit device scratch workspace for its
sweeps; that statement excludes the compiler's internal resource choices and
the owned input and output arrays.

Capture is exercised directly through CUDACore in the tests, rather than through
Tempest's production `Prepared` type. Tests inspect captured node kinds, poison
outputs before replay with changed inputs, retain and check buffer addresses,
and exercise ordinary launches on another stream. Those checks establish the
behavior of the tested research executable; they do not establish general
training-state or concurrent-execution support.

The tested GPU family includes unfamiliar pointwise compositions, dependent and
independent sums, custom scalar callables, column broadcasts, saved
intermediates, and non-power-of-two row counts. Unsupported shapes, types,
operations, and missing purity declarations are rejected before compilation
where the metadata checks cover them. A custom callable must also compile
through cuTile; compiler errors remain visible.

Current restrictions include:

- A matrix iteration domain, with all reductions over the leading row axis.
- Float32 nonconstant values; scalar literals are limited by the schedule check.
- Dense CuArray inputs, with supported singleton/vector broadcasts.
- Escaping outputs retaining the column axis, so each block has distinct writes.
- Positive power-of-two tile sizes bounded at 1024 rows; the row count itself
  has no whole-column tile-size limit.
- No layout operations, arbitrary indexing, opaque calls, or contractions inside
  the selected native region.

Structural extraction can accept graphs that this particular schedule rejects.
The GPU experiment is forward-only: it does not establish automatic backward
compilation, attention fusion, optimizer updates, donation, or end-to-end
training speedups. The later production integration tests graph boundaries and
fallback through the existing provider protocol. See the
[compiler direction](compiler-direction.md) for that distinction.

## Measured on GB10, 21 September 2026

On Julia 1.13.0, cuTile 1.0.0, and NVIDIA GB10, the automatic candidate tests
passed 278 assertions and the native GPU suite passed 304. The separate ABI
probe passed four checks for the working flat entry while reporting the tuple
entry's compiler failure.

The forward fixture returns the normalized result, transformed residual, mean,
and variance. Its custom Julia callable contains nested Float32 parameters and
`exp(ifelse(...))`. All cases below use 32 columns and 256-row tiles.

| Rows | Streamed replay (μs) | Pre-region decomposition (μs) | Streamed output bytes | Current value-buffer bytes |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 16.30 | 51.57 | 65,792 | 98,688 |
| 259 | 8.21 | 49.23 | 66,560 | 99,840 |
| 8192 | 69.70 | 1149.45 | 2,097,408 | 3,146,112 |
| 8193 | 57.31 | 1131.13 | 2,097,664 | 3,146,496 |

Every streamed case captured one kernel; every current-planner case captured
seven. cuTile was loaded for the baseline, whose selected providers were five
cuTile regions and two CUDA regions. The memory columns count output/value
buffers, excluding owned inputs and backend workspace; the new schedule needs
no explicit device scratch. They are not measurements of total GPU residency.

Both implementations finish compilation before timing. Results are medians of
seven 200-replay CUDA-event samples, with measurement order alternating and
warmup before each sample. Host-to-device binding and compilation are excluded.
These are observations from one device and run. The difference between aligned
and irregular row counts has not been explained or tuned.

Compilation remains material: the streamed kernel's first preparation took
19.81 seconds in this process, including 19.10 seconds in Julia/cuTile kernel
compilation. Subsequent shape cases took 0.85–1.37 seconds to prepare, with
0.74–1.27 seconds in kernel compilation. The current planner's first preparation
took 3.41 seconds; later cases reused its already compiled paths. These figures
are not isolated cold-cache compiler benchmarks.

These historical timings compare against an untuned decomposition whose absolute
performance is weak. The ratios do not establish proximity to hardware limits or
competitive normalization performance. The aligned/irregular timing reversal
also makes the small-case ranking unreliable until reproduced and explained.
There was no matched Stiletto/TensorIR timing reference in this experiment.

The later production integration establishes the graph boundary and execution
path. Automatic schedule competition and a trustworthy performance comparison
remain open; see the [revised priorities](compiler-direction.md#Integration-review-and-revised-priorities).

Run from the repository root with the examples environment available:

```sh
julia --project=. examples/research/graph_candidates_tests.jl
julia --project=examples examples/research/cutile_sweeps_tests.jl
julia --project=examples examples/cutile_issues/streaming_abi_probe.jl
julia --project=examples examples/research/cutile_sweeps_benchmark.jl
```

GPU tests and the ABI diagnostic gate on `CUDACore.functional()`; the benchmark
requires a functional device. The benchmark compares captured forward replay
with the current planner using the same custom normalization fixture and
outputs. It reports preparation separately from repeated execution and counts
physical kernel nodes in the captured graphs. It is not a comparison against
all vendor normalization kernels or a full training benchmark.
