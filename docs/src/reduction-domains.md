# Reduction-domain experiment

This extends the production cuTile graph provider to second-axis and full sums.
The semantic graph, region selector, and support/build interface are unchanged.
The schedule separates domain and dependency-phase analysis, generated tile code,
and runtime construction into readable files under `ext/cuTileExt/`.

Axis sums stream within a block. Full sums use parallel partial sums and a
finishing kernel per dependency phase, then write escaping outputs in a final
kernel whose grid covers the output domain. A scalar-only result needs one block. Their scratch is declared through `ImplementationPlan` and shared by the
existing workspace planner. `explain` distinguishes scheduled calls from physical
kernels and prints each cuTile region's axes, tile, and phases.

## Correctness and coverage

The full CPU/CUDA run and final targeted rerun validate 4,970 assertions. The
last vendor-specific test was adjusted to isolate Stiletto workspace sharing;
all 948 assertions in the graph-region and Stiletto files pass in the final rerun.
The docs build also passes.

Production executable tests cover both matrix axes, full reductions over matrices,
vectors and zero-dimensional arrays, singleton broadcasts, irregular dimensions,
independent and dependent sums, and custom Julia scalar callables. They compare
with ordinary CPU array code, check mixed-axis gradients against the CPU reverse
graph and finite differences, poison workspace before replay, and verify stable
addresses and capture without allocation/free nodes. Two normalization regions
separated by a GEMM exercise shared workspace with different capacities.

A cuTile 1.0 bug appeared when a custom scalar callable containing constants was
broadcast over a two-dimensional tile: its inferred and emitted shapes disagreed.
The lowering now expands singleton operands, flattens only the scalar-call
boundary, and restores the tile shape. This preserves the actual Julia callable;
it does not introduce a separate scalar operation vocabulary.

Empty input arrays remain rejected by the frontend. Rank-three graph domains
fall back to the existing execution path. A single region still cannot mix
reduction axes or reduction domains; Float64 intermediates remain a boundary.

The 111-node normalized MLP passes CPU loss/gradient comparisons after capture
and repeated rebinding of epsilon. Relative to the preceding support-guided
selector:

| Measure | Leading-axis provider | Broader reduction domains |
| --- | ---: | ---: |
| Scheduled calls | 24 | 20 |
| Captured kernels | 35 | 33 |
| Value storage bytes | 435,864 | 435,476 |

The new MLP plan selects six graph regions. Fewer scheduled calls do not imply
the same decrease in physical launches: full reductions need multiple launches.
Shared workspace changes from 4,194,404 to 4,194,408 bytes, an increase of four
bytes after reuse. This is a coverage result, not an MLP speed claim.

## Independent measurement

`examples/research/reduction_domains_benchmark.jl` compares the same Float32 sum
and output shape with CUDACore/GPUArrays' public `sum!`. Each path is compiled and
warmed before timing. A CUDA graph contains 100 repetitions, so Julia submission
per operation cannot starve the measurement. Nine trials alternate measurement
order, with ten graph replays immediately before each timed sample. Results
report minimum, median, maximum, graph node counts, and Tempest preparation time
separately.

GPUArrays can allocate partial sums inside `sum!`. The benchmark counts those
allocation nodes and enables automatic release before subsequent graph launches.
Their cost remains included. Tempest's capture must contain no allocation/free
nodes. A device copy provides a cache/launch sanity check. Inputs are repeated
and resident: this is not a measured DRAM bandwidth bound, a normalization
benchmark, or an independent tuned upper bound.

The expanded coverage also changes provider competition. Stiletto's existing
`RegionRequest` implementation does not compete for the earlier `GraphRequest`.
The test specifically checking cuBLAS/Stiletto workspace sharing now disables
graph fusion, while mixed-provider tests retain the default selector. Comparing
alternative partitions and implementations remains a selection-policy task.

## GB10 measurements, 21 September 2026

Julia 1.13.0, cuTile 1.0.0, CUDACore 6.3.0, NVIDIA GB10. Times below are
microseconds per reduction: median [minimum–maximum] across nine trials. Copy
medians are included as the sanity check described above. Device clocks were
not locked. Small cases still vary substantially despite immediate warmup.

| Shape | Axes | Tempest | GPUArrays `sum!` | Copy median |
| --- | --- | ---: | ---: | ---: |
| 64 × 97 | first | 11.78 [11.76–17.14] | 10.26 [7.02–10.29] | 3.29 |
| 64 × 97 | second | 9.44 [7.42–12.51] | 10.50 [10.32–10.66] | 3.31 |
| 64 × 97 | all | 8.35 [7.21–10.24] | 20.30 [19.11–23.93] | 3.30 |
| 259 × 32 | first | 3.98 [3.95–4.00] | 10.27 [10.25–10.28] | 3.32 |
| 259 × 32 | second | 5.69 [5.34–5.71] | 12.26 [12.25–18.79] | 3.32 |
| 259 × 32 | all | 10.40 [7.25–10.54] | 22.33 [18.30–24.12] | 3.27 |
| 512 × 1024 | first | 123.56 [122.01–125.21] | 80.71 [80.24–81.19] | 11.20 |
| 512 × 1024 | second | 74.81 [74.40–75.35] | 72.81 [72.24–73.27] | 11.04 |
| 512 × 1024 | all | 112.99 [112.72–113.45] | 38.93 [38.91–39.09] | 11.37 |

Both axis paths contain one physical kernel per reduction. Full reductions have
three kernels in both implementations here; GPUArrays additionally has one
captured allocation per reduction, while Tempest has none. Tempest's full-sum
workspace is 104, 260, and 8,196 bytes respectively for these shapes. The first
Tempest preparation took 14.48 seconds including initial compilation; subsequent
cases took 0.21–2.28 seconds. That is preparation latency, not a kernel-only
compilation measurement.

The large case is decisive enough to expose scheduling work: leading-axis sums
are about 1.5 times slower than GPUArrays, second-axis sums are close, and full
sums are about 2.9 times slower even though GPUArrays pays for captured temporary
allocation. Some small cases favor Tempest, but variable clocks and timing
spread prevent treating those ratios as robust wins.

The measurements caught an unnecessary full input grid in the scalar output
kernel. The final implementation covers only the escaping output domain. The
table includes that correction; the remaining large-case gap needs different
tile/block schedules and a cost comparison, not another reduction-axis case.
This benchmark does not resolve the earlier 256/259-row normalization anomaly.
A tuned normalization reference and a transformer workload remain outstanding.

## Resident columns and the kernel entry, 22 September 2026

Before retiring the earlier scalar-binding reduction path, its whole-column
cuTile schedule was timed against the graph provider's streamed schedule on the
residual RMSNorm forward fixture (`tanh` epilogue, returned `h` and `rstd`).
Same GB10 and toolchain as above; medians of seven samples of 200
host-issued `replay!` calls, compilation excluded. The older path fell back to
four launches above 4096 rows. The columns come from separate runs as the
schedule changed, with device clocks unlocked; the later same-run
re-measurement below is the better guide to the final comparison.

| Shape | Older whole-column path | Streamed graph region | Resident graph region, flat entry | Resident graph region, tuple entry |
| --- | ---: | ---: | ---: | ---: |
| 256 × 32 | 10.3 μs | 13.6 μs | 8.2 μs | 6.7 μs |
| 259 × 32 | 8.5 μs | 10.2 μs | 7.8 μs | 6.2 μs |
| 8192 × 32 | 1429 μs (4 launches) | 55.8 μs | — | 39.0 μs (streamed) |
| 1024 × 1024 | 59.3 μs | 276 μs | 107 μs | 39.2 μs |
| 128 × 4096 | 88.8 μs | 720 μs | 294 μs | 61.4 μs |

The streamed schedule was four to eight times slower than the older path on
wide matrices. Keeping a column of at most 4096 rows resident in one tile,
with one shared value cache across reduction phases and outputs, recovered part
of that. The rest was the kernel entry: the Tile IR of the two resident
kernels was the same apart from memory-token chaining, yet `nsys` reported 128
threads and 21 registers for the older kernel against 256 threads and 255
registers for the graph kernel. cuTile 1.0's alias analysis starts every
pointer-carrying kernel *argument* in its own alias class, so separate array
parameters leave the loads and stores unordered, while a tuple parameter puts
them in one class. A twelve-line reduced kernel reproduced the effect: 255
registers and 162 μs with flat arguments, 19 registers and 25.6 μs with one
tuple argument. Neither `num_worker_warps` hint changed the register count.

Resident kernels therefore take their arrays as one tuple. Streamed kernels
kept separate parameters because cuTile 1.0.0 rejected their tuple entry. A
later investigation found that failure to be an `AssumeOp` cached across
regions, fixed it locally, and measured the tuple entry *slower* for streamed
kernels: with one alias class, each sweep's loads wait for the previous
stores. See [cuTile issues](cutile-issues.md). With these changes the graph provider matched or beat
the older path at every shape, and the older path, its `fuse_reductions` flag,
and its separate selector were removed.

`examples/research/normalization_schedules_benchmark.jl` repeats the
graph-region and decomposed measurements across these shapes and places each
against the memory-traffic bound, using `examples/research/roofline.jl`. Both
paths are prepared and checked before anything is timed and are measured in
alternating order. Device times are per step from one CUDA graph of 200
back-to-back executions, so host submission gaps are excluded. *Warm* repeats
one input set; *cold* rotates through enough prepared copies, each with its own
inputs and outputs, to cover the 24 MiB L2 at least four times. *Replay* is
host-issued `replay!` on the warm set. *Traffic* counts every input and output
array once, the least a single-pass schedule can move, and effective bandwidth
is traffic over cold time. It is compared with a 1 GiB device copy measured in
the same run: 234 GB/s, 86% of the 273 GB/s the bus width and memory clock
give. Medians of seven samples, 22 September 2026; two consecutive runs agreed
within about 1 μs on the large shapes.

| Shape | Traffic | Graph region warm / cold / replay | Effective | Of copy | Blocks per SM | Decomposed cold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 × 32 | 0.13 MB | 6.9 / — / 10.3 μs | — | — | 0.67 | 57.0 μs (warm) |
| 259 × 32 | 0.13 MB | 5.3 / — / 8.3 μs | — | — | 0.67 | 56.9 μs (warm) |
| 8192 × 32 (streamed) | 4.2 MB | 52.8 / 54.3 / 56.0 μs | 78 GB/s | 33% | 0.67 | 1425 μs, 3.0 GB/s |
| 8193 × 32 (streamed) | 4.2 MB | 53.2 / 71.6 / 56.3 μs | 59 GB/s | 25% | 0.67 | 1409 μs, 3.0 GB/s |
| 1024 × 1024 | 16.8 MB | 53.8 / 88.5 / 57.1 μs | 190 GB/s | 81% | 21.3 | 5611 μs, 3.0 GB/s |
| 128 × 4096 | 8.4 MB | 86.3 / 94.6 / 89.8 μs | 89 GB/s | 38% | 85.3 | 3014 μs, 2.8 GB/s |

Reading the table:

- At 1024 × 1024 the resident schedule reaches 81% of copy bandwidth. The
  warm time is a third lower because a 16.8 MB working set largely fits in L2,
  so warm figures overstate how close a schedule is to the bound.
- The two narrow shapes launch one block per column: 32 blocks on 48 SMs.
  They sit at a quarter to a third of copy bandwidth with most SMs idle.
  Splitting each column's rows across blocks, with a second pass to combine the
  partial sums, is the schedule this shape lacks.
- At 128 × 4096 there are 85 blocks per SM, each reading 512-byte columns,
  and effective bandwidth is 38% of copy. Placing several columns in one block
  is the likely missing schedule; this is inferred from the block count and
  has not been tested.
- 8193 rows is consistently about 18 μs slower cold than 8192 rows, while
  their warm times match. An odd row count leaves every column after the first
  aligned to only 4 bytes; that DRAM loads suffer from the misalignment while
  L2 hides it is the working explanation, also untested.
- The small shapes move 0.13 MB and cannot exceed L2, so they report warm time
  only; at 5–7 μs they are bounded by launch latency, not memory.
- The decomposed schedule's four generic `:cuda` launches reach 3 GB/s, about
  1% of copy bandwidth, at every shape. It is not a meaningful baseline for
  schedule quality; the bound is.

Absolute times depend on clock state, which is not locked on this machine;
the SM clock was sampled at 650–1000 MHz of 3003 MHz in earlier runs. The copy
measurement is taken in the same run for that reason.

## Packed columns and split reductions, 23 September 2026

The two schedules named above now exist, and the decomposed path is no longer
a strawman. Same benchmark, same method; copy measured at 221 GB/s in this run.

| Shape | Traffic | Graph region warm / cold / replay | Effective | Of copy | Schedule | Decomposed cold |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| 256 × 32 | 0.13 MB | 3.2 / — / 6.2 μs | — | — | resident (256, 1) | 10.6 μs (warm) |
| 259 × 32 | 0.13 MB | 2.8 / — / 5.4 μs | — | — | resident (512, 1) | 10.2 μs (warm) |
| 8192 × 32 | 4.2 MB | 37.9 / 42.0 / 41.5 μs | 101 GB/s | 46% | split 8 ways | 73.4 μs |
| 8193 × 32 | 4.2 MB | 30.3 / 38.5 / 34.0 μs | 110 GB/s | 50% | split 8 ways | 56.5 μs |
| 1024 × 1024 | 16.8 MB | 53.9 / 87.5 / 57.0 μs | 192 GB/s | 87% | resident (1024, 1) | 193 μs |
| 128 × 1024 | 2.1 MB | 12.1 / 16.0 / 14.5 μs | 131 GB/s | 59% | resident (128, 4) | 87.6 μs |
| 128 × 4096 | 8.4 MB | 32.9 / 49.2 / 36.3 μs | 171 GB/s | 77% | resident (128, 4) | 301 μs |

- **Packed columns.** Columns of at most 128 rows share a block in tiles of
  about 512 values, keeping at least 128 blocks. Taller columns were measured
  and are not packed: 256 × 1024 fell from 62% to about 40% of copy when
  packed, and at every packed shape tiles above 512 values were slower.
- **Split reductions.** A streamed single-axis reduction with fewer than 64
  kept tiles splits its reduced axis. One launch per dependency phase writes
  partial sums; a finishing launch completes per-row or per-column statistics;
  a final launch writes every tile's outputs in parallel. 8193 rows went from
  25% to 50% of copy.
- **TMA.** With evenly divisible rows, `tileiras` staged the split sweep
  through TMA and shared memory. That was the source of 8192 rows being
  *slower* than 8193 (36% against 50%). The split partial loads now disable
  TMA (46%). Resident tiles need it: without TMA, 128 × 4096 fell to 22%.
- **Decomposed path.** The generated `:cuda` kernels were spilling index
  metadata to local memory. After the fix the decomposed path is 1.7 times
  slower than the graph region at 8192 × 32 and 6 times slower at 128 × 4096,
  where it launches several kernels over small columns. The graph region's
  advantage now measures fusion rather than a broken fallback.

The narrow shapes remain at half of copy bandwidth. A 4.2 MB working set on 48
SMs leaves little to amortize the three launches.

## Taller streamed tiles, 14 October 2026

Streamed first-axis columns now load 1024 rows per step instead of 256. Columns
split only below 32 kept blocks (previously 64), and a split tile shortens, down
to 256 rows, until the grid holds 128 blocks. Resident kernels compile with
occupancy 4 and streamed kernels with no hint, as before. Same benchmark (warm /
cold μs, residual RMSNorm unless marked), median of 7 samples, before and after
the change in one session; copy measured at 217–222 GB/s.

| Shape | 256-row tiles | Schedule | 1024-row tiles | Schedule |
| --- | ---: | --- | ---: | --- |
| 8192 × 32 | 9.9 / 22.6 | split 8 | 6.3 / 20.1 | unsplit |
| 8193 × 32 | 8.0 / 24.3 | split 8 | 9.2 / 24.8 | unsplit |
| 16384 × 32 | 13.6 / 42.1 | split 8 | 10.7 / 39.5 | unsplit |
| 8192 × 48 | 13.0 / 32.3 | split 8 | 7.6 / 29.7 | unsplit |
| 8192 × 64 | 25.2 / 48.3 | unsplit | 11.9 / 43.0 | unsplit |
| 4097 × 256 | 31.0 / 95.1 | unsplit | 25.7 / 93.4 | unsplit |
| 32768 × 128 | 359.5 / 359.7 | unsplit | 355.3 / 355.3 | unsplit |
| 16384 × 16 | 10.0 / 23.1 | split 16 | 7.0 / 23.7 | split 16 |
| 65536 × 16 | 28.4 / 85.3 | split 16 | 26.4 / 84.6 | split 16 |
| 16384 × 8 | 8.4 / 13.9 | split 32 | 5.1 / 13.1 | split 16 |
| 131072 × 8 | 28.5 / 87.4 | split 32 | 28.0 / 86.6 | split 32 |
| 32768 × 4 | 8.4 / 14.1 | split 64 | 5.1 / 13.9 | split 32 |
| 8193 × 5 | 3.7 / — | split 32 | 3.7 / — | split 32, 256-row tiles |
| LayerNorm 16384 × 32 | 20.2 / 46.9 | split 8 | 16.3 / 41.9 | unsplit |
| LayerNorm 8192 × 64 | 35.5 / 55.3 | unsplit | 16.8 / 42.5 | unsplit |
| LayerNorm 8193 × 32 | 11.7 / 27.5 | split 8 | 9.3 / 24.1 | unsplit |

Resident and packed shapes (256 × 32 to 1024 × 1024, 128 × 4096) keep their
schedules and times within 0.1 μs, except the noisy warm 1024 × 1024 (22.7 and
20.5 μs). The domains of the benchmark at the top of this page are all resident
or second-axis and did not move.

- **Height.** 2048-row tiles matched 1024 on power-of-two columns but lost on
  ragged ones, whose last tile is mostly padding: 8193 × 32 took 10.6 / 28.0 μs,
  and 4096-row tiles 12.1 / 30.0 μs. 512-row splits sat between 256 and 1024.
- **Splitting.** With tall tiles, 32 unsplit columns already reach copy
  bandwidth on 48 SMs; splitting 8192 × 32 four ways cost 0.7 / 2.4 μs. At 16
  columns splitting still helps warm times (65536 × 16: 34.8 μs unsplit), with
  cold times within noise. The ragged 8193 × 32 RMSNorm is the one shape that
  lost: 1.2 μs warm and 0.5 μs cold against the old split schedule.
- **Short splits.** At 8193 × 5, 1024-row tiles left 8 chunks per column and 40
  blocks (4.6 μs warm); halving the tile until 128 blocks restores 256 rows and
  3.7 μs. Shortening further, to reach 256 blocks, cost 32768 × 4 1.7 μs warm.
