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

**Time to first execution, cold and warm.** From `using Tempest` to a
program's first result: cold, in a new process with an empty kernel cache
and no record; warm, in a new process whose cache holds every kernel and
whose record holds every time. The GPT step's cold preparation is 142 s
(the schedules trial's table); `test/bench/prepare.jl` times each
guard-rail program's trace, plan, build and first execution and counts the
kernels loaded and compiled, and its totals are the baseline to record
first. What stands trial: the precompile workloads (`src/precompile.jl`,
`ext/cuTileExt/precompile.jl`), the kernel cache
(`src/Compiler/runtime/cache.jl`), the cover's measuring of close calls and
its calibration (`src/Compiler/cover/choose.jl`, `calibrate.jl`), and one
function compiled per kernel (`generated`). Challengers, by what each
solved: tinygrad's cache, every compiled kernel and every search result in
one database keyed by source (`tinygrad/helpers.py`, `diskcache`), for the
warm case needing nothing but loads; JAX's persistent compilation cache and
torch.compile's FX graph cache, which key a whole program rather than a
kernel; and for the cold case, what a precompile workload can hold without
a device, where cuTile's own compile is the floor. Judged by the seconds to
the first result, cold and warm, on prepare.jl's programs and the GPT step,
with the compiled step's time, which must not lose beyond the noise. The
schedule search's cold cost (decided below) is one part of this.

The baseline, each in a new process without the startup file, wall seconds:
prepare.jl's programs cold 98 and warm 34, the GPT step cold 126 and warm 44
(its preparation 120 and 37). Warm was 99% Julia compilation: the workloads
covered only the CPU target, and the launch path compiled a specialization
per kernel. Cold adds one kernel compiled per second, serialized, and three
quarters of them are candidates the search times and discards. Taken so
far: a preparation that builds nothing, so the workloads cover the CUDA
planning path; no specialization per kernel; every proposal's trial planned
in parallel; a grouped five-axis attention in the workloads. After them:
prepare.jl's programs warm 20, the GPT step's preparation warm 19.6 and
cold 85.6. Still open: the kernels compiled and discarded, and their serial
compilation.

Worker processes for compilation (tinygrad's BEAM pool), tried and NOT
adopted. The premise was wrong: a process compiles kernels in parallel on
its threads already (16 distinct GPT kernels: 666 ms each in sequence, 92
ms each on threads); the one-thread run that suggested otherwise was bound
by calibration's references, built one at a time. A pool of eight warmed
workers lost on every measurement: the calibrated GPT step with cold
kernels 60.8 s locally against 69.3 s, 236 ms per kernel through the pool
against 92 on threads, a worker's first kernel 6 to 9 s. The patch is
`compile-workers.patch` in the session's scratchpad. What the intervals
showed instead: compiles arrive in bursts of twelve to sixteen, one group's
candidates at a time, between stretches of one at a time. Taken: every
group's measured candidates built at once before any decision
(`prebuild!`), the calibrated GPT step with cold kernels 65.7 to 51.0 s;
the harness's cold GPT preparation 85.6 to 71.6 s. What remains cold, on a
calibrated device: one burst of compiles (about 20 s), the candidates timed
one at a time, and the warm work; on a fresh device, calibration's
references built one at a time (about 25 s). Taken next: every function a
plan applies priced together, the references built side by side
(`fit_applied_all!`, `price_all!`): one precision's calibration 24.6 s
against about a minute, prices within the noise of the sequential ones, the
GPT step's cold preparation 71.6 to 63.0 s.

Taken next: the staged search of a product's tiles (decided below), the GPT
step's cold preparation 63.0 to 55.1 s, prepare.jl's programs 68.6 to
63.2 s. Then measured and closed without a change, each on the cold GPT
preparation with every compile burst and every timing wrapped. The kernels
of the groups the search never touches, compiled last: 27 kernels in a
2.5 s concurrent burst, and compile throughput is the bound, so an earlier
burst would take as long; 7 of them run on buffers that alias (results
written in place), whose keys a build on the decider's separate buffers
cannot serve. Timing the candidates one at a time: 140 launches timed in
1.2 s. The cover's trials searched together ahead of pricing, rather than
one group at a time inside each comparison (`trials-batch.patch` with the
session): the small bursts fell from about 10 s to 4 s, but the batch
fitted four reference functions and measured ten candidates the
sequential pricing never reaches, and lost the A/B, 54.5, 54.3, 53.9 s
against 55.8, 56.3, 54.5. Each group's claims and estimates found once
rather than by `price_all!` and `options!` each: no change, 55.0, 54.7,
55.3 against 55.4, 55.1, 54.8; the 2.7 s charged to the second pass was
Julia compiling the first. Standing, each in a new process: the GPT step's
preparation warm 37 to 20 s and cold 120 to 55 s; prepare.jl's programs
warm 27 to 13 s in the process, cold 92 to 63. The cold GPT preparation
now: compile bursts 12 to 16 s for 130 kernels, of which the search on the
generator's plan 7 s for 84; timings 1 s; the front end 5.5 s; the device's
calibration 6.7 s; and about 25 s of CPU work in the cover, planning 81
trials of the whole graph and compiling the compiler, which the warm
preparation pays too.

Taken next, the warm floor. The warm GPT preparation's 20 s were 11.7 s of
Julia compilation, 1214 instances of some 10 ms each, and 8.5 s of work: the
cover's loop, which plans the whole graph again after each of the 23
classes it takes and prices 26 proposals again around them, 3.6 s; its
trials planned side by side, 0.5 s of wall; the search on the generator's
plan 0.8 s; the build 1.5 s. Of the compilation, three causes, each with
its number. The second kernel package to load voided the first's cached
cover code: cuBLAS's CuArray constructor from a Diagonal, reached from
`upload`, invalidated 333 instances, and the Implementation protocol's
call sites, bound by inference to the one method a package sees while it
precompiles, some 60 more. The protocol called in the latest world and the
upload filled rather than constructed: the cuTile workload's own program
compiles 1.85 to 0.83 s in a session with both packages (0.19 with cuTile
alone), the GPT step warm 20.2 to 19.6 s, prepare.jl's programs 13.0 to
12.3. The tracer specialized on what it traced: its entry points
unspecialized and a selection's indices data rather than a type parameter,
the trace's compilation 3.62 to 2.31 s, the GPT step warm 19.6 to 18.1,
prepare.jl 12.3 to 11.8. The cuBLAS products had no workload: with one,
the two-package program 0.83 to 0.34 s, the GPT step warm 18.1 to 17.2,
prepare.jl 11.8 to 10.3. Measured and left: cuTile's integer overlays
show in Julia 1.13's log as voiding Tempest's tracing code, but tracing
again after cuTile loads costs 0.03 s. Open, with numbers: the IR's term
types (`Composed`, `Relative`), on which every visitor specializes, about
1 s of compilation; `Pointwise{F}` and `Reduction{F}`, 0.35 s, which the
rewrites dispatch on; the walkers of the arguments' nested structures,
0.3 s; the build's instances on device types, 3 s, which no workload can
run; and the cover's loop, 3.6 s of work.

Taken next, the warm preparation of Qwen3.5-4B's decoding step, 5038 nodes
and 440 kernels: 52 to 59 s in a new process, 31.8 s of it work once Julia
has compiled. The cover's trials, planned side by side, waited 24 s on two
of the 23 it plans: the down projection's fold and an attention layer's
output projection, each recomputed where it is read, planned in 24 and 22 s
where every other took under a second, on one thread as on twelve. Most of
it was hashing: a term was hashed by its fields, so each lookup of a term in
`subterms` hashed its whole subtree, and a projection recomputed puts its
body in every reader's term. Now `Apply` and `Recomputed` keep a hash
computed once from their parts' (`src/Compiler/IR/terms.jl`), which the kernel
cache's text leaves out: that trial 23.0 to 3.6 s; the 4B step's preparation
in a new process 52.4 and 58.7 to 33.8 and 42.9 s, and 31.8 to 12.2 s once
compiled, with the same kernels and tokens; the GPT step's warm preparation
24.8 to 17.9 s, its device time and the guard rails' within the noise. Still
open: a trial that cannot win, a projection recomputed at every element its
readers read, is planned whole before it is priced, 3.6 s of the trials'
wall; and the 4B step's Julia compilation, 20 to 30 s.

Taken next, by the corpus's gate, a cold preparation under 60 s from
`trace` to the first result. Before, each in a new process, SM clocks 2411
to 2431 MHz on a machine other jobs shared: node 53.4 s cold and 31.6 warm,
heat 29.9 and 22.0, the LSTM at batch 64 71.6 and 31.1, the GPT step 78.9
and 42.3. Where the seconds went, each phase timed with a build that wraps
them (a patch with the session, not kept), cold and warm, before the three
changes below and after them, one run each, Julia's compilation summed over
threads:

| phase | LSTM cold | after | LSTM warm | after | GPT cold | after | GPT warm | after |
|---|---|---|---|---|---|---|---|---|
| `using` | 2.8 | 2.8 | 2.8 | 2.7 | 2.7 | 2.7 | 2.7 | 2.7 |
| trace | 4.5 | 4.8 | 4.9 | 4.3 | 3.4 | 3.4 | 3.5 | 3.7 |
| lower, rewrite, partition | 2.4 | 2.3 | 2.6 | 2.1 | 2.7 | 2.5 | 2.8 | 2.7 |
| entering the preparation | 2.1 | 1.2 | 2.3 | 1.1 | 2.1 | 1.2 | 2.1 | 1.2 |
| cover: the device's calibration | 11.4 | 8.9 | 0 | 0 | 12.1 | 10.0 | 0 | 0 |
| cover: compile bursts | 31.2 | 30.8 | 0.9 | 0.5 | 22.7 | 21.6 | 1.0 | 0.7 |
| cover: timing claims | 2.0 | 1.8 | 0 | 0 | 1.5 | 1.4 | 0.5 | 0.5 |
| cover: the rest, planning and Julia's compilation | 18.9 | 13.3 | 18.4 | 10.9 | 38.5 | 27.2 | 31.0 | 20.9 |
| build, bind | 0.7 | 0.6 | 3.4 | 3.2 | 2.0 | 1.9 | 4.0 | 3.6 |
| first execution | 0.0 | 0.0 | 0.6 | 0.5 | 0.0 | 0.0 | 0.4 | 0.3 |
| total | 74.5 | 64.4 | 34.5 | 23.3 | 87.6 | 71.2 | 48.2 | 35.7 |
| Julia's compilation | 45.7 | 38.3 | 33.5 | 21.6 | 79.5 | 64.5 | 42.9 | 29.4 |
| kernels compiled: calibration, options, not in the plan | 28, 141, 125 | 28, 141, 125 | 0 | 0 | 31, 244, 205 | 31, 247, 208 | 0 | 0 |

The GPT's `after` column is before the claims' change; with it, 31, 188
and 149 kernels, its bursts 19.6 s.

Warm was Julia compiling Tempest's code, 33.5 of the LSTM's 34.5 s, and a
warm run's `--trace-compile` listed 1928 instances (LSTM) and 2252 (GPT)
compiled in the session, most of them in the package images already and
rejected as they loaded. Tempest's image is compiled without cuTile,
GPUArrays or SparseArrays, which add methods to `reduce`, `prod`,
`minimum`, `count`, `vcat` and `Base.setindex` as they load; an instance
whose inference called one of them on a type it could not narrow, a
`reduce(vcat, map(f, index); init)` over a `Tuple{Vararg{Affine}}` or a
`[v; x]` that SparseArrays' `vcat` of numbers and vectors now takes, failed
its edges at load, and with it every instance that called it, the
generator's `estimate` and `claims` among them: 4908 instances rejected, 1454
of them Tempest's and its extensions', 1491 under two walkers of an index's
values alone. Taken: those calls on concrete types, a selection's indices
asserted where the node is one, concatenation by `push!` and `append!`
(`IR/expressions.jl`, `plan/planner.jl`, `program/roll.jl` and eleven
more): 1454 to 606, the instances compiled in a warm run 1928 to 1779 and
2252 to 2080, the warm preparation by the corpus 31.1 to 23.1 s (LSTM) and
42.3 to 33.1 (GPT). Still rejected: roots in Base that Tempest's code
reaches, `convert(::Type{Int32}, ::Integer)` under cuTile's `Int32(::Bool)`
(193), `unsafe_read!` under KernelAbstractions (293), and Tempest's own
under ten each; and the instances no device-free workload runs, the
measuring decider's and the build's.

Cold less warm is the device's calibration, the compile bursts, and timing.
The bursts overlap: 148 thread-seconds of compilation in 34 s of wall for
the LSTM, 4.4 kernels at once, 266 in 29 s for the GPT, 9.2, up to twelve
threads in a burst. No lock serializes them: cuTile's `job_results_lock` is
held to look a result up and to publish it, the cache's `RESOLVED_LOCK` to
find a key's task. A burst lasts as long as its slowest kernel, the LSTM's
slowest 4.5 to 4.8 s each, and the stages run in sequence, 19 bursts
with kernels for the LSTM and 17 for the GPT; the same 166 kernels took 24
and 31 s of bursts in two LSTM runs. The calibration's lines and memories
built their references one shape after another. Taken: built side by side
and timed one at a time, as the scalar functions' were (`fit!`,
`fit_memory!`), calibration 11.0 to 8.9 s (LSTM) and 11.4 to 10.0 (GPT).

Of the kernels compiled for options, 125 of 141 (LSTM) and 205 of 244
(GPT) are not in the plan built: candidates the search times and
discards. The LSTM's are close calls, every one measured within 1.17 of the
best estimate. The GPT's included 14 claims of products split four ways,
estimated 3.0 to 4.4 times the best tile and measured 3 to 8 times, never
fastest, measured because `measured_ranks` took every claim of the best's
implementation. Taken: an implementation's own claims past the estimate's
error of its best measured once the record refutes the best's estimate
(`refuted`), the GPT's cold preparation 281 kernels to 219, its bursts 22.4
to 19.6 s; the LSTM's measured options unchanged, 166 kernels.

A cold run's plan differs from another's by its measured close calls, with
or without these changes: the GPT 182, 178, 182 and 182 kernels in four
runs, a cuBLAS candidate taken at 17.1 against 18.1 μs and joins ordered
by their measured gains, and of 70 groups two runs share, 2 to 4 take
another tile within the noise; the LSTM's recurrence step takes its
product at 64×64×64 split four ways (5.70 ms a step) in six runs and at
64×64×32 split eight ways (5.90) in two, one of them without the claims'
change. Standing, by the corpus, cold and warm, two runs each, SM
clocks 2405 to 2418 MHz: node 44.8 and 45.7 s cold, 22.0 and 24.2 warm;
heat 22.7 and 21.2, 15.3 and 14.6; the LSTM 57.7 and 62.6, 22.1 and 23.7;
the GPT 72.0 and 66.6, 33.9 and 35.4. The steps stand within the noise of
before's: node 6.54 ms against 6.57, heat 1.28 against 1.28, the GPT 1.18
against 1.19, the LSTM 5.71 against 5.68, or 5.90 where its step takes the
other tile.

What remains of the GPT's 67 to 72 s cold: compile bursts about 20 s, which
fewer candidates or concurrent stages would shorten; calibration 10 s, once
a device; Julia's compilation of the measuring and building paths, which
only a workload with a device would hold; the cover's planning. The
LSTM's 58 to 63 s: bursts 24 to 31 s, most of it waiting on each
stage's slowest kernel, which cuTile compiles; calibration 9 s; warm 22 s.

The floor, by a system image (PackageCompiler 2.4.5, Julia 1.13.3) holding
Tempest, its extensions, CUDACore, cuTile, cuBLAS, cuDNN and the array
packages, compiled with the precompile statements of the GPT step's and the
LSTM's sessions, cold and warm. Each in a new process, on the shared machine
under its GPU lock, against the base in the same sitting (instances: those
Julia compiled in the session, from `--trace-compile`):

| program | base warm, s | image warm, s | instances | Julia's compilation, s |
|---|---|---|---|---|
| GPT step | 47.0 | 15.9, 20.2 | 1993 to 242, 227 | 35.0 to 3.8, 4.7 |
| LSTM | 34.1 | 4.7, 5.6 | 1680 to 132 | 29.9 to 1.6, 1.4 |
| node, not recorded | 34.5, 31.4 | 9.3, 13.0 | 2161 to 544 | 34.1 to 7.2, 9.3 |
| heat, not recorded | 20.2, 16.6 | 8.3, 9.0 | 1263 to 452 | 19.6 to 7.6, 8.4 |
| prepare.jl's programs, not recorded | 39.9, 29.2 | 20.1, 20.6 | 1551 to 831 | 32.2 to 15.0, 15.3 |

`using` falls from 2.7 to 3.5 s to 0.01 s; cold, the GPT step prepares in
79.7 s and the LSTM in 48.7. On the programs it recorded the image leaves
the programs' own closures and types; on the others, their shapes, 450 to
830 instances. The image is not offered, nor a recipe for one: it takes 8.5
to 11 minutes and 30 GB of memory to build (1530 CPU seconds, a 1.14 GB
library), again after any change to any package in it, and it does not load
as built, since KernelAbstractions' OpenCL backend leaves three device
intrinsics (`julia.opencl.random_keys`, `julia.opencl.random_counters`,
`julia.gpu.state_getter`) as undefined symbols in the host code; a preloaded
library defining them stands in, and KernelAbstractions'
`precompile_workload = false` does not prevent them.

What a warm session compiles, split by what reaches it without a device: the
same program prepared again in a new process for CUDA without building
(`build = false`, as cuTile's workload prepares) or prepared and run on the
CPU. An instance either of those compiles is one a workload without a device
could hold, missing from the images because no workload's program has this
program's shapes; the rest are the device paths, the building, binding,
launching and measuring no workload runs without a device. Seconds of
Julia's compilation in one warm session, as `--trace-compile` times each
instance, on the idle twin machine, at 1c3627d and on the branch that
compiles the compiler's methods once whatever a program's shapes
(prep-latency at cedaf8e); instances for node and heat, counted on the
shared machine at 33ebd32 and on the same branch:

| program | tree | all | reached without a device | device paths | of which on device arrays |
|---|---|---|---|---|---|
| GPT step | main | 33.0 s | 23.4 s | 9.6 s | 2.8 s |
| GPT step | branch | 18.0 s | 10.6 s | 7.4 s | 2.1 s |
| LSTM | main | 22.9 s | 15.7 s | 7.2 s | 2.3 s |
| LSTM | branch | 17.8 s | 11.0 s | 6.9 s | 2.3 s |
| node | main | 2114 | 1385 | 729 | 197 |
| node | branch | 1758 | 1067 | 691 | 185 |
| heat | main | 1263 | 980 | 283 | 89 |
| heat | branch | 1119 | 846 | 273 | 87 |

The branch takes 55% of the GPT's seconds reached without a device and 30%
of the LSTM's, and 23% and 4% of their device paths. The warm sessions
themselves, there: the GPT step 42.1 s to 25.9 (2053 instances to 1457,
31.3 s of compilation to 16.5), the LSTM 24.6 to 19.4 (1675 to 1402, 22.3
to 17.1).

Of the instances compiled again after the images held them, rejected as they
loaded: 55 (GPT) and 52 (LSTM), counted at 33ebd32; of the 51 and 47 whose
roots the log names, none has them in Tempest alone, each calling into
AcceleratedKernels' `mapreduce`, CUDACore's allocation, or Base through
cuTile's overlays. Loading rejects 3571 instances, 854 of them Tempest's and
its extensions', with the kernel packages loaded first, as the corpus loads
them, and 4225, 117 of them Tempest's, with Tempest first, whose code the
kernel packages then invalidate as they load (a second count, by another
method: 1436 of Tempest's rejected with the kernel packages first; 420
rejected and 1478 invalidated with Tempest first). Which a user gets is the
order of their `using` lines. Measured and left: Tempest's call sites the
log names typed (`all`, `prod` and `findmin` with `dims`, `findfirst`,
`convert(String, ...)`, seven files), which took their roots out and raised
Tempest's rejected instances 885 to 996 (with the statement lists below
compiled in), since a narrower type routes a value's conversion through
Base's integer conversions, which cuTile's overlays void. That is Julia's
doing, in 1.12 and 1.13: a method added to a method table of one's own
(`Base.Experimental.@overlay`) invalidates compiled code that calls the
function it overlays on an abstract type, which the overlay never serves.
Reproduced alone: compile `h(r) = Int64(r[]) + 1` and
`k(r) = convert(Int32, r[])` for a `Ref{Integer}`, then overlay
`Int64(x::Bool)` and `Int32(x::Bool)` in a method table of one's own; on
1.11.9 both stay valid, on 1.12.7 and 1.13.3 both are invalidated. With
cuTile's integer overlays removed, in a copy outside the checkout, loading
rejects 2465 rather than 3571, Tempest's 576 rather than 854.

Precompile statements recorded from the sessions, compiled by the extensions
as they precompile, measured and not adopted. From the GPT's and the LSTM's
sessions at 33ebd32, the statements naming nothing numbered by a module's
count and nothing of the program's own: 3753, which cover of the GPT's
seconds above 8.4 of device paths and 19.1 of its shapes, of the LSTM's 6.5
and 12.7, and of node's and heat's instances, which they did not record, 571
and 227 on device paths and 942 and 557 of shapes, for 54 CPU seconds more
of precompiling the extensions. Kept only where both programs compiled it:
1180, the GPT's 5.2 and 3.8, the LSTM's 4.1 and 3.6, node's 393 and 323,
heat's 176 and 168, for 15 CPU seconds. The statements that took the most
time, to 80% of it: 1283, for 41 CPU seconds, mostly the programs' shapes.
Closures numbered by their function (`#price!##0#price!##1`) are renamed as
the function changes; without them the lists cover 5.5 and 3.0 of the GPT's
device paths, not 8.4 and 5.2. Not adopted, since a list stands in for an
interface. The device paths dispatch at run time below every entry point (an
`Executable`'s fields are untyped, its buffers a `Vector{Any}`), so
precompiling 33 entry points by construction, `execute!`, `replay!`,
`capture!`, `bind!`, `build_claim` for cuTile's generator and cuBLAS's
products, `time_launch` for each run, `build_choice`, `prepare`, and
`upload`, `allocate` and `point_at!` for seven element types, gives the GPT
step 0.8 s and the LSTM 0.6 (on the branch 0.8 and 0.6; node 23 instances,
heat 16): a list reaches more only by naming every dispatch it recorded. The
clean form is a refactor of the runtime: every device operation of a
prepared program through the Target interface, the kernel packages' building
and launching written against it (cuTile's module load and `cuLaunchKernel`
are calls to CUDACore today), buffers taken unspecialized, and a stand-in
device under the CUDA target, host memory, launches that do nothing, timings
fixed, for a workload to run every path a session runs. It would reach at
most the device paths not on device arrays, 6.8 s (GPT) and 4.9 (LSTM), 5.3
and 4.6 on the branch, node's 532 instances and heat's 194; no workload
without a device reaches the rest.

Julia keeps ten compiled images of a package
(`JULIA_MAX_NUM_PRECOMPILE_FILES`), and each environment, each Tempest
checkout and each set of flags (`--check-bounds`, coverage) needs its own:
past ten, the one evicted pays a precompile, a minute or more, in its next
`using`. `using` is 2.7 to 3.5 s on the shared machine: Tempest 0.46, cuBLAS
0.45, cuTile's extension 0.39, cuBLAS's 0.22, SparseArrays 0.22,
CUDA_Driver_jll 0.19 (91% of it compiling), GPUArrays 0.18, Pkg 0.15,
CUDA_Runtime_jll 0.13, CUDACore 0.12, REPL 0.11; Pkg loads for the CUDA
libraries' lazy artifacts, REPL for GPUCompiler and cuTile. The workloads'
price at install, on a shared machine: Tempest precompiles in 49 to 101 s
with its workload and 2.9 s without it, cuDNN's extension in 15 to 28 s and
2.6.

Julia's compilation in a warm preparation, by whose code it is. Two
programs are prepared one after the other in one process, without the
device; what the second compiles after the first is what the compiler
compiles because the program differs, with the second's own code and any
operation the first never met. Most of it was Base's generic methods
compiled for the shapes of the compiler's values: `map`, `get!`, `hash`,
`isequal`, `sort` and comprehensions over a node's inputs (`NTuple{21,
Int}` in the LSTM), a definition's operands, a term's arguments, an index's
`Affine`s and the nested tuples of the structural keys; the closures
capturing them; and the planner's methods for each function and fold,
through `Pointwise{F}` and `Reduction{F}`.

Taken, in two kinds. Principles at their place: a `Key` that hashes once
and compares element by element, for the library's tables and lowering's
alike nodes; definitions without type parameters, the index interpreter
taking the function and fold through a barrier; pricing by the rates
table's own function objects; the walkers of the user's nested arguments
walking tuples by their fields; `phase` taking its work unspecialized.
Annotations: `@nospecialize` with `@nospecializeinfer` on the functions
that take an index, a domain's extents or a selection's indices, and on the
constructors of a node, a term, a definition and the views; operations
compared by identity; and the reads, positions and member tables collected
into the element types the code declares. Held back, on branch
prep-latency as the measured reference: the parallel structures, `elements`
and `map_elements` where passes walk operands, arguments and inputs, and
`key_cmp` beside Base's `cmp`; and cuTile's gathered atoms in typed vectors,
three lines for the emitter's refactor. Instances newly inferred by the
second program, and Julia's compilation in the GPT's first preparation
without the device (loaded machine):

| | GPT after LSTM | LSTM after GPT | node after heat | heat after node | GPT compile, s |
|---|---|---|---|---|---|
| main | 14,362 | 8,488 | 8,619 | 3,872 | 26.1, 23.9 |
| principles | 10,421 | 6,352 | 6,040 | 3,454 | 15.7, 17.2 |
| and annotations | 10,196 | 6,337 | 5,937 | 3,289 | 13.0, 13.4, 12.6 |
| and element types (taken) | 8,431 | 6,250 | 5,883 | 3,123 | 11.3 |
| and parallel structures | 6,923 | 5,600 | 5,169 | 2,677 | 10.1, 9.9, 9.5 |

On the idle twin machine, in one sitting at 1c3627d, trace to first
result, each in a new process (the configuration with the annotations
without the element types; Julia's compilation in thread-seconds; the
instances it compiled):

| | main | with the annotations | with the parallel structures |
|---|---|---|---|
| GPT step, warm | 40.4, 41.3 s | 30.3, 31.3 s | 28.6, 26.6 s |
| its compilation | 29.8, 30.8 s; 1917, 1861 | 20.8, 21.2 s; 1555, 1555 | 18.8, 17.0 s; 1397, 1318 |
| GPT step, cold | 94.1 s | | 78.3 s |
| LSTM, warm | 23.9, 25.3 s | 19.4, 20.3 s | 21.9, 19.1 s |
| its compilation | 21.7, 22.9 s | 17.4, 18.2 s | 20.7, 16.9 s |
| LSTM, cold | 63.2 s | | 70.8 s |

The cold runs differ in what the cover measured and built besides (367
kernels against 335 for the GPT, 205 against 223 for the LSTM), and the
first warm run with the parallel structures compiled 5 of the LSTM's. The
second preparation in the same process, the work once compiled: 10.7 and
10.3 s on main, 9.0 and 9.0 with the annotations, 8.8 and 8.6 with the
parallel structures (GPT), 1.8 to 1.9 in all (LSTM). The steps are the
same: the GPT's 1.087 ms in all three, the LSTM's 5.27, 5.29 and 5.26.

The plans are the same: `explain` is identical to main's on the four
corpus programs and on 300 fuzzer seeds, program and gradient, on the CPU
and planned for CUDA.

`test/specialization.jl` asks that a program of three inputs on 5×6,
traced, differentiated, prepared and run on the CPU after one of two
inputs on 3×4 with the same functions and ranks, compile no new instance
of a method defined under src/Compiler/ that had one after the first;
program/tracing.jl and runtime/, which meet the user's function and
arrays, are exempt, as are methods the second program meets first. On main
it finds 13 (the walkers, `phase`, `alike_ops`, `value_term`); on the
principles alone 3; with the annotations none. It cannot tell the taken
configuration from one with the parallel structures: their difference is
Base's methods on larger programs. It covers no extension, no CUDA planning
and no operation the pair lacks. It needs Julia 1.12, where every method's
instances are in one table (`Core.methodtable`); on 1.11 each function has
its own and none lists them all. It counts what a session compiles, not
what loading rejects.

The clean form of what remains on program-sized tuples, 2,385 of the
GPT's 8,431, is one type in IR for a sequence the program sizes,
immutable, its elements in a field typed `Tuple`, so `===` stays
structural and its hash and text are the tuple's, compiled once per element
type whatever the length. It is specified and not built: it fits as a
refactor of the existing types, in nine steps each gated on byte-identical
plans and keys, and would take about 2,000 of those instances and about 2 s
of the GPT's first preparation, for every shape including those no
workload has run. The stand-in Target specified above takes the
device paths, 4.6 to 6.8 s a warm session, which the type does not; the
type makes what a workload holds hold for every shape, so the two compose.

Measured and left. Per-module compiler options on the compiler's modules
and extensions, warm LSTM on the device, trace to prepared: `@optlevel 1`
25.2 s, `@optlevel 0` 22.2, `@max_methods 1` 28.7, against 24.0 and 25.5
without; instances rose to 1,144 to 1,182 against 1,015 to 1,027, the
step and the work once compiled unchanged. A module's options reach only
the methods it defines, and most of what is compiled is Base's,
specialized on Tempest's types. `infer=false`: precompilation fails,
because a fold's state type is inferred (`fold_types`, `Base.promote_op`).
`compile=min` on the compiler's modules, dry, one sitting: the GPT's first
preparation 28.4 s against 31.1, 21.1 s of compilation against 24.9; the
LSTM's 13.5 against 15.4; the second preparation, Tempest's code now
interpreted, 10.2 s against 9.7 (GPT) and 0.69 against 0.52 (LSTM). It
saves 2 to 4 s of compilation at 0.2 to 0.5 s of work. It is an option for
the user, not taken: Julia lets no submodule turn it off, and the module
the generated kernels are evaluated into is under the compiler's, so
CPU-target kernels would run interpreted. It needs the planner in a module
apart from the one kernels are generated into. The tracer's broadcast
walkers with `@nospecializeinfer`, tried and reverted: the result of
`copy` on a traced broadcast stops being concrete, the user's function is
inferred with the branch that materializes the broadcast live, and
inference under the user's call grew from 103 instances to 179 in the CPU
pair. The tracer meets the user's types by design, and the precision of the
user's inference is worth more than the walkers' instances.

What remains of the GPT's 8,431 after the LSTM. The user's own code under
tracing, the model's functions on traced arrays and the corpus's harness:
about 2,900 instances, 1.5 to 2 s. Tracing must compile it, the trace being
the user's function run on Tempest's types; only an interpreter for the
traced call would not. An estimate, not built: the GPT step's trace runs
about a thousand traced operations, each a few dozen statements of the
user's, at microseconds a statement interpreted, well under a second
against 1.5 to 2 s, though the Base broadcast and indexing calls it reaches
are compiled at their dynamic dispatch all the same. The compiler's code on
program-sized tuples, 2,385, the sequence type's. The compiler's code
compiled once whatever the shape, 1,610, and cuTile's emitters, 1,401 (565
by a tile's rank, bounded, so a workload over ranks 1 to 4 holds them): a
workload's. Pricing by the table's functions, 132, bounded. A system image
holding these programs' own warm runs leaves 132 instances on the LSTM and
242 on the GPT.

Taken next, a warm preparation replays the decisions a cover reached
(`src/Compiler/cover/reached.jl`), as tinygrad keeps its search's results
beside its kernels and JAX and torch.compile keep a whole program's
compilation. A preparation keeps the proposals its cover took, in order,
and each group's decision, under a key that writes out everything a
decision reads before deciding: the graph with every function and constant
it holds, the arguments' types and extents, the folds pinned and kept, the
steps of a recurrence, the target with its device and rates, the
implementations loaded and their packages, the TEMPEST_ settings but those
no decision reads, how the decider measures, and Julia's and Tempest's
identities. Beside them it keeps the record's and the rates file's entries
the cover read, each with the value it held or its absence. A preparation
of the same program whose files still hold those values plans the taken
proposals once, checks each group's text against the kept one, and reads
each decision back: no trial planned, nothing priced or measured, no kernel
compiled but the plan's own. An entry read since changed, or a group that
no longer forms, sends it to the cover; TEMPEST_VERIFY reaches the
decisions by the cover too and throws where the two differ, and
TEMPEST_REPLAY=false asks for the cover every time. On the idle GB10, each
run in a new process, each tree with a kernel cache and record of its own,
empty before the cold run, which calibrates the device; main at the
replay's base:

| GPT small | main | replayed |
|---|---|---|
| cold preparation | 95.1 s, 346 kernels compiled | 103.3 s, 3 covers, 348 compiled |
| warm preparation | 38.4 s | 15.5 s, nothing compiled or measured |
| warm, TEMPEST_REPLAY=false | | 37.8 s |
| warm, once Julia has compiled | 10.48 s | 1.04 s |
| step | 1.103, 1.103 ms | 1.105, 1.098 ms |

Of the replayed warm preparation's 15.5 s, 14.2 are Julia compiling. The
replayed plan is the one its cover reached: the warm explain is the cold
one's, and the cover's on the same files, byte for byte.

Decisions are replayed only when settled: reached by a cover that wrote
nothing to the record or the rates file and whose claims all built, so
that a cover in a new process would reach them again. A cover that
measures reads a time absent at first and present after, and on the record
it leaves a cover could rank its proposals otherwise. So a preparation that
measured as it decided covers again on what it left, until a cover writes
nothing, three at most (TEMPEST_SETTLE=false keeps the first). Without
settling, the GPT step's cold cover measured 303 claims, the second
process's cover one more and the third's none, and only the fourth process
would replay, the three plans the same, their explains differing only in
the times they print. Settling runs those covers in the cold process: 8.2 s
and 2 kernels more than main's cold preparation, 5 more times recorded, and
the second process replays. It is the default: its seconds are paid once per
program, and the warm preparation's saving every time after. Behind a first
result given before the search, settling's covers would run after the
result.

Kept decisions rest on their evidence compared exactly, which holds only
where a table in memory holds what its file reads back. It did not: the
first launch the device timed in a session was a Float32 (the replay
overhead was returned as the value of its assignment), kept in memory as a
Float64 and written at a Float32's digits, and a decision set whose
evidence held that time never replayed in the next process. One writer now
keeps and writes the same Float64 for the record and the rates file
(`keep!`), and the timer returns a Float64 every time.

Measured and left: the LSTM's decisions never replay on the device. Eight
cuTile product claims (256×256 and 256×32 at batch 100, TF32 tiles) fail
to build each time they are measured, and the record marks them lost at
Inf in the process but not in the file, so no cover of the LSTM is settled:
replayed 0, covered 12, cold 63.9 s against main's 65.7 and warm 24.8
against 25.5. Its steps, 5.44 and 5.47 ms against main's 5.34 and 5.36,
run plans that differ from main's in 5 of 33 kernels: each tree measured
its own record from empty, and the LSTM is covered as on main. What
removes it is the measuring consolidation, the record holding only
measured times and a failed claim read from the failures, after which a
failure is no reason to cover again: a claim that fails to build fails
under the same key. Also left: the rates the decisions read include the
applied scalar-function lines, 14 in the GPT step's evidence and 7 in the
LSTM's, of which 9 and 4 are zero or under 1e-13 s a position, noise a
recalibration refits; any such refit sends the decisions to the cover.

Also left: the decisions kept are never removed. The store of compiled
kernels beside them is bounded (`JULIA_OBJCACHE_CAPACITY`, 512 MiB by
default, the entries used least recently removed); the `decisions`
directory beside the record is not. A key holds Tempest's own identity, so
every edit of its source keeps a program's decisions again beside the old,
which nothing reads after. One tuning directory after two days of this
trial held 18 files in 2.2 MB, from under 1 KB to 1.9 MB a file. What
removes them is the kernel store's bound applied to them, by when each was
last read.

Taken next, the kernels' part of a cold preparation, each kernel's
compile, each burst and each timing logged by a build that wraps them (a
patch with the session, not kept), one cold run each of the GPT step and
the LSTM at batch 64. The GPT compiled 346 kernels: 30 for the device's
calibration, 20 trials of whether cuTile lowers a scalar function, 296 for
the search's options, of which 36 are in the plan built. The LSTM compiled
203: 23, 16 and 164. Of a cuTile kernel's compile, `tileiras` is 74% of the
thread-seconds (GPT) and 87% (LSTM); inference about 0.10 s a kernel, the
bytecode 0.06, Tempest's dependencies 0.01. Measured and left, the
assembler's flags, the GPT's 336 distinct kernels assembled one at a time:
175.5 s of CPU at `-O3 --lineinfo`, cuTile's default, 172.8 without line
information, 170.6 at `-O2` and 168.4 at `-O1`, 140.2 at `-O0`; `-O2`
emits the code of `-O3` for 330 of the 336, `-O1` other code for 334.
`ptxas`, which `tileiras` runs, takes 0.1 to 0.5 s a kernel, 10 to 15% of
it. The LSTM's slowest kernels, its cell's fused products of 450 to 555
statements, take 3.3 to 5 s at every level. Line information makes the
CUBINs 3.4 times larger for the same code, 70.4 against 20.6 MB for the
GPT's, which the kernel cache stores: an option for cuTile to take for the
cache's size, not for time.

The bursts, against their bound, their slowest kernel's compile or their
thread-seconds over the 20 cores: the GPT's 41.5 s of bursts against
26.7. A burst of 34 to 84 kernels ends about 1.5 s past its bound, its
kernels emitted on one thread and its cores contended; a burst of a few
ends with its slowest. Of the GPT's 41.9 s of search bursts, 22.4 were 14
bursts of 1 to 8 kernels: a comparison decides its groups one at a time,
and a group not searched yet is searched alone, three or four stages of
about 1.6 s each, the next group's after. Searched together, the starting
plan's groups with the trials', and both sides of a comparison before
either is decided, the same groups and so the same kernels, in a build of
the cover that did both (with the rates shipped as well): the LSTM's
searches of one group fell from 6 bursts and 12.1 s to 3 and 0.48 s, its
cold preparation from 77.0 s to 57.0 and its kernels from 203 to 176; the
GPT's did not fall, 17 bursts and 25.1 s, its cold preparation 117.1 s to
101.5 and its kernels 346 to 341. Not taken as built: what is timed and in what order it may be
built is decided in five places now (`measured_ranks`, `expand!`, `compare`,
`search!` and the cover's searches ahead), and batching them is a sixth;
it waits for one function that owns the question, of which the searches
together are callers.

Measured and left, fewer candidates timed. The estimate against the time
of every claim the cold runs timed: the generator's products 0.76 of their
estimate by the median, 70% within the close-call factor of 1.5; cuBLAS's
2.02, 33% within it. Of the GPT's 98 decisions, 38 had one option, 27 were
decided by estimate and 33 by 235 options timed, and the time chose other
than the estimate in 14. Policies replayed on the record, each against
measuring as now: one orientation of each product, 132 timings fewer and
the decisions' time 12.9% more (GPT), 6.0% (LSTM); no options beyond a
tile's (workers, a transposed load, the inner depth, occupancy), 81 fewer
and 3.9% more (GPT); the first stage's claims alone, 9.4% more (LSTM). A
record shipped as a prior, by product shape, from the user's record of 632
groups: the best tile of the nearest shape within 2% of the measured best
for 63% of groups, the best three for 87%, so a prior would still time
three and miss one group in eight. The two orientations of a product differ
by 2.4% at the median and 23% at the 90th percentile, 64% within 5%. The
same claim timed in two runs, 415 claims: the larger time over the smaller
1.086 at the median and 1.266 at the 90th percentile; the generator's
products 1.08 and 1.19, cuBLAS's 1.07 and 1.35, the generator's pointwise
kernels 1.30 and 1.68. One close-call factor of 1.5 for every kind sits
inside the products' noise and at the pointwise kernels' 90th percentile:
the data suggest a factor by kind, not changed here.

Measured and left, estimates fitted on the record. The same estimates
decide a plan where nothing is measured, the plan a first tier runs:
node's step 7.93 ms on the GB10's rates against 6.43 measured, 8.56 on
nominal rates (the tiers lane's). By the time of the option the estimates
pick among those timed, summed over the measured decisions, against the
best timed: as they stand, 8.0% over the best for the GPT's and the LSTM's,
15.6% for node's and heat's. A scale per implementation and claim kind, the
median of time over estimate, fitted on the GPT and the LSTM and tried on
node and heat, 15.6% to 12.0%; fitted on node and heat and tried on the GPT
and the LSTM, 8.0% to 221%, picking cuTile for the LSTM's batched products
at 3.4 to 5.3 times the best; fitted on all four, it loses so in sample. A
line in log time and log estimate does the same. The error is a regime's,
not an implementation's: node's products, 2.1 MFLOP on 2 to 4 blocks, are
estimated at 22 μs and run in 4, a launch of fewer blocks than the 48
multiprocessors priced at that share of the memory's bandwidth; the LSTM's
batched products are estimated alike for every tile, at 511 μs, and run in
648 to 3413, which only measurement ranks. Two terms of a launch of few
blocks, the memory's share saturating at m blocks and a block's loop along
the sum at least L a step, fitted on the record: on node and heat, m 24
and L 0, which takes them from 15.6% to 5.8% and the GPT's and the LSTM's
from 8.0% to 14.9%; on the GPT and the LSTM, nothing changes. The LSTM's
step product, its fused cell, is estimated at 80 μs unsplit on 4 blocks
and at 60 split eight ways, and runs in 42 and 22: priced right in its
memory, the unsplit one is taken, at twice the time. Holding both ways
wants a third term, the fused epilogue's price in this regime, which would
fit the corpus. Left open, with these numbers, so that no scale is fitted
again.

Taken, rates shipped for a device Tempest knows: a file per device in
`src/Compiler/cover/rates`, named as a machine's rates file is, stating the
timing basis its times were taken on and the device's memory bandwidth,
cache and multiprocessors as `device_rates` reports them
(`shipped_rates`), read under the machine's own rates file. Those are the
configuration's rated values, which the driver reports alike for every
unit, so a file is read on any GB10 of this configuration, a second unit
reporting the same; a unit's own clocks and power cap are not in them, and
show as the spread of the rates fitted on each. A file of another name, or
of the name and another identity or basis, is not read,
and the decider fits on first need as before, or prices on nominal rates
where it measures nothing. The GB10's is the file the tests held as a
fixture; `test/bench/rates.jl` writes it again. Rates only price options,
and the close calls among them are measured whichever rates priced them,
so a file that misprices, one package's products a thousand times slower
than it says, makes the plan take the other package and agree with the
interpreter (`test/cuda/cover.jl`). The decider's rates table is by
default the machine's own rates over the shipped ones (`known_rates`),
noted as the evidence of the decisions reached on it; `calibrate!` is given
the machine's alone. The decisions key holds the timing basis beside the
device's reported rates. Cold, by the corpus on the second GB10, in one
sitting: the GPT step 90.8 s on main and 77.8 and 77.9 with the rates
shipped; the LSTM 64.0 and 62.1 against 56.3 and 56.2; warm, and the steps,
the same (the GPT's 39.1 to 41.0 s and 1.09 to 1.10 ms, the LSTM's 24.0 to
24.9 s and 5.25 to 5.32 ms). The second GB10 reports the same identity, and
its two calibrations differ from the shipped file by no more than two
calibrations of one unit do (products' rates by 1 to 2% at the median,
latencies by 3 to 7%). An RTX 6000 Ada reads none of the file, and its
suite calibrates. Shipped rates do not make a cold plan the same run to
run. The GPT step's two cold runs on each GB10 differ, 162 kernels each,
in 3 to 15 of their kernels' claims and in three joins of a residual sum
into the normalization that reads it, close calls measured and falling
either way; the
four plans are of two kinds, 297 or 300 values stored, and a kind seen on
one unit is seen on the other. What makes a plan the same in the next
process is the replay: the warm preparation's plan was the cold one's,
line for line, the GPT's and the LSTM's.

Measured, whether a plan turns on the applied lines of the rates, 190 of
the file's 257 for the swept and inner regimes, fitted at 1e-14 to 1e-11 s
a position, and within one unit 32 of the 63 inner lines zero in one
calibration and not in the other. The GPT step and the LSTM prepared
measuring nothing, where the estimates decide everything, on the shipped
rates, on the shipped rates with those 190 lines zero, and on each of the
second GB10's two calibrations: every plan the same in every kernel and
every claim. The 190 at zero moved the GPT's printed estimates by 0.1 μs
in a few comparisons (47.8 to 47.7) and changed no decision; the steps
1.181 to 1.193 ms and 5.647 to 5.670. Not tested: measuring close calls,
the lines could move a comparison across the close-call factor, which at
deltas of 0.2% would take one sitting at it. The lines are noise to the
plans and not to the decisions kept: a refit across zero changes their
evidence and sends them back to the cover.

Counted and not built, kernels that take their extents as arguments: of
the 336 distinct kernels the GPT's cold preparation compiled, 217 differ in
more than their extents, the tiles' shapes kept; of the LSTM's 176, 105.
Every integer a symbol gives the same counts.

Warm, a kernel's load costs about 12 ms, one after another on one thread:
the kernel store's lookup and its dependencies' check 1.65 ms (0.16 once
the checks are cached), its function defined 3.8 ms, its key 1.5 ms, its
module loaded 5 ms; the GPT's 55 loads 0.7 s of its 3.4 s build, the rest
Julia compiling the build. Left, under a second. The final build compiles
again 8 of the GPT's 22 kernels and 2 of the LSTM's 3 that the search
compiled with other argument types or other buffers aliasing: left, in the
build's one burst. Taken, the floor's kernels keyed as the tiles' are: their
key held a hash of all Tempest's sources, so an edit anywhere compiled them
again, one at a time under CUDA's compiler lock, about 0.35 s each; the
code instances the compiler's inference leaves under its job's owner give
what a kernel used of Tempest's code, as cuTile's inference does for a
tile kernel, and it loads while that is unchanged. After an edit that
changes no code, node's warm preparation compiled the floor's 12 kernels
again, and compiles none now, loading 40; the LSTM's and the GPT step's
none either, loading 31 and 52.

Left unmeasured, the floor: a cold preparation that knows its decisions
and compiles only its plan's kernels. Counted, the GPT step's plan built
36 kernels of the 346 its cold preparation compiled, 24.4 thread-seconds,
the slowest 1.78 s, so in one burst on the 20 cores about 2 s beyond a warm
preparation that compiles nothing (15.5 s with the decisions replayed);
the LSTM's 10 kernels, 1.7 thread-seconds. The distance to it is the
search: the cold preparation's 77.8 s on the rates shipped, before
settling's covers, against about 18. Timing it, with the record and decisions copied and the kernel cache
empty, was cut from the second GB10's queue with node's and heat's runs.

Measured and taken, the LSTM's eight claims that never built. They were one
kernel's, in a trial the cover planned: a sum of the backward's weight
gradient over the steps riding the recurrence, which computes each step's
product of the input and the gate's gradient inside the recurrence's
kernel, taken with an implementation's candidate of that product as a
kernel of its own. That kernel stored nothing and nothing read it, its
products computed in the steps besides; its cuTile claims observed a fold
and stored nothing, which cuTile does not compile ("SSAValue not found in
context" at the loop: of the LSTM's 87 kernels that observe a fold, the 8
that store nothing are the 8 that fail; the reproduction is kept with the
session). The verifier states the fault (invariant 3: a value a riding sum
reads at each step is owned by another kernel), and verifies every plan
the cover tries where verification is on, as the tests have it; the corpus
prepared without it. Prepared with it, the LSTM threw, and the GPT step,
node and heat formed no invalid plan in 145, 102 and 5; `verify` cost 0.61,
0.67, 0.06 and under 0.01 s of their cold preparations, under 1%. A ride
is now about what its steps compute for the sum (`Ride(value, within)`,
`fixes`), so the cover takes no proposal of a kernel of one of those values
with it. The LSTM prepares with every plan verified, and the next process
replays its decisions, 3 replayed and none covered against 0 and 12, its
record holding no failure, in 15.3 s. Its plan is the one it had, group
for group and claim for claim, and so is its step: the rides were refused
then and are not tried now. Priced right they lose too: in the plan with
the product's candidate taken, the ride of an input weight's sum cost 2100
μs against 1810, of which the dead kernel's product was 117 to 129, and of
a recurrent weight's 15300 against 2810 estimated, of which it was 530 to
607.

Open, exclusive proposals. Where two proposals are about one value, as a
ride and the candidate of its product are now, the cover prices each
against the starting plan, ranks them by what they save there, and takes
them in that order; the second, about a value the first took, plans
nothing and is dropped, neither priced in the plan the first made nor
compared with it, and `explain` does not show it. So of a ride and a
candidate the one ranked first wins, by its saving against the starting
plan, with no comparison of the plan with the ride and without the
candidate against the plan with the candidate and without the ride. Before
this fix that comparison was made, on a wrong price; now it is not made.
Pricing both would cost a trial for each such pair in the plan as it is
when the second comes up (eight on the LSTM, against 215 plans its cover
forms), and a measurement where the two are close; a cheaper trial, the
scaling lane's replanning, would make it affordable, a search change to
propose after that lands.

Taken, verifying apart from crosschecking, and the corpus verifying.
`verify` (TEMPEST_VERIFY) checked plans against the invariants and also
compared the decisions a preparation replayed with those a fresh cover
reached: a cover's cost on every warm preparation, the LSTM's 15.3 s to
21.1 with its cover run again. The comparison checks the replay itself, a
shorter way against the whole computation, and is now a crosscheck of its
own (TEMPEST_CROSSCHECK, `crosschecking`), which the tests turn on beside
verification and nothing else does; a trial planned incrementally against
the plan of its whole inputs is the same kind of check. The corpus prepares
with every plan verified, the cover's trials among them, and each row says
so: 0.06 to 0.67 s of a cold preparation, under 1%, and the built plan's
check of a warm one.

Open, a preparation verifying by default. With verifying this cheap and
free of the crosscheck, what stands in the way is that a trial plan the
verifier rejects throws from inside the cover's task, ending a preparation
over a trial that would likely not have won. The cover plans every trial in
one place (`plan_taking`), which refuses a trial that forms no group by
returning nothing; a trial the verifier rejects would be refused there the
same way, counted, and shown by `explain` with the invariant it broke, and
only the plan built would verify or throw. The proposal type needs nothing
for it: a rejected trial is a composition of proposals the planner could
not plan validly, not a kind of proposal. The planner's own check of each
plan it forms would move to the two places a plan is used, `plan_taking`
and the build, and the tests would assert that no trial was refused rather
than rely on the throw.

**How preparation grows with the program.** `test/bench/scaling.jl`
prepares fourteen families of programs dry on the CUDA target, each scaled
by one parameter: the GPT step by layers, length and heads; the neural
ODE's RK4 rolled, and unrolled with steps of lengths of their own; the LSTM
and a stencil, each as a plain loop and as `accumulate`; independent
branches; chains of maps and of normalizations; a step over many leaves;
and a transformer's decoding step. Each size is prepared twice in one
process and the second timed whole, with each phase's seconds and bytes,
the planner's counts (`tallied`), and the exponent of each over the last
doubling, which a phase linear in the program keeps near 1. Seven passes
grew with the program squared or faster, each a pass over everything for
each thing, and each is now an index built once (6ce4025 to 4cff5f4):
ordering a plan's kernels, which looked at 35,713,676 kernels at 32 GPT
layers and looks at 184,856 (exponent 1.97 to 0.99); storage, which ordered
every kernel again for each argument a result could take and scanned every
slot for each value stored; the rewrites, which scanned the graph for each
frame, sum and map; rolling a loop, which walked each run of alike nodes
from each of its nodes; the planner's dependency walks, which allocated
three masks the size of the graph for each value tried; and the cover's
proposals and its bytes of a kernel, which passed over every proposal, or
every term, for each one. No plan changed: the explanations of the fuzzer's
300 seeds (forward and gradient, both targets), the corpus and the families
at two sizes, and of twelve family instances at their largest, are
identical before and after each change, as are the storage pass's slots on
614 plans under three choices of the kernels that write in place.

On grizzle, idle, 12 threads, dry on the CPU target, one run each, main at
1c3627d against 4cff5f4, in seconds (exponent from 16 to 32):

| | 4 | 8 | 16 | 32 | |
|---|---|---|---|---|---|
| GPT layers, 1c3627d | 2.89 | 5.89 | 13.8 | 37.1 | 1.42 |
| GPT layers, 4cff5f4 | 2.72 | 5.17 | 10.2 | 24.6 | 1.27 |
| RK4 unrolled steps, 1c3627d | | 1.97 | 7.05 | 29.7 | 2.08 |
| RK4 unrolled steps, 4cff5f4 | | 1.76 | 6.16 | 25.4 | 2.05 |

At 32 layers the rewrites take 1.97 s where they took 8.84, storage 0.28
where it took 2.92, and the whole allocates 41.6 GB where it allocated
60.6. The worst exponents left at 4cff5f4, from 16 to 32 layers: the
rewrites 2.15, the trace 1.90 (0.62 s), storage 1.64, the cover's loop 1.39
(8.8 s) and its trials 1.22 (2.9 s), the decisions 1.13 (7.8 s); from 16 to
32 unrolled RK4 steps, the cover's trials 2.49 and its loop 2.32 (5.4 and
15.1 s of 25.4), its edits 2.20, the trace 1.77 and lowering 1.76.

Taken since, on prep-scaling-2: relative sums rebuilt the whole graph
after each frame's sums, 64 rebuilds of 9,812 values at 32 layers, and
sought each frame's sums through every map that read it; a pass now
rebuilds once, and seeks them through a difference from the frame, then
the law's function, products, quotients and casts, all a relative sum
reads its frame through. Storage's section found the results that are an
input as it was by comparing every result with every pair that replaces
an input, 343 MB at 32 layers, 0.6 MB now. The rewritten graphs are equal
on 4,038 graphs (the fuzzer's first 1,000 seeds, the corpus and fifteen
family instances, each rounded and not) and the explanations unchanged.

What stands. The cover plans each trial whole, so trials that grow with
the program cost its size squared: the unrolled RK4's plans grow with its
steps, 76 at 8 steps and 248 at 32, and its values tried in groups with
exponent 1.70. A trial changes a few kernels, and a prototype that formed
again only those its change reaches, taking the rest from the plan it
changes where what they read was unchanged and checking each against the
whole plan under `TEMPEST_VERIFY`, tried 50,000 values in groups instead
of 414,273 at 32 GPT layers on four threads, every plan as it was. It is
not taken as built, the change reaching the planner as keywords and every
trial's record kept in a table no one emptied; its form waits for proposals
that state their change to the planner's inputs: a planned plan carrying
what formed it, `replanned(planned, change)`, and formation reading the
plan's state through a context, the plain one costing nothing. A step over
many leaves allocates 9.5 GB at 128 leaves where it allocated 76.7, its
pricing's exponent 2.0 where it was 3.1: one reduction reads every leaf,
each leaf's trial changes it, and no two edits are alike, so each is
decided. It needs a kernel's canonical form invariant to the order of its
members, so that every leaf's trial is one class, priced once. The
decisions grow with the program, about 5.5 ms for each of 1,408 options at
32 layers, most of their bytes in the target's estimate of each option
(normalizing affine indices a quarter, its views a tenth); deciding once
for every kernel alike stands open.

**A trial planned near the plan it changes.** The cover's trials each plan
the graph with a few of what the planner is given changed, and each formed
every kernel again: the unrolled RK4 at 32 steps tried 1,734,161 values in
groups over 248 plans. Forming now reads what is stored and the kernels
formed so far through a context alone, of five functions (whether a value
is stored, whether a kernel owns it, whether values depend on a group's
members, a kernel formed, and the forming of one kernel bracketed); a
recording context notes what each kernel's forming read, and a plan kept
with how it formed (`Planned`) is formed again with a change
(`replanned`), taking each kernel whose noted reads are as they were, asking
again a dependency walk that what differs leaves in doubt, and forming the
rest. The cover keeps its starting plan and each plan it takes so, and each
trial is formed near the plan it changes. Counted
before timed: a trial forms again a median of 4 or 5 kernels, of 126 to
514 in the instances counted; with the test suite's packages loaded, a trial that recomputes a fold
formed again 107 kernels of 254 at 8 RK4 steps, the kernels after it that
noted its storage or walked through a kernel formed again, until a walk
noted the question it was asked and the frontier a kernel grows toward
passed over what depends on its members (36 on average now). The explanations of the
fuzzer's first 1,000 seeds (forward and gradient, on both targets), the
corpus, the families and the twelve large instances are identical, every
trial of them and of the suite compared with the plan formed whole.

What holds a plan formed near another to the plan formed whole is that
comparison, made where TEMPEST_CROSSCHECK asks, as the suite does of every
trial its tests form; a preparation that does not ask takes the plan formed
near as it is. The comparison above was made by a script outside the
repository, its large instances 8 GPT layers and 16 unrolled RK4 steps at
most. At 1c24f8f, under TEMPEST_VERIFY and TEMPEST_CROSSCHECK, no trial
differed in any of: `test/bench/scaling.jl`, dry, at 16 and 32 GPT layers
(228 trials each, 90,129 and 179,409 kernels taken as formed before) and at
16 and 32 unrolled RK4 steps (1,100 and 2,140 trials, 153,170 and 503,570
taken); the corpus's four programs at their small shapes and its GPT at
GPT-2 124M's geometry, each prepared cold and warm on the GB10; and the
fuzzer's seeds 20001 to 20400 on the CPU and 20401 to 20460 on the device.
`test/work.jl` holds the comparison at 8 RK4 steps, where each of 715
trials takes about 170 kernels as formed before; larger is the bench's, run
by hand. Open: a test that a read made outside the context is caught, which
needs a context a test can break.

On grizzle, idle, 12 threads, dry on the CPU target, one run each, 772f16b
against 4a311a3, the change as it was measured before it was rebased onto
main, in seconds (exponent over the last doubling):

| | 4 | 8 | 16 | 32 | |
|---|---|---|---|---|---|
| RK4 unrolled steps, 772f16b | | 1.98 | 6.46 | 25.9 | 2.00 |
| RK4 unrolled steps, 4a311a3 | | 1.20 | 2.84 | 7.16 | 1.33 |
| its cover's trials, 772f16b | | 0.218 | 1.10 | 5.45 | 2.31 |
| its cover's trials, 4a311a3 | | 0.060 | 0.173 | 0.852 | 2.30 |
| its cover's loop, 772f16b | | 0.602 | 3.00 | 15.1 | 2.33 |
| its cover's loop, 4a311a3 | | 0.160 | 0.613 | 1.96 | 1.68 |
| GPT layers, 772f16b | 2.76 | 5.20 | 10.3 | 23.3 | 1.17 |
| GPT layers, 4a311a3 | 1.96 | 3.45 | 6.82 | 13.6 | 0.99 |
| its cover's trials, 772f16b | 0.228 | 0.505 | 1.09 | 2.94 | 1.43 |
| its cover's trials, 4a311a3 | 0.062 | 0.203 | 0.275 | 0.449 | 0.71 |
| its cover's loop, 772f16b | 0.871 | 1.60 | 3.62 | 8.72 | 1.27 |
| its cover's loop, 4a311a3 | 0.195 | 0.414 | 0.848 | 2.01 | 1.24 |

At 32 RK4 steps the values tried are 18,893 (exponent 0.78, where it was
1.70), and a kept plan of the GPT step at 32 layers holds 26 MB, 7.4 of
them how it formed. What grows now is what each trial does whole: it takes
every kernel it does not form again (503,570 kernels taken at 32 steps,
exponent 1.72) and orders every kernel of its plan (508,217, exponent
1.71). Of the replanned trials' 18.3 CPU seconds at 32 RK4 steps, deciding
what is stored takes 6.9, building the plan from its kernels (what each
writes, their order) 5.4, the passes' forming and taking 4.1, and the maps
recomputed where read 1.2; the RK4's trials recompute no fold, and at 32
GPT layers the 5 of 57 that do take 1.0 of 9.3. Deciding storage and
building the plan only where a trial changes them is what stands, and so
does deciding, now the largest phase: 7.6 of the 13.6 s at 32 GPT layers
and 2.15 of 7.16 at 32 RK4 steps, linear in the layers, 99% of it each
kernel finding its options (claims, estimates, variants) though kernels
alike already share one decision. Finding them once for each kernel alike,
its view numbered by first appearance so that no implementation can tell
two alike kernels apart, is specified and not built.

Two compiled programs called at once from two tasks, which a search behind
a program's first result makes the ordinary case, shared the cuBLAS
extension's one handle and one 32 MiB workspace per context. A product's
call sets the handle's stream and workspace and then multiplies: with the
tasks on two threads, the first of 100 rounds failed on GB10 and on RTX
6000 Ada in every run, cuBLAS's internal error on that call and on every
call after it on the handle, each program called alone included (on one
thread, none). A handle per task alone turns those loud errors into
silently wrong results: 780 of 4,000 results wrong, in 99 of 100 rounds
(GB10, 64 by 64 products summing 131,072, which write 1.5 MB of the
workspace), the products then running beside each other on the one
workspace; with a workspace per product, none. So the handle per task must
not land without the workspace. Graphs recorded of products lease the
shared workspace and constants, and CUDA.jl orders their launches one
after the other: none wrong. cuDNN's attention shared one handle per
context the same way, 33 of 1,000 results wrong in 26 of 50 rounds,
silently; its workspace being each launch's own already, it now executes
on its task's handle, CUDA.jl's, and none of 1,000 are wrong in 50 rounds.
The test runner's workers have one thread, on which two tasks never call at
once, so these run in a child process with three
(`test/fixtures/at_once.jl`); a two-task test inside a worker passes on the
shared handles. What closes cuBLAS's is a handle per task and a workspace
per executable, landed together, which the tests held as two gaps until
then and now pass. An implementation is given what one executable's
launches share as they build (`Tempest.Compiler.Implement.resources`), made
through the target as the buffers are, once per executable and once per
decider for the claims it times; cuBLAS's is the workspace and the
constants one and zero, and a call sets its task's handle to the task's
stream and then to its executable's workspace. Two programs' products at
once from three threads, 100 rounds of 20 calls each, 64 by 64 summing
16,384 (1.4 MB of the workspace written) and 131,072 (1.6 MB), uncaptured
and captured: none of 4,000 wrong in any of the four, on GB10 and on the
RTX 6000 Ada; the graphs of two programs lease no memory in common, and 64
programs prepared and let go, and a decider, hold no workspace once
collected. The host's side of a product's launch is unchanged within the
noise (on GB10, alternating, 6.42 and 6.50 μs before and 6.57 and 6.46
after; the task's handle 23 ns). The cost is memory: 32 MiB per executable
with a cuBLAS product where there was one per context, and 32 MiB for a
preparing decider. A cold GPT preparation (the corpus's small shape)
measured on the device's pool: its decider holds 125.1 MiB once it has
decided, as on main, where the context's workspace is made while deciding
and here the decider's; with the executable built beside it, 203.3 MiB
against 171.3, the executable's own workspace. Letting go of the decider
found what held it past its end: a graph captured to time a claim and
released was kept in CUDAExt's table of what graphs launch until the graph
was collected and the table next cleaned, which the release had already
used up the occasion for. A program that never captures again kept its last
timed claims' launches, and with them the decider's trial buffers, 3 of 3
in a probe on main, cleaning the table freeing none; a released graph is
now dropped from the table, and none are held. A captured executable let go
still keeps its buffers until the next graph is captured, on main and here:
the table is cleaned only as it is written.

**The jury is transformer-shaped.** Every number that has decided a trial
came from products, reductions, softmax in its spellings, norms and
attention (`test/bench/guard_rails.jl`, `test/fixtures/compositions.jl`,
`test/bench/gpt.jl`): what a transformer step spends its time on, and what
every compiler benchmarks, but a narrow jury for an algebra. Challengers:
programs outside the transformer, each a test of one part of the algebra.
A cumulative sum or a linear recurrence, which is a scan; a segment sum or
a message pass over an index array, which is indirection in formulas; and a
convolution written as array code, where im2col must become a formula and
not a copy. Judged by whether each runs with no package loaded, by its
kernels against the floor and against a hand-written kernel, and by the
entries it opens. Standing: the scan and the recurrence run on the CPU and
on cuTile, and a segment sum is a sum by key, on cuTile where its key moves
only by gathers (both decided below); a convolution through NNlib does not
trace, since `conv` writes into a typed buffer, and none written as array
code has been judged. The eager slice (`test/bench/eager_slice.jl`): 113 of
its 136 cases trace and agree, Base's 59 of 67.

**A neural ODE's training step: a hundred RK4 steps of a small MLP, the
loop unrolled at trace.** The vector field du/dt = dec(tanh(enc(u))), 32
dimensions through 256 hidden, batch 128, integrated over [0, 1] by 100
fixed RK4 steps written as a plain `for` loop; a mean squared error on the
final state; RAdam at 1e-3. The forward is 400 evaluations of the field,
each two products of 32 by 256 at batch 128, in a chain where every stage
reads the last: kernels too small to fill the device, so the step is
launches and latency, and the trace is 400 times one field's. What stands
trial: the trace and planning of a long unrolled program
(`src/Compiler/program/`, `src/Compiler/plan/planner.jl`), the store or
recompute partition across 400 stage states (`partition.jl`), and the
cover's kernels for products this small (`src/Compiler/cover/`). The
chain group for small batched products, probed and not built, is the
nearest answer Tempest has. Challengers, from PathMNISTReactant.jl at
7684532 (`benchmarks/julia/bench_node.jl`, `benchmarks/pytorch/bench_node.py`,
`benchmarks/results/*_node.jsonl`), on an RTX A6000 in TF32: Reactant
0.2.290 through ReactantNitro, the same plain loop unrolled at trace into one
XLA program, 10.8 ms a step and 99 s to the first; torch.compile
`reduce-overhead` 16.1 ms and 6.4 s, `max-autotune` 21.2 ms and 505 s; eager
PyTorch 127 ms and 0.4 s. Judged on the GB10, with both challengers run
there by their own scripts, by the step's time and the seconds to the first
step, cold and warm, with the losses agreeing with the same code run
eagerly.

The baseline (`test/bench/node.jl`, TF32, 4 November), by RK4 steps: at 1,
36 kernels, prepared in 17.4 s, 0.103 ms a step; at 10, 361 kernels, 69.0
s, 0.995 ms; at 30, the preparation had not finished after ten minutes, at
4.6 GB resident. Losses agree with the interpreter's to 4e-6 and the first
moments to 3e-4 relative at both. A step is about 36 kernels and 0.1 ms
per RK4 step, launches of a few microseconds each; the preparation grows
faster than the program, and the 100 steps of the challengers' program do
not prepare. Planning alone, its kernels cached: 18.4 s at 10 steps (1301
values), 99.2 s at 20 (2511), near the 2.5th power of the program; its
time is the planner's grouping (`group_values`, `try_add!`) and the
trials it prices (`trial`, `src/Compiler/cover/cover.jl`). At 20 the
preparation also compiled kernels for 180 s that the 10 steps' cache did
not hold: the step 1/K is a constant of every kernel, so each K is a
program of its own, while within one the steps share theirs (253 kernels
from 33 keys at 7). Counted, planning
alone at 1, 5, 10 and 20 steps (210, 696, 1301, 2511 values): the cover
plans the whole graph once per trial, 40, 104, 159 and 269 times, linear in
the program; each plan calls `try_add!` 202, 2466, 8625 and 31974 times,
quadratic in it, 3.4% of them adding at 20; and each call costs more as
the program grows, 0.22 ms at 10 and 0.42 ms at 20. The plans' summed
time, across threads, is 297 s at 10 and 3645 s at 20, near the fourth
power; the threads hide part of it in the wall time.

Taken since, 5 November: planning alone 2.1 s at 10 steps, 4.6 s at 20 and
11.6 s at 40, against 18.4 and 99.2 s. A group's loads are found once per
plan and its dependencies from the latest value pending (`plan/context.jl`); the
cover's trials are classed by the groups they change and planned side by
side (`cover.jl`); and a gradient summed over a value's uses accumulates
beside each product that adds to it, cuBLAS adding the product to the sum in
one call, the sum taking its slot in place (`storage.jl`,
`ext/cuBLASExt.jl`). At 100 steps, the challengers' program: 3602 kernels,
prepared in 193 s cold and 184 s warm, of which planning is 148 s; 10.9 ms a
step, against Reactant's 10.8 on the A6000. Open: planning still grows
faster than the program, by the greedy loop's taken trials, 15 at 100 steps,
whose dependencies walk every kernel fixed, 0.22 s each at the first and
0.68 s at the last, and by their pricing. The challengers are not yet run on
the GB10.

**Loops are unrolled: rolled from the tape and stepped, adopted.** A `for`
loop traced as its iterations, a program as long as its trip count, planned
and compiled whole: at 100 RK4 steps, 3602 kernels and 148 s of planning
(above). The rolled loop existed, entered only by
`accumulate(f, eachslice(x); init)`: a `Loop` of the program
(`src/Compiler/program/program.jl`), lowered by batching its body over an
axis of steps with one carried definition, planned as a recurrence group,
differentiated as the loop reversed, and on cuTile one kernel sweeping the
steps with the state on chip, else the floor. A plain loop never reached
it. Challenger: StructuredReactant (AntonOresten/Reactant.jl, branch
`structured`), which rolls structured `for` and `while` loops from Julia's
IR through IRStructurizer, without a macro. Taken instead, two pieces and
no new IR. The tape is rolled where it repeats (`src/Compiler/program/roll.jl`):
the last window whose nodes each read what the node a period back read, or
the node a period after that, a period holding a fold, at least three
periods, is rewritten in place as the loop `accumulate` traces, its first
step read from where the loop starts; inside `withgradient` before the
reverse pass, so its derivative is the loop's, and as the trace ends. The
RK4 loop's tape is periodic up to renaming, 39 nodes a step, the weights
its captures and the state its one carry. And a recurrence no one kernel
holds runs a step at a time (`src/Compiler/runtime/stepped.jl`): its
cycle restricted to one step, the step axis left out of each value and
read of it, is a graph planned and built as any program, its kernels
launched once per step and pointed at the step's slices of the
recurrence's storage, a step's position an argument where the step reads
it; captured, a graph of the steps' launches. Before, the accumulate form
of the bench ran its two recurrences on the floor, 102 ms a step at 10 RK4
steps.

On the GB10, `test/bench/node.jl` unchanged, TF32: at 10 RK4 steps, 24
kernels, 19.6 s to the first result, 0.644 ms a step, against 1.06 ms
unrolled; at 100, 25 kernels, 18.9 s and 6.80 ms, against 194 s and 10.9
ms unrolled, Reactant's 99 s and 10.8 ms and torch.compile's 6.4 s and
16.1 ms on the A6000; at 1000, 19.7 s and 68.8 ms, which unrolled did not
prepare. Losses agree with the interpreter's to 4e-6 and the first
moments to 3e-4 at each. A step forward is 12 kernels and one back 12,
against about 36 unrolled: the weights' gradients are products batched
over the steps. A window of maps alone is not rolled: unrolled it fuses
into one kernel (a calibration's chain of eight maps, a polynomial's
products). Open: a step's own launches are not
fewer than an unrolled step's; a step reading a value along the steps
other than by its last axis, or the step's position other than as itself,
runs on the floor.

The challengers on the GB10, 5 November, TF32, 100 RK4 steps, each alone on
the device: torch 2.14.0+cu130 by its own script (`bench_node.py` at
7684532), eager 50.4 ms a step and 0.43 s to the first, `default` 27.2 ms
and 56 s, `reduce-overhead` 12.1 ms and 48 s, `max-autotune` 10.9 ms and
254 s, without its product autotuning ("not enough SMs" on 48). Reactant
0.2.290 by its own script too (`bench_node.jl` at 7684532), through
ReactantNitro 0.1.6, the first release with the `compile_options` the script
passes (its manifest, naming 0.1.5, re-resolved; no version it pinned
moved): its loop unrolled at trace, as the script has it, 8.53 ms a step and
48.7 s to the first, 10.3 ms at the highest precision; rolled with
`@trace for`, as the Reactant end-to-end post writes it, 9.81 ms and 42.9 s.
Tempest, 6.80 ms and 18.9 s, is the fastest of them at both.

By batch, each stack by its own script with only the batch changed: at
1024, Tempest 16.2 ms a step, torch `reduce-overhead` 17.1, Reactant 18.6;
at 8192, Tempest 113, torch 91.9, Reactant 115. Profiled at 8192, neither
device idles: Tempest's 47 ms after the loops were the parameters'
gradients, each a sum over the steps of every evaluation's part, read back
from the 11 GB a step the backward stored, at the device's bandwidth, where
torch adds each evaluation's part as its backward runs, its operands in the
cache. Taken, 6 November: a sum over a recurrence's steps of what they
compute may ride them (`riding`, `src/Compiler/plan/recurrences.jl`), a fold
of the recurrence's group, as a fold along a scan's axes rides its sweep;
each step's plan adds its part to the sum in the sum's storage, each
product into it as it writes (`step_graph`, `src/Compiler/runtime/stepped.jl`),
and the backward stores nothing only the sums read. Whether a sum rides is
the cover's choice (`Ride`, `src/Compiler/cover/cover.jl`): a stepped run is
priced by its step's plan, and measured whole, under a record name of its
own, where the floor's time of a recurrence as one serial kernel had been
read back as its. At 128 the four sums are measured and stay after the
loops, 6.80 ms; at 1024, 16.0 ms; at 8192 the weights' two and the
encoder's bias ride, 79.0 ms, prepared in 25.5 s to the first result. The
losses and first moments agree with the interpreter's as before.

Checked against main on other loops, 6 November, TF32, 100 steps, 32
inputs through 256 hidden: a GRU at batch 64, 4.62 ms against 4.58, its
nine sums measured or estimated and left after the steps; at 1024, 23.3 ms
against 24.2, one riding; chunked linear attention's gradient, cuTile's
one-kernel recurrences both ways, 0.234 ms against 0.230 at 32 chunks, 1.080
against 1.078 at 128. An LSTM did not plan on main: its hidden state and
cell, each step reading both, were recurrences of their own, kernels
depending on each other in a cycle. Taken: states whose cycles read each
other's are one recurrence's (`recurrences`, `src/Compiler/plan/recurrences.jl`),
their group's step computing its values in the graph's order, a value a
state is carried from that the step reads stored, and the stepped run, the
floor and cuTile's one-kernel recurrence each carrying every state. The LSTM
at batch 64, 5.53 ms a step; at 1024, 32.6 ms; its gradients agree with the
interpreter's to 1e-3. Found on the way: the cover had offered to join a
value of a recurrence's step into a kernel that read it, which then wrote
it too, two kernels owning it; taken where measurements of a microsecond
fell its way, the neural ODE's decoder weights had a gradient 39% off in one
preparation in three. No join takes such a value now.

A loop selecting each step's slice, `xs[:, :, t]`, as most write a
recurrent network, was unrolled: each step's selection differs from the
next at one index, and each use of an array the loop captures records a
copy of it, so no window repeated; at 100 steps the LSTM written so did not
prepare in 30 minutes. Taken, 6 November (`src/Compiler/program/roll.jl`):
selections one index apart from step to step are the loop's scanned arrays,
as `accumulate` over slices has them, the range of the array the steps
select taken once where they do not select it all, the loop reversed where
the index falls; constants of one value are alike, computed once before the
loop; and two states starting from one value, `h, c = h0, h0`, are told
apart by what the next step reads, where they had been one argument of the
step and its result wrong. The LSTM as a plain loop, 17.8 s to the first
result and 5.53 ms a step at batch 64, 19.9 s and 33.2 ms at 1024, as the
`accumulate` form. A selection stepping by more than one stays unrolled, as
does a loop of maps alone.

Checked by loops drawn at random, 6 and 7 November: one to three states
and a number, weights passed or captured, an input's slices forward, back
or by two, states assigned in turn or at once, read after the loop or
collected at each step, a second loop after the first; each run eagerly
in Float64, traced, interpreted and prepared on the CPU, its gradient
against central differences and prepared again with sums made to ride and
folds recomputed at random, and on CUDA in Float32 (a checker outside the
repository). Of some twenty-one thousand drawn, fifteen faults, each taken
with a test of its own. Wrong results: a state no cycle of states runs
through read where nothing had been written (on main too, through
`accumulate`), now its next value one step back; crossing states, one
whose next value only the other's reads, left without what it is carried
from (main's forward 9% off); ties of a maximum sharing its cotangent,
NaN on CUDA where a recomputed sum differed in its last bit, now the
element `findmax` finds, as Zygote's rule gives it; a step's cuTile
kernels built on the first step's slices and pointed at the others, less
aligned than cuTile had compiled them for, a gradient 1.25 off at three
hidden units; and a product in a cuTile recurrence's step whose factor
the step computes, its padded lanes a 0/0 the weight's zeros multiplied,
NaN. Refusals of programs that plan: kernels in a cycle, from states
crossing or swapping, a join, or a sum offered to ride steps it could not
(one summing a batch as long as the steps, one reading a second
recurrence); two states given one value; a name a fold's launch never
bound. Planning exponential in a term's paths, built, shown, hashed and
asked what it mentions along each, and folds proposed for recomputing
found along each path to them: forty maps each reading the last twice
had taken hours, an unrolled loop's gradient over ten minutes, two
unrolled loops' over eleven. And the cover's trials writing one
dictionary from several threads. Four of the fifteen were walks of the
graph through operands, which leave out what a state is carried from.
Refused by name, as before: a scan in a recurrence's cycle, 7% of the
loops drawn; states of two shapes in one recurrence, 2%. Unrolled still:
a loop whose step slices an input it does not read. Preparing the LSTM
below takes 1.40 s warm against main's 1.30, its first 1 to 2 s longer.
Its first state learned as a vector and repeated over the batch had left
the recurrence to the floor's kernel stepping through it, 1.76 s a
training step; now spread where the steps start, 5.6 ms.

Checked by programs drawn at random as well (test/fuzz/run.jl), seed 10's
gradient did not prepare on CUDA: cuTile's estimate priced a kernel's read
of a sum by key as that kernel's own sum, by the elements it combines along
the sum's reduced axes taken as axes of the reader's domain, where the sum
was another kernel's, as a gather's adjoint is wherever the rest of the
gathered array's gradient is added to it. A reader over fewer axes than the
sum reduces threw, and one over as many was priced for other elements than
it reads; such a read is now priced as any read is, by what its loads
reach. Of seeds 1 to 1000, forward and gradient, 185 programs read a sum
by key from another kernel, and so does the corpus's GPT step. Six of
them threw (seeds 10, 228, 268, 326 and 389) and now plan. On the GB10's
rates no decision of the rest changes; the estimates of 135 of their
kernels move, by 0.57 to 1.41 times. Timed alone, the 52 that read the sum
where it lies come nearer their times, 1.46 times off to 1.43 on the
geometric mean, and the 7 that read it through a gather go further, 1.60
to 1.88: a gathered read is priced as its whole buffer, as any gathered
value is, where the wrong price had been smaller. Open, for the estimate:
the reach of a gather, which would need the index's extent; 916-grad's
third kernel reads `%29[%2[i1] - 1, i2]`, priced at 10.27 μs (7.35
before) against 4.11 timed. Open, before and after: seeds 347, 399 and 494
do not plan on CUDA, cuTile's emitter asked for a view's spec where it has
none.

The LSTM's training step against torch 2.14.0 on the GB10, 7 November,
TF32, 32 inputs through 256 hidden, 100 steps, each alone on the device
and captured as a graph: at batch 64, cuDNN's `nn.LSTM` 2.05 ms, the
cell as a loop under `torch.compile` 3.42, the loop eager 6.14, Tempest
5.53; at 1024, 24.8, 20.4, 32.1 and 33.0. Tempest runs a step as about
ten launches, a split product forward and seven kernels back.

**Sizes are constants.** Every extent is fixed at trace (`IR.Domain`,
`src/Compiler/IR/expressions.jl`), a kernel's grid and tiles are chosen for
it, and a program for a new size is a new preparation. Decode escapes it, its
cache of a fixed extent and its length held on the device, which bounds the
sweep (decided below), so Qwen3.5-4B's step prepares once for every token; a
prefill, whose extent is the prompt's length, prepares for each length.
A chunk padded to a fixed extent, its length data and its pad tokens steps
that leave the state as it was, serves every length up to it from one
preparation today (A model written once, below), the model carrying what the
extent would.
Challenger: tinygrad's
symbolic shapes (`Variable` in `tinygrad/shape/shapetracker.py`), a bound
dimension as a runtime argument of one kernel; Futhark's incremental
flattening, several kernel versions chosen by size when the program runs.
Tempest's formulas are quasi-affine in integer variables already; a variable
standing for an extent rather than an index is the smallest step. Judged by
one preparation serving every length in a bound, its time per step against
the kernels prepared for each length, within the noise.

**The schedules are a small set.** A product is claimed under four output
tiles per class, then the depths and the split of the best, and the cover
measures them in stages; every other sketch's parameters are constants
measured on one device (`ext/cuTileExt/cuTileExt.jl`, the constants;
`contract.jl`, `product_tiles` and `product_variants`), and the neighbours
of those constants found nothing on this device (the schedules trial).
Challenger:
TVM's MetaSchedule (`python/tvm/meta_schedule`), where sketch rules
generate the space, random annotation fills the parameters, an evolutionary
search with a learned cost model picks, and a database keeps every result;
tinygrad's BEAM is its smallest form, already taken for products. Judged on
the GPT step and the guard rails, with the cold cost bounded as the staged
search bounds it, and on a second device when there is one, where the
constants are a guess and a search is not.

**A model written once.** Qwen3.5-4B's decode and prefill are run today by
Vallmo (`~/dev/jool/Vallmo`): 1,883 lines of cuTile kernels written by hand
(`src/kernels`, speculative decoding's apart) under a model that calls them.
On trial is whether the model is array code and nothing else: one
`forward(weights, states, tokens, positions, length)` in Base's arrays, a
decode step its case of one token and a prefill its case of T, run eagerly as
it is, compiled by Tempest and traced by Reactant, from one file that names
none of them. What of Tempest stands trial with it: `compile`, which after
each call copies what the program wrote back into the caller's arrays
(`src/Compiler/runtime/compile.jl`), the whole cache a token, where `prepare`
with `replaces` and `replay!` beneath it feed results back with no copy and
are not exported; and a prefill's extent (Sizes are constants, above). A
`push!` of a traced array into a vector of arrays of another element type
converts it, as Base's does (`TracedArray{T, N}(x)`, `tracing.jl`). Challengers: Vallmo's captured step and its eager prefill,
stated most plainly at its commit `16b2df6` (a model of 336 lines, an engine
of 167, kernels of 1,307); the same file under Reactant; the same file eager.
Judged by tokens equal to Vallmo's on its tiny checkpoint (`test/tiny.jl`) and
on the 4B in BFloat16; a captured decode step within 1.1 times Vallmo's at
every length of the cache (the decode entries below hold 39.3 to 43.6 ms
against 39.7 to 44.9, by a script no repository kept); a prefill against
Vallmo's; and the lines that remain against those removed. Recorded 10
November, on the CPU target at toy sizes
(`~/dev/jool/challengers/vallmo-2026-11-10`, the model 268 lines): 59 of 60
programs trace, plan and agree with Float64, in Float32 and in BFloat16, among
them a whole model's decode step, its prefill, a suffix prefill over a
restored state, eight tokens generated greedily and a decode loop of `replay!`
alone; a chunk of 8 padded positions serves lengths 1 to 8 from one
preparation; the one refused was the `push!`, which converts now. None of it has run on a device at
the model's shapes since the script was lost.

**Block-scaled quantized products.** A weight in MXFP8, MXFP4 or NVFP4 is
narrow elements with a scale to each block of 32 or 16 along the axis a
product sums, its value `T(e[i]) * T(s[fld1(i, k)])` (Microscaling's
`BlockScaledArray`, over Microfloats' types and BitPacking's layouts, in
`~/dev/jool`). Nothing in Tempest names any of it, and on trial is whether
anything need: the meaning stays array code in those packages and Tempest's
part is general. What stands trial: an array argument is one input whatever
wraps it (`is_input`, `src/Compiler/program/tracing.jl`), so a
`BlockScaledArray` or a packed `NarrowArray` is made dense on the host at each
call and the program never reads its scales; a traced number converts to
thirteen types (`tracing.jl`), so narrowing to a microfloat is an opaque map
inside a broadcast and overflows the stack outside one; no cast of bits
exists; an index that divides by a constant is not split into an axis of
blocks, so no library's template, whose addresses are affine
(`src/Compiler/cover/library.jl`), matches a block's scale; and the element
types cuTile and cuBLAS take (`TILE_TYPES`, `ext/cuTileExt/cuTileExt.jl`;
`FACTOR_TYPES`, `ext/cuBLASExt.jl`) hold no narrow type, so such a kernel runs
on the floor. Challengers: Vallmo's product that dequantizes in registers
(`src/kernels/gemv.jl`); cuBLASLt's block-scaled products over the 128×4
layout of scales; XLA's scaled dot as Reactant reaches it by a named operation
(the fork in `~/dev/jool/Reactant`, branch `microscaling`: three formats run
right, found by hand, and a quantized weight against BFloat16 activations
never reaches it); MLIR's `arith.scaling_extf`, block scaling as a conversion
broadcast, the nearest to Tempest's way. Judged by a program in the packages'
own types, taken as an argument, planning one kernel a product with its bytes
left packed; that kernel within 1.1 times Vallmo's on the GB10; and no name
for the product in Tempest. Recorded 10 November, on the CPU target at toy
sizes (`~/dev/jool/challengers/microscaling-2026-11-10`): written with
elements and scales as two arrays (a reshape to blocks, `repeat`,
`Float32.(w)` of a Microfloats type, Microscaling's `copy`), MXFP4, MXFP8 and
NVFP4's two levels each plan one kernel that reads the scale at `k ÷ 32` and
agree with Float64; a table indexed by code agrees, its index stored first;
and two faults stand, neither yet a test: the planner's `BoundsError`
(`atom_range`) on `getindex.(Ref(s), k .÷ 32 .+ 1, m)`, and the stack
overflow.

**Whole programs against their challengers.** The corpus (`test/corpus`): a
neural ODE's training step, an LSTM's, a GPT's, and a heat equation's
gradient, each written as a user writes it and run against its challengers
by one command (`run.jl`), cold and warm, timed as a captured graph and
checked against the step run in Float64, one row of a record per run
(`summarize.jl` prints the table). Judged by two gates: a step past 1.1
times its best challenger's, or a cold preparation past 60 s, opens a trial.
On the GB10, 8 November, TF32 where there are products, SM clocks 2400 to
2550 MHz as read: the neural ODE at batch 128, 6.6 to 7.3 ms a step over
four runs and 45 to 48 s cold, against Reactant's 9.9 to 11.0 ms and torch
`reduce-overhead`'s 11.2 (`max-autotune`, compiled from nothing, fails
capturing its graph, an internal cuBLAS error, twice). The LSTM at batch 64,
5.73 ms and 71 s cold, against cuDNN's 2.00 and the cell compiled by torch
3.40; at 1024, 32.8 ms and 92 s, against 24.6 and 20.7 (torch takes no
optimizer step). test/bench/gpt.jl's step, 1.19 ms and 70 s cold, against
the same model under `torch.compile` captured as a graph, 1.31 ms; at GPT-2
124M's geometry, batch 4, 878 ms and 224 s cold, against 264 ms, and since
the GPT-2 step's entry 190.7 ms and 188 s cold, against 264.6, then 189.9 ms
and 186 s, against 264.1. The heat
equation on a 256×256 grid, its steps `accumulate`d so that they roll: 1.89
s at 100 steps and 18.9 s at 1000, against 0.34 and 3.6 ms for a kernel a
step in CUDA.jl, both recurrences of the gradient on the floor; written as a
plain loop it unrolls, 200 kernels and 0.66 ms at 100 steps. Taken: a
recurrence whose step reads its state shifted runs a step at a time (where
cuTile's one kernel, on a grid of up to 2048 positions, had read the state
before writing it, 6% off), and a plain loop whose step selects from the
step before rolls; the heat gradient, a cuTile kernel a step forward and two
back, 1.27 ms at 100 steps and 13.1 ms at 1000 against the hand loop's 0.333
and 3.61 at 2411 MHz, 3.8 and 3.6 times, the forward alone 0.63 ms at 100,
its kernels 2.4 to 3.7 μs each against the hand loop's 1.7. Taken since: the
backward's step recomputes the maps its stencil reads at each neighbour, one
kernel a step; a step's map measures the floor beside its claim, which beats
cuTile's gather at each neighbour; and a state nothing outside its
recurrence reads is not copied after the steps: 0.563 ms at 100 steps and
5.53 at 1000 against the hand loop's 0.344 and 3.67 at 2405 MHz, 1.6 and 1.5
times, what is left each step's state written to a slice of its own, the
hand loop so written 0.494 ms. Agreement with
Float64 is 3e-4 to 1e-3 where the products are TF32, and 2e-6 for the heat
equation in Float32; the GPT's 8e-2 is a weight Adam's first step moves by
its rate times the sign of a gradient near zero, and at GPT-2's geometry,
whose Float64 gradient the interpreter holds past the host's memory, one
layer at batch 1 stands for it, 6e-2 so and its loss to 1e-6.

**The GPT-2 step.** The corpus's GPT at GPT-2 124M's geometry (12 layers,
768 wide, 12 heads of 64, FFN 3072, T 1024, vocabulary 50257, batch 4, TF32,
Adam in place) took 878 ms against `torch.compile`'s 264, where the small
shape is at parity. Profiled by kernel (`TEMPEST_CORPUS_PROFILE`, each kernel timed
alone), 648 ms were attention's backward, and 636 of them two kernels a
layer, 40 and 13 ms, recomputing the log-sum-exp of the scores, an array a
thousandth of their size, in a sweep that read the scores again. The cover
had chosen that by estimate, 3.5 ms against 5.9 for storing it, when both
kernels had already been measured; it now measures a comparison wherever the
record refutes an estimate on either side (`compare`,
`src/Compiler/cover/choose.jl`). And the backward summed the probabilities'
gradient over the keys, as large as the scores, where FlashAttention-2 sums
the output's gradient against the output over the features: the exchange of
the two sums (`exchanged_sums`, `src/Compiler/plan/rewrites.jl`) matched axes
only where a factor reads one alone, and the GPT's values are read from the
projection they reshape, at `i1 + 64·i2`; it now pairs axes read among
others at their multiples. On the GB10 in one session, TF32, the step timed
as a captured graph, the kernels timed alone summed by part, the roofline
the larger of the bytes a part's kernels read once and write at 230 GB/s and
its flops at cuBLAS's 43 TFLOP/s:

| part | ms, before | ms, after | roofline fraction | implementation |
|---|---|---|---|---|
| attention, backward | 648.5 | 54.7 | 0.10 fused, 0.88 as planned | cuTile |
| attention, forward | 21.0 | 42.0 | 0.09 fused, 1.06 as planned | cuTile |
| FFN products and SwiGLU, backward | 49.0 | 47.7 | 0.68 | cuBLAS, cuTile with the SwiGLU's gradient |
| head and cross-entropy, forward and backward | 40.6 | 40.8 | 0.54 | cuBLAS, cuTile online softmax |
| FFN products, forward | 20.6 | 20.6 | 0.79 | cuTile (gate and up as one), cuBLAS |
| Adam, 111 kernels | 18.8 | 19.2 | 1.00 | cuTile |
| QKV and output products, backward | 14.3 | 14.2 | 0.90 | cuBLAS, cuTile |
| QKV and output products, forward | 6.2 | 6.2 | 1.03 | cuBLAS |
| RMSNorm, forward and backward | 9.4 | 9.5 | 1.0 to 1.7 | cuTile |
| rotary positions, forward and backward | 3.6 | 3.7 | 1.4 | cuTile |
| embedding's scatter with the last norm's backward | 1.1 | 1.1 | 1.4 | cuTile |
| step, kernels summed / captured graph | 833 / 874 | 260 / 278 | | |

By the corpus in the same session, SM clocks 2480 to 2550 MHz: 275 ms a
step, 538 kernels, against `torch.compile`'s 262 captured as a graph and 264
without, and torch eager's 356; checked by one layer at batch 1 with the
full vocabulary and length against Float64, the loss to 1.0e-6 and the
worst output 6.0e-2, a weight Adam's first step moves by its rate times the
sign of a gradient near zero, as at the small shape.

Then attention was planned as the eager form, the scores and the
probabilities stored by the forward (201 MB a layer each) and the scores'
gradient by the backward, 97 ms of the step against under 10 for its flops
at TF32. The cover never proposed recomputing them: `recomputable_folds`
(`src/Compiler/cover/proposals.jl`) counted the folds `prepare` keeps stored,
those reading what a result replaces, among the folds recomputed when it
asked whether a fold would be recomputed inside another's sweep; the
projections read the weights Adam replaces in place, so the scores, reading
the projections, and their gradient, reading the output's, were taken for
nested and left out. A fold kept stored now nests nothing. Proposed, the
cover takes both by measurement, one at a time: at one layer, batch 4 and a
vocabulary of 256, the kernels that differ 8.1 ms against 2.9 for the
scores, then 2.6 against 1.1 for their gradient, the forward one kernel and
the backward of the three gradients another, the layer's step 19.1 to 12.4
ms. Recomputing the sum of the output's gradient against the output beside
them was measured too, 1.25 against 1.39, and not taken. By the corpus in one
session, SM clocks 2420 to 2510 MHz:

| | before | after |
|---|---|---|
| step, captured graph | 277.9 ms, 538 kernels | 190.7 ms, 490 kernels |
| attention forward, kernels summed | 41.8 ms | 3.8 ms |
| attention backward, kernels summed | 53.2 ms | 13.6 ms |
| preparation, cold / warm | 235 / 57 s | 188 / 67 s |
| `torch.compile` captured as a graph / without / eager | | 264.6 / 266.2 / 358.2 ms |

The step now runs at 0.72 of `torch.compile`'s; checked by its layer at batch
1 against Float64 as before, the loss to 1.5e-6 and the worst output 6.0e-2,
the same weight's sign. The small shape, whose update replaces its weights
too, recomputes its scores as well: 1.18 to 1.10 ms a step, 178 to 158
kernels, and cold preparation 78 to 100 s, its agreement unchanged.

Then the parts left were taken in turn. By the corpus in one session, SM
clocks 2463 to 2483 MHz, each kernel timed alone and summed by part as
above, the roofline fraction the after's:

| part | ms, before | ms, after | roofline fraction | implementation |
|---|---|---|---|---|
| head's softmax, the cross-entropy and its gradient | 12.4 | 9.5 | 0.75 | cuTile online softmax, steps of 8192 rows |
| head's three products | 28.3 | 28.0 | 0.79 | cuBLAS |
| FFN products and SwiGLU, backward | 49.6 | 49.4 | 0.66 | cuBLAS, cuTile with the SwiGLU's gradient |
| FFN products, forward | 20.9 | 20.8 | 0.78 | cuTile (gate and up as one), cuBLAS |
| attention, backward | 13.7 | 13.6 | 0.37 fused | cuTile |
| attention, forward | 3.8 | 3.7 | 0.81 | cuTile |
| QKV and output products, backward | 15.3 | 14.4 | 0.89 | cuBLAS, cuTile |
| QKV and output products, forward | 6.3 | 6.2 | 1.03 | cuBLAS |
| Adam, 111 kernels | 19.1 | 19.2 | 1.00 | cuTile |
| RMSNorm, rotary, embedding's scatter | 15.2 | 15.4 | 1.0 to 1.7 | cuTile |
| step, kernels summed / captured graph | 185.0 / 194.8 | 180.4 / 189.9 | | |

Taken: a streamed column whose kernel sweeps it again after its folds, as
the cross-entropy's gradient sweeps the logits after their log-sum-exp,
offers steps of 4096, 8192 and 16384 rows beside its 1024 for measurement
(`column_variants`, `ext/cuTileExt/cuTileExt.jl`): blocks holding longer
steps are fewer at once, and the second sweep finds in L2 what the first
read. Over GPT-2's 50257 logits and 4096 columns, alone: 12.4 ms stepping
1024 rows, 11.7 stepping 4096, 9.1 stepping 8192, 9.0 stepping 16384 and
10.3 stepping 32768, against 7.2 for one read and one write; tileiras's
occupancy of one or two, and blocks taking 22 to 86 columns in turn, moved
it by less than a millisecond. The cover takes 8192, 8.8 ms (9.5 timed
alone in the step, written over the logits it reads; see below). The step
takes 189.9 ms against 194.8, `torch.compile`'s 264.1 captured as a graph,
265.1 without and eager 356.5, so 0.72 of its time; 490 kernels against
466, the count moving between preparations where the cover's measured
comparisons fall within the noise, as the attention step at 2048 positions
does too (four kernels on fresh records, once three); cold preparation 186
s, warm 69. Checked by its layer at batch 1 against Float64, the loss to
1.5e-6 and the worst output 6.0e-2, the same weight's sign. The attention
benches beside it on fresh records (BFloat16, causal, 64 features, 8 heads,
batch 2), before and after: the forward 41.3 and 41.2 μs at 1024 positions
and 124.5 and 123.7 at 2048, the step at 2048 596.4 and 601.9; the small
shape 1.09 and 1.11 ms a step, 158 and 162 kernels, its agreement
unchanged.

Measured and not taken. The FFN's backward product fused with the SwiGLU's
gradient (the down projection's gradient, 3072 by 4096 summing 768) takes
1.47 ms on the cover's trial buffers and 1.54 in a layer's step, against
1.65 for cuBLAS's product, 0.54, and the gradient's maps in one pointwise
kernel, 1.11: the estimate's choice, 1.30 against 1.56, stands. The 1.83 the
profile times it at in the full step is the kernel written over the
matrices it reads, timed alone with the cache cleared: a product adding a
stored matrix, 768 by 4096 summing 768, takes 233 μs so and 187 written
apart, and 198 against 206 replayed in a captured graph, so the profile
overstates a kernel written in place and the step does not pay it. The
cover's own trial of cuBLAS's product there leaves the maps as two kernels,
2.0 ms, which as one would still lose. Attention's backward at GPT-2's shape
keeps 128 keys by 64 features, 32 queries a step, its transposed folds
added, 1.13 ms a layer: the search measured 17 options of it (occupancy
two, eight worker warps, products 64 deep inside, 16 queries a step), and
64 keys a block, forced, take 1.46 to 2.58 ms.

What remains open, named. The attention backward is a third of its fused
roofline, 13.6 ms against 5.1, under every tile, depth, occupancy and worker
count the generator offers. The head's three products take 28.0 ms against
21.1 for the same products with the logits' leading dimension padded from
50257 to 50304, measured on cuBLAS alone (10.0, 9.7 and 8.3 ms against 8.1,
6.6 and 6.4): a column of 50257 Float32 is no multiple of 16 bytes, which
cuBLAS's aligned kernels ask of a leading dimension, and Tempest stores
every value densely (`src/Compiler/IR/layout.jl`), so a value stored with a
padded leading dimension is a layout it does not have. The head's softmax
is 9.5 ms against 7.2 for one read and one write. With the FFN's and the
projections', 119 ms of the 180 the kernels sum to are products cuBLAS or
the generator's product kernels run at 0.66 to 0.9 of their roofline. Adam's 111
kernels are at the memory's rate already. A product that recomputes a fold
from a stored array in its steps runs hundreds of times its flops' time:
the value gradient's product reading the probabilities through the
log-sum-exp over keys recomputed from stored scores, 1024 keys by 1024
queries over a layer's 48 heads and batches, takes 471 ms under its first
claim (tile 64×128 by 32) and 185 ms under its best, against 7 ms for
reading the scores once per block of keys, as each recomputes the same sums;
the scores' tile loaded as stored and summed along its first axis instead of
permuted, by hand, 419 ms, so the load's order is not the cause. The cover
measured it in a layer's step (12.8 ms there, against 1.5 estimated) and
refused it. A fold over the keys reading the probabilities so runs at the
memory's rate under each tile tried, 1.65 to 1.90 ms, kept tiles of 1 and 64
alike. Cold preparation is 186 s,
past the gate of a minute; that is the trial of the time to first execution.

**The laws are hand-written matchers.** Every algorithm Tempest knows as a
rewrite lives in `src/Compiler/plan/rewrites.jl` (758 lines) as a function
that walks the index graph and tests its shape by hand: the exchanged sums
(`exchanged_sums`, `matched_axes`, `paired_axes`, `aligned`), the relative
folds (`relative_sums`, `applications`, `dependent`, `admissible`), the
statistics kept beside them (`kept_statistics`) and the folds that observe a
function of others (`compose_observes`). The file's header says laws are
data; the two that are (`Exponential`, `Square`) are the φ of one mechanism,
and everything that decides where the mechanism applies is code. The cost
showed in one day of the corpus and the checker: `dependent` took a
one-operand negation for a difference and crashed; `matched_axes` saw no
axis read through a reshape, so FA2's identity never fired on a GPT and its
attention backward cost 648 of 878 ms; and the same shape of code elsewhere
had the same holes, the stepped run refusing a wrapped index
(`src/Compiler/runtime/stepped.jl`) and cuTile's recurrence claiming a state
read at a neighbour's position. Each was closed by widening one condition,
and nothing says which conditions a law has, so the checker draws programs
over the whole algebra and finds a law's hole only by chance, and the plan
verifier cannot ask whether an application preserved the program. What
stands trial: the laws' matching and preconditions as code, in the
functions named. The challenger: a pattern layer, where a law is a pattern
over definitions and operands with variables, its preconditions as data
beside it (which operands are reads of which values, which axes a sum
reduces, which element types are exact), its rewrite a function of the
match, and one matcher applies every law. Read in tinygrad's `UPat` and
`PatternMatcher` (`tinygrad/uop/ops.py` in the local clone), where every
pass from the tensor graph to the kernel is a list of patterns over one
node type, and in Exo's scheduling rewrites, each checked for equivalence as
it is applied; the strategy review's search report weighs both and the
e-graph systems that take the same shape to its end. Judged on three
counts. First, the four laws written as patterns match what they match
today: every test in `test/plan/rewrites.jl` and every plan of the
rewritten-sums fixtures unchanged in structure, and the corpus's GPT and
attention steps within noise. Second, each law is checked alone: the checker
draws programs that meet its pattern and fails the law, not the program,
when the rewritten graph disagrees with the program run in Float64, and each
application is checked where it is made, the rewritten value against the
value it replaces on the program's own inputs, which is the equivalence
check the verifier lacks. Third, the layer costs no lines: `rewrites.jl`
with the matcher is no longer than it is now, and today's four holes read
as preconditions in the laws' own data. The trial closes against the
challenger if a law cannot be written as a pattern without a general
predicate over the whole graph, that law named; it closes for the code if
the patterns match less, or match the same at more lines.

**A second device: the RTX 6000 Ada faults where tileiras reads a tile
transposed four deep.** On an RTX 6000 Ada (sm_89, 142 multiprocessors, 96
MiB of L2, discrete memory; driver 580.178.04 with CUDA_Driver_jll's
forward-compatible 13.3, runtime 13.4, tileiras 13.4.92 from CUDA_Tile_jll
13.4.1, Tile IR bytecode 13.4, cuTile.jl at tree 491f899), main's suite
meets `CUDA_ERROR_MISALIGNED_ADDRESS` in two tests:
test/runtime/storage.jl's training step on CUDA, in a product of 16×32
summing 4 whose left factor is read transposed, and test/cuda/cover.jl's
products grouped with a product by a constant, in its streamed kernel. The
error is sticky: what runs after it on the worker fails too, 16 errors from
the two. The storage test's fault is cuTile's alone. A Float32 tile loaded
(4, 16), permuted to (16, 4), and its product with a (4, 32) tile
accumulated by `muladd`, in cuTile.jl without Tempest, faults; with 2, 8 or
16 along the sum, or the left tile loaded (16, 4) as it lies, it agrees. The
Tile IR is the same at 4 and 8 but for its shapes. The PTX tileiras emits
for sm_89 at 4 reads the transposed tile with `ld.global.v2.b32` at
`[%rd2+4]` and `[%rd2+12]`, where %rd2 is a multiple of 32 bytes past a
128-byte-aligned base: four overlapping 8-byte loads, two of them at
addresses 4 past a multiple of 8, where four predicated loads of one element
each are meant. The grouped kernel also permutes the tiles it loads; its PTX
is not yet read. Both tests plan alike on the Ada's rates and the GB10's,
fault on the Ada under either, and run clean on the GB10 (sm_121) under the
Ada's rates, as does the cuTile.jl form: it is the device's code generation,
not a plan. Nothing in Tempest works around it. The scripts, the PTX and
Tile IR at 4 and 8, and the outputs are in ~/dev/jool/challengers/sm89/:
`cutile-only.jl` the minimal form, `cutile-ptx.jl` its Tile IR and PTX,
`probe.jl` Tempest's smallest program that meets it, `(w, d) ->
(permutedims(w) * d,)` with w 4×16 and d 4×32 on the generator alone,
`misaligned.jl` the two tests' programs launch by launch. Closes when a
tileiras's PTX for the minimal form has no misaligned load and the two tests
pass on the Ada; whether to report it upstream is open.

Not on the docket: the kernel-author question, how a kernel no implementation
claims is written by hand in a package. Pallas (a kernel in the same language
on explicit refs, transparent to tracing, with its own derivative) and Exo
(plain code scheduled by named, checked transformations) are the two designs
to read when it reopens, which is when such a kernel is first wanted.

## Decided

**BFloat16 attention's forward: matrices from one product to the next,
adopted.** On trial was the product kernel's tiles laid out in the full rank
of the tiling and reshaped to matrices around each product
(`ext/cuTileExt/emit.jl`): reshaped to three dimensions and straight back,
the scores leaving the first product or the weights entering the second
cost cuTile.jl's own attention kernel (`examples/fmha.jl`) 6 to 13 μs,
tileiras moving them through shared memory, and the generated kernel did
both at every step, in 2224 instructions against that kernel's 992. The
challenger was that kernel, FlashAttention-2's algorithm written by hand in
the same tile language, whose scores and weights stay matrices. A product
kernel's term spanning two of its tile axes of more than one lane is now also
computed as a matrix over them, and an exponential fold whose shifted part is
one steps in that plane, its weights entering the next product as they are;
the tiles over the whole tiling beside them, unused, compile to nothing
(`plane`, `in_plane`, `matrix_of`; `step_plane` in `ext/cuTileExt/build.jl`).
The softmax's weights are powers of two with denormals flushed, and products
that recompute a fold are searched at an occupancy of two, as that kernel
asks. Judged on the GB10 at 64 features, 8 heads, batch 2, causal, on fresh
records, the answers agreeing as before:

| μs | 1024 positions | 2048 positions |
|---|---|---|
| forward, before | 67.4 | 220.4 |
| forward, after | 36.2 | 123.7 |
| cuTile's own kernel, 64×64 tile | 37.4 | 121.5 |
| FlashAttention-2 (SDPA flash) | 41 | 128 |
| training step, before / after | 217 / 191 | 735 / 597 |

What is not judged, named: at 1024 positions the staged search's pick
differs between records, 36.2 or 41.1 μs, two schedules within one percent
of each other timed from memory; a 64×64 tile with products 64 deep inside
the scores at an occupancy of two runs in 33.5 μs, and the search, one
variant at a time, does not reach it; TF32 moves little (92.6 μs against
97.2 at 1024); and the backward's transposed sweep still reshapes the query
gradient's matrix to add it at each step.

**Attention's training step: the code wins.** On trial was the step's
cost against the forward's, near six forwards at 1024 positions (TF32,
causal, 64 features, 8 heads, batch 2), judged against FlashAttention-2's,
held near two and a half. The step now runs in three kernels: the forward
writing the log-sum-exp beside the output, the loss, and one backward
kernel whose transposed sweep adds the query gradient atomically. The
challengers ran on the GB10 on the same problem with torch 2.14:
`scaled_dot_product_attention` under each backend that takes it, and
FlexAttention compiled with a causal block mask; the loss the sum of the
output's squares, the gradients those of the queries, keys and values,
replayed as CUDA graphs, TF32 allowed:

| μs, forward / step | 1024 positions | 2048 positions |
|---|---|---|
| Tempest, BFloat16 | 50 / 197 | 166 / 637 |
| FlashAttention-2 (SDPA flash), BFloat16 | 41 / 310 | 128 / 807 |
| FlexAttention, BFloat16 | 43 / 293 | 141 / 841 |
| cuDNN (SDPA), BFloat16 | 60 / 341 | 163 / 938 |
| Tempest, TF32 | 95 / 410 | 330 / 1408 |
| FlexAttention, TF32 | 91 / 746 | 305 / 2511 |
| SDPA memory-efficient, TF32 | 219 / 1098 | 737 / 3690 |

On this device FlashAttention-2's step costs 6.3 to 7.5 of its forwards, not
two and a half, and Tempest's step is the fastest in both precisions: the
ratio was the wrong judge, and the step's time is the right one. What is not
judged, named: the BFloat16 step's gradients against the interpreter, which
these runs did not compare; the forward, which trailed, had its own trial, above.

**A kernel is keyed without Tempest's code and checked by what it uses of it,
adopted.** On trial was the kernel cache's key holding the identity of every
package a kernel names, Tempest's a hash of all its sources while it is
developed: an edit anywhere in Tempest, a comment included, made every
kernel compile again. The challenger, Julia's own answer for its code: a
cuTile kernel's key names Tempest without its identity (`content_key`'s
`checked`, `src/Compiler/runtime/cache.jl`), and the kernel is stored with
what its compilation used of Tempest's code, the methods its inference
reached through the edges it keeps, the globals they read and the types they
were specialized on, each with its lowered code or value
(`dependencies`). A kernel loads while each is unchanged (`current`, checked
once per session per record). A failure has no inference to read that from
and is kept under the whole key (`cached_failure`). Judged on the GB10 by
preparation's wall time over the guard rails and the coverage compositions
(61 programs), the tuning record warm and the store large enough to keep
every kernel:

| | warm | after a comment edit to the planner |
|---|---|---|
| before | 16.5 s, 0 compiled | 39.8 s, 87 of 110 compiled |
| after | 15.2 s, 0 compiled | 17.7 s, 0 compiled |

An edit to a fold's `observe` compiles again the one kernel whose inference
reached it. What is not judged, named: Julia 1.11, whose inference keeps no
such edges, keeps the whole key (`EDGES`); and the store's bound, 512 MiB by
default and shared by every process and checkout on the machine, which the
guard rails and the test suite together outgrow, so that warm runs compile
again what it removed whatever the key.

**What a generator lowers is asked before it claims: a trial kernel per
function, adopted.** On trial was learning it from builds that fail:
cuTile's `expressible` (`ext/cuTileExt/cuTileExt.jl`) checked element types
and folds but not whether a scalar function lowers, so a group applying
`exp10` or `log10` was claimed, failed to build, and was planned again
(`prepare`, `build_group`). The challenger, the answer declared: the cover
asks (`Implement.lowers`), and the generator answers by compiling a kernel
that applies the function once to tiles of its argument types, through the
kernel cache, so that it compiles once per machine and a failure loads as
one (`ext/cuTileExt/lowers.jl`); a map stored apart is proposed on the
answer rather than on a failure (`unlowered`, `src/Compiler/cover/cover.jl`).
A preparation that builds nothing compiles no trial (`compiling`). Judged on
the GB10 by the builds that fail and by preparation's wall time, cold (no
kernel cache, an empty record) and warm, over the guard rails and the
coverage compositions (61 programs), and over the four programs of
`test/cuda/failures.jl`:

| | builds that failed | cold | warm |
|---|---|---|---|
| guard rails and compositions, before | 0 | 114.2 s | 15.6 s |
| after | 0 | 111.7 s | 15.3 s |
| programs applying `exp10` or `log10`, before | 4 | 15.8 s | 2.9 s |
| after | 0 | 13.2 s | 2.0 s |

The trials are 25 of the first set's 323 kernels cold, and cost nothing
measurable: a trial compiles in 0.1 to 0.9 s and loads from the cache in 1 to
40 ms. What is not judged, named: a function the trial lowers and the
group's kernel does not, as at a literal other than the trial's two or at a
shape its one-dimensional tiles do not take, still fails at build and is
planned around.

**Scores past the length the device holds are computed: a tile every mask
rules out is skipped, adopted.** On trial was Qwen3.5-4B's whole decoding
step over a cache of 32768 positions, prepared once and run at lengths 512
to 32256 (prompts the first tokens of Julia's `HISTORY.md`), against
Vallmo's captured step over the same cache. The sum over keys skipped the
steps past the length; the product into the scores did not, since the
positions are among its outputs, and the mask comes after it, one buffer
the layers share: every token read the whole key cache, 64 MiB per layer.
The challengers, FlexAttention's block mask, which skips the blocks of both
products a mask rules out, and FlashAttention's decode, which bounds both
by `cache_seqlens`; Vallmo's split kernel (`mha_decode_split_fwd`) the bar.
Now a block whose tile the masks of everything its kernel stores rule out
skips its sweep and stores their literal (`ruled_out`,
`ext/cuTileExt/mask.jl`), the tile found from the masks' formulas as a
sweep's steps are. Only sums are skipped, a sum over nothing being the
finite statistic a mask's literal takes a fold to be. The sweep is guarded,
not counted: with a count the block reads, the step's kernels over the
cache took 10.1 ms at the whole cache against 4.7.

| ms per token, captured, GB10 | 512 | 4096 | 16384 | 32256 |
|---|---|---|---|---|
| before | 41.37 | 41.57 | 42.40 | 43.72 |
| after | 39.27 | 39.78 | 41.54 | 43.59 |
| Vallmo | 39.69 | 43.96 | 44.36 | 44.87 |

The kernels over the cache take 0.58 ms at 512 positions, against 2.37
before. Tokens: 63 of 63 agree with Vallmo's at 512, 16384 and 32256; at
4096 the fifth is a three-way near-tie, logits 16.354, 16.303 and 16.264,
which Tempest takes the other way, its logits the same bit for bit with no
step skipped. Over a cache of 4096, 38.79 to 38.47 ms, 63 of 63 tokens; the
GPT step 1.230 and 1.247 to 1.255 and 1.243 ms; the guard rails 8992 and
8987 to 8994 and 8995 μs; GPT preparation the same once the changed kernels
are compiled (42.3 and 41.6 to 41.9 and 42.0 s). Vallmo loses from 4096
positions on by its design: its eight splits are of the cache's extent, so
one block sweeps the first 4096 keys. What is not judged, named: a split
sum's partial states and a resident kernel skip nothing; and the skip needs
the masked scores stored, as a mask the layers share makes them, where a
mask written as a formula may let the plan store the scores bare and mask
them where they are read, and then every tile is computed.

**A write at positions the device holds is refused: writes by key, in
place, adopted.** On trial was a write at an array of integers, refused
where the array was traced, and every write a select over its whole array:
an append to a decode step's cache rewrote the cache, at Qwen3.5-4B's
attention over 32768 positions 64 MiB read and 64 MiB written per cache,
per layer, per token, for 2 KiB of new contents
(`src/Compiler/program/updates.jl`). The challengers, XLA's in-place
dynamic-update-slice and scatter, which write only the update's elements
into an operand whose buffer the result shares
(`CanEmitFusedDynamicUpdateSliceInPlaceForGpu`,
`xla/service/gpu/ir_emission_utils.cc`), with vLLM's `reshape_and_cache`
and FlashAttention's `flash_attn_with_kvcache` as the hand-written bars. A
write at index arrays, a traced integer among them, is now a fold by key
that overwrites (`Overwrite`, `src/Compiler/IR/folds.jl`), starting from the old
contents, and a scatter a sum by key in the kernel computing its elements;
a fold by key whose contents nothing reads after it takes their storage, so
the floor and cuTile skip the copy, as XLA's scatter emitter does
(`overwrites`, `src/Compiler/plan/storage.jl`). Judged by Vallmo's append
written as a traced write, k-norm and partial RoPE at the position the
device holds and keys and values into caches the step replaces, one
sequence, captured on the GB10: 4.11 μs per step at 4096 positions and 4.19
at 32768, 4 KiB stored beyond the inputs, against `knorm_rope_append!`'s
4.14 and 4.11. Where several elements land at one position, one of them
does, unspecified which, as in JAX; the reference and the CPU's floor write
in order and the last lands, as Base does, and those are checked on the CPU
(`REPEATED_WRITES`, `test/fixtures/programs.jl`). The derivative masks the
old contents' gradient where the write lands and gathers the values' there,
and where positions may repeat keeps the gradient of the element that
landed, as JAX does. On cuTile, a sum by key whose key moves only along
gathered axes adds a box at a time by a TMA reduction (`atomic_store_add`):
the RMSNorm backward over gathered rows 150 to 14.5 μs, the GPT step 2.127
to 2.09 ms. What is not judged, named: a write at ranges the host holds is
still a mask over its whole array; the cover prices the copy an in-place
fold skips, so the whole 4B step's two cache writes per attention layer are
estimated at 443 and 1760 μs where their timings bound them below 70 and
14; and sums by key are not bitwise reproducible.

**Scans are refused: Base's scans trace as ordered folds, adopted.** On
trial was `cumsum`, `cumprod` and `accumulate` throwing
(`src/Compiler/program/tracing.jl`): the fold algebra had no ordered fold.
The challenger, Futhark's `scanomap`, the fused form of a map into a scan,
and its segmented scans (`futhark/src/Futhark/IR/SOACS`, and `SegScan` in
the GPU backend). A scan is now a fold that keeps the axes it folds along
(`Accumulate`, `src/Compiler/program/program.jl`), as Base's `accumulate`
defines it, forward or reversed, inclusive or exclusive, and a linear
recurrence is `accumulate` over pairs by their combine. A reduction's
combine must associate and commute; a scan's need only associate, which the
recurrence's does without commuting. A scan's group is one kernel sweeping
the scanned axes, the map before it its operand's term and the maps after
it reading the scan where the sweep has it, as `scanomap` fuses them; on
cuTile a block per tile of the kept axes sweeps the scanned ones
(`scan_kernel`, `ext/cuTileExt/build.jl`). A cumulative sum's derivative is
a cumulative sum the other way, any other scan's a linear recurrence the
other way from its combine's partials. Judged by a cumulative sum and a
linear recurrence agreeing with eager Julia, exactly in the interpreter in
Float64 and Float32, and by their time against a hand loop. On the CPU's
floor over 2^20 elements: the cumulative sum 568 μs against the hand loop's
569, the recurrence 1378 against 1389, a map, a scan and a map fused 5661
against 6343. On the GB10 against the floor: a cumulative sum down
4096×1024 147 μs against 1828, across 1024×4096 205 against 1599, the
recurrence down 4096×256 16 against 1015. What is not judged, named: a
vector's scan runs in one block, 2^20 elements in 655 μs against the
floor's 34 ms, where Futhark splits it by chunks; Base's `accumulate` by a
function of the user's, which claims no law, and an exclusive scan, which
only derivatives make, stay on the floor; and no hand-written device scan
(CUB's) was timed.

**Claims are timed from the cache: timed from memory instead, adopted.** On
trial was how the cover times a claim (`time_launch`, `ext/CUDAExt.jl`): a
captured graph of a millisecond of its launches, replayed back to back, so
that a launch whose data fit the 24 MiB cache was timed from it. A step
reads its weights once, from memory. The challengers, nvbench's cold
measurement and Triton's `do_bench` (`triton/testing.py`), clear the cache
before each sample by writing a buffer and time each launch with events.
Written, the buffer leaves the cache dirty, and the launch is timed writing
it back; read through instead, it leaves it clean. Each sample is now the
launch's graph replayed alone between events after a buffer twice the
cache's size is read through, twenty enqueued before one wait, the minimum
taken, less what timing one replay alone adds over its place among a
graph's launches (an empty kernel's 2.5 μs on the GB10, against 0.43 in a
graph). A decoding step's output projection on the GB10 at 2411 MHz,
4096×2560 BFloat16 weights, 21 MB, against the truth: the same launch over
copies of the weights, rotated so that every launch reads memory:

| output projection | truth | from the cache | cleared by writing | cleared by reading |
|---|---|---|---|---|
| one token, step 32 | 106.1 μs | 69.9 μs | 174.8 μs | 106.9 μs |
| one token, step 128 | 92.6 μs | 25.6 μs | 159.6 μs | 91.1 μs |
| 16 tokens, depth 32 | 105.8 μs | 27.5 μs | 173.6 μs | 101.5 μs |
| 16 tokens, depth 128 | 93.5 μs | 22.8 μs | 161.3 μs | 93.3 μs |

From the cache the 16-token projections chose depth 32, at 107 μs in use;
they now choose 128, at 98.5, against Vallmo's 96.8. The other judged, each
from an empty record, before and after:

| case | from the cache | from memory |
|---|---|---|
| Qwen3.5-4B's decode projections per token, one token | 36.90 ms | 36.95 ms |
| the same, 16 tokens | 37.89 ms | 37.80 ms |
| decode attention, trial 2's, lengths 200 to 32768 | 15.5 to 1478 μs | 15.3 to 1478 μs |
| guard rails, 36 cases | 8998 μs | 8915 μs |
| GPT step at TF32, device time, four runs | 1.250 to 1.261 ms | 1.240 to 1.257 ms |
| its preparation from an empty record, kernels compiled | 23.1 s | 20.3 s |
| claims it times, and the seconds timing them | 190, 1.44 s | 143, 1.30 s |

Six guard rails are 9 to 26% faster (Float16 column sums, row sums, softmax
down columns at 1024×4096), one 6% slower (Float16 `sum(abs2)`), timed by a
benchmark that replays from the cache itself. Over ten repeats a timing
spreads ±0.1 μs at 4.3 μs and ±0.27 at 93, and takes 5 to 7 ms. Graphs of
a clearing and a launch less graphs of the clearings alone, which keep the
launch's place in a graph without the correction, spread ±2.9 μs at 94 and
cost 70 ms a timing: not adopted. The first preparation after the change
compiled the claims its search newly reached, 10 s more on the GPT step;
the table's is the next. The rates are timed alike, so the estimates price
from memory too: the projection above is estimated at 83 to 101 μs, where
it was priced at the cache's rate. The record's and the rates' files are
named for the timing (`record_path`, `src/Compiler/cover/record.jl`), so
that times taken from the cache are not compared with these. Timed from
memory, one estimate that decides misses by more than the close-call
factor: cuBLAS's 256×512·384 product at full precision, 9.1 μs against
14.0 (11.8 from the cache), a launch of one wave whose loads and products
add where the cost model overlaps them (a gap in `test/decisions.jl`).
What is not judged, named: a value a launch writes stays in the cache, its
write to memory left to the launches after it, and one that reads it just
after finds it there; timed here, the write is free and the read is from
memory, the one charged in place of the other.

**A length the device holds does not bound a sweep: an integer the block
reads bounds it, and is adopted.** On trial was the generator sweeping every
key of a cache whose keys past each sequence's length are masked,
`ifelse(position <= len[b], score, -Inf32)`: the planner found masks only in
comparisons of index formulas, and this one compares a position with a load.
The challenger, FlashAttention's `flash_attn_with_kvcache`, bounds each
sequence's key loop by its `cache_seqlens`. Here an integer read is a
formula, a gather known to lie in its type's range (`held_formula`,
`src/Compiler/IR/masks.jl`), so the comparison is a mask as a causal one is
(`masked`, `src/Compiler/plan/masks.jl`); where every variable of its index
lies on a kept tile axis of one lane, the generator loads it once per block
into the bound on the sweep (`block_read`, `ext/cuTileExt/mask.jl`). Judged
on the GB10 at 2411 MHz, at Qwen3.5-4B's attention: 256 features, 16 query
heads over 4 key heads, one sequence, BFloat16, a 32768-position cache in
Vallmo's layout, one preparation bound to each length, against Vallmo's
`decode_attention`, the tuning record empty:

| length | streaming plan | the cover's plan | `decode_attention` |
|---|---|---|---|
| 200 | 14.8 μs | 15.5 μs | 52.9 μs |
| 2048 | 94.5 μs | 94.5 μs | 329 μs |
| 16384 | 729 μs | 741 μs | 653 μs |
| 32768 | 1478 μs | 1478 μs | 722 μs |

Before, the two plans took 1500 and 1325 μs at every length. The answers
agree with plain Julia over the keys up to each length to the rounding of
the scores to BFloat16, 2e-3 relative, and the masked-attention table holds:
the causal forward 146 and 463 μs at 1024 and 2048 positions, the step 2.89
and 10.77 ms. Corrected after the close: the table first held only through
the tuning record. The cover timed its claims on buffers of ones, so the
length bounded every timed sweep to one key; on an empty record the 16-key
tile measured best there and took 25.1 and 3243 μs at 200 and 32768
positions, twice the 64-key tile's time. The entries that hid it were
measured before the length bounded anything, under keys that did not change
with the generator; the cover's plan, storing the scores (333 to 1581 μs),
was chosen on those entries, where it was the faster. Now a bound compared
with positions is timed at the far end of them, and an index spread over
its axis (`held_integers`, `src/Compiler/cover/choose.jl`): the cover
chooses the streaming plan itself, and its tile. What is not judged, named.
With a key head per query head, where the scores of a query are one row, the
generator claims no streaming plan, and it runs on the floor (a gap in
`test/runtime.jl`). At the whole cache Vallmo is twice as fast: it splits
the keys across eight blocks per key head and combines them by the
log-sum-exp (`mha_decode_split_fwd`), where the streaming kernel runs on 16
blocks; its split is of the cache's extent, not of the length, so one block
holds every live key below 4096, which is where it loses. A loop whose count
the kernel reads costs the sweep over stored scores 27% at the whole cache
(1288 against 1015 μs, as much with a count computed from the block's index
as with one loaded); a constant count with the dead steps guarded off
recovers it there (957 μs) but costs the streaming plan 30 μs at 200
positions, so the read count stays.

**The schedule search's cold cost: the staged search wins and is adopted.**
On trial was the search over a product's tiles, every tile of every product
compiled and timed once per device and shape (`ext/cuTileExt/contract.jl`,
`product_claims`; `src/Compiler/cover/choose.jl`, `measured_ranks`). The
challenger, the staged form of the same search, which tinygrad's BEAM
(`tinygrad/engine/search.py`) approximates with its beam width: every
output tile at the class's default depth first, then the variants of the
tile that measured best, its other depths and its sum split the other way,
and so on until the best has no variant left. Now an implementation may
stage its claims (`Tempest.Compiler.Implement.variants`), and the cover searches
every group's options in stages, all groups in each: a stage's kernels
compile side by side and are timed one at a time, and every measured option
within the noise of the best is expanded (`search!`, `expand!`); the record
replays the search without the device. Judged on the GB10, each
preparation in a new process with an empty kernel cache and tuning
directory, the same plans agreeing with the interpreter:

| case | before | staged |
|---|---|---|
| GPT step, cold preparation, full precision | 63.0 s | 55.1 s |
| GPT step, cold preparation, tf32 | 65.2 s | 56.7 s |
| GPT step, warm preparation | 19.6 s | 20.2 s |
| prepare.jl's programs, cold | 68.6 s | 63.2 s |
| product claims compiled and timed, GPT step and prepare.jl | 192, 84 | 100, 65 |
| GPT step, device time, two runs | 2.098, 2.102 ms | 2.095, 2.093 ms |
| GPT step, tf32, device time | 1.256 ms | 1.262 ms |
| guard rails, full precision, 36 cases, two runs | 8980, 8948 μs | 8969, 8947 μs |

The claims measured halve and the seconds fall by an eighth: each stage's
kernels compile together, but its candidates are still timed one at a time
and the stages run in sequence. On a 256×4096·4096×256 product at full
precision on cuTile alone, the staged search tried 10 of the 14 tiles and
found the exhaustive search's best, three tiles tied within the noise and
each expanded. What was not judged, named: a product whose best tile lies
off the greedy path, which the step's and the guard rails' times would have
shown.

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
