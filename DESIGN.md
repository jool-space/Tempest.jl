# Tempest design

Written 16 October 2026, and revised the same day after an adversarial review.
This is the design Tempest is built toward, stated on its own terms rather than
as a distance from the current code. It lives with the new core on the `seed`
branch. Tempest's architecture page (`docs/src/architecture.md` on `main`)
describes today's code.

## Purpose

Tempest turns ordinary Julia array code (a loss, a training step, a forward
pass) into a prepared program that runs as fast as the hardware and the best
available kernels allow, and that explains every decision it made. The user
writes what to compute. Tempest decides how: which values are stored, which are
fused, how each kernel is scheduled and which implementation runs it, choosing
among generated tile programs, vendor libraries and the user's own kernels.

It exists because no other system combines three things:

- **Julia functions are first class.** A pointwise function or a fold's
  operator is Julia code. Tempest reads it to plan (its derivative, its cost,
  its canonical form), and the kernel runs the function itself. It is never
  mapped onto a fixed operator set.
- **Custom kernels and custom derivatives compose with fusion.** A named
  operation carries a reference definition, so it fuses like any other code
  when no kernel claims it, and is replaced whole when one does.
- **The whole step is prepared once,** with fixed storage, and replayed as one
  captured graph.

## What makes it a leap

Today a fused kernel exists only when one of several hand-written paths accepts
a group, each path with its own gates, and a new kind of fused kernel means a
new path. Here a kernel exists when a schedule in one language expresses it,
reached from any spelling through one decision. That makes reachable:

- products fused with their neighbours: a norm's scale or a rotary embedding
  as prologue; an activation, a residual or a two-output backward as epilogue;
- attention variants (masks, score modifications, grouped heads, sliding
  windows) as bodies and formulas, not as new kernels;
- convolutions, as contractions whose read formulas shift;
- a user's kernel written as a schedule over index-level bodies, so it fuses
  with its neighbours instead of being opaque;
- derivatives, costs and canonical forms of the user's own scalar functions;
- plans that are deterministic given a record of measurements, explained
  decision by decision, and prepared again in seconds in a new process.

Speed follows from these, and is checked, not hoped for: the speed guard rails
run on every change. The yardstick is torch, eager and compiled, on the same
model at the same precision. Once capture and launch overhead are gone, what
separates two systems is which kernels they use, how they fuse and at what
precision, and that compares like for like.

## Promises

What a user can rely on. Each one is checked by tests.

1. **Layout spelling does not matter.** Programs that differ only in reshape,
   permute, view, slice, broadcast, the placement of transposes, the operand
   order of commutative operations, or the name of a fold (`sum`,
   `reduce(+)`, `mapreduce`) get the same plan. Programs that differ by a
   rewrite the numerics policy permits (reassociation, `sum/n` against
   `sum·(1/n)`, online softmax) get the same plan under that policy. Inside a
   scalar function only operand order and common subexpressions are
   canonical: `x*2` and `x+x` stay different. Where two programs plan
   differently, `explain` names the rewrite that would unify them.
2. **A named call is never worse than its reference.** A claim on a call is
   compared by cost with its reference fused into its neighbours. An
   implementation that claims a call may also claim the same computation
   written out by hand.
3. **One rule decides fusion.** A group becomes one kernel when the estimate
   says it is faster or, for a claim that is measured, when the recorded
   measurement does: never because of which code path found it, and never by
   noise. A measurement overrides the estimate only beyond the device's noise
   margin, and the plan is deterministic given the record.
4. **Every implementation meets the declared numerics policy,** and removing a
   buffer never removes a rounding the program states.
5. **Every decision is visible and can be overridden.** The plan is data.
   `explain` prints each decision with its reason and whether it was estimated
   or measured; tests compare plans; a user pins a value's storage or a group's
   schedule by the value's stable name.
6. **A second preparation takes seconds, in a new process.** Kernels are
   cached by content, not by Julia method, and a hit loads a compiled binary
   without generating code.

## Four levels

### 1. The program

What the user wrote: a typed graph with names and concrete shapes. Its
vocabulary is small.

- `map(f, xs...)`: a Julia callable applied elementwise (see *Scalar
  functions* below).
- `fold(op, x, axes)`: an associative Julia operator with its identity and
  accumulator type. A fold's state may be a tuple, as in the online softmax's
  running maximum, sum and accumulator.
- `contract(a, b, axes)`: a product over named axes. Matrix, batched and
  attention products are all this. Its meaning is a fold of products; it stays
  a primitive because libraries and tensor cores realize it directly.
- `gather` and `scatter` by runtime integer indices.
- `positions`: the index itself, for masks, dropout and embeddings.
- Layout: reshape, permute, view, slice, broadcast, concatenate.
- `call`: a named verb with a reference written in the above, optional
  realizations and an optional derivative rule.

Tracing, differentiation and functional updates work at this level.
Differentiation needs one pullback per primitive; a verb may bring its own. The
backward graph is ordinary program code and is planned together with the
forward. A library may claim a whole named call here, as cuDNN claims
`attention`; a call nobody claims is replaced by its reference.

**Scalar functions.** Tempest lowers a callable's typed IR into a scalar
expression graph of arithmetic, elementary functions, conversions and selects.
The graph gives the derivative, the cost and the canonical form. A kernel runs
the callable itself where its backend compiles it, and the lowered graph
otherwise. Branches whose arms are safe to evaluate either way
become selects, and loops with a constant trip count are unrolled. Three rules
keep that sound:

- Functions that throw on the CPU and trap on the device (`sqrt`, `log`, `^`,
  integer `div` and `rem`) are lowered to total forms: the IEEE operation that
  returns NaN, and divisors guarded by a select. The interpreter uses the same
  forms, so at the program level `sqrt(-1f0)` is NaN, not an error.
- A select stays a select, and its derivative is a select of the arms'
  derivatives. Never `c·a + (1 − c)·b`, which turns `0·∞` in the arm not taken
  into NaN.
- Arms of different types meet through an explicit conversion that `explain`
  shows.

A callable with side effects, an unbounded loop or an arm that cannot be
evaluated speculatively stays opaque: it runs on the floor and needs a user
derivative. The seed starts by tracing callables on scalar stand-ins, as
today's code does, and lowers typed IR after that. Branches reach tile programs
through if-conversion in cuTile's broadcast, upstream. Loops do not need it: for
a tile program the generator emits the lowered graph, with constant-trip loops
unrolled, instead of the callable.

### 2. The index level

Here layout disappears. Every value is an expression over its index variables,
and every read of another value goes through a formula that gives the source
index:

```
q[d, t, h] = qkv[d + D·(h − 1), t]
```

Reshape, permute, slice, view, transpose and broadcast only compose formulas;
they never launch anything. Formulas are quasi-affine (integer sums, products
by constants, and floor division and modulo by sizes) plus one indirect term:
an integer value read at an index, which is what a gather is, so gathers fuse
too. A size may be a bounded symbol (`n ≤ 1024`) even while every size in use
is a constant. The formulas on the inputs come from the layouts of the arrays
passed to `prepare`.

Each value is stored or recomputed inside the kernels that read it, which is
what fusion is.

### 3. Schedules

A schedule says how a group of index-level values becomes one launch, and it is
data:

- the group's domain: index variables with extents;
- per axis: bound to the grid, bound to a tile with an extent, or swept inside
  the kernel, with a bound that may depend on the grid index (a causal mask
  ends its sweep at the diagonal);
- per value: loaded, kept resident, recomputed or stored, and in which phase;
- per fold: its phase, and whether it is split across blocks with a finishing
  launch;
- per contract: its tile extents and whether its contracted axis is swept;
- hints: occupancy, worker warps, CTAs per cluster.

Two examples. RMSNorm over the columns of a 4096 × 4096 matrix, today's
one-kernel schedule:

```
domain   i: 4096 (reduced), j: 4096
axes     j → grid, tile 1;  i → tile 4096, resident
values   x loaded, resident;  x² recomputed, ss = fold(+, i) in phase 1;
         rstd recomputed, y recomputed and stored in phase 2
hints    no occupancy hint at 4096 rows
```

A linear layer with bias and GELU:

```
domain   m: M, n: N, k: K (contracted)
axes     (m, n) → grid with an 8-wide swizzle, tiles 128 × 128;  k → swept, tile 32
values   W, x loaded per k step;  acc = contract over k, resident in Float32;
         b loaded once;  y = gelu(acc + b) recomputed and stored in phase 2
hints    worker warps and CTAs per cluster searched
```

Candidates come from a few sketch rules, each a parameterized shape of schedule:
resident fold, streamed fold, split fold with a finishing pass, contract with a
swept axis and a prologue or epilogue, and contract with an online fold.
Default parameters, such as tile extents and occupancy, come from one
per-device function of the sketch and the tile. The tuning record overrides
them per shape, and a bounded search over them runs for claims that are
measured. The hardware facts that hand-written kernels carry today live in that
function and that record, in one place.

One generator compiles any schedule into a tile program: loads through
formulas, bodies, folds, contracts, sweeps and stores. A sketch only decides
bindings and extents; it contributes no code. One lowering rule is specific: a
fold whose lift is bilinear in the swept tile lowers its combine to a contract.
That rule is how attention's forward pass is a contract scheduled with an online
softmax. Attention's backward pass is, in the seed, a hand-written schedule of
three launches attached to the verb's reverse rule, in the same language;
deriving it from differentiation and planning comes later.

A library call is a schedule with no tile program.

### 4. Kernels

Tile programs, library calls and generic CUDA kernels over concrete buffers, as
a graph with dependencies that capture preserves, so independent branches run
concurrently. Groups with the same schedule and body over many arrays, such as
an optimizer step over every parameter, are one launch over a table of arrays.
Nothing is decided at this level.

## Implementations

Every implementation answers one question in one way: given a group of
index-level values, can you realize it, and how? Each answer is a claim that
states

- the group and its schedule;
- the forms in which it reads each input and writes each output, as a
  predicate on the concrete formula (alignment, contiguity, a unit-stride
  axis, divisibility, writing in place into an input's storage), with a cost
  that may depend on the form, since a form changes which kernel runs;
- its estimated cost as launches with bytes, flops and parallelism, or
  `measure`, for claims whose estimate cannot separate them from their rivals;
- a build function that returns the launches.

Regions, calls, proposals and groups are all claims. The roles:

- **cuTile is the generator.** It claims every group a sketch can schedule,
  with as many schedules as the search allows.
- **cuBLAS and cuDNN are schedules without tile programs.** They claim
  products, products with epilogues and attention, and are chosen when they
  win, usually by measurement.
- **The floor realizes any single value alone:** generic CUDA kernels, with
  cuBLAS for products. It covers what tile programs cannot express, such as
  deterministic scatters and opaque bodies. Because the floor always exists, a
  plan always exists; a claim whose build fails is removed and the cover is
  recomputed.
- **User kernels** arrive through verbs, as claims or as schedules.

An implementation may claim a written-out pattern (attention spelled as two
products around a softmax) as well as the named call. The index level is
canonical, so the match does not depend on spelling.

## The decision procedure

One pass turns the index-level graph into a plan.

1. **Anchors and candidates.** Anchors are folds, contracts, calls, results,
   and values that some reader must load from storage. For each anchor and
   each applicable sketch, grow the largest group the sketch accepts, through
   formulas, pointwise neighbours, and the contracts whose sketch has a prologue
   or epilogue. A group may have several outputs. Candidates are cached by
   canonical numbering, so identical layers share them.
2. **Forms.** Claims that demand a form constrain the layouts of the values
   they read and write. Constraints propagate through formulas, and where two
   demands conflict a copy is inserted and charged. This runs before the cover.
3. **Cover.** Greedy by estimated gain, with a queue ordered by time saved,
   then size, then canonical id. A candidate is accepted when it overlaps no
   accepted group and keeps the partition acyclic, and the neighbours of each
   acceptance are estimated again. What no group takes goes to the floor. The
   order is total and the result deterministic. An exact cover by integer
   programming serves as a test oracle on small graphs.
4. **Estimates and the clock.** The estimate is a roofline with launch
   overhead, a parallelism term and read multiplicity (a streamed schedule
   reads its inputs once per phase), with rates derived from the device.
   Measured claims are compiled in parallel and timed as the minimum of several
   alternating runs, and the times go into the tuning record.
5. **Stored or recomputed.** A value with one reader is recomputed inside it.
   A value with several readers is stored unless every reader's schedule can
   recompute it (it holds the value's operands resident) and recomputing moves
   fewer bytes than storing.
6. **Storage.** Lifetimes over the kernel graph, slot reuse, and donation in
   place where the write formula and the read formula of the aliased value
   agree; staged otherwise.

This is greedy by estimated gain on purpose: XLA and Inductor both settled
there. The leap is in what the candidates are (schedules over formulas,
contracts included) and in what decides (time, with a record), not in the
cover algorithm.

Re-running kernels to save memory, checkpointing, is a different mechanism
from recomputing inside a kernel. It is a cut through the graph, priced by
peak memory over a kernel order. It comes later as its own pass with a memory
budget; until then `checkpoint(f)` marks a region to re-run.

The plan records, for each value, stored (with its layout) or recomputed; for
each kernel, its group, schedule and forms, and whether its choice was
measured; for each buffer, its lifetime and slot.

## Numerics

One policy object, set on the target, read by every implementation and part of
every cache key. It covers product precision (full, TF32, half with Float32
accumulation), accumulator types for folds, the rewrites that may be applied
(reassociation, scaling forms, online softmax), allowed approximations and
determinism. An implementation that cannot meet the policy does not claim.
Removing a buffer never removes a rounding the program states.

## Runtime

A prepared program owns fixed buffers and replays one captured graph per step.
Inputs are bound by copy into owned storage, or by reference when the caller
promises a stable address and layout. Everything that varies between replays is
a device value: scalars, seeds, positions, and the current values of bounded
sizes. A bounded size is served by a few specializations, each padded to its
bucket, with the padding value set per operation (zero for products, −∞ for a
softmax's keys).

Tempest owns two caches on disk. The kernel cache maps a content hash of the
group, schedule, device and policy to a compiled binary and its parameter
layout, and compiles misses in parallel. The tuning record maps the content,
shapes, policy and device of a measured claim to its times; it is local, and
records for known devices can ship with the package.

## Testing

- **A reference interpreter of the program level defines meaning.** Every
  implementation and every plan is tested against it.
- **Generated programs.** Random programs must agree with the interpreter, and
  respellings drawn from the permitted rewrites must produce identical plans.
- **Speed is checked relatively, per device:** the default plan against the
  floor, and estimates against measurements, so a planner change that loses
  time fails a test.

## Lessons taken and left

| From | Take | Leave |
| --- | --- | --- |
| JAX | transformations over a small primitive set; functional state | an opaque backend with custom kernels bolted on |
| PyTorch Inductor | values as index functions; indirect indexing; product templates with epilogues; foreach kernels; autotuning while compiling | a surface of two thousand operators; reduction strategies hard-coded rather than searched |
| tinygrad | layout as index arithmetic; one store-or-recompute decision per value | storing every reduction that a later step broadcasts over |
| Halide, TVM | the schedule as data, separate from the algorithm; sketch rules with a search over their parameters (MetaSchedule) | |
| XLA | whole-program planning, buffer assignment, layout constraints propagated before fusion | fusion heuristics that are hard to inspect or steer |
| Julia packages | Julia functions throughout; extension by dispatch | compile-time blowups; an MLIR boundary that custom kernels and their derivatives cannot cross |

## Not now

Control flow on traced values, mutation, sharding across devices, budgeted
checkpointing, and an MLIR core. An explicit plan leaves room for each later.

## Open questions

- How claims are written: as code over a group first, and as declarative
  patterns once the common cases are known.
- Whether differentiation and planning can derive attention's backward
  schedule (store the log-sum-exp, recompute scores inside the consumer), or
  whether it stays hand-chosen.
- How many schedules to measure per claim, against compile time (1.6 to 4.3 s
  per product configuration on GB10), and how wide each device's noise margin
  is.
- Whether cuDNN reads sliced operands, which decides whether packed QKV needs
  copies.

## How we get there

Not by migrating today's planner. A seed is built beside it.

- **The seed** is a new core implementing the four levels, the claim protocol
  and the decision procedure: the package `TempestSeed` on the orphan branch
  `seed`, checked out as its own worktree beside the main one. It shares no
  files or history with today's code, and loads today's Tempest from the main
  checkout, so old and new run in one session.
- **Today's Tempest is frozen except for fixes,** and serves as the oracle and
  the parts bin. The seed's results and speeds are compared with it, and the
  speed guard rails run on the seed from the start. What today's code
  learned the hard way carries over: the tracing frontend, verbs and fold
  algebra, reverse rules, capture and state donation, cuDNN graph building,
  device rates, and the guard-rail cases. The cuTile schedules are ported as
  schedule records that the one generator compiles, not copied as code; their
  measured thresholds become the per-device defaults. What today's code
  accreted does not carry over: separate paths for calls, proposals, graph
  regions and scalar regions, six request kinds, and decisions about what
  forms a kernel made in five places.
- **Milestones**, each closed by its gate:
  1. The index level with formulas, body lowering, the
     interpreter and generated-program tests, the schedule language with
     sweeps, folds and contracts from the start, and the generator. Today's
     fold schedules are ported as records. Gate: the forward guard-rail cases
     run within the guard rails' limits of today's times, with no special case
     in the generator for any sketch.
  2. Contracts on cuTile with prologues and epilogues, cuBLAS as a
     schedule, forms, the greedy cover, the tuning record, differentiation.
     Gate: the MLP's forward and backward products with bias and GELU pick the
     measured winner for each product and replay it from the record.
  3. Training steps of the MLP, RMSNorm and softmax cases; the kernel cache.
  4. Attention: the forward as a scheduled contract, the backward as a
     hand-written record, cuDNN as a rival. Gate: the GPT step at parity with
     today's Tempest or better, then measured against torch at the same
     precision.
  5. Bounded sizes and decode.
- **Stopping rule.** Stop the seed when a gate fails for a reason the design
  cannot address: the forward guard-rail cases cannot come within 1.10 times
  today's default plan in two consecutive runs; today's fold schedules cannot be
  ported as records without a special case in the generator for some sketch;
  the MLP, RMSNorm and softmax training steps cannot match the interpreter and
  today's Tempest to a relative 1e-5; or the GPT step cannot reach parity. A
  gate not yet met because the work is unfinished is not a reason to stop. On
  stopping, fold what the seed proved (formulas on region inputs, forms, the
  kernel cache, the tuning record) back into today's code step by step.
- **The scorecard:** the spelling battery gives identical plans; generated
  programs agree with the interpreter; the guard rails pass on the seed; the
  GPT step matches or beats today's and holds up against torch at the same
  precision; one place decides what forms a kernel.

When the seed passes milestone 4 it becomes Tempest: the main line moves to it,
and today's code, docs and reports stay in history.
