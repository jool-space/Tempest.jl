# Tempest roadmap

Where Tempest stands, what comes next, what it learned from, and what is still
open. DESIGN.md is the design itself; this file is everything about the
project that is not the design, so the design can stay short.

## History

The design was written on 16 October 2026 and revised the same day after an
adversarial review, then on 17 October to lead with the contract, to separate
operations from their implementations, and to gate on what Tempest can express
rather than on time. On 19 October it was cut to the design alone: the
decision procedure it described (candidates grown from anchors, a greedy
cover) had not been built, and the code's own procedure had drifted from it;
status, open questions and lessons moved here.

Tempest was rebuilt from scratch as the package Tempest itself, on a branch
that became `main` on 18 October. The new code owns every level, from tracing
to the runtime, and imports nothing from the code it replaced. Where a piece of
the old code proved itself (the fold algebra, the measured thresholds and
workarounds in the cuTile schedules, cuDNN graph building, capture and state
donation, device rates, the guard-rail cases), it was rewritten into the design
deliberately. What the old code accreted was left behind: separate paths for
calls, proposals, graph regions and scalar regions, six request kinds, and
decisions about what forms a kernel made in five places. Comparisons with it,
and with torch, run as separate processes against recorded numbers.

## What makes it a leap

Before, a fused kernel existed only when one of several hand-written paths
accepted a group, each path with its own gates, and a new kind of fused kernel
meant a new path. Here a kernel exists when a schedule in one language
expresses it, reached from any spelling through one decision. That makes
reachable:

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

Speed follows from these. Plans are checked by their structure on every change,
and times serve as an alarm against regressions. Speed claims are measured
against torch, eager and compiled, on the same model at the same precision.
Once capture and launch overhead are gone, what separates two systems is which
kernels they use, how they fuse and at what precision, and that compares like
for like.

## Where it stands

Built: the program and index levels, the cuTile generator, the generic
kernels, a cover with cuBLAS as a rival, the tuning record, the kernel cache,
the generated attention forward pass, functional updates of the arrays a
program allocates, the kernel author's interface for named operations, rates
measured on the device, a planner linear in identical layers, and decisions
made once per canonical group. The first gates measured time against the old
Tempest and held (the 29 forward guard-rail cases at 0.94 of its time). From
here the gates are about what Tempest can express; time is an alarm.

Not yet as designed: what remains of the six fronts below, found on 19
October by reading the code against DESIGN.md and by probing plans, and
worked the same day in the order given. None of them is a kernel being slow.
Each is a decision made in the wrong place, by the wrong rule, or stated in
the wrong vocabulary. test/promises.jl records what each front still owes as
`@test_broken`, under a testset naming what closes it.

## Next: the fronts, in order, each closed by its gate

1. **The kernel author's interface.** Done for named operations: a kernel is
   a function of the target and the call's `Spec`s returning a `Kernel`,
   registered as `register!(Package, operation => function)`, and the toys
   in test/calls.jl are written in that form; docs/src/kernels.md is true and
   its *Not built* list says what is not; the hand-written attention is two
   such kernels, and the older call-claim methods are gone. Remaining: the
   forms beyond a pointwise function (a scale, an add), the parameter search,
   and building a body's tile programs at preparation rather than at the
   first execution, so that a kernel that fails to build is withdrawn. Gate:
   that list empties.
2. **Estimates and measurement.** Done: no extension states a rate. Tempest
   measures the device on first use (launch gap, bandwidth in and out of the
   cache) and fits one line, latency and rate, per package and class of
   products from two reference products the package runs, kept in a rates
   file beside the record; `measure` takes `:none`, `:close` or `:all`; and
   a test checks, for every composition in the coverage suite, that each
   estimate that decides is within the close-call factor of its measurement,
   which it is for every composition at both precisions: a line has a
   latency, a rate, a rate for sums split among blocks and one for folds
   recomputed inside products, each from a reference the package runs, and
   each package moves bytes at a memory of its own. Remaining: the parameter
   search within a budget; and `Work`'s shape, which states a kernel's work
   in the vocabulary of the generator's schedules (recomputed flops, split
   sums), to be restated when the kernel interface takes forms. Gate: the
   estimate test holds as the fronts below move plans.
3. **The decision, and compile time.** The shape is settled: the rules make
   the starting plan, every rule-made choice is also a proposal, and a
   proposal is an edit priced on the groups it changes (DESIGN.md, *The
   decision*). Done: a fixed set's group takes its kind from its folds; the
   planner grows each group from a frontier of its readers and is linear in
   identical layers, with every plan unchanged; a decision is made once per
   canonical group; a kernel is loaded or compiled once per content key in a
   session; proposals alike in identical layers are priced once and taken
   together, so the cover's trials do not grow with the layers; and Tempest
   proposes joining a stored value into the kernel that reads it, one
   proposal per edge between groups, which closes Promise 3's test (74 of
   the suite's 1004 plans drop a kernel, none gains one). A join that
   cannot form a group is rejected before any trial, proposals are classed
   by the groups they reach, and the planner's dependency check looks at
   each value once, with ancestors as masks: the GPT step's cover at 4
   layers went from 65 s to 26 s, at 8 layers from 373 s to 75 s, and one
   plan of it at 8 layers from 0.41 s to 0.18 s. Remaining: the other
   rule-made choices (storing a fold the rules recompute, recomputing a map
   in its readers, declining an epilogue) become proposals too, and the
   planner's own constants go with them; and what still grows with depth,
   that each trial is a whole claimed plan (three plans under the `keep`
   loop and a round of claims on every group, which the recompute-or-store
   proposals remove) and that the classes grow (54 at 4 layers, 74 at 8).
   Gate: one place decides what forms a kernel, and preparing 64 layers
   costs the same per layer as 8. The floor
   stays priced only where nothing else claims: offered on every group, its
   estimate was wrong by up to 467 times (a serial sweep is not a roofline
   quantity), it won 283 groups and 44 of them wrongly, and no fit of bytes
   and flops repairs that. Pricing it needs rates of its own, or a
   measurement whenever it is near the best claim.
4. **Canonical forms.** Done: the operands of a commutative function are in
   one order at the index level; a fold a result reads through a permutation
   is stored in the result's order, so a transposed product costs no kernel;
   a quotient by a power of two is the product by its reciprocal. Remaining:
   `x / n` against `x * (1 / n)` where the reciprocal is inexact, a rewrite
   behind the numerics policy; and a group's domain order, where two
   spellings do the same work at the same addresses but print differently.
   The generators read the domain's order, so canonicalizing it would move
   schedules for no gain in time; the open question below is whether the
   promise is the same plan or the same work. Gate: the Promise 1 gaps close
   or the promise is restated.
5. **Forms and layouts.** A call reads dense copies and absorbs only a
   pointwise function aligned with its output. Target: the spec carries a
   layout and a kernel accepts or declines on it; an intermediate is written
   in the layout its consumer accepts rather than copied; and the three forms
   after a result, a scale, an add and a pointwise function, which are
   cuBLAS's coefficients and fused epilogues, a verb kernel's residual, and
   gradient accumulation in place. Rides on fronts 1 and 3. Gate: a product's
   scale and add are coefficients; a verb kernel absorbs the residual after
   it; a permuted argument reaches a kernel that accepts it without a copy.
   Then a cuTENSOR extension: it contracts any label structure over strided
   operands and takes a scale, an add and a short list of operators, which
   are this front's forms and layouts exactly, so it is the extension that
   proves them, written against `Tempest.IR` and `Tempest.Implementations`
   alone with the interfaces test as its gate. Not before: on today's
   programs it would claim what cuBLAS claims at cuBLAS's speed, a third
   rival to calibrate for no plan that changes.
6. **The language and the docs.** Each a day at most, none blocking: ranges
   in a broadcast become positions and `axis_positions` goes; the language
   (tracing, the interpreter, differentiation, operations, updates) in a
   module of its own, out of Compiler; and numerics out of the target and
   into the program. Done: the home page carries the contract and the
   promises as what a user relies on; an einsum spelled with Einops traces as
   one product, through the one method Einops's backend hook asks for, with
   grouped axes as reshapes and three or more operands as the pairs Einops
   orders. Gate: the attention tests unchanged without
   `axis_positions`; TF32 programs agree with the interpreter up to summation
   order and no implementation reads a mode.

Then, as before:

7. The language as declared: operations, folds with tuple states with the
   online softmax as one, size-1 axes out of domains, and gathers by runtime
   indices. Gate: the coverage suite passes as tests that fail, and the eager
   slice is measured.
8. The runtime and the GPT step: storage reuse, donation, capture and
   functional updates. Gate: the GPT training step matches the interpreter,
   then reaches parity with the old Tempest.
9. Speed: schedule search measured into the tuning record, cuDNN as a rival, a
   generated attention backward pass. Gate: measured against torch, eager and
   compiled, at the same precision.
10. Bounded sizes and decode.

**Stopping rule.** Stop and report when a gate fails for a reason the design
cannot address: a composition in the coverage suite needs a new kind of node
or value in the core, or a special case for one pattern in the generator; or
the GPT step cannot match the interpreter. A gate not yet met because the work
is unfinished is not a reason to stop.

**The scorecard:** the coverage suite and the spelling battery pass; the eager
slice grows; generated programs agree with the interpreter; preparing costs
the same per layer however many identical layers a program stacks (it does
from 4 to 16; from 8 to 64 it grows by half, in the planner); the GPT step
matches the old Tempest's and holds up against torch at the same precision;
one place decides what forms a kernel.

## Open questions

- Whether Promise 1 means the same plan or the same work. Two spellings that
  differ in the order of a kernel's domain (a sum over the transpose against
  the transpose of a sum; a map over a reshaped vector against a reshape of
  the map) do the same work at the same addresses, and `explain` prints them
  differently; test/promises.jl judges both with a helper that compares
  kernels up to the naming of their index points, beside the open gap.
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
- What NNlib's `dropout` means under tracing. NNlib draws from a global random
  number generator; Tempest takes an explicit seed.

## Lessons taken and left

| From | Take | Leave |
| --- | --- | --- |
| JAX | transformations over a small primitive set; functional state | an opaque backend with custom kernels bolted on |
| PyTorch Inductor | values as index functions; indirect indexing; product templates with epilogues; foreach kernels; autotuning while compiling | a surface of two thousand operators; reduction strategies hard-coded rather than searched |
| tinygrad | layout as index arithmetic; one store-or-recompute decision per value | storing every reduction that a later step broadcasts over |
| Halide, TVM | the schedule as data, separate from the algorithm; sketch rules with a search over their parameters (MetaSchedule) | |
| XLA | whole-program planning, buffer assignment, layout constraints propagated before fusion | fusion heuristics that are hard to inspect or steer |
| Julia packages | Julia functions throughout; extension by dispatch | compile-time blowups; an MLIR boundary that custom kernels and their derivatives cannot cross |
