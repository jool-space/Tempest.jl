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
the generated attention forward pass, and functional updates of the arrays a
program allocates. The first gates measured time against the old Tempest and
held (the 29 forward guard-rail cases at 0.94 of its time). From here the
gates are about what Tempest can express; time is an alarm.

Not yet as designed: the six fronts below, found on 19 October by reading the
code against DESIGN.md and by probing plans. None of them is a kernel being
slow. Each is a decision made in the wrong place, by the wrong rule, or
stated in the wrong vocabulary. test/promises.jl records what each front
owes as `@test_broken`, under a testset naming what closes it.

## Next: the fronts, in order, each closed by its gate

1. **The kernel author's interface.** A kernel is a struct with five trait
   methods, results are integers, buffers are reached by value id, and a verb
   author and a generator author share one protocol. Target: the interface
   docs/src/kernels.md describes. Gate: that page is true, and the toy
   implementations in test/calls.jl are written in its form.
2. **Estimates and measurement.** Product rates are two hand-typed tables, in
   the cuBLAS and cuTile extensions, and an estimate wrong by more than the
   close-call factor decides without a measurement; the generator's product
   estimate was five times cuBLAS's where the two had measured alike. Target:
   rates measured by Tempest on first use of a device and kept in the tuning
   record; measure levels none, close and all; parameter search within a
   budget. Gate: no rate in any extension, and for every claim the coverage
   suite exercises, the estimate is within the close-call factor of its
   measurement or the claim was measured. Cheap, and it fixes the judge the
   other fronts are judged by.
3. **The decision, and compile time.** Rules decide groups and the cover
   prices only what implementations propose, so Promise 3 does not hold; the
   layer has two ways to claim, two to propose, two builds, a floor that is
   only sometimes an option, and building outside the cover. Decisions are
   cached per group instance, each proposal re-plans the whole program, the
   planner is superlinear, and every instance of a kernel is loaded or
   compiled on its own: preparing 32 identical layers took 29 s where 8 took
   0.4. Target: DESIGN.md, *The decision*, decided per canonical group, a
   trial re-planning only the region its proposal touches, a linear planner,
   and one build per distinct group. Gate: the Promise 3 and Promise 6 gaps
   in test/promises.jl close, and one place decides what forms a kernel.
4. **Canonical forms.** Commutative operands, a domain's axis order, a reshape
   to a vector, a transpose after a product, and the two scaling forms of a
   mean all plan differently. Language-level work, independent of the rest.
   Gate: the Promise 1 gaps close.
5. **Forms and layouts.** A call reads dense copies and absorbs only a
   pointwise function aligned with its output. Target: the spec carries a
   layout and a kernel accepts or declines on it; an intermediate is written
   in the layout its consumer accepts rather than copied; and the three forms
   after a result, a scale, an add and a pointwise function, which are
   cuBLAS's coefficients and fused epilogues, a verb kernel's residual, and
   gradient accumulation in place. Rides on fronts 1 and 3. Gate: a product's
   scale and add are coefficients; a verb kernel absorbs the residual after
   it; a permuted argument reaches a kernel that accepts it without a copy.
6. **The language and the docs.** Each a day at most, none blocking: ranges
   in a broadcast become positions and `axis_positions` goes; an Einops
   extension of one method; the language (tracing, the interpreter,
   differentiation, operations, updates) in a module of its own, out of
   Compiler; numerics out of the target and into the program; and the home
   page carrying the contract and the promises as what a user relies on.
   Gate: the attention tests unchanged without `axis_positions`; TF32
   programs agree with the interpreter up to summation order and no
   implementation reads a mode; the home page states the contract and the
   promises.

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
the same per layer however many identical layers a program stacks; the GPT
step matches the old Tempest's and holds up against torch at the same
precision; one place decides what forms a kernel.

## Open questions

- Which shape the decision takes: candidates grown from anchors and covered
  greedily by gain, or a rule-made starting plan with every rule-made choice
  also offered as a proposal and priced. See DESIGN.md, *The decision*.
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
