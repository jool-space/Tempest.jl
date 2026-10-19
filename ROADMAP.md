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

Not yet as designed: the decision. The planner groups values by fixed rules
and the cover prices only the alternatives implementations propose, so the
planner's own choices are never priced and Promise 3 does not hold. The
decision layer also has more paths than its idea needs: two ways to claim, two
ways to propose, two builds, a floor that is only sometimes a claim, and
building outside the cover. See DESIGN.md, *The decision*, for what it should
be and the one choice still open there.

## Next, in order, each closed by its gate

1. The decision, as DESIGN.md states it. Gate: the gaps in test/promises.jl
   close (each `@test_broken` there names what closes it), and one place
   decides what forms a kernel.
2. The kernel author's interface, as docs/src/kernels.md states it. Gate:
   that page is true, and the toy implementations in test/calls.jl are
   written in its form.
3. The language: operations as declared, folds with tuple states with the
   online softmax as one, size-1 axes out of domains, and gathers by runtime
   indices. Gate: the coverage suite passes as tests that fail, and the eager
   slice is measured.
4. Numerics in the program. Gate: TF32 programs agree with the interpreter up
   to summation order, and no implementation reads a mode.
5. The runtime and the GPT step: storage reuse, donation, capture and
   functional updates. Gate: the GPT training step matches the interpreter,
   then reaches parity with the old Tempest.
6. Speed: schedule search measured into the tuning record, cuDNN as a rival, a
   generated attention backward pass. Gate: measured against torch, eager and
   compiled, at the same precision.
7. Bounded sizes and decode.

**Stopping rule.** Stop and report when a gate fails for a reason the design
cannot address: a composition in the coverage suite needs a new kind of node
or value in the core, or a special case for one pattern in the generator; or
the GPT step cannot match the interpreter. A gate not yet met because the work
is unfinished is not a reason to stop.

**The scorecard:** the coverage suite and the spelling battery pass; the eager
slice grows; generated programs agree with the interpreter; the GPT step
matches the old Tempest's and holds up against torch at the same precision;
one place decides what forms a kernel.

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
