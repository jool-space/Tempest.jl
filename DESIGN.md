# Tempest design

The design Tempest is built toward, stated on its own terms. What people rely
on is the contract and the promises; the sections after them say how Tempest
keeps them, and may change. ROADMAP.md holds where the code stands against
this design, what comes next, the open questions, and what Tempest took from
other systems.

## The contract

Three kinds of people meet in Tempest. Model authors write Julia array code.
Operation authors give operations names and meanings. Kernel authors offer
faster ways to run them. Tempest connects them so that each can work without
knowing the others' code.

- **Tempest owns coverage; implementations own speed.** A program in Tempest's
  primitive language runs with no implementation beyond the generator and the
  generic kernels. Removing an implementation may change how fast a program
  runs, never whether it runs.
- **A new operation changes an operation package. A new kernel changes an
  implementation package. A better optimization changes Tempest.** None of the
  three needs changes in the others, and a model author's code stays the same
  through all of them.
- **What people rely on is the promises below.** Formulas, folds, schedules and
  the planner are how Tempest keeps them today. `explain` shows them, but they
  are not the interface, and they may change.
- **The core stays small.** A new kind of node or value needs a written
  argument that formulas, folds, scalar functions and calls cannot express it.

## Promises

What a user can rely on. Each one is checked by tests.

1. **Layout spelling does not matter.** Programs that differ only in reshape
   (adding or dropping size-1 axes included), permute, view, slice,
   broadcast, the placement of transposes, the operand order of commutative
   operations, or the name of a fold (`sum`, `reduce(+)`, `mapreduce`) get the
   same plan. Programs that differ by a
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
4. **Every implementation computes what the program states,** its types and
   rounding functions included, up to summation order; removing a buffer
   never removes a rounding the program states.
5. **Every decision is visible and can be overridden.** The plan is data.
   `explain` prints each decision with its reason and whether it was estimated
   or measured; tests compare plans; a user pins a value's storage or a group's
   schedule by the value's stable name.
6. **A second preparation takes seconds, in a new process.** Kernels are
   cached by content, not by Julia method, and a hit loads a compiled binary
   without generating code.
7. **Coverage does not depend on a catalogue.** Every program in the primitive
   language plans and runs with only the generator and the generic kernels,
   whatever else is loaded. A composition nobody wrote code for (another score
   function in attention, another spelling of softmax, an extra reduction,
   gradients) gets a plan without anyone adding that combination, and stores
   nothing that a streaming algorithm makes unnecessary; the coverage suite
   checks the kernels and bytes stored (see Testing). An opaque operation is
   outside this promise: it runs where an implementation claims it, and
   preparation fails, naming it, where none does.

## Purpose and scope

Tempest turns ordinary Julia array code (a loss, a training step, a forward
pass) into a prepared program that runs as fast as the hardware and the best
available kernels allow, and that explains every decision it made. The user
writes what to compute. Tempest decides how: which values are stored, which
are fused, how each kernel is scheduled and which implementation runs it,
choosing among generated tile programs, vendor libraries and the user's own
kernels. It exists because no other system combines three things:

- **Julia functions are first class.** A pointwise function or a fold's
  operator is Julia code. Tempest reads it to plan (its derivative, its cost,
  its canonical form), and the kernel runs the function itself. It is never
  mapped onto a fixed operator set.
- **Custom kernels and custom derivatives compose with fusion.** A named
  operation carries a reference definition, so it fuses like any other code
  when no kernel claims it, and is replaced whole when a kernel claims it and
  wins on cost. An operation without a reference is opaque, and still composes
  at its inputs and outputs.
- **The whole step is prepared once,** with fixed storage, and replayed as one
  captured graph.

Tempest is a foundation: the compiler at the base of jool-space, which builds
new infrastructure for deep learning in Julia rather than extending today's
frameworks. Layers, optimizers, data loading and models are packages built on
it, not parts of it; libraries of operations such as NNlib sit beside it.
Integration with today's frameworks is welcome where it comes nearly free, and
does not shape the design.

**The same code runs eagerly.** Every function Tempest adds (folds with custom
operators, `logsumexp`, positions, verbs such as attention and dropout) has a
method on plain arrays, on the CPU or a GPU, so one source runs untraced,
array by array, and traced. The coverage goal is that slice: Julia array code
that runs eagerly should trace, except control flow on array values. Existing
array code traces as it is; only functions whose bodies cannot trace need a
declaration (see Operations).

**Not goals:** another framework's operator list as a checklist; a code path
per model or pattern (every feature is formulas, folds and scalar functions,
or a claim at the edge); ambient numerics (precision is written in the
program, per operation, never a global or task-local mode); speed claims
without their method (hardware, clocks, precision, harness and baseline, all
reproducible from the repository); hardware in the core (formulas, folds,
scalar functions, the planner and the generator's input are backend-neutral,
and vendor specifics live in implementations and device defaults).

## The idea

Tempest describes every computation with three things and plans it with one
decision. Each of the three comes with rules for rewriting it without changing
its result, and those rules are what let the planner rearrange work freely.

- **Formulas: where a value reads from.** Every read is `x[f(i)]`, where `f`
  is index arithmetic: sums, constant multiples, and floor division and
  remainder by constants. Transposes, reshapes, slices, broadcasts and
  concatenations only change `f`, and formulas have one canonical form, so how
  a layout is spelled does not matter.
- **Folds: how a value reduces over axes.** Every reduction is three
  functions: lift an element into a state, combine two states, observe the
  result. Combining is associative and commutative, so every grouping and
  every order is legal: resident, streamed, split, partial-then-merge and
  online evaluation are one fold grouped differently. A fold whose combine is
  associative but not commutative, such as a scan or a linear recurrence, must
  keep its order; that extension is not built.
- **Scalar functions: what happens to each element.** Julia functions, traced
  into primitives with derivative rules. They are pure, so recomputing a value
  instead of storing it is always legal.
- **The decision.** For each value: store it, or recompute it wherever it is
  read. A kernel is a set of maps and folds over one shared index domain, each
  fold reducing its own subset of the axes, laid over tiles by a schedule.

Nothing else is special. A matrix product is a sum fold of a product over
`(m, n, k)`; split-K is the split schedule; an epilogue is a map over the
fold's kept axes. Layer norm's forward and backward are folds in phases. Flash
attention is a fold over keys whose state is the running maximum, the running
sum and the weighted accumulator, with the rescale as the combine. Kernels
follow from the program and a schedule; a hand-written kernel is an escape
hatch at the edge, never a new kind in the core. The core (formulas, folds,
scalar functions, the planner and the generator) stays small enough to read in
an afternoon; backends are realizations of it.

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
  attention products are all this. It is a primitive at this level for
  readable programs and derivatives; at the index level it is a sum fold of a
  pointwise product, and the generator recognizes that bilinear term for tensor
  cores.
- `gather` and `scatter` by runtime integer indices.
- `positions`: the index itself, for masks, dropout and embeddings.
- Layout: reshape, permute, view, slice, broadcast, concatenate.
- `call`: a named operation (see Operations), with a reference written in the
  above or, for an opaque one, declared result shapes; optional implementations
  and an optional derivative rule.

Tracing, differentiation and functional updates work at this level.

**Functional updates.** An array the traced code allocates can be written in
place, by `setindex!`, in-place broadcasts, `copyto!`, `fill!` and the like,
directly or through a view, reshape or transpose that shares its memory as in
Julia. Each write records the array's new contents; later reads see them, and
reads before the write keep what they read. A write of part of an array is a
select between the values written and the old contents, which fuses with its
readers, so no version of the array is stored; its derivative masks the old
contents and selects the values written. Three limits: the contents of
`similar` are zero until written, where Julia leaves them undefined; a write
at positions an array of integers or a mask selects is not a functional update
yet; and a program never writes its inputs, nor anything sharing their memory,
and an operation's reference never writes its arguments. Such a write fails,
saying so: a program is a function of its inputs, and a write into one would
change the caller's array without the program stating it.

Differentiation needs one pullback per primitive; a verb may bring its own. The
backward graph is ordinary program code and is planned together with the
forward. A library may claim a whole named call, as cuDNN claims `attention`:
at the index level a call is its reference expanded, with its place kept, and
a claim replaces it where the claim is cheaper. An opaque call nobody claims
fails preparation.

**Scalar functions.** A callable broadcast over traced arrays enters the
program as primitives: during tracing it runs on tracers that record each
primitive operation it performs (arithmetic, elementary functions, conversions,
comparisons, selects). Its derivative, cost and canonical form follow from the
primitives' rules. A callable that branches on an element value or loops
cannot be traced that way; for those, Tempest lowers the callable's typed IR
into the same primitives, turning branches whose arms are safe to evaluate
either way into selects and unrolling loops with a constant trip count. Three
rules keep that sound:

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
derivative. Tile programs receive primitives, not callables, so loops never
reach cuTile; if-conversion in cuTile's own broadcast, upstream, lets opaque
callables with branches reach tile programs too.

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

An example, a linear layer with bias and GELU:

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
softmax. Attention's backward pass is, at first, a hand-written schedule of
three launches attached to the verb's reverse rule, in the same language;
deriving it from differentiation and planning comes later.

A library call is a schedule with no tile program.

### 4. Kernels

Tile programs, library calls and generic CUDA kernels over concrete buffers, as
a graph with dependencies that capture preserves, so independent branches run
concurrently. Groups with the same schedule and body over many arrays, such as
an optimizer step over every parameter, are one launch over a table of arrays.
Nothing is decided at this level.

## Operations

An operation is a Julia function. Three things about it are kept apart:

- **its meaning,** what the function computes;
- **its decomposition,** a reference that states the meaning in primitives,
  which is optional;
- **its implementations,** ways to run it, of which there may be several or
  none.

Opacity of an operation and opacity of a kernel are separate questions. cuBLAS
runs opaque code for a fully understood product; a custom solver may be an
operation with no reference at all.

A Julia function meets Tempest in one of three ways.

1. **Its body traces.** Generic array code (activations, `logsumexp`, a norm
   written with broadcasts and sums) needs nothing: tracing runs it on traced
   arrays, and Tempest plans its body. The name is not kept, and need not be:
   an implementation recognizes the computation at the index level, where
   spelling does not matter.
2. **It keeps its name.** `@operation f(x::AbstractArray, w::AbstractArray;
   eps)` adds methods of `f` on traced arrays that record a call of `f`: one
   for each mix of traced and plain arrays among the declared arguments, so
   they are more specific than `f`'s own methods and ambiguous with none.
   Eager execution is unchanged, since plain arrays still reach `f`'s own
   methods. The reference is `f`'s method for the declared types run on
   traced arrays, unless one is given with `reference`. A name buys three
   things: a derivative rule, a claim by name, and a boundary that `explain`
   shows.
3. **Its body does not trace.** A function that branches on array values,
   writes its arguments or Julia arrays it allocates, loops over scalars, draws
   from a global random number generator or has only device methods is
   declared the same way, with either a reference
   that traces or, for an opaque operation, a declaration of its result shapes
   and element types: `declared_results`, a function of its arguments' shapes
   and element types and its keywords.

An opaque operation has no decomposition. Its eager method is its meaning and
the oracle its implementations are tested against. It differentiates only
through a rule. At the index level it is an opaque value that reads and writes
whole buffers, and it fuses with its neighbours only where its implementation
takes an epilogue. With no implementation on the target, preparation fails and
names it.

Declarations for packages Tempest does not own go in package extensions, as
methods on Tempest's traced arrays: for NNlib, the functions whose bodies do not
trace, such as `softmax`, `batched_mul` and `conv`. An existing device method,
such as a package's `CuArray` path, can serve as an implementation when it
writes into a buffer it is given, runs on Tempest's stream, and neither
synchronizes nor allocates per call.

Julia dispatch decides which operation is requested; Tempest decides how it
runs. Implementations do not dispatch on array types. They answer claims by
operation (`typeof(f)`), target and input shapes, or by inspecting a group of
index-level values.

An extension author gets more for more effort:

1. a reference: the operation composes, differentiates and fuses;
2. a derivative rule;
3. a kernel with conditions: it runs where it applies, and Tempest generates
   its neighbours;
4. a kernel that takes an epilogue: fusion across its output;
5. a schedule in Tempest's language: fusion through it.

## Meaning, algorithm, schedule

A reference establishes what a computation means, not an acceptable way to run
it: attention that stores its score matrix is correct and useless. Three layers
stay distinguishable, even where they share code.

- **Meaning** is in the reference, owned by the operation author: scores, a
  mask, a softmax, a weighted sum.
- **Algorithm** is in the plan, owned by Tempest: store or stream, an online
  fold, a split, a fold recomputed inside its reader. Algorithms are rewrites of
  folds and the store-or-recompute decision, chosen by cost and permitted by
  the numerics policy. None is tied to one spelling or hidden inside one
  implementation.
- **Schedule** is owned by the implementation: tiles, sweeps, splits, launch
  shapes.

The generic kernels make every plan correct. The generator with Tempest's
algorithms makes coverage worth having.

## Implementations

Every implementation answers one question in one way: given a group of
index-level values, can you run it, and how? Each answer is a claim, which
states

- the group and its schedule;
- the forms in which it reads each input and writes each output, as a
  predicate on the concrete formula (alignment, contiguity, a unit-stride
  axis, divisibility, writing in place into an input's storage), with a cost
  that may depend on the form, since a form changes which kernel runs;
- its estimated cost as launches with bytes, flops and parallelism, or
  `measure`, for claims whose estimate cannot separate them from their rivals;
- a build function that returns the launches.

A claim on a named call, on a written-out pattern, or on a group the
implementation itself proposed is the same kind of claim. The roles:

- **cuTile is the generator.** It claims every group a sketch can schedule,
  with as many schedules as the search allows.
- **cuBLAS and cuDNN are schedules without tile programs.** They claim
  products, products with epilogues and attention, and are chosen when they
  win, usually by measurement.
- **The floor realizes any single value alone** with generic kernels. It
  covers what tile programs cannot express, such as
  deterministic scatters and opaque bodies. Because the floor claims every
  group, a plan always exists. The floor makes every plan correct; it does not
  make coverage worth having (see Meaning, algorithm, schedule).
- **User kernels** arrive as implementations of operations or of groups (see
  Operations).

A library that accepts or declines whole graphs, as cuDNN's engines do, takes
part by proposing groups and claiming those its engines accept. What it
declines is generated around it.

An implementation may claim a written-out pattern (attention spelled as two
products around a softmax) as well as the named call. The index level is
canonical, so the match does not depend on spelling.

Loading a package makes its implementations available; the target decides
which are used. Each implementation is an extension that registers itself when
its packages load (the generator with cuTile, the product claims with cuBLAS).
A target lists the implementations it may use: by default every available one
at preparation, printed by `explain`, and narrowed by naming them, as in
`CUDATarget(cuTile)`, so a session that loads cuBLAS for other reasons can still
plan without it. The floor is always included. Which implementation runs a
kernel changes speed and summation order, never what the program computes.

## The decision

One place decides what forms a kernel, and it decides by cost. Five words
carry the whole of it.

- **Plan.** Groups of index-level values that together cover the program.
  Each group becomes one kernel.
- **Claim.** One way an implementation can run one group, with an estimate
  (see Implementations). The floor claims every group, so a plan always
  exists.
- **Proposal.** An alternative grouping: a set of values to run together, or a
  call with the epilogues that follow it. Implementations propose, and so does
  Tempest, for every choice its own rules would otherwise make alone: joining
  a stored value into a reader's group, recomputing a value in each of its
  readers, applying a rewrite such as the online softmax.
- **Cover.** Chooses the proposals and claims that make the plan cheapest, by
  estimate or, for a measured claim, by the record. Where two claims demand
  conflicting forms, a copy is inserted and charged. The order is total and
  the result deterministic.
- **Build.** Turns the chosen claims into kernels. A claim whose build fails
  is withdrawn and the cover runs again, so the plan reports what was built.

What follows from these, rather than being decided somewhere else: a value
with one reader is recomputed inside it; a value with several readers is
stored unless the cover finds recomputing cheaper; stored values get lifetimes
over the kernel graph, slot reuse, and donation in place where the write
formula and the read formula of the aliased value agree.

**The estimate** is a roofline with launch overhead, a parallelism term and
read multiplicity (a streamed schedule reads its inputs once per phase), with
rates derived from the device. Measured claims are compiled in parallel and
timed as the minimum of several alternating runs, and the times go into the
tuning record. A measurement overrides the estimate only beyond the device's
noise margin.

**The shape.** Tempest's rules make a starting plan, and every rule-made
choice is also a proposal: an edit to the current plan that dissolves the
groups owning its values, forms its group, and regroups what is left with
every other group frozen. The estimate is a sum over groups, so an edit is
priced on the groups it changes, and edits are taken greedily by gain. The
other shape, candidates grown from anchors (folds, contracts, calls, results)
by each applicable sketch and covered greedily by estimated gain as XLA and
Inductor do, prices every grouping the same way, but it would make the
generator a proposer and move every plan at once; this one keeps a plan that
always exists and adds pricing to it. Either way, no grouping may be settled
by a rule the cover never prices. That is Promise 3, and its test is a
program whose best grouping the rules alone would not find.

The plan records, for each value, stored (with its layout) or recomputed; for
each kernel, its group, schedule and forms, and whether its choice was
measured; for each buffer, its lifetime and slot.

## Numerics

A program computes in the types it states, and nothing changes that silently.

- **The defaults are the program's types.** Float32 values multiply and add in
  Float32. Folds over half-precision elements accumulate in Float32 and round
  once, at the end.
- **Reduced precision is written in the program.** TF32 is a rounding function
  on a product's operands: a product of two rounded numbers is exact in
  Float32, so the interpreter computes what tensor cores compute, up to
  summation order. Three-TF32 products split each operand into a rounded part
  and a rounded remainder and sum three products in one fold. BFloat16 and
  Float16 are conversions. An implementation reads these functions on a
  product's operands and chooses its instructions from them; it never takes
  precision from a mode.
- **A policy is a rewrite, not a mode.** A policy such as "TF32 for every
  Float32 product" inserts those functions into a program, or into one region
  of it, before planning. The result is an ordinary program that the
  interpreter, `explain` and the cache key all see.
- **Rewrites that change rounding** (reassociation beyond regrouping a fold,
  scaling forms, online softmax, approximate functions) happen only where the
  policy permits them, and `explain` names them. Regrouping a fold is always
  permitted: it changes summation order, as tile sizes and implementations
  do. Determinism, when asked for, fixes the order.
- An implementation that cannot compute what the program states does not
  claim it. Removing a buffer never removes a rounding the program states.

## Runtime

A prepared program owns fixed buffers and replays one captured graph per step.
Inputs are bound by copy into owned storage, or by reference when the caller
promises a stable address and layout. Everything that varies between replays is
a device value: scalars, seeds, positions, and the current values of bounded
sizes. A bounded size is served by a few specializations, each padded to its
bucket, with the padding value set per operation (zero for products, −∞ for a
softmax's keys).

Tempest keeps two caches on disk. The kernel cache maps a content hash of the
group, schedule, device and policy to a compiled binary and its parameter
layout, and compiles misses in parallel. Its key is Tempest's, computed before
any inference; its entries live in Julia's object cache through CompilerCaching,
beside the kernel packages' own, which bounds their size and evicts the least
recently used. The tuning record, in the depot's scratch space and overridable
by an environment variable, maps the content, shapes, policy and device of a
measured claim to its times; it is local, survives new versions of Tempest, and
records for known devices can ship with the package.

## Testing

- **A reference interpreter of the program level defines meaning.** Every
  implementation and every plan is tested against it.
- **Generated programs.** Random programs must agree with the interpreter, and
  respellings drawn from the permitted rewrites must produce identical plans.
- **The coverage suite.** Compositions nobody wrote code for (attention with
  another score function, a softcap or another spelling of softmax, an extra
  reduction, gradients, a target with an implementation removed) must plan and
  run. Each case checks its plan's structure, the kernels and the bytes
  stored, against a recorded expectation. The check is deterministic and
  independent of the device's speed, and a spelling that stores what another
  spelling streams fails it.
- **The eager slice.** A corpus of Julia array code that runs eagerly, NNlib's
  functions first, is traced and checked against eager execution. How much of
  it traces is reported with every change, and it only grows.
- **Time is an alarm, not a target.** A plan that becomes much slower than its
  recorded time fails a check. Speed claims are made separately, with their
  method (see Purpose and scope).

## Not now

Control flow on traced values, mutation of a program's inputs, writes at
positions an array selects, ordered folds (scans and recurrences, whose
combine is not commutative), ragged sizes, several devices, symbolic sizes
beyond bounded ones, serving, and an MLIR core. Checkpointing, re-running
kernels to save memory, is a cut through the graph priced by peak memory over
a kernel order, and comes later as its own pass with a memory budget; until
then `checkpoint(f)` marks a region to re-run. An explicit plan leaves room
for each later.
