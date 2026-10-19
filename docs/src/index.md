# Tempest.jl

Tempest compiles Julia array programs into fused GPU kernels. A whole training
step, with the loss, its gradient and the optimizer, is traced once from plain
Julia, planned into kernels, and replayed as one CUDA graph. Kernel packages
such as cuTile and cuBLAS compete to run each part, and whatever no package
claims still runs.

```julia
using Tempest, CUDACore, cuTile, cuBLAS

struct Dense{W, B}
    w::W
    b::B
end
(d::Dense)(x) = tanh.(d.w * x .+ d.b)
loss(m, x) = sum(abs2, m(x))

m = Dense(CuArray(randn(Float32, 64, 32)), CuArray(randn(Float32, 64)))
x = CuArray(randn(Float32, 32, 16))

grad = compile((m, x) -> Tempest.gradient(loss, m, x), m, x)
l, dm, dx = grad(m, x)   # dm is a named tuple (w = ..., b = ...)
explain(grad)            # each kernel: what it computes and stores, and what runs it
```

Without a GPU, the same code runs on the CPU with plain arrays.

## The contract

Three kinds of people meet in Tempest. Model authors write Julia array code,
like the example above. Operation authors give operations names and meanings,
such as attention or a norm. Kernel authors offer faster ways to run those
operations on a device. Tempest connects the three so that each can work
without knowing the others' code. Four points say what each can rely on.

Tempest owns coverage and kernels own speed. A program written with Base's
arrays and the primitives on the [API](api.md) page runs with no kernel
package loaded, on the kernels Tempest generates for itself. Loading a package
or removing one may change how fast a program runs, and never whether it runs.
This is what a model author relies on, and what a kernel author may count on:
a kernel never has to cover everything, only to be better at something.

Each kind of person changes one place. A new operation is a change to an
operation package, a new kernel is a change to a kernel package, and a better
optimization is a change to Tempest. None of the three needs a change in the
others, and a model author's code stays the same through all of them.

What people rely on is the promises below, and nothing else. The formulas,
folds, schedules and the planner are how Tempest keeps the promises today.
`explain` shows them, but they are not the interface, and they may change.
A test or a program that depends on one depends on what Tempest has not
promised.

The core stays small. A new kind of node or value in the language Tempest
traces into is added only with a written argument that the existing ones
cannot express it, so that what an operation author or a kernel author learns
today is still what there is to learn.

## The promises

What a user can rely on, by number.

1. **Layout spelling does not matter.** Programs that differ only in reshape
   (adding or dropping size-1 axes included), permute, view, slice, broadcast,
   the placement of transposes, the operand order of commutative operations,
   or the name of a fold (`sum`, `reduce(+)`, `mapreduce`) get the same plan,
   and so do programs that differ by a rewrite the numerics policy permits,
   such as reassociation. Inside a scalar function only operand order and
   common subexpressions are canonical, so `x*2` and `x+x` stay different.
   Where two programs plan differently, `explain` names the rewrite that
   would unify them.
2. **A named call is never worse than its reference.** A kernel offered for an
   operation is compared by cost with the operation's own code fused into its
   neighbours, and runs only if it wins. A kernel that claims an operation may
   also claim the same computation written out by hand.
3. **One rule decides fusion.** Two computations become one kernel when the
   estimate says it is faster or, where a measurement is recorded, when the
   measurement does: never because of which code path found the fusion, and
   never by noise. A measurement overrides the estimate only beyond the
   device's noise margin, and the plan is deterministic given the record.
4. **Every kernel computes what the program states.** Element types and
   rounding functions included, up to summation order. Leaving a value out of
   memory never removes a rounding the program writes.
5. **Every decision is visible and can be overridden.** `explain` prints each
   decision with its reason and whether it was estimated or measured, tests
   can compare plans, and a user can pin where a value is stored or how a
   group of computations is scheduled, by the value's stable name.
6. **A second preparation takes seconds, in a new process.** Kernels are
   cached by content rather than by Julia method, and a cache hit loads a
   compiled binary without generating code.
7. **Coverage does not depend on a catalogue.** Every program in the primitive
   language plans and runs on the generator and the generic kernels alone,
   whatever else is loaded. A composition nobody wrote code for, such as
   another score function in attention, another spelling of softmax, an extra
   reduction or a gradient, gets a plan without anyone adding that
   combination, and stores nothing that a streaming algorithm makes
   unnecessary. An opaque operation is outside this promise: it runs where a
   kernel claims it, and preparation fails, naming it, where none does.

Each promise is checked by tests. A promise not yet met is recorded in the
test suite as a named gap, a broken test under a name that says what would
close it, and is not claimed here as kept.

## Where to go next

Someone offering a kernel for an operation will want [Writing a
kernel](kernels.md), which gives the interface a kernel package implements.
The [API](api.md) page documents the functions Tempest exports for compiling,
differentiating and explaining a program, and for defining operations.
