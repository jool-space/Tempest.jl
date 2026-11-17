# Preparing a program

[`prepare`](@ref Tempest.Compiler.prepare)`(program, inputs...)` plans
`program` for the shapes and types of `inputs` and builds its kernels on
`target`, chosen from the inputs' array types by default. It is the level
beneath [`compile`](@ref), for a step run in place.

## A training loop

A training loop, where the step returns the loss and the new parameters and
optimizer state:

```julia
ex = prepare(trace(step, params, state, batch), params, state, batch; replaces = (2 => 1, 3 => 2))
capture!(ex)
loss = first(outputs(ex))       # where each step leaves the loss
for batch in batches
    bind!(ex, 3 => batch)       # only the batch is new
    replay!(ex)                 # parameters and state are updated where they are
    @show loss[]
end
```

A function that writes its arguments needs none of this: `compile` writes
them where the caller has them, for any arrays a call brings.

## Measuring

Close calls between claims are timed when `measure` asks and the target can,
and kept in the tuning record.

| `measure` | what is timed |
|:--|:--|
| `:none`, the default | no claim, but the times in the record are still used |
| `:close` | the options whose estimates are within the close-call factor of the best |
| `:all` | every option of every group and both sides of every comparison |

`true` and `false` mean `:close` and `:none`. The environment's
`TEMPEST_MEASURE` (`none`, `close` or `all`) overrides the default, and
`measure` given overrides the environment. On a device other jobs share, a
close call timed is decided by chance.

## Decisions kept and replayed

What the cover decided, the proposals it took and the claim that runs each
kernel, is kept beside the tuning record (see `Tempest.Compiler.reach!`),
and a preparation of the same program, on the same target and device, with
the record and the rates file holding what those decisions read, replays it:
it plans the program once, and plans no trial, prices and measures nothing.

Anything else a decision reads that differs, the program's extents, element
types or captured constants, the target's numerics, Tempest's own code or a
time recorded since, makes them the cover's to reach again.

Decisions are replayed only where the cover would reach them again on the
record as it is: a preparation that measured as it decided covers again on
what it measured, until a cover measures nothing, three covers at most, and
keeps the last one's; with the environment's `TEMPEST_SETTLE=false` it keeps
the first's, which the next preparation reaches again.

With `TEMPEST_REPLAY=false` a preparation neither replays decisions nor
keeps its own: the cover reaches them every time. A fresh `TEMPEST_CACHE`
holds no decisions, no record and no kernels.

## A claim that fails to build

A claim that fails to build is no option once it has: the program is planned
again without it, each map of the group it claimed proposed to be stored
apart, so that a scalar function a kernel package cannot lower runs on the
floor and the rest of the group on the package. [`explain`](@ref) names the
claims that failed.

## State

`replaces` declares state, as pairs `result => argument` of positions among
the program's results and its arguments, alike in structure, shapes and
types: `(2 => 1, 3 => 2)` for a step
`(params, state, batch) -> (loss, params, state)`. After each execution the
argument holds the result, as if bound again, so the next execution reads it
where it is.

An input the program writes is replaced so by what it writes; a result that
is the input as it was before the write is copied apart before the kernels
run.

A result is written into the argument's storage when the kernel that writes
it reads the argument only at the address it writes, and the kernels that
read the argument otherwise can run first; else it is written apart and
copied there after the last kernel. `explain` says which.

## Where the inputs live

[`bind!`](@ref Tempest.Compiler.bind!) copies inputs into storage the
executable owns, except the arrays at the positions in `by_reference`, which
must be of the target's array type: they are the executable's storage. The
caller keeps them alive and may write them between executions, and a result
that replaces one writes into it.

[`point!`](@ref Tempest.Compiler.point!) makes the caller's arrays the
storage at each binding instead, as `compile` does.

The preparation binds `inputs` as `bind!` does, or as `point!` does with
`point = true`, as `compile` prepares: the kernels are built on the caller's
arrays where they can read them there, and the executable makes no storage
of its own for those arguments until a binding copies into it, holding the
arrays as `point!` does, until it is bound to others.

## Pinning what the cover decides

`recompute` pins folds to be recomputed where they are read rather than
stored, by their numbers among the values of the program's index graph, as
`explain` prints them: what the cover would otherwise decide by cost (see
`Tempest.Compiler.recomputable_folds`).

`ride` pins values computed from a recurrence's steps to be computed as the
steps run, so numbered: a sum over the steps added at each, a value along
them written at each, one of the last step written at it; what the cover
would otherwise decide by cost (see `Tempest.Compiler.riding_values`).

`partition` decides which of the forward's maps the backward recomputes
rather than reads, by a minimum cut of what saving them costs (see
`Tempest.Compiler.partition`); `false` saves every map the backward reads,
as a trial's comparison.

## Planning without building

`build = false` plans and decides but builds and allocates nothing: the
executable can be explained, not run, on a machine without the target's
device. No claim is timed, and claims are priced on nominal rates.

## Verifying

`verify = true` checks every plan the preparation forms, the cover's trials
among them, and the storage of the one it runs, throwing where one breaks an
invariant the kernels rely on (see [Plan invariants](@ref)). It is the
default where the environment sets `TEMPEST_VERIFY=true`, as the tests do.

Where it sets `TEMPEST_CROSSCHECK=true`, as the tests do too, decisions a
preparation replays are checked against those its cover reaches, which it
reaches as well, at a cover's cost.
