# A launch's work

[`Work`](@ref Tempest.Compiler.Implement.Work) is one launch's work, as an
implementation states it for Tempest to price.

## What a launch states

| field | what it states |
|:--|:--|
| `bytes` | bytes moved |
| `reread` | bytes it reads again (what its blocks read of an argument beyond the first time) |
| `passed` | bytes it passes to or takes from another of its claim's launches (partial results in a workspace, as a split sum's parts) |
| `flops`, `factor` | flops of products whose factors are of type `factor` |
| `recomputed` | flops of folds it recomputes inside them |
| `blocks`, `split` | its blocks, and whether they split the sums among them |
| `applied` | the scalar functions it applies elementwise, each with the number of positions it is applied at |
| `swept` | functions applied inside a product's sweep, to its factors |
| `inner` | functions applied inside a fold the sweep recomputes, to that fold's tiles |
| `steps` | the steps a block takes, where blocks sweep tiles one after another |
| `step_runs` | what a block loads at a step: of each load, the bytes it reads in one run, contiguous, paired with the bytes it reads in the step |
| `stepped` | the functions the steps compute on what they loaded, each with the number of steps it is applied in along their chain |
| `refold_steps` | the steps apart in which a block recomputes a fold whose body is not bilinear along its axis |

## How it is priced

Tempest prices the bytes on its package's memory, those read again from the
cache, those passed within the cache where they fit it, as the launch that
takes them finds them there, the flops on the line its package reaches for
products of that class, and each function on the time it measured for it on
the device (see `seconds`).

Functions applied inside a product's sweep, to its factors, are `swept`:
each step's small tile of them stalls the sweep, and they cost many times
what they cost streamed; those applied inside a fold the sweep recomputes,
to that fold's tiles, are `inner`, at a factor of their own.

## Sweeps in steps

A launch whose blocks sweep tiles one after another, waiting for what each
step loads before the next, states the `steps` a block takes, what a block
loads at a step (`step_runs`: of each load, the bytes it reads in one run,
contiguous, paired with the bytes it reads in the step), and the functions
the steps compute on what they loaded, each with the number of steps it is
applied in along their chain (`stepped`): a step takes its latency and its
bytes at what one block reaches, however few positions it covers, and a
sweep of many narrow steps takes that many times as long, whatever bytes it
moves.

Tempest prices a step on its package's memory and each function on the time
it measured for it in a step (see `seconds`). A launch that states no steps,
or one, is priced by its bytes, flops and functions alone, as a product's
sweep is by its line.

## Folds recomputed

A fold a launch recomputes where it reads it, its flops `recomputed`, a
block sweeps in steps of its own, before the sweep or inside each step the
fold moves along: a product's steps, among the sweep's, where its body is
bilinear along its axis, and otherwise `refold_steps` apart, combining its
body elementwise, which wait on their own loads one after another with the
rest of the block's work, however the sweep hides its bytes; Tempest prices
each as a step of the sweep.
