# Compiling and calling

```@meta
DocTestSetup = :(using Tempest)
```

[`compile`](@ref)`(f, args...)` traces `f` on `args` and prepares it, to be
called like `f`: `compile(f, x)(y)` computes `f(y)` for any `y` alike `x` in
structure, shapes and element types. It returns what `f` returns, in arrays
of its own, with numbers as numbers, and what `f` writes into an argument's
arrays it leaves there, as `f` would.

```jldoctest
julia> c = compile(x -> sum(abs2, x), [0.0, 0.0, 0.0, 0.0]);

julia> c([1.0, 2.0, 3.0, 4.0])
30.0
```

A function that writes its arguments is compiled as it is written, and its
state stays where the caller keeps it:

```jldoctest
julia> f!(s, x) = (s .= 0.5 .* s .+ x; sum(s));

julia> s = [1.0, 2.0]; c = compile(f!, s, [0.0, 0.0]);

julia> c(s, [1.0, 1.0]), s
(3.5, [1.5, 2.0])
```

## Inputs and constants

Which leaves of the arguments are inputs follows
[`trace`](@ref Tempest.Compiler.trace): arrays, also inside tuples, named
tuples, structs, mutable ones too, and vectors of other things than numbers,
whose lengths a call must match. An array the function reads inside any
other leaf, a dictionary's say, would be a constant, and is refused, naming
the argument.

Numbers are constants; a number that varies between calls is an array, `[x]`
say.

`f` is an argument like the others, the call `f(args...)` traced: the arrays
of a callable struct's fields and those a closure captures are inputs too,
read at each call, so a closure over a model sees its weights as they are
when called.

## Globals

A global that `f` reads is no part of `f`, and tracing reads it once. One
holding arrays is refused, since the program would keep them as they were
when traced: pass it as an argument.

In `f`'s own code a global is seen by its name; in a function `f` calls, by
its arrays: an array tracing meets that lives in a global, of any module, is
refused alike, as is a global a function assigned while traced.

Any other global `f`'s own code reads is a constant, and one not declared
`const` is checked at each call. A number a function `f` calls reads, or a
copy it makes of a global's array before using it with an input, is kept as
it was when traced.

## Where the arrays live

An argument's array is the program's storage for it, read and written where
the caller has it, not copied: a call on the arrays of the call before costs
nothing for them, and one on others changes the addresses the captured
graph's kernels read and write, in place (see
[`point!`](@ref Tempest.Compiler.point!)), so a training step whose weights
are new arrays at every call copies none of them.

The program is built on the arrays `compile` is given and holds no storage
of its own for them, so a model's weights are in memory once; it holds the
arrays of its last call, or until the first those it was compiled on, until
a call brings others.

What `f` writes into an argument the kernels write into the caller's array:
where the kernel that writes it reads the array only at the addresses it
writes, and the kernels that read it otherwise can run first, directly, as
an append to a cache writes only its new positions; else into storage of the
program's own and then copied into the array once, after the last kernel
([`explain`](@ref) says which).

Everything else the program stores lives in buffers it owns and reuses at
every call. An array not of the target's array type, an `Array` for a
function compiled for the GPU say, is copied into a buffer the program makes
the first time it needs one and keeps, and, if written, back after the call.

## What a call must match

A call whose arguments, `f`'s own leaves or the globals `f` reads differ in
anything but the values of inputs throws, naming the difference. So does a
call whose arrays share memory with one `f` writes: the program reads each
array apart, where `f` run on them would read what it wrote.

## Measuring

Unless asked, nothing is timed: the first call runs the plan the estimates
pick, with the times the tuning record holds, and the plan does not change
after. Two keywords ask for more.

`measure` says which claims are timed, as for
[`prepare`](@ref Tempest.Compiler.prepare):

| `measure` | what is timed |
|:--|:--|
| `:none`, the default | nothing |
| `:close` | the options whose estimates are close |
| `:all` | every option |

`true` and `false` mean `:close` and `:none`.

`search` says when, where something is timed:

| `search` | when |
|:--|:--|
| `:before`, the default | while preparing, so that the first call runs the plan measurement picks |
| `:after` | behind the first result |

Where nothing is timed, with `measure = :none` or on a target that times
nothing, there is nothing to search and `search` means nothing, as `measure`
means nothing on such a target. The environment overrides the defaults,
`TEMPEST_MEASURE` (`none`, `close` or `all`) and `TEMPEST_SEARCH` (`before`
or `after`), and a keyword given overrides the environment.

## Decisions kept

Measuring once serves the preparations after it. A preparation that measures
close calls keeps the decisions its cover settles, and one that measures
nothing, in this process or another, replays them: its plan is the measured
one, with nothing timed. That holds for

- the same program, the same arguments' shapes and types and the same `f`
  with what it captures,
- on the same target and device,
- with the same toolchain's record and rates file (a file per device and
  toolchain) still holding the times those decisions rest on,
- where the measuring preparation settled (its covers, up to three, ended on
  one that measured nothing new; `TEMPEST_SETTLE=false` keeps unsettled
  decisions, which are not replayed so),
- and no claim failed to build in the preparation that replays.

Decisions reached measuring every option are not replayed by one that
measures nothing, and none are kept or replayed with `TEMPEST_REPLAY=false`
(see [`prepare`](@ref Tempest.Compiler.prepare)).

## Searching behind the first result

With `search = :after`, `compile` returns a [`Searching`](@ref Tempest.Searching),
called as the [`Compiled`](@ref Tempest.Compiled) is, with the plan the
estimates pick, prepared timing nothing, while a task (a
`Tempest.Compiler.Search`) prepares the program again, measuring as `measure`
says, its decisions settled as a preparation's are.

It compiles and builds on a thread of its own where Julia has more than one;
on one, it runs only where a call waits, and holds the calls while it
compiles, and `wait(c)` after the first result takes the whole search at
once.

The calls take its timings, one each after its own work, on the calling
task: so a claim is timed with none of the program's own work beside it, as
when preparing, and the search never captures a graph beside the caller's
work, which CUDA forbids the device's synchronization during. Other
programs' work in the process is not held off while a claim is timed, as it
is not while preparing.

The first call after the search ends runs its plan, captured then, as
`compile` captures the first, where it differs. A program not called
searches no further than its next timing; `wait(c)` takes the search's
timings until it ends, and changes the plan.

The record keeps every time taken, so a search cut short loses none, and the
decisions it settles are kept as it ends: a preparation of the same program
after it replays them, its first result is the measured plan, and there is
nothing to search behind it, so `compile` returns a `Compiled`, as it does
where nothing is timed. A search that fails leaves the program its plan,
with a warning.

Two plans of a program round alike to the program's precision but not to the
bit: a step's results may change in their last bits from the call that runs
the new plan, as between two preparations.

## Capture

With `capture`, the program's launches are recorded once as one graph that
each call replays, as [`capture!`](@ref Tempest.Compiler.capture!) and
[`replay!`](@ref Tempest.Compiler.replay!) do. Results are the same either
way.

## The level beneath

The level beneath is in `Tempest.Compiler`:
[`prepare`](@ref Tempest.Compiler.prepare),
[`bind!`](@ref Tempest.Compiler.bind!), and
[`execute!`](@ref Tempest.Compiler.execute!) or
[`replay!`](@ref Tempest.Compiler.replay!), with
[`point!`](@ref Tempest.Compiler.point!) binding a call's arrays as `compile`
does. `c.executable` is the prepared program, and `explain(c)` shows it.
