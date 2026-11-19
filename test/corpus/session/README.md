# The preparation session

The measure of how long a user waits for Tempest to prepare programs: one
scripted session, every process clocked from outside it, with Tempest's
defaults (nothing measured).

1. `session.sh` makes an empty home directory holding only the depot of
   precompiled dependencies that `deps.sh` made from this environment's
   manifest, and times precompiling Tempest (with its extensions and what
   depends on it) and loading it.
2. Each program then runs three times, each in a new process, from the
   moment the process is started (`outside.sh`) until two calls of the
   program `compile` made have returned and the second's outputs are checked
   against the same function untraced: `first` with empty stores, `kept` with
   the stores that run left, and `doubled` with those stores at the program's
   doubled shape, twice its tokens or its batch. These are scenarios of the
   corpus runner (`../run.jl`), run in this environment, and the worker's
   side of them is `call.jl`. A program's seconds are the sum of its three
   runs, less the seconds spent reading a checkpoint (`checkpoint_s`).
3. A program's factor is its baseline seconds over its seconds now; a set's
   factor the geometric mean of its programs' (`factors.jl`).
4. Each program also runs once after others (`second`): one process prepares
   the corpus's gpt, lstm, heat and node at their small shapes (but the
   program itself where it is one of them), then the program, on stores a
   process preparing the same left, so that every preparation in it finds
   its stores kept.

## What compiles

Every process of the session writes what Julia compiles, as it compiles it,
with each signature's milliseconds (`--trace-compile`, in `OUT/traces`), and
beside it which packages are foreign to its programs: those it loaded but
by a program, Tempest's extensions whoever loaded them. A signature names
the program's own code where any module its types are named under is not
foreign: the program's file and the session's code, which live in `Main`,
or a package the program loaded, Vallmo say. A run's `compiled` counts the
signatures that name none of it, compiled from the moment `compile` was
called to the second call's end; `compiled_process` those of the whole
process. Counts are exact: no other job on the machine moves them, and
work moved elsewhere in the process stays in `compiled_process`.

`compiled.jl BASELINE [CANDIDATE]` reports the counts of the goal of
compiling Tempest once (each the baseline's over the candidate's with a
second session):

1. Second program: the median over the unseen programs of their `second`
   runs' counts.
2. New process: each program's `kept` run's count; a named set's the sum over
   its programs, the development set's (gpt, lstm, the 4B's gradient) and
   the held-out set's (heat, node, the 4B's decode), and the unseen set's
   the median.
3. Copies per function: of the 4B gradient's `first` run, the signatures
   that name none of the program's code over the distinct functions among
   them (`callee`: the function a body is compiled for, each closure one of
   its own).

`compiled.jl --census TRACE [FROM:TO]` lists what a trace's window compiled
that names none of the program's code, by function: its copies, their
milliseconds, and the kinds of type its copies differ by.

The untraced outputs are computed before a program's runs, in a process of
their own, and kept (`TEMPEST_SESSION_REFERENCES`): the corpus programs' by
their steps on their arguments widened to Float64 on the host, the others'
by their own `untraced`, which calls the function twice where it writes
state into its arguments, as a run does. Outputs too large to hold twice are
compared by a digest (`call.jl`).

## The programs

- Development: `gpt`, `lstm` (the corpus's, small; doubled at twice the
  batch) and `qwen-gradient.jl` (Qwen3.5-4B's gradient through Vallmo's
  branch `train`, (16, 1) tokens, doubled (32, 1)).
- Held out: `heat` (100 steps, doubled 200), `node` (batch 128, doubled 256)
  and `qwen-decode.jl` (one decode step of B = 1 sequence, doubled B = 2).
- Unseen: `select.jl COMMIT OUT` draws twelve cases of the fuzzer's
  generator from the commit's hash (`unseen.jl`): three loops, three loops'
  gradients, three cases of other families and three of their gradients, at
  the generator's largest extents, a loop's steps 64 (doubled 128).

## Running one

    bash deps.sh DEPOT                       # once a machine
    TEMPEST_SESSION_DEPS=DEPOT JULIA_NUM_THREADS=20 bash session.sh OUT gpt lstm $PWD/qwen-gradient.jl
    julia factors.jl BASELINE_OUT CANDIDATE_OUT
    julia compiled.jl BASELINE_OUT CANDIDATE_OUT

The environment holds Tempest from this tree (`../../..`), Vallmo from the
tree's sibling `Vallmo-train`, and LMDB from `AntonOresten/LMDB.jl`, branch
`concurrent-transactions`, whose fix keeps a process from hanging at more
than one thread until a release has it.
