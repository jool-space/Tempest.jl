# What a preparation's cover reached, kept and replayed. The cover's outcome
# is a few choices: the proposals it took, in order, which make its plan
# (`plan_taking`), and each group's decision, which claim runs it; and a
# recurrence's step, planned as a program of its own, has its own. They are
# data, named by the values of the program's index graph, and a preparation
# of the same program on the same target reaches them again from the same
# record and rates file. Kept, a preparation that finds them, and finds the
# record and the rates file holding what they rested on, replays them: it
# plans once and prices nothing, plans no trial and measures nothing. JAX's
# persistent compilation cache and torch.compile's FX graph cache keep a
# whole program's compilation so; tinygrad keeps its search's results beside
# its kernels, keyed by the kernel's source.
#
# What they rest on is in two parts. The key: everything a decision reads
# that a preparation knows before it decides, the graph whole (its values'
# definitions with every function and constant they hold, their shapes and
# element types, the results and the arguments), the folds pinned or kept
# and the sums riding, the steps of a recurrence's step, the target and its
# numerics, the device and its rates, the implementations loaded and the
# packages they run, the settings of the environment the implementations
# read, how the decider measures, and the identities of Julia and of
# Tempest's own code, every edit of which is another key (`content_key`).
# And the evidence: each key of the record and of the rates file the
# decisions read, with the value it held once they were reached or its
# absence (see `Noted`), held against the files as they are when they are
# replayed. A time measured since
# for a claim they read, a calibration fitted again, a claim that failed to
# build, each makes them a cover's to reach again.
#
# Only decisions the cover would reach again are replayed: those reached
# with nothing written to the record or the rates file, a `settled` set. A
# cover that measures as it decides reads a time absent at first and present
# after, and on the record it leaves could rank its proposals otherwise:
# what it reached is kept, marked unsettled, for what replaces it.

const REACHED_SCHEMA = 3
const REACHED_MAGIC = "Tempest decisions\n"

"""
What a preparation's cover reached, as data a preparation of the same
program replays (see `reach!`): the proposals it took, in order, whose plan
(`plan_taking`) is the one it reached; each group's text (`canonical`) and
decision; the proposals it compared, each with the times of the two sides,
without the plans; each of its recurrences' steps, by the key of its
decider; and how it was reached: by the decider's `measure`, `settled` when
nothing was measured, fitted or failed to build as it was, and the keys of
the record and of the rates file it read, each with the value it held once
it was reached, or nothing; and, reached measuring nothing, how many options
a cover measuring close calls would have timed at the points this one
decided without a time (see `wanted`).

Reached measuring nothing with none wanted, they are what a cover measuring
close calls reaches on the same record: it would visit the same points,
find a time at each it measures, and measure nothing.
"""
struct Reached
    key::Union{Nothing, String}
    how::Symbol
    settled::Bool
    taken::Vector{Proposal}
    considered::Vector{Considered}
    groups::Vector{String}
    decisions::Vector{Any}
    steps::Vector{Pair{String, Reached}}
    wanted::Int
    record::Vector{Pair{String, Union{Nothing, Float64}}}
    calibration::Vector{Pair{String, Union{Nothing, Float64}}}
end

"Deciders whose decisions were replayed and deciders whose decisions the cover reached, in this session, a recurrence's step's among them."
const REACHED_COUNTS = (replayed = Threads.Atomic{Int}(0), covered = Threads.Atomic{Int}(0))

reached_counts() = (replayed = REACHED_COUNTS.replayed[], covered = REACHED_COUNTS.covered[])

# A preparation whose decisions are being replayed: the steps of its
# recurrences are replayed from it, by their deciders' keys.
const REPLAYING = Base.ScopedValues.ScopedValue{Union{Nothing, Reached}}(nothing)

"A recurrence's step a replayed preparation did not keep: the cover reaches the preparation's decisions."
struct NotReplayed <: Exception end

# The environment's settings that the cover does not read, or reads into
# the decider's `measure`, which the key holds: every other `TEMPEST_`
# setting is part of the key, as the generator's product tiles are.
const UNREAD_SETTINGS = ("TEMPEST_CACHE", "TEMPEST_VERIFY", "TEMPEST_MEASURE", "TEMPEST_SETTLE", "TEMPEST_REPLAY", "TEMPEST_RATES",
    "TEMPEST_CALIBRATE", "TEMPEST_GPU_LOCK")

read_settings() = sort!([k => v for (k, v) in ENV if startswith(k, "TEMPEST_") && !(k in UNREAD_SETTINGS) && !startswith(k, "TEMPEST_CORPUS")])

# The module of the package an implementation's kernels come from, which
# its claims and what it lowers depend on; nothing where none is loaded.
function package_module(name::Symbol)
    for m in Base.loaded_modules_array()
        nameof(m) === name && return m
    end
    nothing
end

"""
    decisions_key(d) -> String or nothing

What the decisions of decider `d` rest on, but for the record and the rates
file, hashed (see `content_key`): the same for the same program on the same
target in every session, and another for any change in what it holds. A
function is written out with what it captures, and one outside any package
with its code, so two closures of one method over different values are two
programs. Nothing where decisions are not kept: where the environment's
`TEMPEST_REPLAY=false` asks for the cover every time (see `keeping`), while
a package precompiles, whose workload is to compile the cover, with nowhere
to keep them, or for an implementation outside any package, whose claims
are methods the key cannot name.
"""
function decisions_key(d::Decider; measure = d.measure)
    parts = key_parts(d; measure)
    parts === nothing ? nothing : content_key(parts...)
end

"""
    keeping() -> Bool

Whether preparations keep their decisions and replay those kept: from the
environment's `TEMPEST_REPLAY`, true unless it says false. Set false, a
preparation neither replays nor keeps any, and its cover reaches them as it
would on the record and the rates file with nothing kept: what the cover
does now, for a probe that times it or a plan being looked into. A fresh
`TEMPEST_CACHE` keeps nothing yet either, and holds no record and no
kernels besides.
"""
keeping() = parse(Bool, get(ENV, "TEMPEST_REPLAY", "true"))

# What `decisions_key` hashes, or nothing: the decider's, at measure level
# `measure`, as a decider measuring nothing looks up those of one that
# measured close calls (see `replayable`).
function key_parts(d::Decider; measure = d.measure)
    dir = tuning_directory()
    (dir === nothing || !keeping() || ccall(:jl_generating_output, Cint, ()) == 1) && return nothing
    implemented = Implementation[implementations(d.target)..., loaded_implementations()...]
    all(i -> identified(parentmodule(typeof(i))), implemented) || return nothing
    g = d.graph
    packages = sort!(unique(filter(!isnothing, [package_module(package_name(i)) for i in implemented])); by = string)
    # Tempest's extensions loaded, whose methods are part of what claims and lowers.
    extensions = sort!([string(nameof(m)) for m in Base.loaded_modules_array() if m !== ROOT && own_root(m)])
    device = measure === :dry ? nothing : device_name(d.target)
    # The device's reported rates, `d.rates`, are its identity, as a shipped
    # rates file states it (see `device_identity`); the record's times, and
    # the shipped rates fitted on them, are on the timing basis.
    ("decisions", REACHED_SCHEMA, TIMING_BASIS, g.values, g.arguments, g.results, g.result_dims, g.result_scalar, g.result_eltype,
        collect(g.backward), sort!(collect(d.pinned)), sort!(collect(d.kept)), sort!(collect(d.rides)), d.steps, measure,
        d.measurable, d.target, device, d.rates, loaded_implementations(), packages, extensions, read_settings())
end

# Where decisions are kept, by key: `decisions` beside the record.
reached_path(key) = joinpath(tuning_directory(), "decisions", key)

"""
    load_reached(key) -> Reached or nothing

The decisions kept under `key`, or nothing: none kept, or a file that is not
a whole one of this schema under this key, as one cut short or another's.
"""
function load_reached(key)
    path = reached_path(key)
    isfile(path) ? read_reached(path, key) : nothing
end

function read_reached(path, key)
    r = try
        open(path) do io
            String(read(io, ncodeunits(REACHED_MAGIC))) == REACHED_MAGIC && read(io, UInt32) == REACHED_SCHEMA ? deserialize(io) : nothing
        end
    catch err
        err isa InterruptException && rethrow()
        nothing
    end
    r isa Reached && r.key == key ? r : nothing
end

"""
    store_reached(r)

Keep decisions under their key: written apart and renamed into place, so
that a preparation reading them, in this process or another, finds a whole
file or none, and of two writing at once the last stands. A claim that
cannot be written out keeps nothing.
"""
function store_reached(r::Reached)
    path = reached_path(r.key)
    mkpath(dirname(path))
    part = string(path, ".", getpid(), ".", objectid(r), ".part")
    try
        open(part, "w") do io
            write(io, REACHED_MAGIC)
            write(io, UInt32(REACHED_SCHEMA))
            serialize(io, r)
        end
        Base.Filesystem.rename(part, path)
    catch err
        err isa InterruptException && rethrow()
        rm(part; force = true)
        @debug "decisions not kept" exception = err
    end
    nothing
end

# A comparison without the plans it compared, as it is kept.
kept_comparison(c::Comparison) = Comparison(c.old, c.new, c.how, (Any[], Any[]))

"""
    reached(d, key, plan, considered, decisions) -> Reached

What the cover of decider `d` reached, as it is kept: the proposals it took,
in order, and each of its recurrences' steps as the decider planned it. How
it was reached is the preparation's to say once it is built (see `settled`).
"""
function reached(d::Decider, key, p::Plan, considered, ds)
    taken = Proposal[c.proposal for c in considered if c.accepted]
    kept = Considered[Considered(c.proposal, c.implementation, kept_comparison(c.comparison), c.accepted) for c in considered]
    steps = Pair{String, Reached}[]
    for g in p.groups
        key === nothing && break
        g.kind === :recurrence || continue
        sp = get(d.stepped, group_identity(p, g), nothing)
        sp === nothing || any(s -> first(s) == sp.reached.key, steps) || push!(steps, sp.reached.key => sp.reached)
    end
    Reached(key, d.measure, false, taken, kept, [canonical(p, g) for g in p.groups], Any[ds...], steps, length(d.wanted),
        Pair{String, Union{Nothing, Float64}}[], Pair{String, Union{Nothing, Float64}}[])
end

"""
    settled(r, d) -> Reached

Decisions `r` as a preparation by decider `d` leaves them: with the keys of
the record and of the rates file its deciders read, and settled where
nothing was written to either and no claim failed to build, so that the
cover, on the files as they now are, reaches them again.
"""
settled(r::Reached, d::Decider) = Reached(r.key, r.how, !written(d.record) && !written(d.calibration) && isempty(d.failed),
    r.taken, r.considered, r.groups, r.decisions, r.steps, r.wanted, evidence(d.record), evidence(d.calibration))

"""
    replayable(d, key) -> Reached or nothing

The decisions decider `d` may replay, kept under its `key` or, for a
decider that measures nothing, under the key it would have measuring close
calls, which it prefers: settled, and the record and the rates file it reads
holding what they rested on, no claim having failed to build in this
preparation.

A cover that measures nothing reaches what a settled cover measuring close
calls reached, on the record that cover left: the two read the record alike
(`measured_ranks`, `refuted`, `choose`, `expand!`, `compare`), and differ in
the measuring alone, which wrote nothing, and in the searches ahead of the
decisions, whose groups the cover measuring nothing decides too. One
measuring every option is not so replayed: it measured options a decision
stands by the estimate against.
"""
function replayable(d::Decider, key)
    isempty(d.failed) || return nothing
    for level in (d.measure === :none ? (:close, :none) : (d.measure,))
        r = load_reached(level === d.measure ? key : decisions_key(d; measure = level))
        r !== nothing && r.settled && r.how === level && holds(d.record, r.record) && holds(d.calibration, r.calibration) && return r
    end
    nothing
end

"""
A preparation's decisions and the plan they form: the plan, the proposals
the cover compared, each group's decision, and what was reached as it is
kept (`Reached`), replayed or by the cover (see `reach!`).
"""
struct Decided
    plan::Plan
    considered::Vector{Any}
    decisions::Vector{Any}
    reached::Reached
end

"""
    replay(d, r) -> Decided or nothing

Decisions `r` replayed on decider `d`'s graph: the plan of the proposals
taken, planned once, each group's text checked against the one kept, and
each recurrence's step replayed from those kept with them. Nothing where a
group differs or a step was not kept: the cover reaches them instead.
"""
function replay(d::Decider, r::Reached)
    p = plan_taking(d, r.taken)
    p === nothing && return nothing
    length(p.groups) == length(r.groups) || return nothing
    all(k -> canonical(p, p.groups[k]) == r.groups[k], eachindex(p.groups)) || return nothing
    # Each recurrence's step, replayed here rather than as the groups build
    # side by side.
    try
        Base.ScopedValues.with(REPLAYING => r) do
            foreach(g -> g.kind === :recurrence && stepped_plan(d, p, g), p.groups)
        end
    catch err
        err isa NotReplayed || rethrow()
        return nothing
    end
    Threads.atomic_add!(REACHED_COUNTS.replayed, 1)
    Decided(p, Any[r.considered...], Any[r.decisions...], r)
end

"""
    reach!(d; settle = false) -> Decided

The decisions of decider `d` and the plan they form: replayed where
decisions kept under the decider's key may be (see `replayable`), else
reached by the cover, settled with `settle` (see `covered!`). A recurrence's step within a
preparation replayed is replayed from the preparation's. A preparation that
verifies reaches its decisions by the cover too, and throws where the two
differ.
"""
function reach!(d::Decider; settle = false)
    within = REPLAYING[]
    if within !== nothing
        # A step's decisions, keyed as the decisions replayed were reached.
        key = decisions_key(d; measure = within.how)
        i = findfirst(s -> first(s) == key, within.steps)
        i === nothing && throw(NotReplayed())
        step = last(within.steps[i])
        replayed = replay(d, step)
        replayed === nothing && throw(NotReplayed())
        return replayed
    end
    key = decisions_key(d)
    stored = key === nothing ? nothing : replayable(d, key)
    stored === nothing && return covered!(d, key; settle = settle && key !== nothing)
    replayed = phase(() -> replay(d, stored), :replay)
    replayed === nothing || VERIFYING[] || return replayed
    # The steps replayed are the cover's to plan again.
    empty!(d.stepped)
    found = covered!(d, key; settle)
    if VERIFYING[]
        difference = replayed === nothing ? "the plan of the proposals taken" : differs(stored, found.reached)
        difference === nothing || error("decisions replayed differ from the cover's: ", difference)
    end
    found
end

# The covers a preparation runs at most to settle its decisions.
const SETTLES = 3

# Whether a preparation settles its decisions: `prepare`'s default, from the
# environment's TEMPEST_SETTLE, true unless it says false. Settling adds a
# cover's work to a preparation that measured, for the next to replay.
settling() = parse(Bool, get(ENV, "TEMPEST_SETTLE", "true"))

"""
    covered!(d, key; settle = false) -> Decided

The cover's decisions, with what was reached. With `settle`, a cover that
measured or fitted as it decided is followed by another on what it left,
its decisions and options forgotten, until one writes nothing, `SETTLES`
covers at most: the decisions of the last are those a preparation in a new
process would reach, and replay (see `settled`).
"""
function covered!(d::Decider, key; settle = false)
    found = cover_once!(d, key)
    for _ in 2:(settle ? SETTLES : 1)
        (written(d.record) || written(d.calibration)) && isempty(d.failed) || break
        unwritten!(d.record)
        unwritten!(d.calibration)
        forget_decisions!(d)
        empty!(d.stepped)
        found = cover_once!(d, key)
    end
    found
end

function cover_once!(d::Decider, key)
    Threads.atomic_add!(REACHED_COUNTS.covered, 1)
    p, considered = phase(() -> cover(d), :cover)
    # The sections after the cover (see `lap!`).
    clock = mark()
    search!(d, p)
    clock = lap!(:search, clock)
    ds = decisions(d, p)
    lap!(:decisions, clock)
    Decided(p, considered, ds, reached(d, key, p, considered, ds))
end

"""
    given!(d, r) -> Decided

Decisions `r` replayed on decider `d`, as a search in this process has just
reached them for it: what they rest on is not held against the files, and
decisions that do not form their plan throw.
"""
function given!(d::Decider, r::Reached)
    replayed = replay(d, r)
    replayed === nothing && error("the decisions given do not form the plan they were reached with")
    replayed
end

"""
    differs(a, b; plan = false) -> String or nothing

How decisions `a` differ from `b`, in a line, or nothing where they are the
same: the proposals taken, the groups, each group's decision, the proposals
compared, and the steps of recurrences, each alike. With `plan`, only what
builds the program: each group's choice by its key, and whether every option
of it writes in place, which its storage reads (see `storage`); not how it
was made, its time or its options, nor the proposals compared.
"""
function differs(a::Reached, b::Reached; plan::Bool = false)
    a.taken == b.taken || return "the proposals taken"
    a.groups == b.groups || return "the groups"
    length(a.decisions) == length(b.decisions) || return "the number of decisions"
    text = plan ? runs_text : decision_text
    for (k, (x, y)) in enumerate(zip(a.decisions, b.decisions))
        text(x) == text(y) || return string("the decision of group ", k, ": ", text(x), " against ", text(y))
    end
    plan || map(considered_text, a.considered) == map(considered_text, b.considered) || return "the proposals compared"
    map(first, a.steps) == map(first, b.steps) || return "the recurrences' steps"
    for ((_, x), (_, y)) in zip(a.steps, b.steps)
        difference = differs(x, y; plan)
        difference === nothing || return string("a recurrence's step: ", difference)
    end
    nothing
end

"""
    same_plan(a, b) -> Bool

Whether decisions `a` and `b` build the same program (see `differs`).
"""
same_plan(a::Reached, b::Reached) = differs(a, b; plan = true) === nothing

runs_text(x::Decision) = string(x.choice.key, " ", all(o -> o.implementation === nothing || Base.invokelatest(in_place, o.implementation, o.claim), x.options))
runs_text(::Nothing) = "nothing"

decision_text(x::Decision) = string(x.choice.key, " ", x.how, " ", x.measured, " ", x.choice.estimate, " ",
    join((string(o.key, " ", o.estimate) for o in x.options), "; "), " ", join((string(f.implementation, " ", f.reason) for f in x.failed), "; "))
decision_text(::Nothing) = "nothing"

considered_text(c::Considered) = string(repr(c.proposal), " ", c.implementation === nothing ? "the planner" : label(c.implementation), " ",
    c.comparison.old, " ", c.comparison.new, " ", c.comparison.how, " ", c.accepted)
