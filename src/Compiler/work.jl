# The compiler's own work, as a preparation does it: counts of what grows with
# a program's size, which a test bounds where a time would be the machine's;
# the seconds and bytes of each phase, to fit against a program's size, while
# `tallied` runs; and each phase's boundaries as events, for an instrument
# that reads what else happened meanwhile, while `OBSERVER` holds one. Off,
# a count, a mark or an event is a load and a branch.
#
# The phases a preparation marks: lower, rewrite, partition, decider and
# planned, and bind after it; within planned, cover, then the sections search,
# decisions, storage, verify, build and executable (what the build leaves to
# assemble); within cover, the sections cover_start (the generator's plan),
# cover_search, cover_proposals, cover_keys, cover_trials, cover_search_ahead,
# cover_pricing, cover_edits and cover_loop. Wherever they run: calibration,
# burst (claims built side by side to be timed), timing, and compile, on the
# task that compiles a kernel. A point event, kernel, names each kernel asked
# for by its key.

const TALLYING = Ref(false)

"""
What hears each phase's boundaries, or nothing: called as `f(name, :begin)`
and `f(name, :end)` around a phase, `f(name, :lap)` where a section `name`
of the enclosing phase ends on this task, and `f(name, x)` at a point event.
What it throws is dropped.
"""
const OBSERVER = Ref{Any}(nothing)

# What is counted. Each is counted on whichever thread does it: the cover
# plans its trials side by side.
const COUNTED = (
    :plans,      # plans formed (`plan`)
    :groupings,  # times the planner formed every kernel (`group_values`)
    :tries,      # values tried in a group (`try_add!`)
    :walked,     # dependency walks (`depends`)
    :terms,      # terms built (`built_term`)
    :trials,     # the cover's trials, each a plan of the whole graph unless refused before
    :compared,   # the cover's comparisons of a trial with the plan it changes
    :options,    # groups whose claims and estimates were found (`options`)
    :ordered,    # kernels looked at to order them (`order_groups`)
)

const COUNTS = Dict{Symbol, Threads.Atomic{Int}}(k => Threads.Atomic{Int}(0) for k in COUNTED)

# The phases timed, in the order each first ended: seconds, bytes allocated
# and seconds collecting garbage. Phases end on any task.
const PHASES = Pair{Symbol, Tuple{Float64, Int, Float64}}[]
const PHASES_LOCK = ReentrantLock()

"Whether the compiler's work is tallied or observed."
@inline watching() = TALLYING[] || OBSERVER[] !== nothing

"Count `n` more of `what`, while watched."
@inline tally!(what::Symbol, n::Integer = 1) = (watching() && Threads.atomic_add!(COUNTS[what], Int(n)); nothing)

"A point event, `name` with `x`, for the observer."
observe(name::Symbol, x = nothing) = (OBSERVER[] === nothing || heard(name, x); nothing)

# The observer called, what it throws dropped: an instrument does not take a
# preparation down.
function heard(name, what)
    o = OBSERVER[]
    o === nothing && return
    try
        o(name, what)
    catch
    end
    nothing
end

"""
`f()` as phase `name`: its boundaries heard by the observer, and while
tallying its seconds, the bytes allocated meanwhile on every thread and the
seconds collecting garbage added to the phase's, though `f` throw.
"""
function phase(f, name::Symbol)
    watching() || return f()
    start = mark()
    heard(name, :begin)
    try
        f()
    finally
        heard(name, :end)
        TALLYING[] && spend!(name, start)
    end
end

"Where a section timed by `lap!` starts, while tallying."
mark() = TALLYING[] ? (time_ns(), Base.gc_bytes(), Base.gc_num().total_time) : (UInt64(0), Int64(0), Int64(0))

"""
Section `name` of the enclosing phase, started at `start` (see `mark`), ends
on this task: the observer hears it, and while tallying its seconds, bytes
and seconds collecting garbage are added to its; the next section starts.
"""
function lap!(name::Symbol, start)
    watching() || return start
    heard(name, :lap)
    TALLYING[] || return start
    spend!(name, start)
    mark()
end

function spend!(name, start)
    now = mark()
    spent = (1.0e-9 * (now[1] - start[1]), now[2] - start[2], 1.0e-9 * (now[3] - start[3]))
    @lock PHASES_LOCK begin
        k = findfirst(p -> first(p) === name, PHASES)
        k === nothing ? push!(PHASES, name => spent) : (PHASES[k] = name => PHASES[k][2] .+ spent)
    end
    nothing
end

"""
    tallied(f) -> (result, counts, phases)

`f()` with the compiler's work tallied: what it counted, by name, and each
phase's and section's seconds, bytes allocated and seconds collecting
garbage, in the order each first ended. Each is inclusive: a phase's seconds
hold those of the phases within it, and of what other tasks did meanwhile,
whose bytes it counts too; so phases nested, as calibration in search, or
run on other tasks, as compile, add to more than the whole. One tally at a
time.
"""
function tallied(f)
    TALLYING[] && throw(ArgumentError("the compiler's work is tallied already"))
    foreach(c -> c[] = 0, Base.values(COUNTS))
    @lock PHASES_LOCK empty!(PHASES)
    TALLYING[] = true
    x = try
        f()
    finally
        TALLYING[] = false
    end
    x, Dict(k => COUNTS[k][] for k in COUNTED), @lock(PHASES_LOCK, copy(PHASES))
end
