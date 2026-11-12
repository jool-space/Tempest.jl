# A corpus program compiled measuring close calls with its search behind the
# first result (`compile`'s `measure = :close, search = :after`), then called until its plan changes and a
# few calls after: the seconds from `trace` to the first result and the
# kernels compiled before it; the calls, and the seconds, until the search
# ends, its settled decisions kept, and until the change; each call's
# seconds before and after it, the call that changes the plan among them;
# and the step on each plan. A third task, where asked, calls a
# small program of its own meanwhile, and its slowest call says what the
# timings cost a call beside them. Run it in a new process, cold with an
# empty cache and warm (with the GB10's rates in TEMPEST_CACHE/tuning where
# Tempest ships none):
#   TEMPEST_CACHE=dir JULIA_OBJCACHE_PATH=dir/objcache julia --threads=4 --project=test test/bench/tiers.jl PROGRAM [SHAPE] [--before|--defaults] [--beside]
# `--before` compiles measuring close calls with `search = :before`, the
# search's default where something is measured, and calls as many times;
# `--defaults` compiles as `compile` does unless asked, which measures
# nothing and searches nothing unless the environment says otherwise. `--wait` calls `wait(c)` after the first result, which takes
# the search's timings until it ends, and prints its seconds. With
# TEMPEST_CORPUS_EXPLAIN=path, the first plan's `explain` is written there and
# the searched one's, where the plan changed, to path.searched. The calls are
# the step on the inputs as given, its results discarded: the plan, not the
# training, is what is measured.
import cuBLAS, cuTile
using Tempest, CUDACore, Statistics
const C = Tempest.Compiler
const CORPUS = joinpath(@__DIR__, "..", "corpus")

m = Core.eval(Main, :(module Program end))
Base.include(m, joinpath(CORPUS, ARGS[1] * ".jl"))
Base.include(m, joinpath(CORPUS, "measure.jl"))

milliseconds(t) = round(1.0e3 * t; sigdigits = 3)

# A call's seconds, its results on the host.
function timed_call(c, args)
    started = time()
    c(args...)
    CUDACore.synchronize()
    time() - started
end

# A small program called in a loop on a task of its own until `done[]`: its
# calls' seconds.
function beside(done)
    x = CUDACore.rand(Float32, 1024)
    small = compile(x -> sum(abs2, 2.0f0 .* x), x)
    small(x)
    Threads.@spawn begin
        seconds = Float64[]
        while !done[]
            started = time()
            small(x)
            push!(seconds, time() - started)
        end
        seconds
    end
end

function main(name, shape, before, defaults, besides, waiting)
    case = Base.invokelatest(m.instance, shape)
    args = map(m.device, case.args)
    counts, started = C.cache_counts(), time()
    c = defaults ? compile(case.step, args...; target = case.target) :
        compile(case.step, args...; target = case.target, measure = :close, search = before ? :before : :after)
    timed_call(c, args)
    first_s, now = time() - started, C.cache_counts()
    executable(c) = c isa Tempest.Searching ? c.compiled.executable : c.executable
    search = c isa Tempest.Searching ? c.search : nothing
    first_plan = executable(c)
    reached = C.reached_counts()
    println(name, " ", shape, defaults ? ", as compile does unless asked" : before ? ", searched before" : ", searched after", ": first result ", round(first_s; digits = 1),
        " s, ", now.compiled - counts.compiled, " kernels compiled, ", length(first_plan.kernels), " in the plan; deciders covered ",
        reached.covered, ", replayed ", reached.replayed, "; a ", nameof(typeof(c)), search === nothing ? ", nothing to search" : "")
    if waiting
        t = time()
        wait(c)
        println("  wait(c) ", round(time() - t; digits = 1), " s, ", C.cache_counts().compiled - now.compiled, " kernels compiled")
    end
    done = Ref(false)
    other = besides ? beside(done) : nothing
    # Calls until the plan changes, or the search ends on the same plan, and
    # twenty more; at most ten minutes.
    calls, at, changed, ended = Float64[], nothing, nothing, nothing
    counts = C.cache_counts()
    while time() - started < 600
        push!(calls, timed_call(c, args))
        if ended === nothing && search !== nothing && C.ended(search)
            ended = (length(calls), time() - started - first_s)
        end
        if changed === nothing && executable(c) !== first_plan
            changed, at = length(calls), time() - started - first_s
        end
        searching = c isa Tempest.Searching && c.search !== nothing
        (changed !== nothing || !searching) && length(calls) >= something(changed, 0) + 20 && break
    end
    done[] = true
    now = C.cache_counts()
    before_change = calls[1:(something(changed, length(calls) + 1) - 1)]
    after_change = changed === nothing ? Float64[] : calls[(changed + 1):end]
    ended === nothing || println("  search ended, its settled decisions kept, by call ", ended[1], ", ", round(ended[2]; digits = 1),
        " s after the first result; deciders covered ", C.reached_counts().covered - reached.covered, " and replayed ",
        C.reached_counts().replayed - reached.replayed, " in the search")
    if changed === nothing
        println("  no change of plan in ", length(calls), " calls")
    else
        println("  plan changed at call ", changed, ", ", round(at; digits = 1), " s after the first result, ",
            now.compiled - counts.compiled, " kernels compiled meanwhile; that call ", milliseconds(calls[changed]), " ms")
    end
    spread(seconds) = string("mean ", milliseconds(mean(seconds)), " ms, median ", milliseconds(median(seconds)), " ms, slowest ",
        milliseconds(maximum(seconds)), " ms")
    isempty(before_change) || println("  calls before: ", spread(before_change))
    isempty(after_change) || println("  calls after: ", spread(after_change))
    if other !== nothing
        seconds = fetch(other)
        println("  a small program beside: ", length(seconds), " calls, ", spread(seconds))
    end
    # The step on each plan, now that the calls are done with them.
    step_first = Base.invokelatest(m.step_milliseconds, first_plan)
    step_last = executable(c) === first_plan ? step_first : Base.invokelatest(m.step_milliseconds, executable(c))
    println("  step ", round(step_first; sigdigits = 3), " ms on the first plan, ", round(step_last; sigdigits = 3), " ms on the last")
    path = get(ENV, "TEMPEST_CORPUS_EXPLAIN", "")
    isempty(path) || open(io -> C.explain(io, first_plan), path, "w")
    isempty(path) || executable(c) === first_plan || open(io -> C.explain(io, executable(c)), path * ".searched", "w")
end

main(ARGS[1], Symbol(get(filter(a -> !startswith(a, "--"), ARGS), 2, "small")), "--before" in ARGS, "--defaults" in ARGS, "--beside" in ARGS,
    "--wait" in ARGS)
