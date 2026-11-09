# A preparation that runs behind a program already running: it decides and
# builds on a task of its own, and has the program's calls time its claims.

"""
    Search(work; enter) -> Search

A preparation run on a task of its own while the program it prepares runs:
`work(timer)` decides with `timer` (see `decide!`) and returns the
executable it builds, or nothing where the plan the program runs stands.
`enter()`, called first in the task, gives it the state on the target of the
task that made it (see `task_state`).

Its timings are not taken on its task: each is asked of the program's calls
(`time_asked!`), one a call, after the call's own work, so that a claim is
timed where the calls run and with none of the program's work beside it, and
the task never captures a graph beside the caller's work, which CUDA forbids
the device's synchronization during. `wait` takes the rest. A search that
throws warns, and leaves the program its plan; a timing that throws fails the
search, not the call that took it.
"""
struct Search
    task::Task
    # The timings asked for, each `(target, launch, answer)`; closed as the
    # search ends.
    asked::Channel{Any}
end

function Search(work; enter = () -> nothing)
    asked = Channel{Any}(Inf)
    timer = asking(asked)
    task = Threads.@spawn try
        enter()
        work(timer)
    catch err
        @warn "the search behind the first result failed: the program keeps the plan the estimates picked" exception = (err, catch_backtrace())
        nothing
    finally
        close(asked)
    end
    Search(task, asked)
end

# A search's timer: each timing asked of the calls, and waited for. What the
# timing runs is compiled first, on the search's task, so that the call that
# takes it times the claim and does not compile it.
function asking(asked::Channel{Any})
    function (target, launch)
        precompile(time_launch, (typeof(target), typeof(launch)))
        answer = Channel{Any}(1)
        put!(asked, (target, launch, answer))
        seconds = take!(answer)
        seconds isa Exception && throw(seconds)
        seconds
    end
end

"""
    time_asked!(s::Search)

Take one timing the search asked for, if one waits, on the calling task: a
call takes at most one, so that it costs at most one timing's slice more.
"""
function time_asked!(s::Search)
    ask = @lock s.asked isready(s.asked) ? take!(s.asked) : nothing
    ask === nothing || answer!(ask)
    nothing
end

# A timing asked for, taken on the calling task, its seconds or what it
# threw handed back: the search, not the call, fails by it.
function answer!((target, launch, answer))
    seconds = try
        timed(target, launch)
    catch err
        CapturedException(err, catch_backtrace())
    end
    put!(answer, seconds)
end

"Whether the search has ended, with or without an executable."
ended(s::Search) = istaskdone(s.task)

"""
    wait(s::Search) -> executable or nothing

Take the search's timings on the calling task until it ends; the executable
it built, or nothing.
"""
function Base.wait(s::Search)
    foreach(answer!, s.asked)
    fetch(s.task)
end
