# The tables a session keeps: answers to questions whose key holds everything
# the answer was derived from that can differ between two questions in one
# session (see `SessionTable`). Such a table only saves time: emptied at any
# moment, even while a preparation on another task reads it, it costs that
# time again and never gives another answer. What a preparation keeps of its
# own program (interned terms, matches) is not one of them: `prepare` empties
# those as it starts, which keeps them its own while one preparation runs at a
# time. Nor is a device resource, released rather than forgotten, nor a store
# on disk, which `TEMPEST_CACHE` and `JULIA_OBJCACHE_PATH` point elsewhere.

"""
    SessionTable{K, V}()

A table of answers kept for the session, read and filled under its own lock:
how the compiler and the packages that implement its targets keep what they
find once a session. A table is one only if its key holds everything its
answer was derived from that can differ between two questions in one
session, so that emptying it (`forget_session!`) costs time and never a wrong
plan; any other table of theirs is named, with why, in `NOT_SESSION_TABLES`.
A key holds, of what can differ:

  - code, named as the stores' content keys name it (`definitions`): a
    package's by the function itself, since no code changes within a session
    (below); any other, a function of `Main` or a closure over one, by its
    definition, which holds the values of the globals it reads, which change
    without a definition;
  - the device, by what the answer read of it: its architecture, its context;
  - the environment's settings the answer read, and what the preparation that
    asked was asked, as whether it compiles (`Implement.compiling`);
  - content, as a kernel's, by its content key.

Answers are their session's: a table filled in another process, as a
package's precompile workload fills its tables in the package's image,
compiled on another machine perhaps and compiling nothing, holds none in the
session that loads it, and a session ends when code is defined that is not
Tempest's own (`session`). Revise evaluating an edited file, an `@eval` into
a package and a function redefined at the prompt are one event to Julia, a
world after the last, and end a session alike, so that the code a table's
answer was derived from is the code the process runs. Tempest's own
definitions, its kernels' functions, which every build makes, leave it
standing (`own_definition`).


An answer missing is found outside the lock and kept with
`get!(table, key, answer, asked)`, `asked` the session read as the question
was asked (`session`), the first kept winning where two tasks found one at
once: the same answer found twice. An answer is its session's: one found
while the session ended, by a definition on another thread, is of the code
before and is returned to its caller but not kept. `get!(f, table, key)`
does all three, finding it by `f()`; a caller whose `f` would be a closure
over a value of its own type each time, a kernel's function say, reads the
session, looks up with `get` and keeps with `get!`, so as not to compile a
closure for each. Where finding an
answer twice is more than waste, as a kernel's compilation is, making it
once is the table's own business: its value is what makes it, created under
the lock (`cached_kernel`).
"""
struct SessionTable{K, V}
    table::Dict{K, V}
    lock::ReentrantLock
    # The session its answers are of.
    session::Threads.Atomic{UInt}
end

SessionTable{K, V}() where {K, V} = SessionTable{K, V}(Dict{K, V}(), ReentrantLock(), Threads.Atomic{UInt}(SESSION[]))

"""
The session this process is in: begun as Tempest loads, and again whenever
code is defined that is not Tempest's own (`session`), other in every process.
A table whose answers are of another session, as those a package image holds
are, holds none in this one.
"""
const SESSION = Threads.Atomic{UInt}(0)

# The world the session knows of: the world counter as the session began or
# as Tempest's last own definition left it. A world past it is code defined
# by someone else since.
const KNOWN_WORLD = Threads.Atomic{UInt}(0)

# The world as Tempest's definition in flight began, zero while none is.
const DEFINING_SINCE = Threads.Atomic{UInt}(0)

# The fewest worlds one of Tempest's definitions has made in this process:
# what one makes, learned rather than written down, since what a definition
# counts (a method, a binding for its function's type) is Julia's to change.
# Unknown until the first, which ends the session as though someone else's
# had been among its worlds.
const OWN_WORLDS = Threads.Atomic{UInt}(typemax(UInt))

# Tempest's definitions, one at a time, so that the worlds of each are
# counted alone.
const DEFINING = ReentrantLock()

# Held while a session begins, so that a world past the known one begins one.
const BEGINNING = ReentrantLock()

# As Tempest loads: the process's first session, its own definitions' worlds
# not yet counted, whatever the process that wrote its image counted.
start_session!() = (OWN_WORLDS[] = typemax(UInt); begin_session!())

# A session differs from that of any process before it, the one that wrote
# the package images it loads among them, and from any before it in the
# process. The world is known after the session is set, so that whoever
# reads the known world reads the session of it.
function begin_session!()
    world = Base.get_world_counter()
    SESSION[] = hash((getpid(), time_ns(), world))
    KNOWN_WORLD[] = world
    nothing
end

"""
    session() -> UInt

The session whose answers hold now: the one before, unless code was defined
since that is not Tempest's own, which begins another. Julia counts a world
for every definition, of a method, a type or a constant, by whatever made it;
the world counter is read once per question. A question asked while one of
Tempest's definitions makes its worlds, some microseconds, waits for that one
to end, which counts them (`own_definition`), and for no other.
"""
function session()
    world = Base.get_world_counter()
    world == KNOWN_WORLD[] && return SESSION[]
    since = DEFINING_SINCE[]
    if since != 0
        while DEFINING_SINCE[] == since
            yield()
        end
        return session()
    end
    @lock BEGINNING moved!()
    SESSION[]
end

# Under `BEGINNING`: a world past the known one begins a session.
function moved!()
    Base.get_world_counter() == KNOWN_WORLD[] || begin_session!()
    nothing
end

"""
    own_definition(into, source) -> what evaluating `source` in `into` returns

Define code of Tempest's own, as a generated kernel's function is
(`generated`), leaving the session standing. Every build defines its
kernels' functions, so a session that ended at them would never keep an
answer to a later preparation; and a function defined once and never again,
as a kernel's is, changes no code an answer was derived from.

The source is lowered first, side by side with any other, and evaluated
lowered, one definition at a time: the worlds a definition makes are made in
that evaluation, some microseconds, rather than over the lowering, a
thousand times longer, so that a question asked meanwhile waits that long.
A definition by someone else before it ends the session, and so does one
made on another thread while it evaluates: the worlds between its start and
its end are more than one of Tempest's definitions makes (`OWN_WORLDS`).
Where they cannot be told apart, as for the first of Tempest's definitions,
the session ends.
"""
function own_definition(into::Module, source::Expr)
    lowered = Meta.lower(into, source)
    @lock DEFINING begin
        @lock BEGINNING moved!()
        before = KNOWN_WORLD[]
        DEFINING_SINCE[] = before
        try
            defined = Core.eval(into, lowered)
            made = Base.get_world_counter() - before
            fewest = OWN_WORLDS[]
            OWN_WORLDS[] = min(fewest, made)
            if made > fewest || fewest == typemax(UInt)
                @lock BEGINNING begin_session!()
            else
                KNOWN_WORLD[] = before + made
            end
            defined
        finally
            DEFINING_SINCE[] = 0
        end
    end
end

# The table's answers, under its lock: none if they are of another session.
function answers(t::SessionTable)
    now = session()
    if t.session[] != now
        empty!(t.table)
        t.session[] = now
    end
    t.table
end

struct Absent end

Base.get(t::SessionTable, key, default) = @lock t.lock get(answers(t), key, default)
function Base.get!(t::SessionTable, key, x, asked::UInt)
    @lock t.lock begin
        table = answers(t)
        t.session[] == asked ? get!(table, key, x) : x
    end
end

function Base.get!(f, t::SessionTable, key)
    asked = session()
    found = get(t, key, Absent())
    found isa Absent ? get!(t, key, f(), asked) : found
end

Base.empty!(t::SessionTable) = (@lock t.lock empty!(answers(t)); t)
Base.length(t::SessionTable) = @lock t.lock length(answers(t))

# The table-like globals of Tempest's modules that are not session tables,
# each with why. test/runtime/session.jl holds every one `tempest_globals`
# finds to being a `SessionTable` or named here.
const NOT_SESSION_TABLES = (
    "Tempest.Compiler.TALLYING" => "a switch: whether the compiler's work is tallied",
    "Tempest.Compiler.OBSERVER" => "a switch: what hears the phases",
    "Tempest.Compiler.TUNING" => "a setting: the tuning record's directory where TEMPEST_CACHE names none",
    "Tempest.Compiler.COUNTS" => "the tally's counts, which `tallied` resets",
    "Tempest.Compiler.PHASES" => "the tally's phases, which `tallied` empties",
    "Tempest.Compiler.GENERATED_SOURCES" => "definitions: the source a generated kernel's key holds",
    "Tempest.Compiler.STEP_LATENCIES" => "constants: the latencies the fit of a sweep's step searches",
    "Tempest.Compiler.BLOCK_SHARES" => "constants: the shares of the bandwidth the fit of a block's rate searches",
    "CUDAExt.OVERHEAD" => "a measurement, once a session: measured again it may differ",
    "CUDAExt.CLEARING" => "device resources: the launch per device that clears its cache before one is timed",
    "CUDAExt.GRAPHS" => "device resources: the graphs kept while their programs are",
    "cuDNNExt.GRAPHS" => "device resources: a graph built per context and problem",
)

# Tempest's modules and those of its extensions loaded: each module whose
# package is Tempest or one of its extensions (an extension's UUID is derived
# from its parent's and its name), with the modules within them.
function tempest_modules()
    root = Base.PkgId(ROOT).uuid
    tops = filter(Base.loaded_modules_array()) do m
        id = Base.PkgId(m)
        m === ROOT || (id.uuid !== nothing && id.uuid == Base.uuid5(root, id.name))
    end
    modules = Module[]
    within!(m) = m in modules || (push!(modules, m); foreach(within!, submodules(m)))
    foreach(within!, tops)
    modules
end

submodules(m::Module) = [x for x in (getglobal(m, n) for n in names(m; all = true) if isdefined(m, n)) if x isa Module && x !== m && parentmodule(x) === m]

# What may hold answers: a table, a set, an array or a reference.
const TABLE_LIKE = Union{SessionTable, AbstractDict, AbstractSet, AbstractArray, Base.RefValue}

"""
    tempest_globals() -> Vector{Pair{String, Any}}

Each table-like constant of Tempest's modules and of its extensions loaded, by
its full name: the session's tables among them, which are found here rather than
registered as each is defined. An extension's tables are made while it
precompiles, after Tempest's image is saved, so a list of Tempest's they
joined then would not hold them in the next session; and registering in each
extension's `__init__` is a step every new table would have to remember. The
walk is the registry: `forget_session!` empties what it finds, and the test of
the session's tables holds what else it finds to `NOT_SESSION_TABLES`.
"""
function tempest_globals()
    found = Pair{String, Any}[]
    for m in tempest_modules(), n in names(m; all = true)
        # Names Julia makes, as a docstring's table, begin with '#'. A table
        # is a constant: what is not, as the variables a precompile
        # workload's setup leaves in its module on Julia 1.11, is no table.
        (startswith(string(n), '#') || !isdefined(m, n) || !isconst(m, n)) && continue
        x = getglobal(m, n)
        x isa TABLE_LIKE || continue
        push!(found, join((fullname(m)..., n), '.') => x)
    end
    found
end

"""
    forget_session!()

Empty every table of answers the session keeps (its compiled kernels by
content, the functions loaded from them, which functions cuTile lowers, what
identifies each package's code, among others), so that a preparation after
finds each again, as one in a new process would. Julia's compiled code, the
device's resources and the stores on disk stay: a new `TEMPEST_CACHE` and
`JULIA_OBJCACHE_PATH` leave those behind too. The process's counts
(`cache_counts`, `reached_counts`) stay, which are read by their differences. Each table's answer is keyed by
everything it reads, so that emptying them costs time and never gives another
answer.
"""
forget_session!() = foreach(((_, t),) -> t isa SessionTable && empty!(t), tempest_globals())
