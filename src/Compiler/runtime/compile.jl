# The front door: a function traced and prepared once, then called like the
# function. `prepare` and `execute!` are the level beneath it.

public compile, Compiled, Searching

"""
    compile(f, args...; target, measure = :none, search = :before, capture = true) -> Compiled or Searching

`f` traced on `args` and prepared, to be called like `f`: `compile(f, x)(y)`
computes `f(y)` for any `y` alike `x` in structure, shapes and element types.
It returns what `f` returns, in arrays of its own, with numbers as numbers,
and what `f` writes into an argument's arrays it leaves there, as `f` would.

The arrays among the arguments are inputs, those `f` holds or captures among
them, read and written where the caller has them; numbers are constants. A
call that differs from the one compiled in anything but its inputs' values
throws, naming the difference.

- `target`: where the program runs; that of the first array among `f` and
  the arguments that has one, else the CPU's.
- `measure`: which claims are timed: `:none`, the default, none, so that the
  plan is the one the estimates pick; `:close`, the options whose estimates
  are close; `:all`, every option. `true` and `false` mean `:close` and
  `:none`.
- `search`: when they are timed: `:before`, the default, while preparing, so
  that the first call runs the plan measurement picks; `:after`, behind the
  first result, where `compile` returns a `Searching`.
- `capture`: whether the program's launches are recorded once as one graph
  that each call replays. Results are the same either way.

The environment overrides the defaults of `measure` and `search`,
`TEMPEST_MEASURE` (`none`, `close` or `all`) and `TEMPEST_SEARCH` (`before`
or `after`), and a keyword given overrides the environment.

```jldoctest
julia> c = compile(x -> sum(abs2, x), [0.0, 0.0, 0.0, 0.0]);

julia> c([1.0, 2.0, 3.0, 4.0])
30.0
```

The manual's page [Compiling and calling](@ref) says which leaves are inputs,
how globals are read, where the program keeps its arrays, what a call must
match, and how measuring and searching go. The level beneath is in
`Tempest.Compiler`: `prepare`, `bind!`, and `execute!` or `replay!`.
`c.executable` is the prepared program, and `explain(c)` shows it.
"""
function compile(f, @nospecialize(args...); target = default_target((f, args...)), measure = measuring(), search::Symbol = searching(),
        capture::Bool = true)
    search in (:before, :after) || throw(ArgumentError("search is :before or :after, not :$search"))
    refuse_devices(target, (f, args...), leaf_paths)
    passed = argument_arrays((f, args...))
    globals = read_globals(f, args, passed)
    program, met = trace_meeting(call, f, args...)
    refuse_held(met, (f, args...), passed)
    behind = search === :after && level(measure) !== :none && measurable(target)
    ex = prepare(program, f, args...; target, measure = behind ? :none : measure, point = true)
    capture && capture!(ex)
    c = Compiled(f, ex, signature((f, args...), leaves(program.input_structure)), globals)
    behind && !replays(ex, measure) || return c
    Searching(c, Search(timer -> searched(program, (f, args...), target, measure, timer); enter = task_state(target)))
end

# Whether a preparation's decisions are those a search measuring as
# `measure` says would reach: a settled set it replayed, reached so.
replays(ex::Executable, measure) = ex.reached.settled && ex.reached.how === level(measure)

# The program prepared again behind its first result, measuring as `measure`
# says, its timings by `timer`, its arguments bound as `compile` binds them.
function searched(program, inputs, target, measure, timer)
    s = setup(program; target, placement = argument_devices(target, program, inputs), measure)
    prepared(s, decide!(s; timer), inputs...; point = true)
end

# The call `f(args...)`, traced with `f` among its arguments, so that the arrays
# `f` holds are inputs as the arguments' are.
call(f, args...) = f(args...)

# The globals `f`'s own code reads by name that a call must find as they were:
# those not declared `const`, with their values when traced. One holding arrays
# throws, since the program would keep them as they were when traced, as a
# model in a global would be; `passed` are the arguments' arrays, to tell one
# passed too. What a function `f` calls reads is `refuse_held`'s.
function read_globals(f, args, passed)
    values = Pair{GlobalRef, Any}[]
    for g in named_globals(f, args)
        value = getglobal(g.mod, g.name)
        arrays = filter(is_input, leaves(value))
        isempty(arrays) || throw(ArgumentError(holds_arrays(g.name, any(x -> wraps(passed, x), arrays))))
        isconst(g.mod, g.name) || push!(values, g => value)
    end
    values
end

# The bindings `f`'s own code reads by name, other than Base's and Core's. A
# closure's code names a global where it would name a variable it captures.
function named_globals(f, args)
    refs = GlobalRef[]
    f isa Core.Builtin && return refs
    for code in code_lowered(f, map(Core.Typeof, args))
        foreach(s -> named!(refs, s), code.code)
    end
    filter(g -> isdefined(g.mod, g.name) && !(Base.moduleroot(Base.binding_module(g.mod, g.name)) in (Base, Core)), unique(refs))
end

named!(refs, x::GlobalRef) = push!(refs, x)
named!(refs, x::Expr) = foreach(a -> named!(refs, a), x.args)
named!(refs, x::Core.ReturnNode) = isdefined(x, :val) && named!(refs, x.val)
named!(refs, x::Core.GotoIfNot) = named!(refs, x.cond)
named!(refs, x) = nothing

holds_arrays(name, passed) = string("the function reads the global ", name, ", which holds arrays: compiled, it would ",
    "keep them as they were when traced. ", passed ? string(name, " is an argument too: read the argument rather than ",
    "the global") : string("Pass ", name, " as an argument, or bind it to a local the function captures"))

# The arguments' arrays, and those they wrap.
function argument_arrays(args)
    arrays = Base.IdSet{Any}()
    for x in leaves(args)
        is_input(x) || continue
        while true
            push!(arrays, x)
            p = parent(x)
            p === x && break
            x = p
        end
    end
    arrays
end

# Whether `x`, or an array it wraps, is among `arrays`.
function wraps(arrays, x)
    while true
        x in arrays && return true
        p = parent(x)
        p === x && return false
        x = p
    end
end

# The arrays tracing met as constants that live in a global, refused as those
# `f`'s own code names are: what a function `f` calls reads, seen by its arrays
# rather than its name. The globals of every module loaded are looked through,
# only when tracing met an array. A global a function assigned while traced
# holds stand-ins of the program, and is refused first. An array met inside a
# leaf of the call `(f, args...)` that is no input, a dictionary or a matrix
# of layers, was reached through that argument, whatever global holds it too,
# and the argument is named.
function refuse_held(met, call, passed)
    isempty(met) && return
    held = held_arrays()
    if !isempty(held.traced)
        name = global_name(first(held.traced))
        throw(ArgumentError(string("the function assigned the global ", name, " while traced, which now holds ",
            "stand-ins of the program: assign ", name, " again, then pass it as an argument and assign what the ",
            "compiled function returns")))
    end
    hidden = hidden_arrays(call)
    for x in met
        within = holder(hidden.arrays, x)
        within === nothing || throw(ArgumentError(holds_hidden(within...)))
        at = holder(held.arrays, x)
        at === nothing || throw(ArgumentError(holds_arrays(global_name(at), wraps(passed, x))))
    end
end

holds_hidden(path, leaf) = string(path, ", a ", summary(leaf), ", holds arrays Tempest does not look inside: compiled, ",
    "the function would keep them as they were when traced. Hold them in tuples, named tuples, structs or vectors, ",
    "whose arrays are inputs")

# The arrays held inside the leaves of the call that are no inputs, each with
# the leaf's path and the leaf.
function hidden_arrays(call)
    hidden = Held(IdDict{Any, Tuple{String, Any}}(), Tuple{String, Any}[])
    seen = Base.IdSet{Any}()
    for (path, leaf) in zip(leaf_paths(call), leaves(call))
        is_input(leaf) || hold!(hidden, seen, Any[leaf], (path, leaf))
    end
    hidden
end

"""
The arrays that live in globals, or in a call's leaves, each with where it
lives, and the globals that hold stand-ins.
"""
struct Held{A}
    arrays::IdDict{Any, A}
    traced::Vector{A}
end

# The globals of every module loaded, of Main and of each package, and their
# modules within.
function held_arrays()
    held = Held(IdDict{Any, Tuple{Module, Symbol}}(), Tuple{Module, Symbol}[])
    seen, stack = Base.IdSet{Any}(), Any[]
    modules, done = Module[Main; Base.loaded_modules_array()], Base.IdSet{Module}()
    while !isempty(modules)
        m = pop!(modules)
        m in done && continue
        push!(done, m)
        for n in names(m; all = true)
            isdefined(m, n) && !Base.isdeprecated(m, n) || continue
            value = getglobal(m, n)
            if value isa Module
                parentmodule(value) === m && push!(modules, value)
            else
                push!(stack, value)
                hold!(held, seen, stack, (m, n))
            end
        end
    end
    held
end

# What a global's value holds, reached by fields as a dict's are, and by
# elements in an array of anything but bits; each mutable value once, which
# ends every cycle. Only a mutable array is found by identity: an immutable
# one, a range, is equal to any like it, and holds what it wraps. A value
# without fields, a function's, or of bits holds none, nor do Julia's own
# records of code, its method tables and docstrings.
function hold!(held, seen, stack, at)
    while !isempty(stack)
        x = pop!(stack)
        T = typeof(x)
        (!ismutabletype(T) && fieldcount(T) == 0 || isbitstype(T) || x isa CODE) && continue
        if x isa Union{Traced, TracedArray} || x isa AbstractArray && eltype(x) <: Traced
            at in held.traced || push!(held.traced, at)
            continue
        end
        if ismutable(x)
            x in seen && continue
            push!(seen, x)
        end
        if x isa AbstractArray && eltype(x) <: Number && ismutable(x)
            held.arrays[x] = at
        elseif x isa Union{Array, Core.Memory}
            isbitstype(eltype(x)) && continue
            for i in 1:min(length(x), 100_000)
                isassigned(x, i) && push!(stack, x[i])
            end
        else
            for i in 1:nfields(x)
                isdefined(x, i) && push!(stack, getfield(x, i))
            end
        end
    end
end

const CODE = Union{Module, Type, Symbol, AbstractString, Number, Ptr, Expr, Task, GlobalRef, Method, Core.MethodInstance,
    Core.CodeInstance, Core.CodeInfo, Core.MethodTable, Core.TypeMapEntry, Core.TypeMapLevel, Core.TypeName,
    Core.SimpleVector, Base.Docs.DocStr, Base.Docs.MultiDoc}

# A global of the user's code named as `f`'s own are, a package's with its module.
global_name((m, n)) = Base.moduleroot(m) === Main ? string(n) : string(m, ".", n)

# The global an array lives in, or one it wraps, a view or an adjoint, or `nothing`.
function holder(arrays, x)
    while true
        at = get(arrays, x, nothing)
        at === nothing || return at
        p = parent(x)
        p === x && return nothing
        x = p
    end
end

"""
A function traced and prepared by `compile`. Calls run one at a time, from
any task. `c.executable` is the prepared program the calls run.
"""
struct Compiled{F}
    f::F
    executable::Executable
    signature::Tuple
    globals::Vector{Pair{GlobalRef, Any}}
    # Held by each call: the calls share the executable's buffers.
    lock::ReentrantLock
end

Compiled(f, ex::Executable, signature, globals) = Compiled(f, ex, signature, globals, ReentrantLock())

function (c::Compiled)(@nospecialize(args...))
    call = (c.f, args...)
    # Its arrays' addresses are asked, its arrays bound and its results
    # read each on its device (see `on_each_device`), and its run puts each
    # step on its own (see `walk`).
    b = bound(c, call)
    @lock c.lock called(c.executable, b, call)
end

# A call of the compiled function, its arrays `b` as `bound` finds them, run
# by executable `ex`.
function called(ex::Executable, b::Passed, @nospecialize(call))
    point_each!(ex, b)
    ex.captured === nothing ? execute!(ex) : replay!(ex)
    returned(ex)
end

# The results of a call apart from the buffers the next call overwrites: a
# number where the program computed a number, else an array, though it have
# no dimensions. The arrays are copied first: each copy waits for the
# kernels on the device while the host goes on to allocate the next, where
# reading a number waits for them on the host, and the allocations after it,
# which grow the device's memory pool while the last call's results live,
# would wait in turn.
function returned(ex::Executable)
    scalar = ex.plan.graph.result_scalar
    out = ex.program.output_structure
    ks = leaves(out)
    # Each result read on its device.
    held = [i for (i, k) in enumerate(ks) if k isa Int]
    device(i) = ex.result_devices[ks[i]]
    arrays = Vector{Any}(nothing, length(ks))
    on_each_device(i -> scalar[ks[i]] || (arrays[i] = copy(ex.results[ks[i]])), device, ex.target, held)
    on_each_device(i -> scalar[ks[i]] && (arrays[i] = Array(ex.results[ks[i]])[]), device, ex.target, held)
    i = 0
    rebuild(out) do k
        i += 1
        k isa Int ? arrays[i] : constant_leaf(k)
    end
end

"""
    wait(c::Compiled) -> c

At once: a compiled function searches nothing behind its calls (see
`compile`'s `search`, which returns a `Searching` that does).
"""
Base.wait(c::Compiled) = c

"""
A function compiled with its search behind the first result (see
`compile`'s `search`), called as a `Compiled` is: `s.compiled` is the one
the calls run, its plan the estimates' until the search ends and the first
call after runs the search's, where it differs. Each call takes one of the
search's timings after its own work; `wait(s)` takes the rest.
"""
mutable struct Searching{F}
    compiled::Compiled{F}
    # The search, until a call takes what it built.
    search::Union{Nothing, Search}
end

function (s::Searching)(@nospecialize(args...))
    @lock s.compiled.lock begin
        changed!(s)
        results = s.compiled(args...)
        s.search === nothing || time_asked!(s.search)
        results
    end
end

# The calls given the search's plan once it has ended, under the calls'
# lock: captured in the calling task where the calls replay a graph, as
# `compile` captures the first, unless its plan is the one the calls run.
# The old plan's graph is destroyed once the device has run its last
# launches; its buffers go with its executable. A capture that fails leaves
# the calls their plan, with a warning.
function changed!(s::Searching)
    search = s.search
    search !== nothing && ended(search) || return
    s.search = nothing
    ex = fetch(search.task)
    c = s.compiled
    old = c.executable
    (ex === nothing || same_choices(old, ex)) && return
    try
        old.captured === nothing || capture!(ex)
    catch err
        @warn "the plan the search found could not be captured: the program keeps the plan it runs" exception = (err, catch_backtrace())
        return
    end
    s.compiled = Compiled(c.f, ex, c.signature, c.globals, c.lock)
    quiesce(old.target)
    release!(old)
    nothing
end

# Whether two executables of one program run the same plan: the same groups,
# each run by the same choice.
same_choices(a::Executable, b::Executable) = plan_choices(a) == plan_choices(b)
plan_choices(ex::Executable) = [(canonical(ex.plan, g), x === nothing ? "" : x.choice.key) for (g, x) in zip(ex.plan.groups, ex.decisions)]

"""
    wait(s::Searching) -> s

Take the timings of the search behind `s`'s first result until it ends, and
give the calls its plan where it differs (see `compile`'s `search`); at once
where the calls have taken it.
"""
function Base.wait(s::Searching)
    search = s.search
    search === nothing && return s
    wait(search)
    @lock s.compiled.lock changed!(s)
    s
end

"""
An input as a call must match it, its element type and its size, and the
argument of the program it is.
"""
struct InputType
    eltype::Type
    dims::Tuple
    argument::Int
end

# The call `(f, args...)` as a call must match it: each input by its type,
# with the argument of the program it is, as `positions`, the leaves of the
# program's input structure, number them; any other leaf as it is.
function signature(call, positions)
    i = 0
    structure_of(call) do x
        i += 1
        is_input(x) ? InputType(eltype(x), size(x), positions[i]) : x
    end
end

# The arrays of the call `(f, args...)`, one per argument of the program,
# found by one walk of its leaves beside the signature, which checks each
# leaf as it finds it; the call's other leaves and the globals `f` reads
# are checked against what was traced. A call that differs throws, naming
# where.
function bound(c::Compiled, call)
    s = c.signature
    n, m = length(s) - 1, length(call) - 1
    n == m || throw(ArgumentError(compiled_for(c, string("it takes ", n, n == 1 ? " argument" : " arguments", ", not ", m))))
    b = Passed(c.executable)
    for k in eachindex(s)
        d = match!(b, s[k], call[k])
        d === nothing || throw(ArgumentError(compiled_for(c, differs(d, k == 1 ? "f" : "argument $(k - 1)"))))
    end
    isempty(b.elsewhere) || refuse_elsewhere(c.executable, b, leaves(c.executable.program.input_structure), call, leaf_paths)
    for (g, value) in c.globals
        now = getglobal(g.mod, g.name)
        isequal(now, value) ||
            throw(ArgumentError(compiled_for(c, string("the global ", g.name, " is ", shown(now), ", not ", shown(value), " as when traced"))))
    end
    located!(b, c.executable, call, leaf_paths, string(c.f))
end

compiled_for(c, difference) = string(c.f, " was compiled for other arguments: ", difference)

"Where a call differs from its signature: what it has there, what the signature has, and the path to them, innermost first."
struct Mismatch
    expected::Any
    got::Any
    path::Vector{String}
end

# The mismatch `d` within `segment` of what holds it.
within(d::Mismatch, segment) = (push!(d.path, segment); d)

# Mismatch `d` in words, at the argument named `name`.
differs(d::Mismatch, name) = string(name, join(reverse(d.path)), " is ", describe(signature_of(d.got)), ", not ", describe(d.expected))

# A part of a call as a signature would have it, its inputs no argument's,
# to describe it.
signature_of(@nospecialize(x)) = structure_of(y -> is_input(y) ? InputType(eltype(y), size(y), 0) : y, x)

# Leaf by leaf, `x`, a part of a call, against `e`, the signature's part
# there: each input is placed among `b`'s arrays at its argument, and the
# first leaf or structure that differs is returned, a `Mismatch`, else
# `nothing`. A struct matches one of its kind, a vector of other things than
# numbers one of its length, tuples and named tuples their like, and any
# other leaf one equal to it. The walk takes both unspecialized, as `leaves`
# does, reading types by their parameters, a named tuple's names and a
# vector's elements', and only an input's check is compiled for its type: a
# model's hundreds of structs and arrays walked in well under a microsecond
# each, where building the call's structure to compare it took several.
Base.@nospecializeinfer function match!(b::Passed, @nospecialize(e), @nospecialize(x))
    if e isa InputType
        placed!(b, e, x) || return Mismatch(e, x, String[])
    elseif e isa Tuple
        x isa Tuple && nfields(x) == nfields(e) || return Mismatch(e, x, String[])
        for k in 1:nfields(e)
            d = match!(b, getfield(e, k), getfield(x, k))
            d === nothing || return within(d, "[$k]")
        end
    elseif e isa NamedTuple
        x isa NamedTuple && typeof(x).parameters[1] === typeof(e).parameters[1] || return Mismatch(e, x, String[])
        for k in 1:nfields(e)
            d = match!(b, getfield(e, k), getfield(x, k))
            d === nothing || return within(d, ".$(keys(e)[k])")
        end
    elseif e isa Parts
        T, parts = typeof(e).parameters[1], e.parts
        if T <: Vector
            x isa Vector && !(typeof(x).parameters[1] <: Number) && length(x) == nfields(parts) || return Mismatch(e, x, String[])
            for k in 1:nfields(parts)
                d = match!(b, getfield(parts, k), x[k])
                d === nothing || return within(d, "[$k]")
            end
        else
            Base.typename(typeof(x)) === Base.typename(T) || return Mismatch(e, x, String[])
            for k in 1:nfields(parts)
                d = match!(b, getfield(parts, k), getfield(x, k))
                d === nothing || return within(d, ".$(keys(parts)[k])")
            end
        end
    else
        isequal(e, x) || return Mismatch(e, x, String[])
    end
    nothing
end

# Whether `x` is the input `e` describes, placed among `b`'s arrays at its
# argument if it is: one call a leaf, specialized on the leaf's type.
function placed!(b::Passed, e::InputType, x)
    is_input(x) && eltype(x) === e.eltype && size(x) === e.dims || return false
    place!(b, e.argument, x)
    true
end

shown(x) = sprint(show, x; context = :limit => true)

# The path of each leaf of the call `(f, args...)`, as a difference names it,
# in the order `leaves` gives them.
leaf_paths(call) = foldl((out, (k, x)) -> leaf_paths!(out, x, k == 1 ? "f" : "argument $(k - 1)"), enumerate(call); init = String[])
leaf_paths!(out, x::Tuple, path) = foldl((out, (k, y)) -> leaf_paths!(out, y, "$path[$k]"), enumerate(x); init = out)
leaf_paths!(out, x::NamedTuple, path) = foldl((out, k) -> leaf_paths!(out, x[k], "$path.$k"), keys(x); init = out)
leaf_paths!(out, x, path) = nested(x) ? leaf_paths!(out, parts(x), path) : push!(out, path)

describe(x::InputType) = string("a ", x.eltype, " array of size ", x.dims)
describe(x::Parts{T}) where {T} = string("a ", nameof(T))
describe(x::Parts{<:Vector}) = string("a vector of ", length(x.parts))
describe(x::Tuple) = string("a tuple of ", length(x))
describe(x::NamedTuple) = string("a named tuple of ", join(keys(x), ", "))
describe(x) = string("the constant ", sprint(show, x; context = :limit => true))

function Base.show(io::IO, c::Compiled)
    ex = c.executable
    print(io, "Compiled(", c.f, ": ", flow(ex.program), "; ", ex.target, ", ", counted(length(ex.kernels), "kernel"), ")")
end

function Base.show(io::IO, s::Searching)
    search = s.search
    print(io, "Searching(", s.compiled, search === nothing ? "" : ended(search) ? ", searched" : ", searching", ")")
end

explain(io::IO, c::Compiled) = explain(io, c.executable)
explain(io::IO, s::Searching) = explain(io, s.compiled)
explain(c::Union{Compiled, Searching}) = explain(stdout, c)
