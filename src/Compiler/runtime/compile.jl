# The front door: a function traced and prepared once, then called like the
# function. `prepare` and `execute!` are the level beneath it.

public compile, Compiled

"""
    compile(f, args...; target, measure = :close, capture = true) -> Compiled

`f` traced on `args` and prepared, to be called like `f`: `compile(f, x)(y)`
computes `f(y)` for any `y` alike `x` in structure, shapes and element types.
It returns what `f` returns, in arrays of its own, with numbers as numbers,
and what `f` writes into an argument's arrays it leaves there, as `f` would.

`measure` says which claims are timed when preparing, as for `prepare`:
`:none` none, `:close` the options whose estimates are close, `:all` every
option; `true` and `false` mean `:close` and `:none`. The default is
`TEMPEST_MEASURE`'s where the environment sets it.

Which leaves of the arguments are inputs follows `trace`: arrays, also inside
tuples, named tuples, structs, mutable ones too, and vectors of other things
than numbers, whose lengths a call must match. An array the function reads
inside any other leaf, a dictionary's say, would be a constant, and is refused,
naming the argument. Numbers are constants; a
number that varies between calls is an array, `[x]` say. `f` is an argument
like the others, the call `f(args...)` traced: the arrays of a callable
struct's fields and those a closure captures are inputs too, read at each
call, so a closure over a model sees its weights as they are when called.

A global that `f` reads is no part of `f`, and tracing reads it once. One
holding arrays is refused, since the program would keep them as they were when
traced: pass it as an argument. In `f`'s own code a global is seen by its name;
in a function `f` calls, by its arrays: an array tracing meets that lives in a
global, of any module, is refused alike, as is a global a function assigned
while traced. Any other global `f`'s own code reads is a constant, and one not
declared `const` is checked at each call. A number a function `f` calls reads,
or a copy it makes of a global's array before using it with an input, is kept
as it was when traced.

An array the program only reads is read where the caller has it, not
copied: a call on the arrays of the call before costs nothing for them, and
one on others changes the addresses the captured graph's kernels read, in
place (see `Tempest.Compiler.point!`), so a training step whose weights
are new arrays at every call copies none of them.

A call whose arguments, `f`'s own leaves or the globals `f` reads differ in
anything but the values of inputs throws, naming the difference. So does a
call whose arrays share memory with one `f` writes: the program reads each
array apart, where `f` run on them would read what it wrote.

```jldoctest
julia> c = compile(x -> sum(abs2, x), [0.0, 0.0, 0.0, 0.0]);

julia> c([1.0, 2.0, 3.0, 4.0])
30.0
```

With `capture`, the program's launches are recorded once as one graph that
each call replays, as `capture!` and `replay!` do. Results are the same either
way.

For a step that updates its state where it is and replays without copies,
use the level beneath, in `Tempest.Compiler`: `prepare`, `bind!`, and
`execute!` or `replay!`. `c.executable` is the prepared program, and
`explain(c)` shows it.
"""
function compile(f, @nospecialize(args...); target = default_target((f, args...)), measure = measuring(), capture::Bool = true)
    passed = argument_arrays((f, args...))
    globals = read_globals(f, args, passed)
    program, met = trace_meeting(call, f, args...)
    refuse_held(met, (f, args...), passed)
    ex = prepare(program, f, args...; target, measure)
    capture && capture!(ex)
    Compiled(f, ex, signature((f, args...)), globals)
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

"A function traced and prepared by `compile`."
struct Compiled{F}
    f::F
    executable::Executable
    signature::Tuple
    globals::Vector{Pair{GlobalRef, Any}}
end

function (c::Compiled)(@nospecialize(args...))
    d = difference(c, args)
    d === nothing || throw(ArgumentError("$(c.f) was compiled for other arguments: $d"))
    ex = c.executable
    isempty(ex.program.writes) || refuse_shared(ex, (c.f, args...))
    point!(ex, c.f, args...)
    ex.captured === nothing ? execute!(ex) : replay!(ex)
    # What the program wrote into an input, the caller's array takes.
    if !isempty(ex.program.writes)
        xs = argument_leaves(ex.program, (c.f, args...))
        foreach(((_, a),) -> copyto!(vec(xs[a]), ex.arguments[a]), ex.program.writes)
    end
    # The results apart from the buffers the next call overwrites: a number
    # where the program computed a number, else an array, though it have no
    # dimensions. The arrays are copied first: each copy waits for the
    # kernels on the device while the host goes on to allocate the next,
    # where reading a number waits for them on the host, and the allocations
    # after it, which grow the device's memory pool while the last call's
    # results live, would wait in turn.
    scalar = ex.plan.graph.result_scalar
    out = ex.program.output_structure
    arrays = Any[k isa Int && !scalar[k] ? copy(ex.results[k]) : nothing for k in leaves(out)]
    i = 0
    rebuild(out) do k
        i += 1
        k isa Int ? (scalar[k] ? Array(ex.results[k])[] : arrays[i]) : constant_leaf(k)
    end
end

"An input as a call must match it: its element type and its size."
struct InputType
    eltype::Type
    dims::Tuple
end

# The arguments as a call must match them: inputs by type, any other leaf as it is.
signature(args) = structure_of(x -> is_input(x) ? InputType(eltype(x), size(x)) : x, args)

# Where a call differs from the compiled one, in words, or `nothing`: in its
# arguments, in `f`'s own leaves, or in a global `f` reads.
function difference(c::Compiled, args)
    d = difference(c.signature, signature((c.f, args...)))
    d === nothing || return d
    for (g, value) in c.globals
        now = getglobal(g.mod, g.name)
        isequal(now, value) || return string("the global ", g.name, " is ", shown(now), ", not ", shown(value), " as when traced")
    end
    nothing
end

shown(x) = sprint(show, x; context = :limit => true)

# A call whose arrays share memory with one the program writes throws: the
# program reads each array as one of its own, which the function itself,
# run on the same arrays, would not, as `f!(x, x)` writing its first
# argument and reading its second reads what it wrote. Which arrays may
# share memory is Base's `mightalias`, as broadcasting's.
function refuse_shared(ex::Executable, call)
    positions, xs = leaves(ex.program.input_structure), leaves(call)
    for (_, a) in ex.program.writes
        i = findfirst(==(a), positions)
        for (j, y) in enumerate(xs)
            (j == i || positions[j] === nothing || !Base.mightalias(xs[i], y)) && continue
            paths = leaf_paths(call)
            throw(ArgumentError(string(paths[j], " shares memory with ", paths[i], ", which ", call[1],
                " writes: a compiled call reads its arrays apart, so pass arrays that share none")))
        end
    end
end

# The path of each leaf of the call `(f, args...)`, as `difference` names it,
# in the order `leaves` gives them.
leaf_paths(call) = foldl((out, (k, x)) -> leaf_paths!(out, x, k == 1 ? "f" : "argument $(k - 1)"), enumerate(call); init = String[])
leaf_paths!(out, x::Tuple, path) = foldl((out, (k, y)) -> leaf_paths!(out, y, "$path[$k]"), enumerate(x); init = out)
leaf_paths!(out, x::NamedTuple, path) = foldl((out, k) -> leaf_paths!(out, x[k], "$path.$k"), keys(x); init = out)
leaf_paths!(out, x, path) = nested(x) ? leaf_paths!(out, parts(x), path) : push!(out, path)

# Where the call `got`, `f` and its arguments, differs from `expected`, in
# words, or `nothing`. The function is `f`, and the arguments are counted after it.
function difference(expected::Tuple, got::Tuple)
    n, m = length(expected) - 1, length(got) - 1
    n == m || return string("it takes ", n, n == 1 ? " argument" : " arguments", ", not ", m)
    for k in eachindex(expected)
        d = difference(expected[k], got[k], k == 1 ? "f" : "argument $(k - 1)")
        d === nothing || return d
    end
    nothing
end

function difference(expected, got, path)
    e, g = inner(expected), inner(got)
    if e !== nothing && g !== nothing && kind(expected) == kind(got) && keys(e) == keys(g)
        for k in keys(e)
            d = difference(e[k], g[k], expected isa Tuple ? "$path[$k]" : "$path.$k")
            d === nothing || return d
        end
        return nothing
    end
    isequal(expected, got) ? nothing : string(path, " is ", describe(got), ", not ", describe(expected))
end

inner(x::Union{Tuple, NamedTuple}) = x
inner(x::Parts) = x.parts
inner(x) = nothing

kind(::Tuple) = Tuple
kind(::NamedTuple) = NamedTuple
kind(::Parts{T}) where {T} = Base.typename(T).wrapper

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

explain(io::IO, c::Compiled) = explain(io, c.executable)
explain(c::Compiled) = explain(stdout, c)
