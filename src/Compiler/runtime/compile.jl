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

An argument's array is the program's storage for it, read and written where
the caller has it, not copied: a call on the arrays of the call before costs
nothing for them, and one on others changes the addresses the captured
graph's kernels read and write, in place (see `Tempest.Compiler.point!`), so
a training step whose weights are new arrays at every call copies none of
them. What `f` writes into an argument the kernels write into the caller's
array: where the kernel that writes it reads the array only at the
addresses it writes, and the kernels that read it otherwise can run first,
directly, as an append to a cache writes only its new positions; else into
storage of the program's own and then copied into the array once, after the
last kernel (`explain` says which). Everything else the program stores lives
in buffers it owns and reuses at every call. An array not of the target's
array type, an `Array` for a function compiled for the GPU say, is copied
into the program's storage and, if written, back after the call.

A call whose arguments, `f`'s own leaves or the globals `f` reads differ in
anything but the values of inputs throws, naming the difference. So does a
call whose arrays share memory with one `f` writes: the program reads each
array apart, where `f` run on them would read what it wrote.

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

With `capture`, the program's launches are recorded once as one graph that
each call replays, as `capture!` and `replay!` do. Results are the same either
way.

The level beneath is in `Tempest.Compiler`: `prepare`, `bind!`, and
`execute!` or `replay!`, with `point!` binding a call's arrays as `compile`
does. `c.executable` is the prepared program, and `explain(c)` shows it.
"""
function compile(f, @nospecialize(args...); target = default_target((f, args...)), measure = measuring(), capture::Bool = true)
    passed = argument_arrays((f, args...))
    globals = read_globals(f, args, passed)
    program, met = trace_meeting(call, f, args...)
    refuse_held(met, (f, args...), passed)
    ex = prepare(program, f, args...; target, measure)
    capture && capture!(ex)
    Compiled(f, ex, signature((f, args...), leaves(program.input_structure)), globals)
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
    call = (c.f, args...)
    b = bound(c, call)
    ex = c.executable
    isempty(ex.program.writes) || refuse_shared(ex, b, call)
    point_each!(ex, b)
    ex.captured === nothing ? execute!(ex) : replay!(ex)
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
    b = Passed(length(c.executable.program.arguments))
    for k in eachindex(s)
        d = match!(b, s[k], call[k])
        d === nothing || throw(ArgumentError(compiled_for(c, differs(d, k == 1 ? "f" : "argument $(k - 1)"))))
    end
    for (g, value) in c.globals
        now = getglobal(g.mod, g.name)
        isequal(now, value) ||
            throw(ArgumentError(compiled_for(c, string("the global ", g.name, " is ", shown(now), ", not ", shown(value), " as when traced"))))
    end
    b
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

# A call whose arrays share memory with one the program writes throws: the
# program reads each array as one of its own, which the function itself,
# run on the same arrays, would not, as `f!(x, x)` writing its first
# argument and reading its second reads what it wrote. Which arrays may
# share memory is Base's `mightalias`, as broadcasting's. `b` holds the
# call's arrays.
function refuse_shared(ex::Executable, b::Passed, call)
    xs = b.arrays
    for (_, a) in ex.program.writes, k in eachindex(xs)
        k == a || !Base.mightalias(xs[a], xs[k]) || refuse_sharing(ex, call, k, a)
    end
end

# Argument `k` of the call shares memory with argument `a`, which the program writes.
function refuse_sharing(ex, call, k, a)
    paths, positions = leaf_paths(call), leaves(ex.program.input_structure)
    path(j) = paths[findfirst(==(j), positions)]
    throw(ArgumentError(string(path(k), " shares memory with ", path(a), ", which ", call[1],
        " writes: a compiled call reads its arrays apart, so pass arrays that share none")))
end

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

explain(io::IO, c::Compiled) = explain(io, c.executable)
explain(c::Compiled) = explain(stdout, c)
