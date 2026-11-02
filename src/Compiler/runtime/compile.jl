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
option; `true` and `false` mean `:close` and `:none`.

Which leaves of the arguments are inputs follows `trace`: arrays, also inside
tuples, named tuples and structs, mutable ones too. Numbers are constants; a
number that varies between calls is an array, `[x]` say. `f` is an argument
like the others, the call `f(args...)` traced: the arrays of a callable
struct's fields and those a closure captures are inputs too, read at each
call, so a closure over a model sees its weights as they are when called.

A global that `f`'s own code reads by name is no part of `f`, and tracing reads
it once. One holding arrays is refused, since the program would keep them as
they were when traced: pass it as an argument. Any other is a constant, and
one not declared `const` is checked at each call. A global read deeper, by a
function `f` calls, is not seen, and kept as it was when traced.

A call whose arguments, `f`'s own leaves or the globals `f` reads differ in
anything but the values of inputs throws, naming the difference.

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
function compile(f, @nospecialize(args...); target = default_target((f, args...)), measure = :close, capture::Bool = true)
    globals = read_globals(f, args)
    ex = prepare(trace(call, f, args...), f, args...; target, measure)
    capture && capture!(ex)
    Compiled(f, ex, signature((f, args...)), globals)
end

# The call `f(args...)`, traced with `f` among its arguments, so that the arrays
# `f` holds are inputs as the arguments' are.
call(f, args...) = f(args...)

# The globals `f`'s own code reads by name that a call must find as they were:
# those not declared `const`, with their values when traced. One holding arrays
# throws, since the program would keep them as they were when traced, as a
# model in a global would be. What a function `f` calls reads is not seen.
function read_globals(f, args)
    values = Pair{GlobalRef, Any}[]
    for g in named_globals(f, args)
        value = getglobal(g.mod, g.name)
        any(is_input, leaves(value)) && throw(ArgumentError(string("the function reads the global ", g.name,
            ", which holds arrays: compiled, it would keep them as they were when traced. Pass ", g.name,
            " as an argument, or bind it to a local the function captures")))
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
    bind!(ex, c.f, args...)
    ex.captured === nothing ? execute!(ex) : replay!(ex)
    # What the program wrote into an input, the caller's array takes.
    if !isempty(ex.program.writes)
        xs = argument_leaves(ex.program, (c.f, args...))
        foreach(((_, a),) -> copyto!(vec(xs[a]), ex.arguments[a]), ex.program.writes)
    end
    rebuild(k -> k isa Int ? fresh(ex.results[k], ex.plan.graph.result_scalar[k]) : constant_leaf(k), ex.program.output_structure)
end

# A result apart from the buffers the next call overwrites: a number where the
# program computed a number, else an array, though it have no dimensions.
fresh(x, scalar) = scalar ? Array(x)[] : copy(x)

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
inner(x::Fields) = x.fields
inner(x) = nothing

kind(::Tuple) = Tuple
kind(::NamedTuple) = NamedTuple
kind(::Fields{T}) where {T} = Base.typename(T).wrapper

describe(x::InputType) = string("a ", x.eltype, " array of size ", x.dims)
describe(x::Fields{T}) where {T} = string("a ", nameof(T))
describe(x::Tuple) = string("a tuple of ", length(x))
describe(x::NamedTuple) = string("a named tuple of ", join(keys(x), ", "))
describe(x) = string("the constant ", sprint(show, x; context = :limit => true))

function Base.show(io::IO, c::Compiled)
    ex = c.executable
    print(io, "Compiled(", c.f, ": ", flow(ex.program), "; ", ex.target, ", ", counted(length(ex.kernels), "kernel"), ")")
end

explain(io::IO, c::Compiled) = explain(io, c.executable)
explain(c::Compiled) = explain(stdout, c)
