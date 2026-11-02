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
call, so a closure over a model sees its weights as they are when called. A
global that `f` reads is no part of `f`, and a constant, copied when traced.
A call whose arguments, or `f`'s own leaves, differ in anything but the values
of inputs throws, naming the difference.

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
    ex = prepare(trace(call, f, args...), f, args...; target, measure)
    capture && capture!(ex)
    Compiled(f, ex, signature((f, args...)))
end

# The call `f(args...)`, traced with `f` among its arguments, so that the arrays
# `f` holds are inputs as the arguments' are.
call(f, args...) = f(args...)

"A function traced and prepared by `compile`."
struct Compiled{F}
    f::F
    executable::Executable
    signature::Tuple
end

function (c::Compiled)(@nospecialize(args...))
    d = difference(c.signature, signature((c.f, args...)))
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
