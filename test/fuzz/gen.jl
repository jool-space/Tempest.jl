# Random programs over Tempest's algebra, as source. A case is a function of
# its arguments, a few statements and a loss summing what they compute; each
# statement is drawn from a family: maps (masks by position or value, integer
# arithmetic on positions), folds, scans, layouts, gathers, writes, products,
# or the whole case is a loop over states. Its extents are named, and its
# source and arguments are rendered from them, so that the shrinker can make
# them smaller; its arguments are drawn from its seed and their names, so that
# they are the same for the same extents however the statements change.
module Gen

using Random

"""
An argument: its name, its extents as expressions in the case's extent names,
and how its contents are drawn: `:float` from a normal distribution times
`scale`, `:index` from `1:of`, `:distinct` from `1:of` without repeats, and
`:position` a single integer of `1:of`.
"""
struct Arg
    name::String
    dims::Vector{String}
    kind::Symbol
    of::String
    scale::Float64
end

Arg(name, dims; scale = 1.0) = Arg(name, dims, :float, "", scale)

struct Case
    seed::Int
    family::Symbol
    extents::Dict{String, Int}
    args::Vector{Arg}
    globals::Vector{Arg}
    lines::Vector{String}
    terms::Vector{String}
    notes::Dict{Symbol, Any}
end

# A case with some of its parts replaced, as the shrinker makes them.
function Base.copy(c::Case; extents = c.extents, args = c.args, lines = c.lines, terms = c.terms)
    Case(c.seed, c.family, extents, args, c.globals, lines, terms, c.notes)
end

# An extent's expression evaluated: names, integers, and Base's arithmetic.
measure(text::AbstractString, extents) = measure(Meta.parse(text), extents)
measure(x::Integer, _) = Int(x)
measure(x::Symbol, extents) = extents[string(x)]
measure(x::Expr, extents) = getfield(Base, x.args[1])(map(a -> measure(a, extents), x.args[2:end])...)

# In a template, `{0.5}` is a literal of the element type and `{A - 1}` an
# extent's expression, an integer.
const LITERAL = r"^-?([0-9]+\.[0-9]*(e-?[0-9]+)?|Inf)$"

literal(x, E) = E === Float32 ? (isinf(x) ? (x > 0 ? "Inf32" : "-Inf32") : string(x) * "f0") : string(x)

function render(template, E, extents)
    replace(template, r"\{([^{}]+)\}" => function (m)
        inner = m[2:(end - 1)]
        occursin(LITERAL, inner) ? literal(parse(Float64, inner), E) : string(measure(inner, extents))
    end)
end

function source(c::Case, E = Float64)
    io = IOBuffer()
    println(io, "function f(", join((a.name for a in c.args), ", "), ")")
    foreach(line -> println(io, "    ", render(line, E, c.extents)), c.lines)
    println(io, "    ", render(join(c.terms, " + "), E, c.extents))
    println(io, "end")
    String(take!(io))
end

# The seed an argument's contents are drawn from: the case's and the name's.
data_seed(seed, name) = seed * 1_000_003 + foldl((h, c) -> (31h + Int(c)) % 1_000_003, codeunits(name); init = 7)

# The expression that draws an argument's contents, in Float64.
function construction(a::Arg, seed, extents)
    s = data_seed(seed, a.name)
    dims = join((measure(d, extents) for d in a.dims), ", ")
    a.kind === :float && return "$(a.scale) .* randn(Xoshiro($s), $dims)"
    n = measure(a.of, extents)
    a.kind === :index && return "Int32.(rand(Xoshiro($s), 1:$n, $dims))"
    a.kind === :distinct && return "Int32.(randperm(Xoshiro($s), $n)[1:$dims])"
    "rand(Xoshiro($s), 1:$n)"
end

const PRELUDE = """
using Tempest, Random, LinearAlgebra
import Tempest as T, Tempest.Compiler as C
σ(x) = 1 / (1 + exp(-x))
"""

"""
    standalone(case, E = Float64) -> String

The case as a script: the function in element type `E`, its globals, and its
arguments in Float64 as `args`.
"""
function standalone(c::Case, E = Float64)
    io = IOBuffer()
    print(io, PRELUDE)
    for g in c.globals
        println(io, g.name, " = ", E, ".(", construction(g, c.seed, c.extents), ")")
    end
    print(io, source(c, E))
    for a in c.args
        println(io, a.name, " = ", construction(a, c.seed, c.extents))
    end
    println(io, "args = (", join((a.name for a in c.args), ", "), length(c.args) == 1 ? ",)" : ")")
    String(take!(io))
end

"The arguments' contents: floats in Float64, indices in Int32."
function materialize(c::Case)
    m = Module(:Draw)
    Core.eval(m, :(using Random))
    [Core.eval(m, Meta.parse(construction(a, c.seed, c.extents))) for a in c.args]
end

# ---- drawing extents

# Small extents, one often among them, and sizes either side of the tiles'
# multiples, with no value of more than LIMIT elements, which the CPU's
# interpreter still runs in seconds.
const SMALL = (1, 1, 2, 2, 3, 3, 4, 5, 7)
const TILED = (31, 32, 33, 127, 128, 129)
const LIMIT = 8192

struct Value
    name::String
    dims::Vector{String}
    # It may hold elements equal because they are one element twice (a
    # gather by repeated indices, a concatenation of a value with itself, a
    # product of a matrix with itself): finite differences split a maximum's
    # cotangent between such ties, and Tempest gives it to the first, as
    # Zygote does, so no maximum or minimum is taken of it.
    tied::Bool
end

mutable struct Builder
    rng::Xoshiro
    big::Bool
    extents::Dict{String, Int}
    args::Vector{Arg}
    globals::Vector{Arg}
    lines::Vector{String}
    pool::Vector{Value}
    count::Int
    ops::Vector{Symbol}
end

Builder(rng, big) = Builder(rng, big, Dict{String, Int}(), Arg[], Arg[], String[], Value[], 0, Symbol[])

elements(b, dims) = prod(d -> measure(d, b.extents), dims; init = 1)

function draw(b, room)
    sizes = b.big && rand(b.rng) < 0.5 ? TILED : SMALL
    fits = filter(<=(room), sizes)
    isempty(fits) ? 1 : rand(b.rng, fits)
end

# A new extent, named `prefix` and a number, of at most `room`.
function extent!(b, prefix, room = LIMIT)
    name = prefix * string(count(k -> startswith(k, prefix), keys(b.extents)) + 1)
    b.extents[name] = draw(b, room)
    name
end

fresh!(b) = "v$(b.count += 1)"

function emit!(b, dims, expression; tied = false)
    elements(b, dims) <= LIMIT || return false
    name = fresh!(b)
    push!(b.lines, "$name = $expression")
    push!(b.pool, Value(name, dims, tied))
    true
end

# A value of the pool that `wanted` admits, the last two more often.
function pick(b, wanted = v -> true)
    candidates = filter(wanted, b.pool)
    isempty(candidates) && return nothing
    rand(b.rng) < 0.6 ? rand(b.rng, candidates[max(1, end - 1):end]) : rand(b.rng, candidates)
end

ranked(v) = !isempty(v.dims)

# A weight, passed as an argument or captured from a global.
function weight!(b, dims; scale = 1 / sqrt(max(1, measure(last(dims), b.extents))))
    if rand(b.rng) < 0.35
        push!(b.globals, Arg("G$(length(b.globals) + 1)", dims; scale))
        return b.globals[end].name
    end
    push!(b.args, Arg("W$(count(a -> startswith(a.name, "W"), b.args) + 1)", dims; scale))
    b.args[end].name
end

function index!(b, dims, kind, of)
    prefix = kind === :position ? "t" : "i"
    push!(b.args, Arg(prefix * "$(count(a -> startswith(a.name, prefix), b.args) + 1)", dims, kind, of, 1.0))
    b.args[end].name
end

# ---- rendering pieces

# A dimension in a template: a number as it is, an extent in braces.
D(d) = all(isdigit, d) ? d : "{" * d * "}"
wrap(x) = occursin(r"^[A-Za-z0-9]+$", x) ? x : "(" * x * ")"
tup(xs) = "(" * join(xs, ", ") * (length(xs) == 1 ? ",)" : ")")
dims_keyword(ds) = length(ds) == 1 ? string(only(ds)) : tup(ds)

# The positions along dimension `i`, as a range broadcast along the others;
# in Int32 with `int`, since a plain range is Int64 and would promote.
function positions(dims, i; int = false)
    range = int ? "Int32(1):Int32($(D(dims[i])))" : "1:$(D(dims[i]))"
    length(dims) == 1 && return "($range)"
    "reshape($range, " * join((j == i ? ":" : "1" for j in eachindex(dims)), ", ") * ")"
end

selection(r, i, sel) = join((j == i ? sel : ":" for j in 1:r), ", ")

# Whether `a` and `c` broadcast together, by their extents' names.
function compatible(a, c)
    all(1:max(length(a.dims), length(c.dims))) do i
        x, y = get(a.dims, i, "1"), get(c.dims, i, "1")
        x == y || x == "1" || y == "1"
    end
end

function broadcast_dims(a, c)
    map(1:max(length(a.dims), length(c.dims))) do i
        x = get(a.dims, i, "1")
        x == "1" ? get(c.dims, i, "1") : x
    end
end

# A dimension to fold or scan along, more often one of more than one element.
function along(b, v)
    long = findall(!=("1"), v.dims)
    isempty(long) || rand(b.rng) < 0.15 ? rand(b.rng, eachindex(v.dims)) : rand(b.rng, long)
end

# ---- maps

function unary!(b)
    a = pick(b)
    a === nothing && return false
    x = a.name
    e = rand(b.rng, ["tanh.($x)", "σ.($x)", "sin.($x)", "abs.($x)", "$x .^ 2", "sqrt.(abs.($x) .+ {1.0})",
        "max.($x, {0.1})", "clamp.($x, {-0.5}, {0.5})", "exp.(-abs2.($x))", "{0.5} .* $x .+ {0.25}"])
    emit!(b, a.dims, e; tied = a.tied)
end

function binary!(b)
    a = pick(b, ranked)
    a === nothing && return false
    c = something(pick(b, v -> v !== a && compatible(a, v)), a)
    x, y = a.name, c.name
    tied = a.tied || c.tied
    choices = ["$x .+ $y", "$x .* $y", "$x .- $y", "$x ./ ({1.0} .+ abs.($y))", "$x .* tanh.($y)"]
    tied || push!(choices, "max.($x, $y)", "ifelse.($x .> $y, $x, $y)")
    emit!(b, broadcast_dims(a, c), rand(b.rng, choices); tied)
end

function position_mask!(b)
    a = pick(b, ranked)
    a === nothing && return false
    r = length(a.dims)
    i, j = rand(b.rng, 1:r), rand(b.rng, 1:r)
    p = positions(a.dims, i)
    q = i == j ? "{cld($(a.dims[i]), 2)}" : positions(a.dims, j)
    mask = rand(b.rng, ["$p .<= $q", "mod.($p .+ $q, 3) .!= 0", "isodd.($p .+ $q)"])
    x = a.name
    emit!(b, a.dims, rand(b.rng, ["ifelse.($mask, $x, {0.0})", "ifelse.($mask, $x, {-0.5} .* $x)"]); tied = a.tied)
end

function value_mask!(b)
    a = pick(b, ranked)
    a === nothing && return false
    x = a.name
    e = rand(b.rng, ["ifelse.($x .> {0.0}, $x, {0.1} .* $x)", "ifelse.(abs.($x) .< {0.5}, $x, {0.5} .* tanh.($x))"])
    emit!(b, a.dims, e; tied = a.tied)
end

function index_values!(b)
    a = pick(b, ranked)
    a === nothing && return false
    r = length(a.dims)
    p = positions(a.dims, rand(b.rng, 1:r); int = true)
    q = positions(a.dims, rand(b.rng, 1:r); int = true)
    x = a.name
    e = rand(b.rng, ["$x .+ {0.1} .* mod.(Int32(3) .* $p .+ $q, Int32(5))", "$x .* {0.1} .* (fld.($p, Int32(2)) .+ $q)"])
    emit!(b, a.dims, e; tied = a.tied)
end

# ---- folds

function fold!(b)
    a = pick(b, ranked)
    a === nothing && return false
    r = length(a.dims)
    ds = r >= 2 && rand(b.rng) < 0.15 ? sort!(randperm(b.rng, r)[1:2]) : [along(b, a)]
    x = a.name
    k = dims_keyword(ds)
    choices = ["sum($x; dims = $k)", "sum(abs2, $x; dims = $k)", "prod({0.9} .+ {0.1} .* tanh.($x); dims = $k)"]
    a.tied || push!(choices, "maximum($x; dims = $k)", "minimum($x; dims = $k)")
    e = rand(b.rng, choices)
    dims = [i in ds ? "1" : d for (i, d) in enumerate(a.dims)]
    # Dropped from a vector, the dimension would leave an array of none.
    if length(ds) == 1 && r > 1 && rand(b.rng) < 0.3
        e = "dropdims($e; dims = $k)"
        dims = deleteat!(copy(a.dims), only(ds))
    end
    emit!(b, dims, e; tied = a.tied)
end

function whole_fold!(b)
    a = pick(b, ranked)
    a === nothing && return false
    x = a.name
    choices = ["sum($x)", "sum(abs2, $x) * {0.1}"]
    a.tied || push!(choices, "maximum($x)", "minimum($x)")
    emit!(b, String[], rand(b.rng, choices))
end

# Folds whose state is a tuple, as Base spells them: a mean weighted by
# exponentials, and a sum beside a maximum.
function tuple_fold!(b)
    a = pick(b, ranked)
    a === nothing && return false
    d = along(b, a)
    x = a.name
    choices = ["(s -> s[2] / s[1]).(mapreduce(e -> (exp(e), e * exp(e)), (p, q) -> p .+ q, $x; dims = $d, init = ({0.0}, {0.0})))"]
    a.tied || push!(choices, "(s -> s[1] + {0.5} * s[2]).(mapreduce(e -> (e, abs2(e)), (p, q) -> (p[1] + q[1], max(p[2], q[2])), $x; dims = $d, init = ({0.0}, {0.0})))")
    emit!(b, [i == d ? "1" : e for (i, e) in enumerate(a.dims)], rand(b.rng, choices); tied = a.tied)
end

# The log-sum-exp, alone or as a softmax, maybe of a causal mask's scores:
# the mask keeps each slice's first position, so no slice is all -Inf.
function logsumexp!(b)
    a = pick(b, ranked)
    a === nothing && return false
    r = length(a.dims)
    d = along(b, a)
    x = a.name
    others = filter(!=(d), 1:r)
    s = isempty(others) || rand(b.rng) < 0.6 ? x :
        "ifelse.($(positions(a.dims, d)) .<= $(positions(a.dims, rand(b.rng, others))), $x, {-Inf})"
    rand(b.rng) < 0.5 && return emit!(b, a.dims, "exp.($s .- T.logsumexp($s; dims = $d))"; tied = a.tied)
    emit!(b, [i == d ? "1" : e for (i, e) in enumerate(a.dims)], "T.logsumexp($s; dims = $d)"; tied = a.tied)
end

# ---- scans

function scan!(b)
    a = pick(b, ranked)
    a === nothing && return false
    d = along(b, a)
    x = a.name
    k = length(a.dims) == 1 && rand(b.rng) < 0.5 ? "" : "; dims = $d"
    init = isempty(k) ? "; init = {1.0}" : ", init = {1.0}"
    choices = ["cumsum($x$k)", "accumulate(+, $x$k)", "cumprod({0.9} .+ {0.1} .* tanh.($x)$k)",
        "accumulate((s, e) -> s / 2 + e, $x$k$init)"]
    a.tied || push!(choices, "accumulate(max, $x$k)", "accumulate(min, $x$k)")
    emit!(b, a.dims, rand(b.rng, choices); tied = a.tied)
end

function reversed_scan!(b)
    a = pick(b, ranked)
    a === nothing && return false
    k = "; dims = $(along(b, a))"
    x = a.name
    choices = ["reverse(cumsum(reverse($x$k)$k)$k)", "reverse(accumulate((s, e) -> s / 2 + e, reverse($x$k)$k, init = {1.0})$k)"]
    a.tied || push!(choices, "reverse(accumulate(max, reverse($x$k)$k)$k)")
    emit!(b, a.dims, rand(b.rng, choices); tied = a.tied)
end

# ---- layouts

function reshape_merge!(b)
    a = pick(b, v -> length(v.dims) >= 2)
    a === nothing && return false
    i = rand(b.rng, 1:(length(a.dims) - 1))
    dims = [a.dims[1:(i - 1)]; wrap(a.dims[i]) * " * " * wrap(a.dims[i + 1]); a.dims[(i + 2):end]]
    emit!(b, dims, "reshape($(a.name), $(join(D.(dims), ", ")))"; tied = a.tied)
end

# A product of two extents, split back into them; nothing for any other.
function factors(d)
    e = Meta.parse(d)
    e isa Expr && e.head === :call && e.args[1] === :* && length(e.args) == 3 || return nothing
    (string(e.args[2]), string(e.args[3]))
end

function reshape_split!(b)
    a = pick(b, v -> any(d -> factors(d) !== nothing, v.dims) && length(v.dims) < 4)
    a === nothing && return false
    i = rand(b.rng, findall(d -> factors(d) !== nothing, a.dims))
    dims = [a.dims[1:(i - 1)]; factors(a.dims[i])...; a.dims[(i + 1):end]]
    emit!(b, dims, "reshape($(a.name), $(join(D.(dims), ", ")))"; tied = a.tied)
end

function singleton!(b)
    a = pick(b, ranked)
    a === nothing && return false
    units = findall(==("1"), a.dims)
    if !isempty(units) && length(a.dims) > 1 && rand(b.rng) < 0.5
        i = rand(b.rng, units)
        return emit!(b, deleteat!(copy(a.dims), i), "dropdims($(a.name); dims = $i)"; tied = a.tied)
    end
    length(a.dims) < 3 || return false
    dims = insert!(copy(a.dims), rand(b.rng, 1:(length(a.dims) + 1)), "1")
    emit!(b, dims, "reshape($(a.name), $(join(D.(dims), ", ")))"; tied = a.tied)
end

function permute!(b)
    a = pick(b, v -> length(v.dims) in (2, 3))
    a === nothing && return false
    length(a.dims) == 2 && return emit!(b, reverse(a.dims), rand(b.rng, ["permutedims($(a.name))", "transpose($(a.name))"]); tied = a.tied)
    perm = rand(b.rng, [[1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]])
    emit!(b, a.dims[perm], "permutedims($(a.name), $(tup(perm)))"; tied = a.tied)
end

function select!(b)
    a = pick(b, ranked)
    a === nothing && return false
    r = length(a.dims)
    i = rand(b.rng, 1:r)
    x = a.dims[i]
    # A view's selection names its last position, which `end` cannot outside
    # brackets.
    viewed = rand(b.rng) < 0.3
    last_ = viewed ? "{$x}" : "end"
    less = viewed ? "{$(wrap(x)) - 1}" : "(end - 1)"
    choices = [("1:2:$last_", "cld($x, 2)"), ("$last_:-1:1", x), ("{cld($x, 2)}", nothing), ("1:1", "1")]
    measure(x, b.extents) >= 2 && push!(choices, ("2:$last_", wrap(x) * " - 1"), ("1:$less", wrap(x) * " - 1"), ("2:2:$last_", "fld($x, 2)"))
    sel, d = rand(b.rng, choices)
    r == 1 && d === nothing && return false
    dims = d === nothing ? deleteat!(copy(a.dims), i) : [j == i ? d : e for (j, e) in enumerate(a.dims)]
    e = viewed ? "view($(a.name), $(selection(r, i, sel)))" : "$(a.name)[$(selection(r, i, sel))]"
    emit!(b, dims, e; tied = a.tied)
end

function concat!(b)
    a = pick(b, ranked)
    a === nothing && return false
    k = rand(b.rng, 1:min(3, length(a.dims)))
    alike(v) = v !== a && length(v.dims) == length(a.dims) && all(i -> i == k || v.dims[i] == a.dims[i], eachindex(a.dims))
    c = pick(b, alike)
    other, d, tied = c === nothing ? ("{0.5} .* $(a.name)", a.dims[k], true) : (c.name, c.dims[k], a.tied || c.tied)
    e = k == 1 ? "vcat($(a.name), $other)" : k == 2 ? "hcat($(a.name), $other)" : "cat($(a.name), $other; dims = 3)"
    dims = [i == k ? wrap(a.dims[k]) * " + " * wrap(d) : e for (i, e) in enumerate(a.dims)]
    emit!(b, dims, e; tied)
end

function reverse!(b)
    a = pick(b, ranked)
    a === nothing && return false
    e = length(a.dims) == 1 && rand(b.rng) < 0.5 ? "reverse($(a.name))" : "reverse($(a.name); dims = $(along(b, a)))"
    emit!(b, a.dims, e; tied = a.tied)
end

# ---- gathers

# By an index array along a dimension, its indices repeated more often than
# not: the gradient is then a sum by key.
function gather!(b)
    a = pick(b, ranked)
    a === nothing && return false
    r = length(a.dims)
    i = rand(b.rng, 1:r)
    rest = elements(b, a.dims) ÷ measure(a.dims[i], b.extents)
    distinct = rand(b.rng) < 0.3
    room = distinct ? min(LIMIT ÷ rest, measure(a.dims[i], b.extents)) : LIMIT ÷ rest
    if !distinct && r < 3 && rand(b.rng) < 0.3
        k, l = extent!(b, "K", isqrt(room)), extent!(b, "L", isqrt(room))
        idx = index!(b, [k, l], :index, a.dims[i])
        return emit!(b, [a.dims[1:(i - 1)]; k; l; a.dims[(i + 1):end]], "$(a.name)[$(selection(r, i, idx))]"; tied = true)
    end
    k = extent!(b, "K", room)
    idx = index!(b, [k], distinct ? :distinct : :index, a.dims[i])
    emit!(b, [j == i ? k : e for (j, e) in enumerate(a.dims)], "$(a.name)[$(selection(r, i, idx))]"; tied = a.tied || !distinct)
end

function gather_linear!(b)
    a = pick(b, v -> length(v.dims) >= 2)
    a === nothing && return false
    k = extent!(b, "K")
    idx = index!(b, [k], :index, join(wrap.(a.dims), " * "))
    emit!(b, [k], "$(a.name)[$idx]"; tied = true)
end

# By a single integer argument, as a loop's step slices its input.
function gather_position!(b)
    a = pick(b, v -> length(v.dims) >= 2)
    a === nothing && return false
    r = length(a.dims)
    i = rand(b.rng, 1:r)
    t = index!(b, String[], :position, a.dims[i])
    emit!(b, deleteat!(copy(a.dims), i), "$(a.name)[$(selection(r, i, t))]"; tied = a.tied)
end

# ---- writes

# At distinct indices: where several land at one position a device lands any
# one of them, and the result is not one answer.
function write_indexed!(b)
    a = pick(b, ranked)
    a === nothing && return false
    r = length(a.dims)
    i = rand(b.rng, 1:r)
    k = extent!(b, "K", measure(a.dims[i], b.extents))
    sel = selection(r, i, index!(b, [k], :distinct, a.dims[i]))
    y = fresh!(b)
    value = rand(b.rng, ["{0.0}", "tanh.($(a.name)[$sel])", "{2.0} .* $(a.name)[$sel] .+ {0.5}"])
    push!(b.lines, "$y = copy($(a.name))", "$y[$sel] .= $value")
    push!(b.pool, Value(y, a.dims, a.tied))
    true
end

function write_range!(b)
    a = pick(b, v -> any(d -> measure(d, b.extents) >= 2, v.dims))
    a === nothing && return false
    r = length(a.dims)
    i = rand(b.rng, findall(d -> measure(d, b.extents) >= 2, a.dims))
    y = fresh!(b)
    x = a.name
    k = extent!(b, "K", measure(a.dims[i], b.extents))
    shift = "$y[$(selection(r, i, "2:end"))] = $x[$(selection(r, i, "1:(end - 1)"))]"
    write = rand(b.rng, [shift, "$y[$(selection(r, i, "1:2:end"))] .= {0.0}",
        "$y[$(selection(r, i, "1:{$k}"))] = tanh.($x[$(selection(r, i, "1:{$k}"))])"])
    push!(b.lines, "$y = copy($x)", write)
    # Shifted by one, the first element is there twice (seed 309).
    push!(b.pool, Value(y, a.dims, a.tied || write == shift))
    true
end

function write_scalar!(b)
    a = pick(b, ranked)
    a === nothing && return false
    y = fresh!(b)
    s = pick(b, v -> isempty(v.dims))
    value = s === nothing ? "sum($(a.name)) * {0.1}" : s.name
    push!(b.lines, "$y = copy($(a.name))", "$y[$(join(fill("1", length(a.dims)), ", "))] = $value")
    push!(b.pool, Value(y, a.dims, a.tied))
    true
end

# ---- products

function product!(b)
    a = pick(b, v -> length(v.dims) in (1, 2))
    a === nothing && return false
    rest = elements(b, a.dims) ÷ measure(a.dims[1], b.extents)
    m = extent!(b, "M", min(LIMIT ÷ rest, LIMIT ÷ measure(a.dims[1], b.extents)))
    w = weight!(b, [m, a.dims[1]])
    emit!(b, [m; a.dims[2:end]], "$w * $(a.name)"; tied = a.tied)
end

function transposed_product!(b)
    a = pick(b, v -> length(v.dims) == 2)
    a === nothing && return false
    x = a.name
    if rand(b.rng) < 0.5
        c = pick(b, v -> v !== a && length(v.dims) == 2 && v.dims[1] == a.dims[1])
        c === nothing && return emit!(b, [a.dims[2], a.dims[2]], "transpose($x) * $x"; tied = true)
        return emit!(b, [a.dims[2], c.dims[2]], "transpose($x) * $(c.name)"; tied = a.tied || c.tied)
    end
    c = pick(b, v -> v !== a && length(v.dims) == 2 && v.dims[2] == a.dims[2])
    c === nothing && return emit!(b, [a.dims[1], a.dims[1]], "$x * transpose($x)"; tied = true)
    emit!(b, [a.dims[1], c.dims[1]], "$x * transpose($(c.name))"; tied = a.tied || c.tied)
end

# A product by `contract`, its labels drawn: one axis contracted, another
# maybe a batch, a new one from the weight, the result's in any order.
function contract!(b)
    a = pick(b, v -> length(v.dims) in (2, 3))
    a === nothing && return false
    r = length(a.dims)
    labels = randperm(b.rng, r)
    contracted = labels[1]
    batch = r == 3 && rand(b.rng) < 0.6 ? [labels[2]] : Int[]
    free = setdiff(1:r, [contracted; batch])
    room = min(LIMIT ÷ elements(b, a.dims[[free; batch]]), LIMIT ÷ elements(b, a.dims[[contracted; batch]]))
    m = extent!(b, "M", room)
    new = r + 1
    lb = shuffle(b.rng, [contracted; batch; new])
    lo = shuffle(b.rng, [free; batch; new])
    dim(l) = l == new ? m : a.dims[l]
    w = weight!(b, dim.(lb); scale = 1 / sqrt(max(1, measure(a.dims[contracted], b.extents))))
    emit!(b, dim.(lo), "C.contract($(a.name), $w, $(tup(1:r)), $(tup(lb)), $(tup(lo)))"; tied = a.tied)
end

# ---- the cases

const FAMILY_OPS = Dict(
    :map => [unary!, binary!, position_mask!, value_mask!, index_values!],
    :fold => [fold!, whole_fold!, tuple_fold!, logsumexp!],
    :scan => [scan!, reversed_scan!],
    :layout => [reshape_merge!, reshape_split!, singleton!, permute!, select!, concat!, reverse!],
    :gather => [gather!, gather_linear!, gather_position!],
    :write => [write_indexed!, write_range!, write_scalar!],
    :product => [product!, transposed_product!, contract!],
)
const FAMILIES = (:map, :fold, :scan, :layout, :gather, :write, :product, :loop)
const OPS = reduce(vcat, values(FAMILY_OPS))

function inputs!(b)
    r = rand(b.rng, (1, 2, 2, 2, 3))
    dims = String[]
    for name in ("A", "B", "C")[1:r]
        b.extents[name] = draw(b, LIMIT ÷ elements(b, dims))
        push!(dims, name)
    end
    push!(b.args, Arg("x", dims))
    push!(b.pool, Value("x", dims, false))
    rand(b.rng) < 0.5 || return
    other = rand(b.rng) < 0.5 ? dims : dims[1:1]
    push!(b.args, Arg("y", other))
    push!(b.pool, Value("y", other, false))
end

function loss(b, v)
    isempty(v.dims) && return "$(v.name) * {0.5}"
    rand(b.rng, ["sum(abs2, $(v.name)) * {0.1}", "sum(sin.($(v.name)))", "sum($(v.name) .* {0.3})"])
end

"""
    generate(seed; big = false) -> Case

The case drawn from `seed`: a family, a statement of it among up to five
others, extents from the small sweep, or with `big` also around the tiles'
multiples.
"""
function generate(seed; big = false)
    rng = Xoshiro(seed)
    family = rand(rng, FAMILIES)
    family === :loop && return loop_case(seed, rng, big)
    b = Builder(rng, big)
    inputs!(b)
    inputs = length(b.pool)
    n = rand(rng, 1:5)
    at = rand(rng, 0:n)
    for k in 0:n
        ops = k == at ? FAMILY_OPS[family] : rand(rng) < 0.4 ? FAMILY_OPS[:map] : OPS
        for _ in 1:20
            op = rand(rng, ops)
            op(b) || continue
            push!(b.ops, nameof(op))
            break
        end
    end
    made = b.pool[(inputs + 1):end]
    terms = [loss(b, isempty(made) ? b.pool[end] : made[end])]
    length(made) > 1 && rand(rng) < 0.5 && push!(terms, loss(b, rand(rng, made[1:(end - 1)])))
    notes = Dict{Symbol, Any}(:family => family, :ops => Tuple(b.ops))
    Case(seed, family, b.extents, b.args, b.globals, b.lines, terms, notes)
end

# ---- loops

# A recurrence over one to three states (and maybe a number), stepping through
# slices of an input or not, its weights passed in or captured from globals,
# its states read after the loop or collected at each step.
function loop_case(seed, rng, big)
    b = Builder(rng, big)
    b.extents["H"] = big ? rand(rng, (16, 31, 32, 33, 64)) : rand(rng, (1, 2, 3, 4))
    batch = big ? rand(rng, (0, 1, 8, 32, 33)) : rand(rng, (0, 1, 2, 3))
    b.extents["D"] = big ? rand(rng, (4, 16, 31)) : rand(rng, 1:3)
    b.extents["T"] = rand(rng, (1, 2, 3, 3, 4, 5, 7))
    matrix = batch > 0
    matrix && (b.extents["B"] = batch)
    shape = matrix ? ["H", "B"] : ["H"]
    form = rand(rng, (:none, :whole, :whole, :whole, :middle, :sub, :rev, :step2))
    !matrix && form === :middle && (form = :whole)
    range, L = form === :sub ? ("2:{T + 1}", "T + 2") : form === :rev ? ("{T}:-1:1", "T") :
        form === :step2 ? ("1:2:{2 * T - 1}", "2 * T") : ("1:{T}", "T")
    scalar_input = rand(rng) < 0.2
    notes = Dict{Symbol, Any}(:family => :loop, :form => form, :scalar_input => scalar_input)
    form === :none || push!(b.args, Arg("xs", form === :middle ? ["D", L, "B"] : matrix ? ["D", "B", L] : ["D", L]))
    scalar_input && push!(b.args, Arg("ts", [L]; scale = 0.5))
    slice = form === :none ? nothing : form === :middle ? "xs[:, t, :]" : matrix ? "xs[:, :, t]" : "xs[:, t]"
    nstates = rand(rng, 1:3)
    scalar_state = rand(rng) < 0.25
    notes[:states] = nstates
    notes[:scalar_state] = scalar_state
    starts = String[]
    for k in 1:nstates
        r = rand(rng)
        if k > 1 && r < 0.3
            push!(starts, "s1")
        elseif k > 1 && r < 0.4
            push!(starts, "tanh.(s1)")
        else
            push!(b.args, Arg("a$k", shape; scale = 0.5))
            push!(starts, "a$k")
        end
    end
    function weight(name, dims; scale = 0.5 / sqrt(b.extents[last(dims)]))
        if rand(rng) < 0.35
            push!(b.globals, Arg("G$(length(b.globals) + 1)_$name", dims; scale))
            return b.globals[end].name
        end
        push!(b.args, Arg(name, dims; scale))
        name
    end
    Ws = [weight("W$k", ["H", "H"]) for k in 1:rand(rng, 1:3)]
    U = form === :none ? nothing : weight("U", ["H", "D"])
    bs = [weight("b$k", ["H"]; scale = 0.2) for k in 1:rand(rng, 0:2)]
    # The step: temporaries from a pool of values of the state's shape, the
    # states updated among them, in turn or at once.
    function body!(lines, indent)
        pool = ["s$k" for k in 1:nstates]
        scalars = String[]
        scalar_state && push!(scalars, "r")
        scalar_input && push!(scalars, "ts[t]")
        if slice !== nothing
            push!(lines, indent * "x = " * slice)
            push!(pool, "($U * x)")
        end
        temps = 0
        operand() = rand(rng) < 0.5 && temps > 0 ? "z$(rand(rng, max(1, temps - 2):temps))" : rand(rng, pool)
        function temp!()
            a, c = operand(), operand()
            op = rand(rng, 1:14)
            e = op == 1 ? "$(rand(rng, Ws)) * $a" :
                op == 2 ? "transpose($(rand(rng, Ws))) * $a" :
                op == 3 ? "$a .+ $c" :
                op == 4 ? "$a .* $c" :
                op == 5 ? "$a .- $c" :
                op == 6 ? "tanh.($a)" :
                op == 7 ? "σ.($a)" :
                op == 8 ? "{0.5} .* $a .+ " * (isempty(bs) ? "{0.1}" : rand(rng, bs)) :
                op == 9 ? "$a .- sum($a; dims = 1) ./ {H}" :
                op == 10 ? "$a .* " * (isempty(scalars) ? "(sum($c) * {0.1})" : rand(rng, scalars)) :
                op == 11 ? "exp.($a .- maximum($a; dims = 1))" :
                op == 12 ? "cumsum($a; dims = 1) .* {0.3}" :
                op == 13 ? "$a ./ (1 .+ abs.($c))" :
                "$(rand(rng, Ws)) * $a .+ " * (isempty(bs) ? "$c" : rand(rng, bs))
            temps += 1
            push!(lines, indent * "z$temps = " * e)
        end
        foreach(_ -> temp!(), 1:rand(rng, 1:3))
        at_once = nstates > 1 && rand(rng) < 0.3
        news = String[]
        for k in randperm(rng, nstates)
            rand(rng) < 0.5 && temp!()
            z, z2 = "z$temps", "z$(rand(rng, 1:temps))"
            m = rand(rng, 1:3)
            e = m == 1 ? "tanh.($z)" : m == 2 ? "{0.5} .* s$k .+ {0.5} .* tanh.($z)" : "σ.($z) .* s$k .+ (1 .- σ.($z)) .* tanh.($z2)"
            at_once ? push!(news, e) : push!(lines, indent * "s$k = " * e)
        end
        at_once && push!(lines, indent * join(("s$k" for k in 1:nstates), ", ") * " = " * join(news, ", "))
        scalar_state && push!(lines, indent * "r = {0.5} * r + {0.01} * sum(abs2, z$temps)")
    end
    collected = rand(rng) < 0.3
    second = rand(rng) < 0.15
    notes[:collect] = collected
    notes[:second] = second
    lines = b.lines
    append!(lines, ["s$k = " * starts[k] for k in 1:nstates])
    scalar_state && push!(lines, "r = sum(s1) * {0.1}")
    collected && push!(lines, "acc = []")
    push!(lines, "for t in " * range)
    body!(lines, "    ")
    collected && push!(lines, "    push!(acc, s1)")
    push!(lines, "end")
    if second
        push!(lines, "for t in " * range)
        body!(lines, "    ")
        push!(lines, "end")
    end
    terms = ["sum(abs2, s1)"; ["sum(s$k) * {0.3}" for k in 2:nstates]]
    scalar_state && push!(terms, "r")
    collected && push!(terms, "sum(abs2, stack(acc)) * {0.1}", "sum(acc[end])")
    Case(seed, :loop, b.extents, b.args, b.globals, lines, terms, notes)
end

end
