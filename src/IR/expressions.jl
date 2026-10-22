# Index expressions: quasi-affine integer expressions over the variables of a
# domain, kept in a canonical form so that two spellings of one layout give
# equal expressions. Indices are zero-based. Everything here is immutable and
# built from tuples, so `==` and `hash` are structural.

public Affine, Var, Div, Mod, Indirect, Domain, affine_range, coefficient, is_affine,
    indirect_values, substitute

abstract type Atom end

"The `n`th variable of the enclosing domain."
struct Var <: Atom
    n::Int
end

"`constant + Σ coefficient·atom`, atoms sorted, coefficients nonzero."
struct Affine
    constant::Int
    terms::Tuple{Vararg{Pair{Atom, Int}}}
end

"Floor division of an expression by a positive constant."
struct Div <: Atom
    e::Affine
    d::Int
end

"The remainder, in `0:d-1`, of an expression divided by a positive constant."
struct Mod <: Atom
    e::Affine
    d::Int
end

"An integer read from `value` at `index`, known to lie in `lo:hi`: a gather."
struct Indirect <: Atom
    value::Int
    index::Tuple{Vararg{Affine}}
    lo::Int
    hi::Int
end

# A total order on atoms, so sorted terms are canonical.
atom_key(a::Var) = (0, a.n)
atom_key(a::Div) = (1, affine_key(a.e), a.d)
atom_key(a::Mod) = (2, affine_key(a.e), a.d)
atom_key(a::Indirect) = (3, a.value, map(affine_key, a.index), a.lo, a.hi)
affine_key(a::Affine) = (a.constant, map(t -> (atom_key(t.first), t.second), a.terms))

constant(c::Integer) = Affine(Int(c), ())
var(n::Integer) = Affine(0, (Var(n) => 1,))
isconstant(a::Affine) = isempty(a.terms)

"""
    Domain(extents)

The variables an expression ranges over: variable `n` takes the values
`0:extents[n]-1`.
"""
struct Domain
    extents::Tuple{Vararg{Int}}
end

Domain(extents::AbstractVector) = Domain(Tuple(extents))
Base.length(d::Domain) = length(d.extents)

# Inclusive bounds of an expression over a domain.
atom_range(a::Var, d::Domain) = (0, d.extents[a.n] - 1)
atom_range(a::Indirect, ::Domain) = (a.lo, a.hi)

function atom_range(a::Div, d::Domain)
    lo, hi = affine_range(a.e, d)
    (fld(lo, a.d), fld(hi, a.d))
end

function atom_range(a::Mod, d::Domain)
    lo, hi = affine_range(a.e, d)
    hi - lo < a.d && mod(lo, a.d) <= mod(hi, a.d) ? (mod(lo, a.d), mod(hi, a.d)) : (0, a.d - 1)
end

function affine_range(a::Affine, d::Domain)
    lo = hi = a.constant
    for (atom, c) in a.terms
        l, h = atom_range(atom, d)
        lo += c > 0 ? c * l : c * h
        hi += c > 0 ? c * h : c * l
    end
    (lo, hi)
end

# Collect like atoms, drop zeros and single-valued atoms, sort, and fold
# `c·(e mod d) + c·d·(e ÷ d)` back into `c·e`, the quotient in its simplified
# form: `(i ÷ 32) mod 2` pairs with `i ÷ 64`.
function normalize(constant::Int, terms, d::Domain)
    collected = Dict{Atom, Int}()
    for (atom, c) in terms
        c == 0 && continue
        lo, hi = atom_range(atom, d)
        if lo == hi
            constant += c * lo
            continue
        end
        collected[atom] = get(collected, atom, 0) + c
    end
    filter!(p -> p.second != 0, collected)
    changed = true
    while changed
        changed = false
        for (atom, c) in collected
            atom isa Mod || continue
            partner = quotient(atom, d)
            get(collected, partner, 0) == c * atom.d || continue
            delete!(collected, atom)
            delete!(collected, partner)
            whole = scale(atom.e, c)
            constant += whole.constant
            for (a, k) in whole.terms
                collected[a] = get(collected, a, 0) + k
            end
            filter!(p -> p.second != 0, collected)
            changed = true
            break
        end
    end
    sorted = sort!(collect(collected); by = p -> atom_key(p.first))
    Affine(constant, Tuple(sorted))
end

# The division a remainder pairs with, as `fdiv` writes it when that is one
# atom.
function quotient(m::Mod, d::Domain)
    q = fdiv(m.e, m.d, d)
    q.constant == 0 && length(q.terms) == 1 && q.terms[1].second == 1 ? q.terms[1].first : Div(m.e, m.d)
end

scale(a::Affine, c::Integer) =
    c == 0 ? constant(0) : Affine(a.constant * c, map(t -> t.first => t.second * c, a.terms))

add(a::Affine, b::Affine, d::Domain) = normalize(a.constant + b.constant, (a.terms..., b.terms...), d)
add(a::Affine, c::Integer, d::Domain) = normalize(a.constant + c, a.terms, d)
mul(a::Affine, c::Integer, d::Domain) = normalize(a.constant * c, scale(a, c).terms, d)

# Split `e` into `q·k + r` where every term of `q·k` is divisible by `k`.
function split_divisible(e::Affine, k::Int)
    q = Pair{Atom, Int}[]
    r = Pair{Atom, Int}[]
    for (atom, c) in e.terms
        c % k == 0 ? push!(q, atom => c ÷ k) : push!(r, atom => c)
    end
    (fld(e.constant, k), q, mod(e.constant, k), r)
end

# The divisors of `k` between 2 and `k - 1`, largest first.
function proper_divisors(k::Int)
    ds = Int[]
    for g in 2:isqrt(k)
        k % g == 0 || continue
        push!(ds, g)
        g != k ÷ g && push!(ds, k ÷ g)
    end
    sort!(ds; rev = true)
end

# `e = g·b + s` with `s` in `0:g-1`, for the largest `g` that divides `k` and
# leaves such a remainder, or nothing. Then `fld(e, k) = fld(b, k ÷ g)` and
# `mod(e, k) = g·mod(b, k ÷ g) + s`: a small term below a common factor of
# the others and the divisor drops out of the division.
function split_common(e::Affine, k::Int, d::Domain)
    for g in proper_divisors(k)
        big = Pair{Atom, Int}[]
        small = Pair{Atom, Int}[]
        for (atom, c) in e.terms
            c % g == 0 ? push!(big, atom => c ÷ g) : push!(small, atom => c)
        end
        s = normalize(mod(e.constant, g), small, d)
        lo, hi = affine_range(s, d)
        lo >= 0 && hi < g && return (g, normalize(fld(e.constant, g), big, d), s)
    end
    nothing
end

"""
    fdiv(e, k, domain)

`fld(e, k)` for a positive constant `k`, simplified: terms divisible by `k`
come out of the division, a remainder known to lie in `0:k-1` vanishes, and a
factor common to the other terms and `k` divides out.
"""
function fdiv(e::Affine, k::Int, d::Domain)
    k > 0 || throw(ArgumentError("division by a nonpositive constant $k"))
    k == 1 && return e
    qc, q, rc, r = split_divisible(e, k)
    rest = normalize(rc, r, d)
    lo, hi = affine_range(rest, d)
    quotient = normalize(qc, q, d)
    lo >= 0 && hi < k && return quotient
    if fld(lo, k) == fld(hi, k)
        return add(quotient, fld(lo, k), d)
    end
    # Nested divisions compose: fld(fld(x, a), b) == fld(x, a·b).
    if length(rest.terms) == 1 && rest.constant == 0 && rest.terms[1].second == 1 &&
       rest.terms[1].first isa Div
        inner = rest.terms[1].first
        return add(quotient, fdiv(inner.e, inner.d * k, d), d)
    end
    common = split_common(rest, k, d)
    common === nothing || return add(quotient, fdiv(common[2], k ÷ common[1], d), d)
    normalize(quotient.constant, (quotient.terms..., Div(rest, k) => 1), d)
end

"""
    fmod(e, k, domain)

`mod(e, k)` for a positive constant `k`, simplified: terms divisible by `k`
vanish, a remainder known to lie in `0:k-1` is returned as it is, and a factor
common to the other terms and `k` comes out.
"""
function fmod(e::Affine, k::Int, d::Domain)
    k > 0 || throw(ArgumentError("modulo by a nonpositive constant $k"))
    k == 1 && return constant(0)
    _, _, rc, r = split_divisible(e, k)
    rest = normalize(rc, r, d)
    lo, hi = affine_range(rest, d)
    lo >= 0 && hi < k && return rest
    # mod(mod(x, a), b) == mod(x, b) when b divides a.
    if length(rest.terms) == 1 && rest.constant == 0 && rest.terms[1].second == 1 &&
       rest.terms[1].first isa Mod && rest.terms[1].first.d % k == 0
        return fmod(rest.terms[1].first.e, k, d)
    end
    common = split_common(rest, k, d)
    if common !== nothing
        g, b, small = common
        return add(mul(fmod(b, k ÷ g, d), g, d), small, d)
    end
    normalize(0, (Mod(rest, k) => 1,), d)
end

"""
    substitute(e, arguments, domain)

Replace variable `n` of `e` by `arguments[n]`, an expression over `domain`,
and simplify there.
"""
function substitute(e::Affine, arguments, d::Domain)
    result = constant(e.constant)
    for (atom, c) in e.terms
        result = add(result, scale(substitute(atom, arguments, d), c), d)
    end
    result
end

substitute(a::Var, arguments, ::Domain) = arguments[a.n]
substitute(a::Div, arguments, d::Domain) = fdiv(substitute(a.e, arguments, d), a.d, d)
substitute(a::Mod, arguments, d::Domain) = fmod(substitute(a.e, arguments, d), a.d, d)
substitute(a::Indirect, arguments, d::Domain) = normalize(
    0,
    (Indirect(a.value, map(x -> substitute(x, arguments, d), a.index), a.lo, a.hi) => 1,),
    d,
)

# Evaluation, for interpreters and tests. `indirect(value, index)` reads a gather.
evaluate_index(a::Affine, x, indirect = nothing) =
    a.constant + sum(t -> t.second * evaluate_index(t.first, x, indirect), a.terms; init = 0)
evaluate_index(a::Var, x, _) = x[a.n]
evaluate_index(a::Div, x, indirect) = fld(evaluate_index(a.e, x, indirect), a.d)
evaluate_index(a::Mod, x, indirect) = mod(evaluate_index(a.e, x, indirect), a.d)
evaluate_index(a::Indirect, x, indirect) =
    indirect(a.value, map(i -> evaluate_index(i, x, indirect), a.index))

# The coefficient of variable `n` when `e` is affine in it, or `nothing`.
function coefficient(e::Affine, n::Int)
    c = 0
    for (atom, k) in e.terms
        if atom isa Var
            atom.n == n && (c += k)
        elseif mentions(atom, n)
            return nothing
        end
    end
    c
end

mentions(a::Var, n) = a.n == n
mentions(a::Union{Div, Mod}, n) = any(t -> mentions(t.first, n), a.e.terms)
mentions(a::Indirect, n) = any(i -> any(t -> mentions(t.first, n), i.terms), a.index)
mentions(e::Affine, n) = any(t -> mentions(t.first, n), e.terms)

"True when `e` is `constant + Σ cₙ·xₙ` with no division, modulo or gather."
is_affine(e::Affine) = all(t -> t.first isa Var, e.terms)

"The values an expression gathers from."
indirect_values(e::Affine) = reduce(vcat, (indirect_values(a) for (a, _) in e.terms); init = Int[])
indirect_values(::Var) = Int[]
indirect_values(a::Union{Div, Mod}) = indirect_values(a.e)
indirect_values(a::Indirect) = [a.value; reduce(vcat, map(indirect_values, a.index); init = Int[])]

function Base.show(io::IO, a::Affine)
    parts = String[]
    for (atom, c) in a.terms
        s = sprint(show, atom)
        push!(parts, c == 1 ? s : c == -1 ? "-" * s : string(c, "·", s))
    end
    (a.constant != 0 || isempty(parts)) && push!(parts, string(a.constant))
    print(io, join(parts, " + "))
end

Base.show(io::IO, a::Var) = print(io, "i", a.n)
Base.show(io::IO, a::Div) = print(io, "(", a.e, ") ÷ ", a.d)
Base.show(io::IO, a::Mod) = print(io, "(", a.e, ") mod ", a.d)
Base.show(io::IO, a::Indirect) = print(io, "%", a.value, "[", join(map(string, a.index), ", "), "]")
