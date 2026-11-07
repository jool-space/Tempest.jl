# A kernel's body: scalar terms over the kernel's domain. Values the kernel
# recomputes are inlined, so a term mentions only loads of stored values,
# literals, index expressions, the kernel's own reductions and the folds it
# recomputes.

public Term, Load, Lit, Ix, Apply, Stat, Recomputed, Cast, term_eltype, subterms, loads, mentions,
    product_factors

abstract type Term end

"A stored value read at an index of the kernel's domain."
struct Load <: Term
    value::Int
    index::Tuple{Vararg{Affine}}
end

struct Lit{T} <: Term
    value::T
end

"An integer computed from the kernel's indices, converted to `T`."
struct Ix{T} <: Term
    expr::Affine
end

# Exact under IEEE arithmetic in either order, `min` and `max` included.
commutative(f) = f in (+, *, min, max, &, |, xor, ==, !=)

# `f` is untyped: terms are data to plan and emit, and an `Apply{typeof(f)}` would
# compile every method that handles terms again for each function applied.
# Its hash is computed once, from its parts' (see `Recomputed`), and it is
# interned (see `intern`). A commutative function's arguments are kept in one
# order (`term_order`), wherever the term is built: planned from the graph,
# or with its indices renamed, as a transposed fold's is.
struct Apply <: Term
    f::Any
    args::Tuple{Vararg{Term}}
    eltype::DataType
    hash::UInt
    # The variables it mentions, from its parts' (see `variables`).
    variables::UInt64
    function Apply(f, args, eltype)
        commutative(f) && length(args) > 1 && (args = Tuple(sort!(collect(Term, args); lt = (a, b) -> term_order(a, b) < 0, alg = InsertionSort)))
        intern(new(f, args, eltype, hash(f, hash(args, hash(eltype, 0x5d3e7a6b1c4f9e21))),
            mapreduce(variables, |, args; init = UInt64(0))))
    end
end

Base.hash(t::Apply, h::UInt) = hash(t.hash, h)

"A conversion to `T`, as a primitive with its own derivative rule."
struct Cast{T} end
(::Cast{T})(x) where {T} = T(x)
Base.show(io::IO, ::Cast{T}) where {T} = print(io, T)

"A reduction of the kernel, at the kernel's current position along its kept axes."
struct Stat <: Term
    value::Int
end

"""
A fold the kernel recomputes where it reads it, at `index`: the fold of value
`value`, whose operand is `body` over the reader's domain extended by the
fold's reduced axes, of `extents`, numbered after the domain's own variables.
"""
struct Recomputed <: Term
    value::Int
    index::Tuple{Vararg{Affine}}
    extents::Tuple{Vararg{Int}}
    body::Term
    # Computed once, from its parts' hashes: a term hashed by its fields is
    # hashed whole at every lookup, and a fold recomputed at each read puts
    # its body in every reader's term, so that planning Qwen3.5-4B's step
    # with a projection recomputed took 23 s, most of it hashing.
    hash::UInt
    variables::UInt64
    Recomputed(value, index, extents, body) = intern(new(value, index, extents, body, hash((value, index, extents, body), 0x2b8f4c1d7e6a3095), variables(body)))
end

Base.hash(t::Recomputed, h::UInt) = hash(t.hash, h)

# An order on terms by what they compute at the kernel's indices, never by
# the numbers of the values they read, so that groups alike up to the
# numbering of their values, as identical layers are whichever the program
# takes first, have one term and one kernel. The graph orders a value's
# operands by number, in the value's own indices, which a group's domain
# renames; ties keep that order. Values come first, then positions, then
# constants, as the graph orders operands.
term_order(a::Term, b::Term) = a === b ? 0 : (r = cmp(term_rank(a), term_rank(b))) != 0 ? r : same_rank_order(a, b)

term_rank(::Load) = 0
term_rank(::Stat) = 1
term_rank(::Recomputed) = 2
term_rank(::Apply) = 3
term_rank(::Term) = 4
term_rank(::Ix) = 5
term_rank(::Lit) = 6

index_order(a, b) = cmp(map(affine_key, a), map(affine_key, b))
same_rank_order(a::Load, b::Load) = index_order(a.index, b.index)
same_rank_order(a::Ix{S}, b::Ix{T}) where {S, T} = (r = cmp(string(S), string(T))) != 0 ? r : cmp(affine_key(a.expr), affine_key(b.expr))
same_rank_order(a::Lit, b::Lit) = cmp(string(typeof(a.value), " ", repr(a.value)), string(typeof(b.value), " ", repr(b.value)))
function same_rank_order(a::Apply, b::Apply)
    r = cmp(string(a.f), string(b.f))
    r != 0 && return r
    r = cmp(string(a.eltype), string(b.eltype))
    r != 0 && return r
    r = cmp(length(a.args), length(b.args))
    r != 0 && return r
    for (x, y) in zip(a.args, b.args)
        r = term_order(x, y)
        r != 0 && return r
    end
    0
end
function same_rank_order(a::Recomputed, b::Recomputed)
    r = index_order(a.index, b.index)
    r != 0 && return r
    r = cmp(a.extents, b.extents)
    r != 0 ? r : term_order(a.body, b.body)
end
same_rank_order(a::Term, b::Term) = 0

# Terms are interned as they are built: one equal to a term built before is
# that term. Each of the cover's trials plans the program again, building
# anew the terms of every group it leaves as it was, and `===` on equal terms
# built apart walks both whole: comparing the groups of two plans took most
# of preparing a chunked gated DeltaNet. Interned, their parts are the same
# objects, at which `===` stops at once. The table only saves time: a term
# not in it, as after `forget_terms!`, is equal to its like by its fields.
const INTERNED = [(ReentrantLock(), Dict{UInt, Vector{Term}}()) for _ in 1:64]

function intern(t::T) where {T <: Union{Apply, Recomputed}}
    lock, table = INTERNED[t.hash % length(INTERNED) + 1]
    @lock lock begin
        alike = get!(() -> Term[], table, t.hash)
        k = findfirst(u -> u === t, alike)
        k === nothing ? (push!(alike, t); t) : alike[k]::T
    end
end

"Empty the table of interned terms, which holds every term built since."
forget_terms!() = foreach(((lock, table),) -> @lock(lock, empty!(table)), INTERNED)

term_eltype(t::Load, graph) = graph.values[t.value].eltype
term_eltype(t::Lit, graph) = typeof(t.value)
term_eltype(::Ix{T}, graph) where {T} = T
term_eltype(t::Apply, graph) = t.eltype
term_eltype(t::Stat, graph) = graph.values[t.value].eltype
term_eltype(t::Recomputed, graph) = graph.values[t.value].eltype

# Every subterm once, operands before their users; with `nested`, also those
# of recomputed folds' bodies, which are over a longer domain. Terms for which
# `known` holds are left out, with their operands unless something else needs
# them.
function subterms(roots; nested = false, known = t -> false)
    seen = Dict{Term, Int}()
    order = Term[]
    function visit(t)
        (haskey(seen, t) || known(t)) && return
        t isa Apply && foreach(visit, t.args)
        nested && t isa Recomputed && visit(t.body)
        seen[t] = length(order) + 1
        push!(order, t)
    end
    foreach(visit, roots)
    order
end

loads(t::Term) = [s for s in subterms((t,); nested = true) if s isa Load]

"""
    product_factors(fold, term) -> (x, y) or nothing

The two factors whose products a fold of `term` sums: those of a sum of a
product; for an exponential fold of `(y, g...)`, the first `g` that is not
a constant, and `y`, whose `exp(y - m)` weighs it.
"""
product_factors(::Fold, t) = nothing
product_factors(::SumFold, t) = t isa Apply && t.f === (*) && length(t.args) == 2 ? t.args : nothing
function product_factors(::Relative{Exponential}, t)
    t isa Apply && t.f === tuple || return nothing
    i = findnext(a -> !(a isa Lit), t.args, 2)
    i === nothing ? nothing : (t.args[i], t.args[1])
end
product_factors(::Recomputed, t) = nothing

"The values a term reads from storage: those it loads, and those its formulas gather from."
function stored_reads(t::Term)
    reads = Int[]
    for s in subterms((t,); nested = true)
        if s isa Load
            push!(reads, s.value)
            foreach(e -> append!(reads, indirect_values(e)), s.index)
        elseif s isa Ix
            append!(reads, indirect_values(s.expr))
        end
    end
    reads
end

# The variables a term mentions, as bits: variable `n` at bit `n` up to 63,
# bit 64 for any after. A term reaching a load along many paths, as a chain
# of maps each reading the last twice does, is asked of each variable once
# per term, not once per path: an unrolled loop's gradient, walked as a tree,
# took longer than ten minutes to emit.
variable_bit(n::Int) = n < 64 ? UInt64(1) << (n - 1) : UInt64(1) << 63
variables(a::Var) = variable_bit(a.n)
variables(a::Union{Div, Mod}) = variables(a.e)
variables(a::Indirect) = mapreduce(variables, |, a.index; init = UInt64(0))
variables(e::Affine) = mapreduce(t -> variables(t.first), |, e.terms; init = UInt64(0))
variables(t::Load) = mapreduce(variables, |, t.index; init = UInt64(0))
variables(t::Ix) = variables(t.expr)
variables(t::Union{Apply, Recomputed}) = t.variables
variables(::Union{Lit, Stat}) = UInt64(0)

"Whether term `t` depends on variable `n` of its domain."
mentions(t::Load, n) = any(e -> mentions(e, n), t.index)
mentions(t::Ix, n) = mentions(t.expr, n)
mentions(t::Union{Apply, Recomputed}, n) = n < 64 ? t.variables & variable_bit(n) != 0 :
    t.variables & variable_bit(n) != 0 && (t isa Apply ? any(a -> mentions(a, n), t.args) : mentions(t.body, n))
mentions(::Union{Lit, Stat}, n) = false
stats(t::Term) = [s.value for s in subterms((t,); nested = true) if s isa Stat]

# A term is shown as the tree it computes, but for a subterm it reaches
# along several paths that would be long written out: named `#k=` where it is
# first written, `#k` after. A chain of maps each reading the last twice
# would otherwise be written out in text doubling with each map, and a term's
# text is the key its kernel's times are recorded under.
const NAMED_TERM_SIZE = 64

struct TermNames
    sizes::Dict{Term, Int}
    paths::Dict{Term, Int}
    names::Dict{Term, Int}
end

children(t::Apply) = t.args
children(t::Recomputed) = (t.body,)
children(::Term) = ()

function TermNames(t::Term)
    sizes, paths = Dict{Term, Int}(), Dict{Term, Int}()
    # Each distinct subterm's size as a tree, saturating, and the paths to it.
    function visit(t)
        paths[t] = get(paths, t, 0) + 1
        haskey(sizes, t) && return sizes[t]
        sizes[t] = min(1 + sum(visit, children(t); init = 0), typemax(Int) ÷ 4)
    end
    visit(t)
    TermNames(sizes, paths, Dict{Term, Int}())
end

show_term(io, t::Load, _) = print(io, "%", t.value, "[", join(map(string, t.index), ", "), "]")
show_term(io, t::Lit, _) = print(io, repr(t.value))
show_term(io, t::Ix{T}, _) where {T} = print(io, T, "(", t.expr, ")")
show_term(io, t::Stat, _) = print(io, "⟨%", t.value, "⟩")
function show_term(io, t::Union{Apply, Recomputed}, n::TermNames)
    if n.sizes[t] > NAMED_TERM_SIZE && n.paths[t] > 1
        haskey(n.names, t) && return print(io, "#", n.names[t])
        n.names[t] = length(n.names) + 1
        print(io, "#", n.names[t], "=")
    end
    show_parts(io, t, n)
end
function show_parts(io, t::Recomputed, n)
    print(io, "⟪%", t.value, " = fold over ", join(t.extents, "×"), " of ")
    show_term(io, t.body, n)
    print(io, "⟫")
end
function show_parts(io, t::Apply, n)
    print(io, t.f, "(")
    for (k, a) in enumerate(t.args)
        k > 1 && print(io, ", ")
        show_term(io, a, n)
    end
    print(io, ")")
end
Base.show(io::IO, t::Term) = show_term(io, t, TermNames(t))
