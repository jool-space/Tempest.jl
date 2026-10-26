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

# `f` is untyped: terms are data to plan and emit, and an `Apply{typeof(f)}` would
# compile every method that handles terms again for each function applied.
# Its hash is computed once, from its parts' (see `Recomputed`), and it is
# interned (see `intern`).
struct Apply <: Term
    f::Any
    args::Tuple{Vararg{Term}}
    eltype::DataType
    hash::UInt
    Apply(f, args, eltype) = intern(new(f, args, eltype, hash(f, hash(args, hash(eltype, 0x5d3e7a6b1c4f9e21)))))
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
    Recomputed(value, index, extents, body) = intern(new(value, index, extents, body, hash((value, index, extents, body), 0x2b8f4c1d7e6a3095)))
end

Base.hash(t::Recomputed, h::UInt) = hash(t.hash, h)

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

"Whether term `t` depends on variable `n` of its domain."
mentions(t::Load, n) = any(e -> mentions(e, n), t.index)
mentions(t::Ix, n) = mentions(t.expr, n)
mentions(t::Apply, n) = any(a -> mentions(a, n), t.args)
mentions(t::Recomputed, n) = mentions(t.body, n)
mentions(::Union{Lit, Stat}, n) = false
stats(t::Term) = [s.value for s in subterms((t,); nested = true) if s isa Stat]

show_term(io, t::Load) = print(io, "%", t.value, "[", join(map(string, t.index), ", "), "]")
show_term(io, t::Lit) = print(io, repr(t.value))
show_term(io, t::Ix{T}) where {T} = print(io, T, "(", t.expr, ")")
show_term(io, t::Stat) = print(io, "⟨%", t.value, "⟩")
function show_term(io, t::Recomputed)
    print(io, "⟪%", t.value, " = fold over ", join(t.extents, "×"), " of ")
    show_term(io, t.body)
    print(io, "⟫")
end
function show_term(io, t::Apply)
    print(io, t.f, "(")
    for (k, a) in enumerate(t.args)
        k > 1 && print(io, ", ")
        show_term(io, a)
    end
    print(io, ")")
end
Base.show(io::IO, t::Term) = show_term(io, t)
