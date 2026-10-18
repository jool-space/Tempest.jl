# A kernel's body: scalar terms over the kernel's domain. Values the kernel
# recomputes are inlined, so a term mentions only loads of stored values,
# literals, index expressions, the kernel's own reductions and the folds it
# recomputes.

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

struct Apply{F} <: Term
    f::F
    args::Tuple{Vararg{Term}}
    eltype::DataType
end

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
end

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
