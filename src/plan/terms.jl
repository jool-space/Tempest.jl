# A kernel's body: scalar terms over the kernel's domain. Values the kernel
# recomputes are inlined, so a term mentions only loads of stored values,
# literals, index expressions and the kernel's own reductions.

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

term_eltype(t::Load, graph) = graph.values[t.value].eltype
term_eltype(t::Lit, graph) = typeof(t.value)
term_eltype(::Ix{T}, graph) where {T} = T
term_eltype(t::Apply, graph) = t.eltype
term_eltype(t::Stat, graph) = graph.values[t.value].eltype

# Every subterm once, operands before their users.
function subterms(roots)
    seen = Dict{Term, Int}()
    order = Term[]
    function visit(t)
        haskey(seen, t) && return
        t isa Apply && foreach(visit, t.args)
        seen[t] = length(order) + 1
        push!(order, t)
    end
    foreach(visit, roots)
    order
end

loads(t::Term) = [s for s in subterms((t,)) if s isa Load]
stats(t::Term) = [s.value for s in subterms((t,)) if s isa Stat]

show_term(io, t::Load) = print(io, "%", t.value, "[", join(map(string, t.index), ", "), "]")
show_term(io, t::Lit) = print(io, repr(t.value))
show_term(io, t::Ix{T}) where {T} = print(io, T, "(", t.expr, ")")
show_term(io, t::Stat) = print(io, "⟨%", t.value, "⟩")
function show_term(io, t::Apply)
    print(io, t.f, "(")
    for (k, a) in enumerate(t.args)
        k > 1 && print(io, ", ")
        show_term(io, a)
    end
    print(io, ")")
end
Base.show(io::IO, t::Term) = show_term(io, t)
