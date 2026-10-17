# Algorithms as rewrites of folds, applied to the index-level graph before
# planning where the target's numerics permit them, since they change
# rounding. A sum weighted by `exp(y - m)`, where `m` is the maximum or the
# logsumexp of `y` over the summed axes, becomes an online softmax: one fold
# that keeps the running maximum and rescales its sums as the maximum grows,
# so that `m` is never computed in a pass of its own. Its factors constant
# along the sum leave it, as the softmax's denominator does: Σ v·e/l is
# (Σ v·e)/l. A quotient of two online softmaxes with the same maximum becomes
# one. So the two spellings of a softmax, `exp(s - logsumexp(s))` and
# `exp(s - maximum(s)) / sum(exp(s - maximum(s)))`, weighted and summed, reach
# the same fold.

"""
    rewrite(graph) -> IndexGraph

The graph with its sums weighted by a softmax made online softmax folds. The
graph's `rewrites` notes each, and `explain` prints them.
"""
rewrite(graph::IndexGraph) = compact(merge_softmaxes(online_softmaxes(graph)))

## Building a graph from another, in order.

# The new graph's values, each old value's new id (zero when left out), and
# notes on the values rewrites made.
struct Builder
    values::Vector{Value}
    ids::Vector{Int}
    rewrites::Vector{Pair{Int, String}}
end

"Add a value to the new graph; its id."
function add!(b::Builder, d::Definition, dims, T, scalar = false)
    push!(b.values, Value(d, dims, T, scalar))
    length(b.values)
end

# The graph with old value `id` replaced by the value `f(builder, id)` returns
# the id of (or kept, for `nothing`), leaving out the values `keep` excludes;
# and each old value's new id.
function rebuild(f, graph::IndexGraph, keep = trues(length(graph.values)))
    b = Builder(Value[], zeros(Int, length(graph.values)), Pair{Int, String}[])
    for (id, v) in enumerate(graph.values)
        keep[id] || continue
        new = f(b, id)
        b.ids[id] = new === nothing ? add!(b, remap(v.definition, b.ids), v.dims, v.eltype, v.scalar) : new
    end
    notes = [[b.ids[v] => s for (v, s) in graph.rewrites if b.ids[v] != 0]; b.rewrites]
    g = IndexGraph(b.values, map(a -> b.ids[a], graph.arguments), Operand[remap(o, b.ids) for o in graph.results],
        graph.result_dims, graph.result_scalar, graph.result_eltype, notes)
    g, b.ids
end

remap(e::Affine, ids) = Affine(e.constant, map(t -> remap(t.first, ids) => t.second, e.terms))
remap(a::Var, ids) = a
remap(a::Div, ids) = Div(remap(a.e, ids), a.d)
remap(a::Mod, ids) = Mod(remap(a.e, ids), a.d)
remap(a::Indirect, ids) = Indirect(ids[a.value], map(e -> remap(e, ids), a.index), a.lo, a.hi)
remap(o::Read, ids) = Read(ids[o.value], map(e -> remap(e, ids), o.index))
remap(o::Literal, ids) = o
remap(o::IndexOperand{T}, ids) where {T} = IndexOperand{T}(remap(o.expr, ids))
remap(d::Pointwise, ids) = Pointwise(d.f, map(o -> remap(o, ids), d.operands))
remap(d::Reduction, ids) = Reduction(d.fold, remap(d.operand, ids), d.iteration, d.reduced)
remap(d::Opaque, ids) = Opaque(d.node, map(o -> remap(o, ids), d.operands), d.shapes, d.epilogue)
remap(d::Union{Input, ConstantArray}, ids) = d

# The values something reads: the results and what they read, inputs and
# opaque work, and constant arrays.
function live_values(graph::IndexGraph)
    live = falses(length(graph.values))
    for o in graph.results, r in read_values(o)
        live[r] = true
    end
    for id in length(graph.values):-1:1
        d = graph.values[id].definition
        live[id] |= !(d isa Union{Pointwise, Reduction})
        live[id] || continue
        foreach(o -> foreach(r -> live[r] = true, read_values(o)), operands(d))
    end
    live
end

"The graph without the values nothing reads."
compact(graph::IndexGraph) = first(rebuild((b, id) -> nothing, graph, live_values(graph)))

## Products of operands, seen through the values they read.

eltype_of(graph, o::Read) = graph.values[o.value].eltype
eltype_of(graph, o::Literal) = typeof(o.value)
eltype_of(graph, ::IndexOperand{T}) where {T} = T

# An operand of a value read at `index`, as read from the reader's domain.
through(x::Read, index, d::Domain) = Read(x.value, map(e -> substitute(e, index, d), x.index))
through(x::Literal, index, d::Domain) = x
through(x::IndexOperand{T}, index, d::Domain) where {T} = IndexOperand{T}(substitute(x.expr, index, d))

pointwise(graph, o, f) = o isa Read && (d = graph.values[o.value].definition) isa Pointwise && d.f === f

"The operands of the pointwise value `o` reads, as read from the reader's domain."
inner(graph, o::Read, d::Domain) = map(x -> through(x, o.index, d), graph.values[o.value].definition.operands)

mentions(o::Read, k) = any(e -> mentions(e, k), o.index)
mentions(o::IndexOperand, k) = mentions(o.expr, k)
mentions(::Literal, k) = false

# The factors of the product `o` reads, through the products and quotients of
# type `T` it reads: each an operand and its power, 1 or -1.
function factors(graph, o, d::Domain, T = eltype_of(graph, o), power = 1)
    if (pointwise(graph, o, *) || pointwise(graph, o, /)) && eltype_of(graph, o) === T
        xs = inner(graph, o, d)
        if length(xs) == 2 && all(x -> x isa Literal || eltype_of(graph, x) === T, xs)
            quotient = graph.values[o.value].definition.f === (/)
            return [factors(graph, xs[1], d, T, power); factors(graph, xs[2], d, T, quotient ? -power : power)]
        end
    end
    [(o, power)]
end

# The product of `factors`, operands of the new graph, over `dims`: the one
# factor, or a new value.
function product!(b::Builder, factors, dims, T, scalar = false)
    x = Literal(one(T))
    for (k, (o, power)) in enumerate(factors)
        k == 1 && power == 1 && (x = o; continue)
        id = add!(b, Pointwise(power == 1 ? (*) : (/), (x, o)), dims, T, scalar)
        x = Read(id, identity_index(dims))
    end
    x isa Literal ? Literal(convert(T, x.value)) : x
end

new_factors(b::Builder, fs) = [(remap(o, b.ids), power) for (o, power) in fs]

## The rewrites.

# The index at which fold `F`, read at `index` from a domain whose axes `axes`
# it reduces, in order, reads its operand: its kept axes at `index`, its
# reduced axes the reader's. Nothing when the axes differ in number or
# extent, or the read moves along them.
function aligned(graph, F, index, axes, d::Domain)
    def = graph.values[F].definition
    def isa Reduction && length(def.reduced) == length(axes) || return nothing
    all(j -> def.iteration[def.reduced[j]] == d.extents[axes[j]], eachindex(axes)) || return nothing
    any(e -> any(k -> mentions(e, k), axes), index) && return nothing
    iteration_index(def, index, map(k -> identity_index(d.extents)[k], axes))
end

# A sum whose factors include `exp(y - F)`, where `F` folds `y` over the summed
# axes: `y` and the other factors, or nothing.
function weighted(graph, r, F)
    d = graph.values[r].definition
    domain = Domain(d.iteration)
    T = graph.values[r].eltype
    fs = factors(graph, d.operand, domain, T)
    for (j, (o, power)) in enumerate(fs)
        power == 1 && pointwise(graph, o, exp) && eltype_of(graph, o) === T || continue
        x = only(inner(graph, o, domain))
        pointwise(graph, x, -) || continue
        y, m = inner(graph, x, domain)
        m isa Read && m.value == F && eltype_of(graph, y) === T || continue
        index = aligned(graph, F, m.index, d.reduced, domain)
        (index === nothing || through(graph.values[F].definition.operand, index, domain) != y) && continue
        return y, fs[1:end .!= j]
    end
    nothing
end

# Σ c·g·exp(y - m) over the axes `m` reduces, `c` constant along them, is `c`
# times an online softmax of `(y, g)` for `m` the maximum of `y`; for `m` the
# logsumexp of `y`, `exp(y - m)` is `exp(y - max) / Σ exp(y - max)`, so the
# online softmax is one of `(y, g, 1)` that divides. Done for a maximum or a
# logsumexp when every sum it weighs can be, so that nothing else reads it and
# its pass disappears.
function online_softmaxes(graph::IndexGraph)
    F = 1
    while F <= length(graph.values)
        v = graph.values[F]
        frame = v.definition isa Reduction{<:Union{MaxFold, LogSumExpFold}} && !(v.eltype <: HalfFloat)
        sums = frame ? [r for (r, w) in enumerate(graph.values) if w.definition isa Reduction{SumFold} && weighted(graph, r, F) !== nothing] : Int[]
        if !isempty(sums)
            rewritten, ids = rebuild(graph) do b, r
                r in sums || return nothing
                online!(b, graph, r, F)
            end
            if !live_values(rewritten)[ids[F]]
                graph, F = compact(rewritten), 0
            end
        end
        F += 1
    end
    graph
end

function online!(b::Builder, graph, r, F)
    v = graph.values[r]
    d = v.definition
    y, rest = weighted(graph, r, F)
    along = [any(k -> mentions(o, k), d.reduced) for (o, _) in rest]
    g = product!(b, new_factors(b, rest[along]), d.iteration, v.eltype)
    lse = graph.values[F].definition.fold isa LogSumExpFold
    elements = lse ? (remap(y, b.ids), g, Literal(one(v.eltype))) : (remap(y, b.ids), g)
    E = Tuple{map(_ -> v.eltype, elements)...}
    t = add!(b, Pointwise(tuple, elements), d.iteration, E)
    fold = OnlineSoftmax(lse ? (/) : identity)
    id = add!(b, Reduction(fold, Read(t, identity_index(d.iteration)), d.iteration, d.reduced), v.dims, v.eltype, v.scalar)
    push!(b.rewrites, id => string("online softmax: a sum weighted by exp(y - m), m the ", lse ? "logsumexp" : "maximum", " of y"))
    all(along) && return id
    # The factors constant along the sum, read from its own axes, a numerator
    # first so that a quotient divides.
    kept = iteration_index(d, identity_index(v.dims), map(_ -> constant(0), d.reduced))
    scale = sort(new_factors(b, [(through(o, kept, Domain(v.dims)), power) for (o, power) in rest[.!along]]); by = f -> -f[2])
    product!(b, [(Read(id, identity_index(v.dims)), 1); scale], v.dims, v.eltype, v.scalar).value
end

# The online softmax of `(y, g...)` that `o` reads at the identity of a domain
# of `dims`, observing its one sum; or nothing.
function plain_softmax(graph, o)
    o isa Read && graph.values[o.value].definition isa Reduction{OnlineSoftmax{typeof(identity)}} || return nothing
    o.index == identity_index(graph.values[o.value].dims) ? graph.values[o.value].definition : nothing
end

# `f(a, b)` of two online softmaxes, each observing its one sum, whose
# maximum is of the same `y` over the same axes: one online softmax observing
# `f` of both sums.
function merge_softmaxes(graph::IndexGraph)
    first(rebuild(graph) do b, id
        v = graph.values[id]
        d = v.definition
        d isa Pointwise && length(d.operands) == 2 || return nothing
        a, l = d.operands
        A = plain_softmax(graph, a)
        A !== nothing && l isa Read && l.value != a.value && graph.values[a.value].dims == v.dims || return nothing
        L = graph.values[l.value].definition
        L isa Reduction{OnlineSoftmax{typeof(identity)}} || return nothing
        # `l` as read over the iteration of `A`, whose kept axes are `v`'s.
        domain = Domain(A.iteration)
        whole = identity_index(A.iteration)
        kept = Tuple(whole[k] for k in eachindex(whole) if !(k in A.reduced))
        index = aligned(graph, l.value, map(e -> substitute(e, kept, domain), l.index), A.reduced, domain)
        index === nothing && return nothing
        (ya, ga), (yl, gl) = inner(graph, A.operand, domain), inner(graph, through(L.operand, index, domain), domain)
        ya == yl || return nothing
        elements = map(o -> remap(o, b.ids), (ya, ga, gl))
        t = add!(b, Pointwise(tuple, elements), A.iteration, Tuple{map(o -> eltype_of(graph, o), (ya, ga, gl))...})
        new = add!(b, Reduction(OnlineSoftmax(d.f), Read(t, identity_index(A.iteration)), A.iteration, A.reduced), v.dims, v.eltype, v.scalar)
        push!(b.rewrites, new => "online softmax: $(d.f) of two sharing their maximum")
        new
    end)
end
