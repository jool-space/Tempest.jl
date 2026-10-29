# Algorithms as rewrites of folds, applied to the index-level graph before
# planning where the target's numerics permit them, since they change
# rounding. One mechanism: a sum of `g · φ(y - m)`, where `m` is a statistic
# of `y` over the summed axes, is a relative fold (see `Relative`) when `φ`
# has a law that moves a partial sum from one statistic to another, so that
# `m` is never computed in a pass of its own. Laws are data: `exp` against
# the maximum (the online softmax), the square against the mean (Welford's
# variance). Factors constant along the sum leave it, as the softmax's
# denominator does: Σ v·e/l is (Σ v·e)/l. Then a pointwise function of such
# folds over the same `y`, and of the statistic, is one fold observing that
# function. So the spellings of a softmax, `exp(s - logsumexp(s))` and
# `exp(s - maximum(s)) / sum(exp(s - maximum(s)))`, weighted and summed, reach
# the same fold, and the two-pass variance reaches Welford's. Before them, a
# sum over another sum's product exchanges the two where the sum it moves
# inside is one the program computes (see `exchanged_sums`).

"""
    rewrite(graph) -> IndexGraph

The graph with its sums over a sum's product exchanged where that reads a sum
it computes, then its sums relative to a statistic made relative folds, and
pointwise functions of them composed into the folds' observes, until neither
finds anything; then the statistics something else still reads computed
where the relative folds over the same `y` are. The graph's `rewrites` notes
each, and `explain` prints them.
"""
function rewrite(graph::IndexGraph)
    # A sum is found by its factors before the online rewrite makes it a
    # relative fold.
    graph = exchanged_sums(graph)
    # Each rewrite can make another possible. Compositions nest, `m + log(s)`
    # over `log(s)`, and each pass composes one level; a log-sum-exp written
    # out is a statistic only once composed, and a sum weighted by its
    # exponential is relative to it only then.
    while true
        next = compose_observes(relative_sums(graph))
        length(next.rewrites) == length(graph.rewrites) && return compact(kept_statistics(next))
        graph = next
    end
end

## Building a graph from another, in order.

# The new graph's values, each old value's new id (zero when left out), and
# notes on the values rewrites made.
struct Builder
    values::Vector{Value}
    ids::Vector{Int}
    rewrites::Vector{Pair{Int, String}}
    # Whether the values added now belong to the backward: those of the old
    # value being rebuilt.
    backward::BitVector
    in_backward::Base.RefValue{Bool}
end

"Add a value to the new graph; its id."
function add!(b::Builder, d::Definition, dims, T, scalar = false)
    push!(b.values, Value(d, dims, T, scalar))
    push!(b.backward, b.in_backward[])
    length(b.values)
end

# The graph with old value `id` replaced by the value `f(builder, id)` returns
# the id of (or kept, for `nothing`), leaving out the values `keep` excludes;
# and each old value's new id.
function rebuild(f, graph::IndexGraph, keep = trues(length(graph.values)))
    b = Builder(Value[], zeros(Int, length(graph.values)), Pair{Int, String}[], BitVector(), Ref(false))
    for (id, v) in enumerate(graph.values)
        keep[id] || continue
        b.in_backward[] = graph.backward[id]
        new = f(b, id)
        b.ids[id] = new === nothing ? add!(b, remap(v.definition, b.ids), v.dims, v.eltype, v.scalar) : new
    end
    notes = [[b.ids[v] => s for (v, s) in graph.rewrites if b.ids[v] != 0]; b.rewrites]
    g = IndexGraph(b.values, map(a -> b.ids[a], graph.arguments), Operand[remap(o, b.ids) for o in graph.results],
        graph.result_dims, graph.result_scalar, graph.result_eltype, notes, b.backward)
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
remap(d::Scan, ids) = Scan(d.fold, remap(d.operand, ids), d.axes, d.reverse, d.exclusive)
remap(d::Keyed, ids) = Keyed(d.fold, remap(d.operand, ids), d.iteration, map(e -> remap(e, ids), d.key),
    d.into === nothing ? nothing : remap(d.into, ids))
remap(d::Union{Input, ConstantArray}, ids) = d

# The values something reads: the results and what they read, inputs and
# constant arrays.
function live_values(graph::IndexGraph)
    live = falses(length(graph.values))
    for o in graph.results, r in read_values(o)
        live[r] = true
    end
    for id in length(graph.values):-1:1
        d = graph.values[id].definition
        live[id] |= !(d isa Union{Pointwise, Reduction, Scan, Keyed})
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

applies(graph, o, f) = o isa Read && (d = graph.values[o.value].definition) isa Pointwise && d.f === f

"The operands of the pointwise value `o` reads, as read from the reader's domain."
inner(graph, o::Read, d::Domain) = map(x -> through(x, o.index, d), graph.values[o.value].definition.operands)

mentions(o::Read, k) = any(e -> mentions(e, k), o.index)
mentions(o::IndexOperand, k) = mentions(o.expr, k)
mentions(::Literal, k) = false

# Whether elements of `S` convert to `T` exactly, so that a product in `T` of
# a factor of `S` is the product of the factor converted: narrower floats,
# Booleans, and integers the float holds exactly.
exact(S, T) = S === T || S <: HalfFloat && T === Float32 || S <: Union{HalfFloat, Float32} && T === Float64 ||
    S === Bool && T <: AbstractFloat || S <: Integer && T === Float64 && sizeof(S) <= 4 || S <: Integer && T === Float32 && sizeof(S) <= 2

# Whether an operand of a product in `T` converts to `T` exactly: a literal, an
# operand of an exact type, or a position whose range the float holds exactly.
function exact_operand(graph, x, T, d::Domain)
    x isa Literal && return true
    exact(eltype_of(graph, x), T) && return true
    x isa IndexOperand && T <: AbstractFloat || return false
    lo, hi = affine_range(x.expr, d)
    max(abs(lo), abs(hi)) <= maxintfloat(T)
end

# The factors of the product `o` reads, through the products and quotients of
# type `T` it reads: each an operand and its power, 1 or -1. A factor may be
# of a type that converts to `T` exactly, as a half-precision factor of a
# Float32 product is, and is read through that conversion.
function factors(graph, o, d::Domain, T = eltype_of(graph, o), power = 1)
    if (applies(graph, o, *) || applies(graph, o, /)) && eltype_of(graph, o) === T
        xs = inner(graph, o, d)
        if length(xs) == 2 && all(x -> exact_operand(graph, x, T, d), xs)
            quotient = graph.values[o.value].definition.f === (/)
            return [factors(graph, xs[1], d, T, power); factors(graph, xs[2], d, T, quotient ? -power : power)]
        end
    elseif applies(graph, o, Cast{T}())
        x = only(inner(graph, o, d))
        exact(eltype_of(graph, x), T) && return factors(graph, x, d, eltype_of(graph, x), power)
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

# Σ_R a·(Σ_K b) is Σ_K Σ_R a·b, and the sum over R moves inside past the
# factors constant along R. Where the sum it moves inside is a sum the
# program computes, read at some index, it is that sum, and the sum over R is
# not computed at all: FlashAttention-2's identity, where the softmax
# backward's Σ_j p·dp, dp = Σ_d v·dO, is Σ_d dO·o, o = Σ_j p·v the forward's
# output. Only where the exchanged sum runs over no more elements than the
# sum it replaces.
function exchanged_sums(graph::IndexGraph)
    sums = Pair{Int, Vector{Tuple{Operand, Int}}}[]
    found = Dict{Int, Any}()
    for (r, v) in enumerate(graph.values)
        v.definition isa Reduction{SumFold} || continue
        e = exchange(graph, r, sums)
        e === nothing || (found[r] = e)
        v.scalar || push!(sums, r => factors(graph, v.definition.operand, Domain(v.definition.iteration), v.eltype))
    end
    isempty(found) && return graph
    first(rebuild((b, r) -> haskey(found, r) ? exchanged!(b, graph, r, found[r]) : nothing, graph))
end

# The exchange of sum `r` over a factor that is a sum, where the sum it moves
# inside is among `sums`, the factors of the sums before `r`: whether `r`
# negates its product, the factors left outside over the exchanged domain,
# the sum read, that domain and its axes the outer sum keeps. Or nothing.
function exchange(graph, r, sums)
    v = graph.values[r]
    d = v.definition
    n, domain = length(d.iteration), Domain(d.iteration)
    x, negated = d.operand, false
    if applies(graph, x, -) && length(graph.values[x.value].definition.operands) == 1
        x, negated = only(inner(graph, x, domain)), true
    end
    T = v.eltype
    eltype_of(graph, x) === T || return nothing
    fs = factors(graph, x, domain, T)
    for (j, (f, power)) in enumerate(fs)
        power == 1 && f isa Read && graph.values[f.value].definition isa Reduction{SumFold} && graph.values[f.value].eltype === T || continue
        F = graph.values[f.value].definition
        extents = (d.iteration..., map(k -> F.iteration[k], F.reduced)...)
        wide = Domain(extents)
        index = iteration_index(F, f.index, map(m -> var(n + m), Tuple(eachindex(F.reduced))))
        spread = [fs[1:end .!= j]; factors(graph, through(F.operand, index, wide), wide, T)]
        along = [any(k -> mentions(o, k), d.reduced) for (o, _) in spread]
        kept = Tuple(k for k in eachindex(extents) if !(k in d.reduced))
        prod(k -> extents[k], kept; init = 1) <= prod(d.iteration; init = 1) || continue
        s = computed_sum(graph, sums, spread[along], extents, d.reduced, kept, T)
        s === nothing && continue
        return (; negated, outside = spread[.!along], sum = s, extents, kept)
    end
    nothing
end

# A sum among `sums` that is the sum over `reduced` of the product of `fs`,
# operands of type `T` over `extents`: read at its index over the axes `kept`.
function computed_sum(graph, sums, fs, extents, reduced, kept, T)
    for (V, gs) in sums
        S = graph.values[V].definition
        graph.values[V].eltype === T && length(gs) == length(fs) && length(S.reduced) == length(reduced) || continue
        σ = matched_axes(gs, fs, S.iteration, extents)
        σ !== nothing && Set(σ[k] for k in S.reduced) == Set(reduced) || continue
        values = Tuple(k for k in eachindex(S.iteration) if !(k in S.reduced))
        return Read(V, map(k -> var(findfirst(==(σ[k]), kept)), values))
    end
    nothing
end

# The axes of `extents` that a sum's iteration axes are, such that its
# factors `gs`, read over them, are the factors `fs` in some order; nothing
# when no order makes them so. Each axis is matched where a factor reads it
# alone along one of its own axes.
function matched_axes(gs, fs, iteration, extents)
    sole(e) = e.constant == 0 && length(e.terms) == 1 && e.terms[1].second == 1 && e.terms[1].first isa Var ? e.terms[1].first.n : 0
    function match(i, σ, used)
        if i > length(gs)
            all(!iszero, σ) && allunique(σ) && all(k -> iteration[k] == extents[σ[k]], eachindex(σ)) || return nothing
            index = map(var, Tuple(σ))
            pairs = zip(gs, (fs[j] for j in used))
            return all(((g, f),) -> g[2] == f[2] && through(g[1], index, Domain(extents)) == f[1], pairs) ? σ : nothing
        end
        (g, p) = gs[i]
        for (j, (f, q)) in enumerate(fs)
            (j in used || p != q || typeof(g) != typeof(f)) && continue
            τ = copy(σ)
            if g isa Read
                g.value == f.value || continue
                ok = true
                for (a, b) in zip(g.index, f.index)
                    k, l = sole(a), sole(b)
                    k == 0 && continue
                    l == 0 || τ[k] == 0 || τ[k] == l || (ok = false; break)
                    l == 0 || (τ[k] = l)
                end
                ok || continue
            end
            found = match(i + 1, τ, [used; j])
            found === nothing || return found
        end
        nothing
    end
    match(1, zeros(Int, length(iteration)), Int[])
end

# Sum `r` exchanged: the factors left outside and the sum read, over the
# exchanged domain's kept axes, summed over those the inner sum reduced.
function exchanged!(b::Builder, graph, r, e)
    v = graph.values[r]
    dims = map(k -> e.extents[k], e.kept)
    domain = Domain(dims)
    renumber = ntuple(k -> (p = findfirst(==(k), e.kept)) === nothing ? constant(0) : var(p), length(e.extents))
    outside = new_factors(b, [(through(o, renumber, domain), power) for (o, power) in e.outside])
    fs = sort([outside; (remap(e.sum, b.ids), 1)]; by = f -> -f[2])
    g = product!(b, fs, dims, v.eltype)
    reduced = Tuple((length(v.dims) + 1):length(dims))
    id = add!(b, Reduction(v.definition.fold, g, dims, reduced), v.dims, v.eltype, v.scalar)
    push!(b.rewrites, id => "sums exchanged: Σ a·(Σ b) is Σ (Σ a·b), a sum computed already")
    e.negated || return id
    add!(b, Pointwise(-, (Read(id, identity_index(v.dims)),)), v.dims, v.eltype, v.scalar)
end

"""
A value that is a statistic of `y` over some axes: the law it is the
statistic of, the fold whose state it is (the maximum, a sum, a relative
fold), what it observes of that fold, and `y` as the fold's operand over the
fold's iteration domain.
"""
struct Frame
    law::Law
    fold::Int
    observe::Any
    iteration::Tuple{Vararg{Int}}
    reduced::Tuple{Vararg{Int}}
    y::Operand
end

# `y` of a relative fold's element: the first component of a tuple, or the
# element itself.
element_y(graph, o, d::Domain) = applies(graph, o, tuple) ? first(inner(graph, o, d)) : o

# The sum fold a value is the mean of: the value divides it by the extent of
# its reduced axes, or multiplies it by the exact reciprocal (see
# `map_definition`), reading it over its own domain.
function mean_of(graph, F)
    v = graph.values[F]
    d = v.definition
    d isa Pointwise && length(d.operands) == 2 || return nothing
    o, c = d.operands
    o isa Read && c isa Literal && c.value isa Real && o.index == identity_index(v.dims) || return nothing
    S = graph.values[o.value]
    S.definition isa Reduction{SumFold} && S.dims == v.dims && S.eltype === v.eltype || return nothing
    n = prod(k -> S.definition.iteration[k], S.definition.reduced; init = 1)
    (d.f === (/) && c.value == n || d.f === (*) && c.value * n == 1) ? o.value : nothing
end

function frame(graph, F)
    v = graph.values[F]
    d = v.definition
    if d isa Reduction{MaxFold}
        return Frame(Exponential(), F, Statistic(), d.iteration, d.reduced, d.operand)
    elseif d isa Reduction{<:Relative}
        return Frame(d.fold.law, F, d.fold.observe, d.iteration, d.reduced, element_y(graph, d.operand, Domain(d.iteration)))
    elseif d isa Pointwise
        S = mean_of(graph, F)
        S === nothing && return nothing
        r = graph.values[S].definition
        return Frame(Square(), S, Statistic(), r.iteration, r.reduced, r.operand)
    end
    nothing
end

# How a sum of `g · φ(y - F)` reads the law's sums through what the frame
# observes: the extra elements the fold carries, and the observe of the sum's
# fold as a function of the index of its own sum among the elements. Through
# the statistic itself, the sum; through the log-sum-exp,
# `exp(y - lse) = exp(y - m) / Σ exp(y - m)`: an element of one, and the
# quotient of the two sums.
function through_observe(law::Law, observe)
    observe isa Statistic && return ((), i -> Sum(i))
    law isa Exponential && observe == LOGSUMEXP && return ((1,), i -> Composed(/, (Sum(i), Sum(i + 1))))
    nothing
end

# Whether a law's sums may weigh each element differently: squares about the
# mean move together only when weighted alike.
weighted(::Exponential) = true
weighted(::Square) = false

# The law's function applied among the factors: `exp(x)` for the
# exponential; `abs2(x)`, or `x` twice among the factors, for the square. Its
# argument, and the factors without it.
function applications(graph, fs, law, d::Domain)
    found = Tuple{Operand, Vector{Tuple{Operand, Int}}}[]
    for (j, (e, power)) in enumerate(fs)
        power == 1 || continue
        applies(graph, e, law_function(law)) && push!(found, (only(inner(graph, e, d)), fs[1:end .!= j]))
        law isa Square || continue
        k = findnext(((o, p),) -> p == 1 && o == e, fs, j + 1)
        k === nothing || push!(found, (e, fs[(1:end .!= j) .& (1:end .!= k)]))
    end
    found
end

# A sum whose factors include `φ(y - F)` for the frame's law, where `F` is the
# statistic of `y` over the summed axes: `y` and the other factors, or
# nothing.
function dependent(graph, r, F, fr::Frame)
    d = graph.values[r].definition
    domain = Domain(d.iteration)
    for (x, rest) in applications(graph, factors(graph, d.operand, domain), fr.law, domain)
        applies(graph, x, -) || continue
        y, m = inner(graph, x, domain)
        m isa Read && m.value == F && eltype_of(graph, y) === eltype_of(graph, x) === graph.values[F].eltype || continue
        index = aligned(graph, fr.fold, m.index, d.reduced, domain)
        (index === nothing || through(fr.y, index, domain) != y) && continue
        return y, rest
    end
    nothing
end

function admissible(graph, r, F, fr::Frame)
    found = dependent(graph, r, F, fr)
    found === nothing && return false
    d = graph.values[r].definition
    weighted(fr.law) || all(((o, _),) -> !any(k -> mentions(o, k), d.reduced), found[2])
end

# Every sum relative to a frame becomes a relative fold. The frame stays as
# it is when something else reads it: a fold over the same axes, in the same
# phase as the folds that were relative to it.
function relative_sums(graph::IndexGraph)
    F = 1
    while F <= length(graph.values)
        fr = frame(graph, F)
        seen = fr === nothing ? nothing : through_observe(fr.law, fr.observe)
        rs = seen === nothing ? Int[] : [r for (r, w) in enumerate(graph.values) if w.definition isa Reduction{SumFold} && admissible(graph, r, F, fr)]
        if !isempty(rs)
            graph = first(rebuild(graph) do b, r
                r in rs || return nothing
                relative!(b, graph, r, F, fr, seen)
            end)
        end
        F += 1
    end
    graph
end

# Σ c·g·φ(y - m) over the axes `m` reduces, `c` constant along them, is `c`
# times the relative fold of `(y, g, extras...)` observed as the law says.
function relative!(b::Builder, graph, r, F, fr::Frame, (extras, observe))
    v = graph.values[r]
    d = v.definition
    y, rest = dependent(graph, r, F, fr)
    along = [any(k -> mentions(o, k), d.reduced) for (o, _) in rest]
    # The weights and the sums are of the summed product's type; `y` keeps its
    # own, and the fold accumulates as its element's types say.
    G = eltype_of(graph, d.operand)
    g = product!(b, new_factors(b, rest[along]), d.iteration, G)
    elements = (remap(y, b.ids), g, map(x -> Literal(convert(G, x)), extras)...)
    E = Tuple{eltype_of(graph, y), map(_ -> G, Base.tail(elements))...}
    t = element!(b, graph, elements, E, d.iteration)
    fold = Relative(fr.law, observe(1))
    id = add!(b, Reduction(fold, t, d.iteration, d.reduced), v.dims, v.eltype, v.scalar)
    push!(b.rewrites, id => string(law_note(fr.law), ": a sum of ", law_function(fr.law), "(y - m), m the ",
        fr.observe isa Statistic ? statistic_name(fr.law) : fold_name(graph.values[F].definition.fold), " of y"))
    all(along) && return id
    # The factors constant along the sum, read from its own axes, a numerator
    # first so that a quotient divides.
    kept = iteration_index(d, identity_index(v.dims), map(_ -> constant(0), d.reduced))
    scale = sort(new_factors(b, [(through(o, kept, Domain(v.dims)), power) for (o, power) in rest[.!along]]); by = f -> -f[2])
    product!(b, [(Read(id, identity_index(v.dims)), 1); scale], v.dims, v.eltype, v.scalar).value
end

# A statistic of `y` that something still reads after the sums relative to it
# have become relative folds, where a relative fold over more axes folds the
# same `y` under the same law: the group that computes that fold has the
# statistic too, as FlashAttention's forward writes the log-sum-exp beside its
# output for the backward. The statistic becomes the same fold over the
# relative fold's iteration, which does not vary along the axes it does not
# keep, read at the first of them; nothing computes it apart.
function kept_statistics(graph::IndexGraph)
    live = live_values(graph)
    found = Dict{Int, Tuple{Int, Tuple{Vararg{Affine}}}}()
    for F in eachindex(graph.values)
        live[F] && graph.values[F].definition isa Reduction || continue
        fr = frame(graph, F)
        fr === nothing && continue
        for R in eachindex(graph.values)
            live[R] && R != F || continue
            e = graph.values[R].definition
            e isa Reduction{<:Relative} && e.fold.law == fr.law && length(e.iteration) > length(fr.iteration) || continue
            index = embedding(graph, fr, R)
            index === nothing && continue
            found[F] = (R, index)
            break
        end
    end
    isempty(found) && return graph
    first(rebuild(graph) do b, id
        haskey(found, id) || return nothing
        kept!(b, graph, id, found[id]...)
    end)
end

# The iteration index of the frame's fold over relative fold `R`'s domain at
# which the frame's `y` is `R`'s, its reduced axes `R`'s, each of its other
# axes one of `R`'s other axes of the same extent; nothing when there is none.
# Axes of the same extent are told apart by `y`, as attention's keys and
# queries are.
function embedding(graph, fr::Frame, R)
    e = graph.values[R].definition
    domain = Domain(e.iteration)
    y = element_y(graph, e.operand, domain)
    kept = [k for k in eachindex(fr.iteration) if !(k in fr.reduced)]
    others = [k for k in eachindex(e.iteration) if !(k in e.reduced)]
    length(fr.reduced) == length(e.reduced) && length(kept) <= length(others) || return nothing
    function search(chosen)
        if length(chosen) == length(kept)
            index = ntuple(length(fr.iteration)) do k
                j = findfirst(==(k), fr.reduced)
                var(j === nothing ? chosen[findfirst(==(k), kept)] : e.reduced[j])
            end
            all(j -> fr.iteration[fr.reduced[j]] == e.iteration[e.reduced[j]], eachindex(fr.reduced)) || return nothing
            return through(fr.y, index, domain) == y ? index : nothing
        end
        k = kept[length(chosen) + 1]
        for a in others
            (a in chosen || e.iteration[a] != fr.iteration[k]) && continue
            found = search([chosen; a])
            found === nothing || return found
        end
        nothing
    end
    search(Int[])
end

# Statistic `F` as fold `R`'s group computes it: the fold over `R`'s iteration,
# its operand read at `index`, and its value at `F`'s positions, `R`'s other
# kept axes at zero.
function kept!(b::Builder, graph, F, R, index)
    v, w = graph.values[F], graph.values[R]
    d, e = v.definition, w.definition
    over = add!(b, Reduction(d.fold, remap(through(d.operand, index, Domain(e.iteration)), b.ids), e.iteration, e.reduced), w.dims, v.eltype)
    kept = [k for k in eachindex(d.iteration) if !(k in d.reduced)]
    slice = map(k for k in eachindex(e.iteration) if !(k in e.reduced)) do a
        j = findfirst(k -> index[k] == var(a), kept)
        j === nothing ? constant(0) : var(j)
    end
    id = add!(b, Pointwise(identity, (Read(over, Tuple(slice)),)), v.dims, v.eltype, v.scalar)
    push!(b.rewrites, id => string("the ", fold_name(d.fold), " of y computed beside a relative fold of y over more axes"))
    id
end

law_note(::Exponential) = "online softmax"

# The operand of a relative fold of `elements` over `iteration`: a tuple
# value, or `y` itself when the elements are `y` and a literal one, which is
# what a number is as an element (see `fold_lift`).
function element!(b::Builder, graph, elements, E, iteration)
    y, gs = first(elements), Base.tail(elements)
    one_only = length(gs) == 1 && only(gs) isa Literal && isone(only(gs).value) && fieldtype(E, 2) === fieldtype(E, 1)
    one_only && return y
    Read(add!(b, Pointwise(tuple, elements), iteration, E), identity_index(iteration))
end
law_note(::Square) = "Welford"

# A relative fold, or a maximum, that `o` reads: its law, what it observes,
# its value, iteration and reduced axes, its operand, and whether it has sums
# (a maximum has none).
function relative_read(graph, o)
    o isa Read || return nothing
    d = graph.values[o.value].definition
    if d isa Reduction{<:Relative}
        return (law = d.fold.law, observe = d.fold.observe, value = o.value, iteration = d.iteration, reduced = d.reduced, operand = d.operand, index = o.index, sums = true)
    elseif d isa Reduction{MaxFold}
        return (law = Exponential(), observe = Statistic(), value = o.value, iteration = d.iteration, reduced = d.reduced, operand = d.operand, index = o.index, sums = false)
    end
    nothing
end

# The sums `g...` of a relative fold's operand, as read over a domain: those
# of a tuple element, one for a number, none for a maximum.
function element_sums(graph, r, operand, d::Domain)
    r.sums || return Operand[]
    applies(graph, operand, tuple) ? collect(Operand, Base.tail(inner(graph, operand, d))) : Operand[Literal(one(eltype_of(graph, operand)))]
end

# An observe with its sums numbered from `offset` on.
shift(o::Sum, offset) = Sum(o.i + offset)
shift(o::Composed, offset) = Composed(o.f, map(a -> a isa Number ? a : shift(a, offset), o.args))
shift(o, offset) = o

# `f` of relative folds over the same `y` and axes under one law, and of the
# statistic, and of numbers, where `f` alone reads each of them: one relative
# fold over the union of their elements, observing `f` of what each observed.
# The folds may iterate over different domains, as the softmax's denominator
# and the weighted sum of attention do: each is read over the domain of the
# fold with the most axes.
#
# A pointwise function of a generic fold's result, read over the fold's own
# domain directly or through pointwise values of it, is the fold observing
# that function: a fold whose state is a tuple, as Base's `mapreduce` writes
# one, is observed by the broadcast after it, which reads the state's
# components.
function compose_observes(graph::IndexGraph)
    live = live_values(graph)
    users = readers(graph)
    results = Set(o.value for o in graph.results if o isa Read)
    first(rebuild(graph) do b, id
        v = graph.values[id]
        d = v.definition
        d isa Pointwise && d.f !== tuple && all(o -> o isa Literal || o isa Read, d.operands) || return nothing
        live[id] && (observed = observed_fold!(b, graph, id, users, live, results)) !== nothing && return observed
        reads = [relative_read(graph, o) for o in d.operands]
        folds = filter(!isnothing, reads)
        isempty(folds) && return nothing
        all(((o, r),) -> o isa Literal || r !== nothing, zip(d.operands, reads)) || return nothing
        # Composing replaces each fold: nothing else that runs reads it.
        all(r -> !(r.value in results) && all(u -> u == id || !live[u], users[r.value]), folds) || return nothing
        a = folds[argmax(map(r -> length(r.iteration), folds))]
        a.index == identity_index(v.dims) && graph.values[a.value].dims == v.dims || return nothing
        all(r -> r.law == a.law, folds) || return nothing
        domain = Domain(a.iteration)
        whole = identity_index(a.iteration)
        kept = Tuple(whole[k] for k in eachindex(whole) if !(k in a.reduced))
        ya = element_y(graph, a.operand, domain)
        # Each fold's sums after those before it; a sum equal to an earlier one
        # is that one.
        gs = Operand[]
        args = Any[]
        for (o, r) in zip(d.operands, reads)
            o isa Literal && (push!(args, o.value); continue)
            index = aligned(graph, r.value, map(e -> substitute(e, kept, domain), r.index), a.reduced, domain)
            index === nothing && return nothing
            operand = through(r.operand, index, domain)
            element_y(graph, operand, domain) == ya || return nothing
            places = map(g -> (k = findfirst(==(g), gs)) === nothing ? (push!(gs, g); length(gs)) : k, element_sums(graph, r, operand, domain))
            arg = renumbered(r.observe, places)
            arg === nothing && return nothing
            push!(args, arg)
        end
        elements = map(o -> remap(o, b.ids), (ya, gs...))
        t = element!(b, graph, elements, Tuple{map(o -> eltype_of(graph, o), (ya, gs...))...}, a.iteration)
        new = add!(b, Reduction(Relative(a.law, Composed(d.f, Tuple(args))), t, a.iteration, a.reduced), v.dims, v.eltype, v.scalar)
        push!(b.rewrites, new => string(law_note(a.law), ": $(d.f) of folds sharing their $(statistic_name(a.law))"))
        new
    end)
end

# The generic fold the pointwise definition `d`, over `dims`, is a function
# of, through the pointwise values over `dims` it reads, which go to
# `between`: zero when it reads no fold, nothing when it reads anything else
# or two folds.
function state_source(graph, d::Pointwise, dims, between)
    F = 0
    for o in d.operands
        o isa Literal && continue
        o isa Read && o.index == identity_index(dims) && graph.values[o.value].dims == dims || return nothing
        e = graph.values[o.value].definition
        G = if e isa Reduction{<:GenericFold}
            o.value
        elseif e isa Pointwise && e.f !== tuple
            push!(between, o.value)
            state_source(graph, e, dims, between)
        end
        G === nothing && return nothing
        F == 0 || G == 0 || F == G || return nothing
        G == 0 || (F = G)
    end
    F
end

# `d` as an observe of the state of fold `F`: the fold's own observe where it
# reads `F`.
observing(graph, d::Pointwise, F) = Composed(d.f, map(d.operands) do o
    o isa Literal && return o.value
    o.value == F ? graph.values[F].definition.fold.observe : observing(graph, graph.values[o.value].definition, F)
end)

# Value `id` as the fold it is a function of, observing it, when nothing else
# that runs reads the fold or the values between; or nothing.
function observed_fold!(b::Builder, graph, id, users, live, results)
    v = graph.values[id]
    between = Int[]
    F = state_source(graph, v.definition, v.dims, between)
    F isa Int && F != 0 || return nothing
    inside = Set([id; between])
    all(u -> !(u in results) && all(w -> w in inside || !live[w], users[u]), [F; between]) || return nothing
    r = graph.values[F].definition
    fold = GenericFold(r.fold.init, r.fold.lift, r.fold.combine, observing(graph, v.definition, F), nothing)
    new = add!(b, Reduction(fold, remap(r.operand, b.ids), r.iteration, r.reduced), v.dims, v.eltype, v.scalar)
    push!(b.rewrites, new => string("a fold observing ", v.definition.f, " of its result"))
    new
end

# An observe with its `i`th sum the `places[i]`th of a larger element; nothing
# for an observe that does not renumber.
renumbered(o::Sum, places) = Sum(places[o.i])
renumbered(o::Composed, places) = (args = map(a -> a isa Number ? a : renumbered(a, places), o.args); any(isnothing, args) ? nothing : Composed(o.f, args))
renumbered(o::Statistic, places) = o
renumbered(o, places) = nothing
