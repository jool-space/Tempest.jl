# Writes alike, joined. A loop over the slices of an axis whose bodies write
# the same elements at the same keys but for a position along that axis
# traces as one write per slice, each over the contents the one before left:
# `K[:, positions[:, b], :, b] .= k[:, :, :, b]` for each `b` is a chain of
# folds by key that differ only in the constant `b` (see `Keyed`). Where those
# constants step evenly and the key holds them apart, so that no two writes
# land at one position, the order between them states nothing, and the chain
# is one fold by key over its iteration and an axis more, along which the
# constant steps: a map over the slices, as one statement over the whole
# array would trace. Elements a write computes from its slice alone, the
# slice converted to the array's type say, are alike the same way, each a
# value of its own: they are stacked into one value over that axis too. The
# rewrite is exact, and made whatever the target's numerics.

"""
    joined_writes(graph) -> IndexGraph

The graph with each run of folds by key that write into one another, alike
but for constants that step evenly from one to the next and that keep their
positions apart, joined into one fold over the run's steps (see `joined`).
A fold whose contents something else reads, between two writes, ends a run.
The graph's `rewrites` notes each.
"""
function joined_writes(graph::IndexGraph)
    joins = written_runs(graph)
    isempty(joins) && return graph
    keep = trues(length(graph.values))
    last_of = Dict{Int, WriteJoin}()
    for j in joins
        foreach(t -> keep[t] = false, j.run[1:(end - 1)])
        foreach(u -> keep[u] = false, j.members)
        last_of[last(j.run)] = j
    end
    first(rebuild(graph, keep) do b, id
        haskey(last_of, id) || return nothing
        j = last_of[id]
        # The stacked values first, each read by those after it.
        ids = [b.ids; zeros(Int, length(j.stacked))]
        for (k, w) in enumerate(j.stacked)
            ids[length(graph.values) + k] = add!(b, Pointwise(w.f, map(o -> remap(o, ids), w.operands)), w.dims, w.eltype)
        end
        v = graph.values[id]
        new = add!(b, remap(j.fold, ids), v.dims, v.eltype, v.scalar)
        push!(b.rewrites, new => string(length(j.run), " writes alike but for a position each lands apart at, joined: one over their steps"))
        new
    end)
end

"""
A map of `operands` by `f` over `dims`, the axes of the values a step of a
join reads and the steps: the operands as found, in the order the maps of
each step have theirs, which a `Pointwise` of a commutative `f` reorders.
"""
struct Stacked
    f::Any
    operands::Tuple{Vararg{Operand}}
    dims::Tuple{Vararg{Int}}
    eltype::DataType
end

"""
A join being found, of the folds by key `run`, `steps` of them, into `fold`:
its elements may read maps `stacked`, numbered after the graph's values;
`members` are the values of each step they stack, which nothing else reads.
"""
mutable struct WriteJoin
    const graph::IndexGraph
    const counts::Vector{Int}
    const steps::Int
    const stacked::Vector{Stacked}
    const members::Vector{Int}
    run::Vector{Int}
    fold::Union{Nothing, Keyed}
end

WriteJoin(graph, counts, steps) = WriteJoin(graph, counts, steps, Stacked[], Int[], Int[], nothing)

# How many times each value is read: by an operand of each definition, a
# state's value carried from, and the results.
function read_counts(graph::IndexGraph)
    counts = zeros(Int, length(graph.values))
    count!(o) = foreach(r -> counts[r] += 1, read_values(o))
    for v in graph.values
        foreach(count!, operands(v.definition))
        v.definition isa Carried && count!(v.definition.from)
    end
    foreach(count!, graph.results)
    counts
end

# The joins to make: along each chain of folds by key, each over the
# contents of the one before and read by nothing else, the longest runs from
# its first that join, in turn.
function written_runs(graph::IndexGraph)
    values = graph.values
    counts = read_counts(graph)
    # The fold each one writes over, where it continues a chain.
    before = zeros(Int, length(values))
    for (id, v) in enumerate(values)
        d = v.definition
        d isa Keyed && d.into isa Read && d.into.index == identity_index(v.dims) || continue
        p = d.into.value
        w = values[p]
        w.definition isa Keyed && w.dims == v.dims && w.eltype === v.eltype && counts[p] == 1 && (before[id] = p)
    end
    after = zeros(Int, length(values))
    foreach(id -> before[id] == 0 || (after[before[id]] = id), eachindex(values))
    joins = WriteJoin[]
    for id in eachindex(values)
        before[id] == 0 && after[id] != 0 || continue
        chain = [id]
        while after[last(chain)] != 0
            push!(chain, after[last(chain)])
        end
        s = 1
        while s < length(chain)
            j = longest_join(graph, counts, chain[s:end])
            j === nothing ? (s += 1) : (push!(joins, j); s += length(j.run))
        end
    end
    joins
end

# The longest run from the first of `chain` that joins, of two writes or
# more; or nothing.
function longest_join(graph, counts, chain)
    term(t) = graph.values[t].definition::Keyed
    # The constants are found from the first two; how far they keep
    # stepping, over the whole chain; the fold, over the steps that do.
    j = joined(WriteJoin(graph, counts, length(chain)), term(chain[1]), term(chain[2]))
    j === nothing && return nothing
    steps = something(findfirst(k -> !at_step(j, k - 1, term(chain[k])), eachindex(chain)), length(chain) + 1) - 1
    steps >= 2 || return nothing
    steps == length(chain) || (j = joined(WriteJoin(graph, counts, steps), term(chain[1]), term(chain[2])))
    j === nothing && return nothing
    empty!(j.members)
    all(k -> at_step(j, k - 1, term(chain[k])), 1:steps) && apart(j.fold) && allunique(j.members) || return nothing
    j.run = chain[1:steps]
    j
end

"""
    joined(j, a, b) -> WriteJoin or nothing

`j` with its fold by key over the iteration of folds `a` and `b` and an axis
of its steps more, whose first step is `a` and second `b`, where the two are
alike but for constants: each constant of `b` that differs from `a`'s, `δ`
further on, is `a`'s plus `δ` times the new axis's variable. A value one
reads where the other reads another, alike it so, is read from the two
stacked. It starts from the contents `a` starts from. Its later steps are
whatever the constants step to; `at_step` asks whether they are a chain's.
"""
function joined(j::WriteJoin, a::Keyed, b::Keyed)
    isequal(a.fold, b.fold) && a.iteration == b.iteration && length(a.key) == length(b.key) || return nothing
    v = length(a.iteration) + 1
    d = Domain((a.iteration..., j.steps))
    operand = sliding(j, a.operand, b.operand, v, d)
    key = map((x, y) -> sliding(j, x, y, v, d), a.key, b.key)
    (operand === nothing || any(isnothing, key)) && return nothing
    j.fold = Keyed(a.fold, operand, (a.iteration..., j.steps), key, a.into)
    j
end

# The expression over a domain of one variable `v` more that is `a` where
# `v` is zero and `b` where it is one, the two alike but for constants; or
# nothing.
function sliding(j, a::Affine, b::Affine, v, d::Domain)
    length(a.terms) == length(b.terms) && all(((x, y),) -> x.second == y.second, zip(a.terms, b.terms)) || return nothing
    e = add(mul(var(v), b.constant - a.constant, d), a.constant, d)
    for ((x, c), (y, _)) in zip(a.terms, b.terms)
        z = sliding(j, x, y, v, d)
        z === nothing && return nothing
        e = add(e, scale(z, c), d)
    end
    e
end

sliding(j, x::Var, y::Var, v, d) = x == y ? var(x.n) : nothing
function sliding(j, x::Div, y::Div, v, d)
    e = x.d == y.d ? sliding(j, x.e, y.e, v, d) : nothing
    e === nothing ? nothing : fdiv(e, x.d, d)
end
function sliding(j, x::Mod, y::Mod, v, d)
    e = x.d == y.d ? sliding(j, x.e, y.e, v, d) : nothing
    e === nothing ? nothing : fmod(e, x.d, d)
end
function sliding(j, x::Indirect, y::Indirect, v, d)
    x.value == y.value && x.lo == y.lo && x.hi == y.hi && length(x.index) == length(y.index) || return nothing
    index = map((i, k) -> sliding(j, i, k, v, d), x.index, y.index)
    any(isnothing, index) && return nothing
    normalize(0, (Indirect(x.value, index, x.lo, x.hi) => 1,), d)
end

function sliding(j, a::Read, b::Read, v, d)
    length(a.index) == length(b.index) || return nothing
    index = map((i, k) -> sliding(j, i, k, v, d), a.index, b.index)
    any(isnothing, index) && return nothing
    a.value == b.value && return Read(a.value, index)
    s = stack!(j, a.value, b.value)
    s === nothing ? nothing : Read(s, (index..., var(v)))
end
function sliding(j, a::IndexOperand{T}, b::IndexOperand{T}, v, d) where {T}
    e = sliding(j, a.expr, b.expr, v, d)
    e === nothing ? nothing : IndexOperand{T}(e)
end
sliding(j, a::Literal, b::Literal, v, d) = isequal(a, b) ? a : nothing
# Unlike atoms or operands.
sliding(j, a, b, v, d) = nothing

# The value over the axes of maps `x` and `y` and the steps, whose first step
# is `x` and second `y`, where the two are alike but for constants and
# nothing else reads them; its number after the graph's values, or nothing.
function stack!(j::WriteJoin, x, y)
    values = j.graph.values
    a, b = values[x], values[y]
    a.definition isa Pointwise && b.definition isa Pointwise && isequal(a.definition.f, b.definition.f) &&
        a.dims == b.dims && a.eltype === b.eltype && !a.scalar && !b.scalar && j.counts[x] == j.counts[y] == 1 &&
        length(a.definition.operands) == length(b.definition.operands) || return nothing
    v = length(a.dims) + 1
    d = Domain((a.dims..., j.steps))
    operands = map((o, p) -> sliding(j, o, p, v, d), a.definition.operands, b.definition.operands)
    any(isnothing, operands) && return nothing
    push!(j.stacked, Stacked(a.definition.f, operands, (a.dims..., j.steps), a.eltype))
    length(values) + length(j.stacked)
end

# Whether the joined fold of `j` at step `k` of its last axis is the fold
# `t`, but for the contents each starts from; the values `t` reads where the
# fold reads stacked ones are the join's members.
function at_step(j::WriteJoin, k, t::Keyed)
    d = j.fold
    n = length(t.iteration)
    d.iteration[1:n] == t.iteration && isequal(d.fold, t.fold) || return false
    at = (ntuple(var, n)..., constant(k))
    own = Domain(t.iteration)
    matches(j, k, d.operand, t.operand, at, own) && map(e -> substitute(e, at, own), d.key) == t.key
end

# Whether operand `o` of the join, at step `k` of the variables `at` over a
# step's domain `own`, is `p`.
function matches(j::WriteJoin, k, o::Read, p::Read, at, own)
    index = map(e -> substitute(e, at, own), o.index)
    o.value <= length(j.graph.values) && return o.value == p.value && index == p.index
    last(index) == constant(k) && index[1:(end - 1)] == p.index && member(j, k, o.value, p.value) || return false
    push!(j.members, p.value)
    true
end
matches(j::WriteJoin, k, o::IndexOperand{T}, p::IndexOperand{T}, at, own) where {T} = substitute(o.expr, at, own) == p.expr
matches(j::WriteJoin, k, o::Literal, p::Literal, at, own) = isequal(o, p)
matches(j::WriteJoin, k, o, p, at, own) = false

# Whether stacked value `s` at step `k` is the value `u`, which nothing else reads.
function member(j::WriteJoin, k, s, u)
    w = j.stacked[s - length(j.graph.values)]
    x = j.graph.values[u]
    n = length(x.dims)
    x.definition isa Pointwise && isequal(x.definition.f, w.f) && w.dims[1:n] == x.dims && length(w.dims) == n + 1 &&
        x.eltype === w.eltype && !x.scalar && j.counts[u] == 1 || return false
    at = (ntuple(var, n)..., constant(k))
    own = Domain(x.dims)
    length(w.operands) == length(x.definition.operands) &&
        all(((o, p),) -> matches(j, k, o, p, at, own), zip(w.operands, x.definition.operands))
end

# Whether no two steps of the joined fold `d` land at one position: some
# component of its key is `e + s·v`, `v` the steps' variable and `e` not
# reading it, where `e` spans less than `s`.
function apart(d::Keyed)
    v = length(d.iteration)
    own = Domain(d.iteration[1:(v - 1)])
    any(d.key) do e
        s = coefficient(e, v)
        (s === nothing || s == 0) && return false
        lo, hi = affine_range(Affine(e.constant, filter(t -> t.first != Var(v), e.terms)), own)
        hi - lo < abs(s)
    end
end
