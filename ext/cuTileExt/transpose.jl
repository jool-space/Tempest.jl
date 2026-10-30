# Folds that read a product's tiles transposed. Attention's backward sums dV
# and dK over the queries after each key, and dQ over the keys before each
# query: three folds over one domain, a position kept and one summed, whose
# terms recompute the scores. A block that keeps a tile of positions and
# sweeps the others computes dV and dK with the scores as Sᵀ, keys by
# queries, and dQ with them as S, queries by keys, so each pair of tiles
# recomputes the scores and the probabilities' gradient twice: seven products
# where FlashAttention-2 computes five. With its kept and summed positions
# exchanged, dQ's term is a fold over the block's keys at the step's queries
# that reads the tiles dK's term reads. Summed over the block's tile at each
# step, and added into dQ at the step's tile atomically, as FlashAttention-2
# adds its dQ, it is two products fewer: at 1024 positions, 64 features, 8
# heads, causal, TF32, the backward's kernel took 418 μs against 594 at the
# same tiles with 8 worker warps (see `ProductOptions`), and 307 at the tiles
# and options the search then found, 16 queries a step and the scores'
# products 64 deep.

"""
    transposing(c) -> Context

The context of a product schedule that `transposes`: its folds that share no
fold they recompute with the group's first fold, but would with a kept
position and the position they sum exchanged, so exchanged. Each becomes a
fold over that kept tile axis, the product's rows or columns, at the swept
one's positions, which the sweep adds into its value at each step (see
`added_folds`). The positions are one domain axis each, of one extent; the
folds are sums of 32-bit floats, as an atomic addition takes them, read by
no fold of the group (`takes_additions`), and the group's folds are one
phase. The context as it is otherwise. MLA's backward keeps the features as
the rows, 96 of them, and the queries or keys as the columns.
"""
function transposing(c::Context)
    s, g = c.s, c.g
    s.sketch === :product && s.options.transposes && s.splits == 1 && sweeps(s) > 1 || return c
    p = sweep_axis(s)
    swept(s) == [p] && length(reduction_phases(c)) == 1 && length(s.tiling[p].axes) == 1 || return c
    kp = only(s.tiling[p].axes)
    d = Domain(g.domain)
    for m in (1, 3)
        ax = s.tiling[m]
        length(ax.axes) == 1 && ax.tile > 1 && g.domain[only(ax.axes)] == g.domain[kp] || continue
        km = only(ax.axes)
        σ = ntuple(k -> Affine(0, (Var(k == kp ? km : k == km ? kp : k) => 1,)), length(g.domain))
        folds = transposed_folds(c, σ, m, d)
        isempty(folds) || return exchanged(c, folds, σ, km, d)
    end
    c
end

# The context with `folds` exchanged by `σ`: each its term, its placement,
# and the axis it reduces, `km`.
function exchanged(c::Context, folds, σ, km, d)
    g = c.g
    terms, placement, values = copy(g.terms), copy(g.placement), copy(g.values)
    for (r, t) in folds
        v = values[r]
        terms[r] = t
        placement[r] = map(e -> IR.substitute(e, σ, d), g.placement[r])
        values[r] = ValueView(v.eltype, v.dims, v.scalar, v.stored, v.fold, (km,), v.keyed, v.into, v.scan, v.masked, v.carried)
    end
    Context(GroupView(g.kind, g.domain, g.reductions, g.outputs, terms, g.phase, g.primary_axes, values, placement, g.bytes, g.reference), c.s, c.held)
end

# The folds to exchange, by their terms exchanged: those of the smaller of two
# sides, the folds that share what they recompute with the first fold, through
# each other, and the rest, when every fold of that side can be exchanged and
# then shares with the other side; none otherwise.
function transposed_folds(c::Context, σ, m, d)
    g = c.g
    phase = primary_folds(c)
    found = Dict{Int, Term}()
    isempty(phase) && return found
    recomputed(t) = Set{Term}(x for x in subterms((t,)) if x isa Recomputed)
    own = Dict(r => recomputed(g.terms[r]) for r in phase)
    aligned, seen = [first(phase)], copy(own[first(phase)])
    while (i = findfirst(r -> !(r in aligned) && !isempty(intersect(own[r], seen)), phase)) !== nothing
        push!(aligned, phase[i])
        union!(seen, own[phase[i]])
    end
    rest = filter(!in(aligned), phase)
    isempty(rest) && return found
    function side(rs, others)
        for r in rs
            t = transposable(c, r, σ, m, d)
            (t === nothing || isempty(intersect(recomputed(t), others))) && return nothing
            found[r] = t
        end
        found
    end
    chosen = length(rest) <= length(aligned) ? side(rest, seen) : nothing
    if chosen === nothing
        empty!(found)
        chosen = side(aligned, reduce(union, (own[r] for r in rest)))
    end
    chosen === nothing ? empty!(found) : found
end

# Fold `r`'s term with its positions exchanged by `σ`, when it can be added
# from the sweep: a sum of 32-bit floats whose value takes the additions
# (`takes_additions`), whose exchanged term the generator lowers, a product
# over kept tile axis `m`, and whose value is a view at the exchanged
# positions; nothing otherwise.
function transposable(c::Context, r, σ, m, d)
    g = c.g
    v = g.values[r]
    v.fold isa SumFold && !v.keyed && v.carried === nothing && v.eltype === Float32 && term_eltype(g.terms[r], g) === Float32 || return nothing
    g.placement[r] === nothing && return nothing
    takes_additions(g, r) || return nothing
    t = renamed(g.terms[r], σ, d)
    (t === nothing || bilinear_over(c, SumFold(), t, m) === nothing) && return nothing
    terms_expressible(c, (t,)) || return nothing
    view_of(g, r, map(e -> IR.substitute(e, σ, d), g.placement[r]), groups(c)...) === nothing ? nothing : t
end

# Term `t` over domain `d` with its variables renamed by `σ`, an index over
# `d`, a fold it recomputes keeping its own variables after the domain's;
# nothing when it reads a fold of the group, which a block holds at its own
# position only.
renamed(t::Lit, σ, d) = t
renamed(t::Stat, σ, d) = nothing
renamed(t::Load, σ, d) = Load(t.value, map(e -> IR.substitute(e, σ, d), t.index))
renamed(t::Ix, σ, d) = typeof(t)(IR.substitute(t.expr, σ, d))
function renamed(t::Apply, σ, d)
    args = map(a -> renamed(a, σ, d), t.args)
    any(isnothing, args) ? nothing : Apply(t.f, args, t.eltype)
end
function renamed(t::Recomputed, σ, d)
    n = length(d.extents)
    inner = (σ..., ntuple(j -> Affine(0, (Var(n + j) => 1,)), length(t.extents))...)
    body = renamed(t.body, inner, Domain((d.extents..., t.extents...)))
    body === nothing ? nothing : Recomputed(t.value, map(e -> IR.substitute(e, σ, d), t.index), t.extents, body)
end

# Whether fold `r`'s value can take the additions of a sweep: what reads it in
# the group are outputs of it alone, which a launch after the sweep computes
# from it (`late_outputs`), and it is stored or read by one; a workspace holds
# it unstored. BFloat16 attention's query gradient is so, the rounding of a
# sum of 32-bit floats that is not stored itself.
function takes_additions(g::GroupView, r)
    reads(m) = any(s -> s isa Stat && s.value == r, subterms((g.terms[m],); nested = true))
    alone(m) = m in g.outputs && !(m in g.reductions) && all(s -> !(s isa Stat) || s.value == r, subterms((g.terms[m],); nested = true))
    readers = [m for m in members(g) if haskey(g.terms, m) && reads(m)]
    all(alone, readers) && (g.values[r].stored || !isempty(readers))
end

"""
    added_folds(c) -> folds

The folds over a kept tile axis that a schedule which `transposes` adds into
their values from its sweep: sums of 32-bit floats, each a product over that
axis, whose step is the product over the block's tile at the step's tile of
the value, added atomically (see `add_step!`), into a workspace when the
value is not stored. They start from zeros, written by a launch before
(`initial_states_kernel`), and the outputs that read them are computed by a
launch after (`late_kernel`).
"""
function added_folds(c::Context)
    c.s.options.transposes && c.s.splits == 1 && sweeps(c.s) > 1 || return Int[]
    filter(partial_folds(c)) do r
        v, tiles = c.g.values[r], fold_tiles(c, r)
        v.fold isa SumFold && v.eltype === Float32 && takes_additions(c.g, r) && tiles !== nothing && length(tiles) == 1 &&
            !c.s.tiling[only(tiles)].reduced && bilinear_over(c, SumFold(), c.g.terms[r], only(tiles)) !== nothing
    end
end

# The step of an added fold at the tile of block indices `indices`, in the
# step's `scope`: its product over the block's tile, added into the value at
# the step's tile.
function add_step!(k, st, c::Context, r, scope, indices)
    m = only(fold_tiles(c, r))
    product = bilinear_over(c, SumFold(), c.g.terms[r], m)
    x, y, i, j = product
    emit_tiles!(k, st, c, [product_factor(c, x), product_factor(c, y)], scope, indices...)
    acc = gensym(:sum)
    push!(st, :($acc = fill(0.0f0, ($(tile(c)[i]), $(tile(c)[j])))))
    accumulate_product!(st, c, scope, product, acc, indices[m], m)
    v, spec = fold_view(c, r)
    push!(st, store_nd(argument!(k, spec), v, tile(c), indices, from_matrix(c, acc, i, j); add = true))
end

# The outputs of the group that read a fold the sweep adds into, which only a
# launch after the sweep sees complete.
function late_outputs(c::Context)
    added = added_folds(c)
    isempty(added) && return Int[]
    [m for m in c.g.outputs if any(s -> s isa Stat && s.value in added, subterms((c.g.terms[m],); nested = true))]
end

# The launch after a transposed sweep that computes outputs `ms` from the
# folds `rs` it added into, in the schedule's own orientation `c`: each tile
# of the positions the folds keep on a block of its own, the folds loaded from
# where the sweep added into them.
function late_kernel(c::Context, ms, rs, workspaces)
    k = TileKernel("outputs of the folds added", (), nothing)
    kept = [i for i in eachindex(c.s.tiling) if !c.s.tiling[i].reduced]
    counts = tile_counts(c, kept)
    indices = Any[:(Int32(1)) for _ in c.s.tiling]
    indices[kept] = linear_indices!(k.body, counts, :tile_)
    k.grid = (prod(counts; init = 1),)
    names = Dict{Term, Any}()
    load_folds!(k, k.body, c, rs, names, indices, workspaces)
    emit_tiles!(k, k.body, c, [c.g.terms[m] for m in ms], names, indices...)
    store_values!(k, k.body, c, ms, names, indices...)
    k
end
