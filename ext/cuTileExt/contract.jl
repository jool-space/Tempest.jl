# The product sketch: tile axes for reduction groups whose sums are of
# bilinear products, over the axes `Tempest.Compiler.Implement.product_axes`
# finds. The reduced axes (K) are swept. Tile order (M, K, N, batch) puts each
# factor's tile in matrix order, so the fold algebra's product rule applies.
# M, N and the batch each collapse into one tile axis where every view chains
# over them; otherwise the longest axis of M and of N is tiled and every other
# kept axis is a grid axis of its own. K collapses likewise, or its longest
# piece that chains is tiled and each other piece is swept an index per step,
# as grouped heads and queries are in attention's backward pass. Everything
# else is the ordinary resident or streamed reduction kernel.

# The tiles a product may take: its output tile's rows and columns, the depth
# of each step of its sum, and whether its sum is split among blocks. The
# generator claims the group under every output tile at the class's default
# depth, split as the grid's size suggests, and offers the variants of the
# claim that measures best: its other depths, and its sum split the other way
# (`variants`). So the cover searches the tiles on the device in stages,
# tinygrad's BEAM search at width one, and the tile is found there and kept
# in the tuning record, not chosen here. The set is what is searched:
# full-precision Float32 products on cuTile run far below the tensor cores'
# rates and favour small tiles; TF32 and half products large ones, and a
# product of few columns, a decoding step's projection, steps 128 deep as
# Vallmo's kernel does, reading the weights at the memory's rate; a product
# whose factor recomputes a fold (attention's scores, inside the sum over
# keys) does that work once per step and per block, so its rows stay in one
# tile when they fit, and the depth of its steps is its own question.
# `TEMPEST_PRODUCT_TILE=m,n,k` names one tile instead, with no variants, for
# experiments. The products inside a recomputed fold step at the depth
# measured best for the class: 16 at full precision, 32 otherwise; a
# schedule's options may name another, a variant measured as the tiles are
# (see `product_variants`).
product_step(precision, T) = T === Float32 && precision === :full ? 16 : 32

full_precision(target, T) = T === Float32 && target.precision === :full

product_outputs(target, T; recomputing) = recomputing ? ((128, 128), (64, 128)) :
    full_precision(target, T) ? ((64, 64), (128, 64), (64, 128), (32, 64)) : ((128, 128), (128, 64), (64, 128), (64, 64))
product_depths(target, T; recomputing) = full_precision(target, T) ? (16, 32) : recomputing ? (16, 32, 64) : (32, 64, 128)

# A tile within a product's extents.
fitted((tm, tn, tk), m, n, k) = (min(tm, nextpow(2, m)), min(tn, nextpow(2, n)), min(tk, nextpow(2, k)))

# The tiles the search begins with: every output tile at the default depth.
function product_tiles(target, T, m, n, k; recomputing)
    override = get(ENV, "TEMPEST_PRODUCT_TILE", "")
    tiles = isempty(override) ? [(tm, tn, product_step(target.precision, T)) for (tm, tn) in product_outputs(target, T; recomputing)] :
        [Tuple(parse.(Int, split(override, ",")))]
    unique(fitted(t, m, n, k) for t in tiles)
end

# What the product sketch needs of a group beyond its axes: the type of its
# factors, the folds recomputed inside its sum, and whether every fold is a
# plain sum, which a split may merge; nothing when the group is no product.
function product_facts(target, g::GroupView)
    target isa CUDATarget && g.kind === :reduction || return nothing
    folds = [g.values[r].fold for r in g.reductions]
    all(r -> g.phase[r] == 1, g.reductions) && all(f -> f isa Union{SumFold, Relative}, folds) || return nothing
    # The factors of the first fold that is a product: a fold of weights
    # alone, such as a log-sum-exp beside attention's weighted sum, has none.
    first_product = findfirst(r -> product_factors(g.values[r].fold, g.terms[r]) !== nothing, g.reductions)
    first_product === nothing && return nothing
    with_product = g.reductions[first_product]
    c = Context(g, TileSchedule(:product, TileAxis[], 1, nothing, 0, :full))
    recomputed = [s for r in g.reductions for s in subterms((g.terms[r],)) if s isa Recomputed]
    (T = product_eltype(c, product_factors(g.values[with_product].fold, g.terms[with_product])), recomputed,
        sums = all(f -> f isa SumFold, folds))
end

# The type a product of factors `x` and `y` runs in: a half type when either
# is one, read through its conversion to Float32, since the other then rounds
# to it (see `accumulate_product!`). Priced and tiled by its first factor
# alone, BFloat16 attention's backward at full precision, whose first product
# weighs a Float32 probability, was estimated at 3570 μs and measured at 320.
function product_eltype(c::Context, (x, y))
    Tx, Ty = term_eltype(product_factor(c, x), c.g), term_eltype(product_factor(c, y), c.g)
    Tx <: HalfFloat || !(Ty <: HalfFloat) ? Tx : Ty
end

# Whether a product's sum over the `swept` axes may be split among blocks: not
# when a recomputed fold does not move along them, so is computed once per
# block before the sweep, which every split would repeat; nor for a relative
# fold.
splittable(facts, swept) = facts.sums && !any(s -> !any(a -> mentions(s, a), swept), facts.recomputed)

# How many ways the sum of tile `(tm, tn, tk)` over `tiling` (M, K and N, then
# the grid's axes and those swept in turn) splits among blocks: a grid of few
# output tiles leaves the device idle, so the summed axis is split among
# blocks (split-K), merging the partial sums in a second launch.
function product_split(tiling, tm, tn, tk, splittable)
    M, K, N = tiling[1:3]
    blocks = cld(M.extent, tm) * cld(N.extent, tn) * prod((ax.extent for ax in tiling[4:end] if !ax.reduced); init = 1)
    k = K.extent * prod((ax.extent for ax in tiling[4:end] if ax.reduced); init = 1)
    splittable && blocks < SPLIT_KEPT_BLOCKS[1] ? split_count(k, tk, blocks) : 1
end

# The claim under tile `(tm, tn, tk)`, its sum split `split` ways, over
# `tiling`; nothing when the generator cannot express the group over it.
function product_claim(g::GroupView, tiling, tm, tn, tk, split, precision; options = PLAIN)
    M, K, N = tiling[1:3]
    t = [TileAxis(M.axes, M.extent, tm, false), TileAxis(K.axes, K.extent, tk, true), TileAxis(N.axes, N.extent, tn, false), tiling[4:end]...]
    s = TileSchedule(:product, t, split, nothing, 8, precision, options)
    c = Context(g, s)
    all(r -> products_of(c, r), g.reductions) && any(r -> has_product(c, r), g.reductions) && expressible_over(c) ? Claim(s) : nothing
end

function product_claims(target, g::GroupView)
    facts = product_facts(target, g)
    facts === nothing && return Claim[]
    found = product_axes(g)
    found === nothing && return Claim[]
    M, N, K, B = found
    extent(axes) = prod(k -> g.domain[k], axes; init = 1)
    m, n, k, b = extent(M), extent(N), extent(K), extent(B)
    max(m, n, k, b) < typemax(Int32) ÷ 2 || return Claim[]
    longest(axes) = axes[argmax(map(a -> g.domain[a], axes))]
    recomputing = !isempty(facts.recomputed)
    may_split = splittable(facts, K)
    # Summed axes the factors do not read as one strided axis are swept in
    # turn: the longest piece over which they do is tiled, every other piece
    # an index per step.
    found, main = pieces(K, view_addresses(g), g.domain)
    summed = main == 0 || length(found) == 1 ? [(K, [])] : [(K, []), (found[main], found[1:end .!= main])]
    for collapsed in (true, false), (Kt, others) in summed
        collapsed && !isempty(others) && continue
        Mt, Nt = collapsed ? (M, N) : ([longest(M)], [longest(N)])
        grid = collapsed ? (isempty(B) ? Vector{Int}[] : [B]) : [[a] for a in sort!([setdiff(M, Mt); setdiff(N, Nt); B])]
        mt, nt, kt = extent(Mt), extent(Nt), extent(Kt)
        tiling = [TileAxis(Mt, mt, 1, false), TileAxis(Kt, kt, 1, true), TileAxis(Nt, nt, 1, false)]
        append!(tiling, [TileAxis(a, extent(a), 1, false) for a in grid])
        append!(tiling, [TileAxis(a, extent(a), 1, true) for a in others])
        found = Claim[]
        for (tm, tn, tk) in product_tiles(target, facts.T, mt, nt, kt; recomputing)
            c = product_claim(g, tiling, tm, tn, tk, product_split(tiling, tm, tn, tk, may_split), target.precision;
                options = something(named_options(), PLAIN))
            c === nothing || push!(found, c)
        end
        isempty(found) || return found
    end
    Claim[]
end

# The variants of a product claim, offered once it measures best: the other
# depths at its output tile, split as it is, and its sum split the other way;
# and, for a product that recomputes a fold, its other options at its tile
# (`product_options`), and the products inside what it recomputes twice as
# deep as the class's steps, or back at them. A split sum transposes nothing.
# Transposed with eight worker warps, attention's backward took 395 μs with
# its scores' products 64 deep against 418 at 32 (1024 positions, 64
# features, causal, TF32); plainly, 603 against 594.
function product_variants(target, g::GroupView, c::Claim)
    s = c.schedule
    s.sketch === :product && isempty(get(ENV, "TEMPEST_PRODUCT_TILE", "")) || return Claim[]
    facts = product_facts(target, g)
    facts === nothing && return Claim[]
    M, K, N = s.tiling[1:3]
    may_split = splittable(facts, [K.axes; [a for ax in s.tiling[4:end] if ax.reduced for a in ax.axes]])
    natural(tk) = product_split(s.tiling, M.tile, N.tile, tk, may_split)
    depths = unique(min(tk, nextpow(2, K.extent)) for tk in product_depths(target, facts.T; recomputing = !isempty(facts.recomputed)))
    tiles = [(K.tile, s.splits > 1 ? 1 : natural(K.tile)); [(tk, s.splits > 1 ? natural(tk) : 1) for tk in depths]]
    found = Claim[]
    for (tk, split) in unique(tiles)
        (tk, split) == (K.tile, s.splits) && continue
        options = split > 1 ? ProductOptions(s.options.workers, false, s.options.inner) : s.options
        x = product_claim(g, s.tiling, M.tile, N.tile, tk, split, s.precision; options)
        x === nothing || push!(found, x)
    end
    !isempty(facts.recomputed) && named_options() === nothing || return found
    o = s.options
    step = product_step(s.precision, facts.T)
    deeper = any(t -> any(>(step), t.extents), facts.recomputed) ? [ProductOptions(o.workers, o.transposes, o.inner > 0 ? 0 : 2step)] : ProductOptions[]
    for options in [[ProductOptions(x.workers, x.transposes, o.inner) for x in product_options(target, g, s)]; deeper]
        options == o && continue
        x = product_claim(g, s.tiling, M.tile, N.tile, K.tile, s.splits, s.precision; options)
        x === nothing || push!(found, x)
    end
    found
end

# `TEMPEST_PRODUCT_OPTIONS` names every product's options instead, with no
# variants of them, for experiments: its worker warps or `default`, then
# `transposed` to add its transposed folds, and the depth of the products
# inside what it recomputes, as `8,transposed,64`.
function named_options()
    text = get(ENV, "TEMPEST_PRODUCT_OPTIONS", "")
    isempty(text) && return nothing
    parts = split(text, ",")
    depth = something(findfirst(x -> tryparse(Int, x) !== nothing, parts[2:end]), 0)
    ProductOptions(parts[1] == "default" ? nothing : parse(Int, parts[1]), "transposed" in parts,
        depth == 0 ? 0 : parse(Int, parts[depth + 1]))
end

# The options a product's sweep may take: tileiras's worker warps as it
# chooses them, or eight; each with its transposed folds added, where its
# schedule has any (see `transposing`) and the target lets a sum's rounding
# depend on the order of its additions (`deterministic`). Offered together,
# since they are measured together: at 1024 positions, 64 features, causal,
# TF32, attention's backward took 594 μs plainly, 808 with eight worker
# warps, 1180 transposed, which spilled 1728 bytes a thread, and 418
# transposed with eight worker warps.
function product_options(target, g::GroupView, s::TileSchedule)
    options = [ProductOptions(w, false) for w in (nothing, 8)]
    s.splits == 1 && !Tempest.Compiler.deterministic(target) || return options
    transposed = TileSchedule(s.sketch, s.tiling, s.splits, s.occupancy, s.swizzle, s.precision, ProductOptions(nothing, true))
    isempty(added_folds(transposing(Context(g, transposed)))) && return options
    [options; [ProductOptions(w, true) for w in (nothing, 8)]]
end

# A product of a matrix and one column, a matrix-vector product, has no kept
# axis that only the column reads, so it is not a product (`product_axes`)
# but the streamed reduction of its sum. That kernel steps 32 along the sum,
# 64 bytes of each row of a half-precision matrix stored along the sum, as a
# checkpoint stores its weights: on GB10 it reads such a matrix at 205 GB/s
# where Vallmo's decode kernel, stepping 128, reads it at 233. Its longer
# steps are offered as a product's depths are, and measured.
const VECTOR_PRODUCT_STEPS = (64, 128)

function vector_product_variants(g::GroupView, c::Claim)
    s = c.schedule
    # A sum that recomputes a product steps at a product's depths already.
    s.sketch === :streamed && !recomputing(g) || return Claim[]
    all(r -> g.values[r].fold isa SumFold && product_factors(g.values[r].fold, g.terms[r]) !== nothing, g.reductions) || return Claim[]
    p = sweep_axis(s)
    ax = s.tiling[p]
    found = Claim[]
    for step in VECTOR_PRODUCT_STEPS
        step > ax.tile && cld(ax.extent, step) > 1 || continue
        tiling = copy(s.tiling)
        tiling[p] = TileAxis(ax.axes, ax.extent, step, true, ax.chunk)
        x = TileSchedule(s.sketch, tiling, s.splits, s.occupancy, s.swizzle, s.precision)
        fits(Context(g, x)) && push!(found, Claim(x))
    end
    found
end

# Whether a fold accumulates in a product's sweep: a sum of a bilinear
# product, or an exponential fold whose sums are products or sums of its
# weights alone; and whether it has a product among them.
function products_of(c::Context, r)
    sums = fold_sums(c, r)
    sums !== nothing && all(((term, product),) -> product !== nothing || term.f isa Weight, sums)
end
has_product(c::Context, r) = (sums = fold_sums(c, r); sums !== nothing && any(s -> s[2] !== nothing, sums))

# Every load a view over the tiling, every stored value too, and nothing the
# generator cannot lower.
function expressible_over(c::Context)
    g = c.g
    for m in members(g)
        val = g.values[m]
        val.eltype <: TILE_TYPES || return false
        m in g.reductions && !tile_fold(val.fold, term_eltype(g.terms[m], g)) && return false
        m in g.outputs && val.dims == g.domain && return false
        terms_expressible(c, (g.terms[m],); stats = true) || return false
        (g.values[m].stored || m in g.outputs) && output_view(c, m) === nothing && return false
    end
    true
end
