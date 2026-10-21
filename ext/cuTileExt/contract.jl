# The product sketch: tile axes for reduction groups whose sums are of
# bilinear products, over the axes `Tempest.Compiler.product_axes` finds. The reduced
# axes (K) are swept. Tile order (M, K, N, batch) puts each factor's tile in
# matrix order, so the fold algebra's product rule applies. M, N and the batch
# each collapse into one tile axis where every view chains over them;
# otherwise the longest axis of M and of N is tiled and every other kept axis
# is a grid axis of its own. K collapses likewise, or its longest piece that
# chains is tiled and each other piece is swept an index per step, as grouped
# heads and queries are in attention's backward pass. Everything else is the
# ordinary resident or streamed reduction kernel.

# The tiles a product may take: its output tile's rows and columns, and the
# depth of each step of its sum. The generator claims the group under each,
# and the cover measures the claims whose estimates are close, which for
# tiles of one product they are, so the tile is found on the device and kept
# in the tuning record, not chosen here (tinygrad's BEAM search, the small
# version: a fixed set of options per kernel, timed). The set is what is
# searched: full-precision Float32 products on cuTile run far below the
# tensor cores' rates and favour small tiles; TF32 and half products large
# ones; a product whose factor recomputes a fold (attention's scores, inside
# the sum over keys) does that work once per step and per block, so its rows
# stay in one tile when they fit, and the depth of its steps is its own
# question. `TEMPEST_PRODUCT_TILE=m,n,k` names one tile instead, for
# experiments. The products inside a recomputed fold step at the depth
# measured best for the class: 16 at full precision, 32 otherwise.
product_step(precision, T) = T === Float32 && precision === :full ? 16 : 32

function product_tiles(target, T, m, n, k; recomputing)
    full = T === Float32 && target.precision === :full
    outputs = recomputing ? ((128, 128), (64, 128)) :
        full ? ((64, 64), (128, 64), (64, 128), (32, 64)) : ((128, 128), (128, 64), (64, 128), (64, 64))
    depths = full ? (16, 32) : recomputing ? (16, 32, 64) : (32, 64)
    override = get(ENV, "TEMPEST_PRODUCT_TILE", "")
    candidates = isempty(override) ? [(tm, tn, tk) for (tm, tn) in outputs for tk in depths] : [Tuple(parse.(Int, split(override, ",")))]
    unique((min(tm, nextpow(2, m)), min(tn, nextpow(2, n)), min(tk, nextpow(2, k))) for (tm, tn, tk) in candidates)
end

function product_claims(target, g::GroupView)
    target isa CUDATarget && g.kind === :reduction || return Claim[]
    folds = [g.values[r].fold for r in g.reductions]
    all(r -> g.phase[r] == 1, g.reductions) && all(f -> f isa Union{SumFold, Relative}, folds) || return Claim[]
    found = product_axes(g)
    found === nothing && return Claim[]
    M, N, K, B = found
    extent(axes) = prod(k -> g.domain[k], axes; init = 1)
    m, n, k, b = extent(M), extent(N), extent(K), extent(B)
    max(m, n, k, b) < typemax(Int32) ÷ 2 || return Claim[]
    # The factors of the first fold that is a product: a fold of weights
    # alone, such as a log-sum-exp beside attention's weighted sum, has none.
    first_product = findfirst(r -> product_factors(g.values[r].fold, g.terms[r]) !== nothing, g.reductions)
    first_product === nothing && return Claim[]
    with_product = g.reductions[first_product]
    factor = product_factor(Context(g, TileSchedule(:product, TileAxis[], 1, nothing, 0, :full)),
        first(product_factors(g.values[with_product].fold, g.terms[with_product])))
    longest(axes) = axes[argmax(map(a -> g.domain[a], axes))]
    recomputed = [s for r in g.reductions for s in subterms((g.terms[r],)) if s isa Recomputed]
    recomputing = !isempty(recomputed)
    # A recomputed fold that does not move along the summed axes is computed
    # once per block, before the sweep.
    hoisted = any(s -> !any(a -> mentions(s, a), K), recomputed)
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
        T = term_eltype(factor, g)
        found = Claim[]
        for (tm, tn, tk) in product_tiles(target, T, mt, nt, kt; recomputing)
            tiling = [TileAxis(Mt, mt, tm, false), TileAxis(Kt, kt, tk, true), TileAxis(Nt, nt, tn, false)]
            append!(tiling, [TileAxis(a, extent(a), 1, false) for a in grid])
            append!(tiling, [TileAxis(a, extent(a), 1, true) for a in others])
            # A grid of few output tiles leaves the device idle: the summed
            # axis may be split among blocks (split-K), merging the partial
            # sums in a second launch, and the cover measures both; not when
            # a fold is computed before the sweep, which every split would
            # repeat, nor for a relative fold.
            blocks = cld(mt, tm) * cld(nt, tn) * b * (m ÷ mt) * (n ÷ nt)
            sums = all(f -> f isa SumFold, folds)
            splits = blocks < SPLIT_KEPT_BLOCKS[1] && !hoisted && sums ? split_count(k, tk, blocks) : 1
            for split in unique((1, splits))
                s = TileSchedule(:product, tiling, split, nothing, 8, target.precision)
                c = Context(g, s)
                all(r -> products_of(c, r), g.reductions) && any(r -> has_product(c, r), g.reductions) && expressible_over(c) && push!(found, Claim(s))
            end
        end
        isempty(found) || return found
    end
    Claim[]
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
