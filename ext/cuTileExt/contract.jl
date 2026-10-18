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

# Tiles measured on GB10 (the scratchpad benchmark of cuTile against cuBLAS):
# 128×128 output tiles with 32-deep steps for TF32 up to K = 1024 and 64-deep
# otherwise. Full-precision Float32 products are much slower on cuTile and
# fastest in small tiles (64×64×16 measured best, 2.6 TFLOPS).
product_step(precision, T) = T === Float32 && precision === :full ? 16 : 32

function product_tiles(target, T, m, n, k)
    if T === Float32 && target.precision === :full
        tm, tn, tk = 64, 64, 16
    else
        tm, tn = 128, 128
        tk = T === Float32 && k <= 1024 ? 32 : 64
    end
    override = get(ENV, "TEMPEST_PRODUCT_TILE", "")
    isempty(override) || ((tm, tn, tk) = Tuple(parse.(Int, split(override, ","))))
    (min(tm, nextpow(2, m)), min(tn, nextpow(2, n)), min(tk, nextpow(2, k)))
end

# A product whose factor recomputes a fold (attention's scores, inside the sum
# over keys) does that work once per step and per block: M in one tile when it
# fits, so no block repeats it, 128 columns, and the product's steps. Measured
# on GB10 with GPT-2's attention (64 features, 256 positions): 128 queries per
# block beat 32, 64 and 256, and 32 keys per step beat 16 and 64 in TF32 (243
# µs); in full precision 16-key steps with 16-deep products inside (1281 µs).
function recomputing_tiles(target, T, m, n, k)
    tm, tn, tk = 128, 128, product_step(target.precision, T)
    override = get(ENV, "TEMPEST_RECOMPUTING_TILE", "")
    isempty(override) || ((tm, tn, tk) = Tuple(parse.(Int, split(override, ","))))
    (min(tm, nextpow(2, m)), min(tn, nextpow(2, n)), min(tk, nextpow(2, k)))
end

function product_claim(target, p::Plan, g::Group)
    target isa CUDATarget && g.kind === :reduction || return nothing
    folds = [p.graph.values[r].definition.fold for r in g.reductions]
    all(r -> g.phase[r] == 1, g.reductions) && all(f -> f isa Union{Tempest.Compiler.SumFold, Tempest.Compiler.OnlineSoftmax}, folds) || return nothing
    found = Tempest.Compiler.product_axes(p.graph, g)
    found === nothing && return nothing
    M, N, K, B = found
    extent(axes) = prod(k -> g.domain[k], axes; init = 1)
    m, n, k, b = extent(M), extent(N), extent(K), extent(B)
    max(m, n, k, b) < typemax(Int32) ÷ 2 || return nothing
    factor = product_factor(Context(p.graph, p.stored, g, TileSchedule(:product, TileAxis[], 1, nothing, 0, :full)),
        first(Tempest.Compiler.product_factors(first(folds), g.terms[first(g.reductions)])))
    longest(axes) = axes[argmax(map(a -> g.domain[a], axes))]
    recomputed = [s for r in g.reductions for s in Tempest.Compiler.subterms((g.terms[r],)) if s isa Tempest.Compiler.Recomputed]
    recomputing = !isempty(recomputed)
    # A recomputed fold that does not move along the summed axes is computed
    # once per block, before the sweep.
    hoisted = any(s -> !any(a -> Tempest.Compiler.mentions(s, a), K), recomputed)
    # Summed axes the factors do not read as one strided axis are swept in
    # turn: the longest piece over which they do is tiled, every other piece
    # an index per step.
    found, main = pieces(K, view_addresses(p.graph, p.stored, g), g.domain)
    summed = main == 0 || length(found) == 1 ? [(K, [])] : [(K, []), (found[main], found[1:end .!= main])]
    for collapsed in (true, false), (Kt, others) in summed
        collapsed && !isempty(others) && continue
        Mt, Nt = collapsed ? (M, N) : ([longest(M)], [longest(N)])
        grid = collapsed ? (isempty(B) ? Vector{Int}[] : [B]) : [[a] for a in sort!([setdiff(M, Mt); setdiff(N, Nt); B])]
        mt, nt, kt = extent(Mt), extent(Nt), extent(Kt)
        T = Tempest.Compiler.term_eltype(factor, p.graph)
        tm, tn, tk = recomputing ? recomputing_tiles(target, T, mt, nt, kt) : product_tiles(target, T, mt, nt, kt)
        tiling = [TileAxis(Mt, mt, tm, false), TileAxis(Kt, kt, tk, true), TileAxis(Nt, nt, tn, false)]
        append!(tiling, [TileAxis(a, extent(a), 1, false) for a in grid])
        append!(tiling, [TileAxis(a, extent(a), 1, true) for a in others])
        # A grid of few output tiles leaves the device idle: split the summed
        # axis among blocks (split-K), merging the partial sums in a second
        # launch; not when a fold is computed before the sweep, which every
        # split would repeat, nor for an online softmax.
        blocks = cld(mt, tm) * cld(nt, tn) * b * (m ÷ mt) * (n ÷ nt)
        sums = all(f -> f isa Tempest.Compiler.SumFold, folds)
        splits = blocks < SPLIT_KEPT_BLOCKS[1] && !hoisted && sums ? split_count(k, tk, blocks) : 1
        s = TileSchedule(:product, tiling, splits, nothing, 8, target.precision)
        c = Context(p.graph, p.stored, g, s)
        all(r -> products_of(c, r), g.reductions) && expressible_over(c) && return Claim(s)
    end
    nothing
end

# Whether a fold accumulates as products of tiles: a sum of a bilinear
# product, or an online softmax whose sums are, but for sums of its weights.
function products_of(c::Context, r)
    sums = fold_sums(c, r)
    sums !== nothing && any(s -> s[2] !== nothing, sums) && all(((term, product),) -> product !== nothing || term.f isa Weight, sums)
end

# Every load a view over the tiling, every stored value too, and nothing the
# generator cannot lower.
function expressible_over(c::Context)
    graph, g = c.graph, c.g
    for m in Tempest.Compiler.members(g)
        val = graph.values[m]
        val.eltype <: TILE_TYPES || return false
        m in g.reductions && !tile_fold(val.definition.fold, Tempest.Compiler.term_eltype(g.terms[m], graph)) && return false
        m in g.outputs && val.dims == g.domain && return false
        terms_expressible(c, (g.terms[m],); stats = true) || return false
        (c.stored[m] || m in g.outputs) && output_view(c, m) === nothing && return false
    end
    true
end
