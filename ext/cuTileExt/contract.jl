# The product sketch: tile axes for reduction groups whose sums are of
# bilinear products. The kept axes split into those only one factor reads (M),
# those only the other reads (N), and those both read (a batch, on the grid);
# the reduced axes (K) are swept. Tile order (M, K, N, batch) puts each factor's
# tile in matrix order, so the fold algebra's product rule applies. Everything
# else is the ordinary resident or streamed reduction kernel.

# The domain's axes a term reads, or nothing when it reads reductions.
function axes_read(t::Term, n)
    axes = Set{Int}()
    for s in Tempest.subterms((t,))
        s isa Stat && return nothing
        exprs = s isa Load ? s.index : s isa Ix ? (s.expr,) : ()
        for e in exprs, k in 1:n
            Tempest.mentions(e, k) && push!(axes, k)
        end
    end
    axes
end

function product_groups(graph, g::Group)
    n = length(g.domain)
    K = sort!(collect(Tempest.primary_axes(graph, g)))
    kept = [k for k in 1:n if !(k in K)]
    found = nothing
    for r in g.reductions
        t = g.terms[r]
        t isa Apply && t.f === (*) && length(t.args) == 2 || return nothing
        ax, ay = axes_read(t.args[1], n), axes_read(t.args[2], n)
        (ax === nothing || ay === nothing) && return nothing
        all(k -> k in ax && k in ay, K) || return nothing
        M = [k for k in kept if k in ax && !(k in ay)]
        N = [k for k in kept if k in ay && !(k in ax)]
        B = [k for k in kept if k in ax && k in ay]
        length(M) + length(N) + length(B) == length(kept) && !isempty(M) && !isempty(N) || return nothing
        groups = (M, N, K, B)
        found === nothing || found == groups || found == (N, M, K, B) || return nothing
        found === nothing && (found = groups)
    end
    found
end

# Tiles measured on GB10 (the scratchpad benchmark of cuTile against cuBLAS):
# 128×128 output tiles with 32-deep steps for TF32 up to K = 1024 and 64-deep
# otherwise. Full-precision Float32 products are much slower on cuTile and
# fastest in small tiles (64×64×16 measured best, 2.6 TFLOPS).
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

function product_claim(target, p::Plan, g::Group)
    target isa CUDATarget && g.kind === :reduction || return nothing
    all(r -> g.phase[r] == 1 && p.graph.values[r].definition.fold isa Tempest.SumFold, g.reductions) || return nothing
    found = product_groups(p.graph, g)
    found === nothing && return nothing
    M, N, K, B = found
    extent(axes) = prod(k -> g.domain[k], axes; init = 1)
    m, n, k, b = extent(M), extent(N), extent(K), extent(B)
    max(m, n, k, b) < typemax(Int32) ÷ 2 || return nothing
    factor = product_factor(Context(p.graph, p.stored, g, TileSchedule(:product, TileAxis[], 1, nothing, 0, :full)),
        first(g.terms[first(g.reductions)].args))
    tm, tn, tk = product_tiles(target, Tempest.term_eltype(factor, p.graph), m, n, k)
    tiling = [TileAxis(M, m, tm, false), TileAxis(K, k, tk, true), TileAxis(N, n, tn, false)]
    isempty(B) || push!(tiling, TileAxis(B, b, 1, false))
    # A grid of few output tiles leaves the device idle: split the summed axis
    # among blocks (split-K), merging the partial sums in a second launch.
    blocks = cld(m, tm) * cld(n, tn) * b
    splits = blocks < SPLIT_KEPT_BLOCKS[1] ? split_count(k, tk, blocks) : 1
    s = TileSchedule(:product, tiling, splits, nothing, 8, target.precision)
    c = Context(p.graph, p.stored, g, s)
    all(r -> bilinear(c, r) !== nothing, g.reductions) || return nothing
    expressible_over(c) || return nothing
    Claim(s)
end

# Every load a view over the tiling, every stored value too, and nothing the
# generator cannot lower.
function expressible_over(c::Context)
    graph, g = c.graph, c.g
    for m in Tempest.members(g)
        val = graph.values[m]
        val.eltype <: TILE_TYPES || return false
        m in g.reductions && !(val.definition.fold isa FOLDS) && return false
        m in g.outputs && val.dims == g.domain && return false
        for s in Tempest.subterms((g.terms[m],))
            s isa Ix && return false
            s isa Apply && !(s.eltype <: TILE_TYPES) && return false
            s isa Lit && s.value isa Float64 && return false
            s isa Load && view_of(graph, s.value, s.index, g, groups(c)...) === nothing && return false
        end
        (c.stored[m] || m in g.outputs) && output_view(c, m) === nothing && return false
    end
    true
end
