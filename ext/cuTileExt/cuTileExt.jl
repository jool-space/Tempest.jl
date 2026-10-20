# cuTile: one generator for every group a sketch can schedule.
#
# A schedule collapses the group's domain into tile axes: each a group of the
# domain's axes, with an extent, a tile size, and whether it is reduced. Every
# load and store must be a strided view of its buffer over those groups (a
# formula affine in the group's variables, with coefficients that chain within
# each group), except a load whose formula reads an index from a value (a
# gather), which reads element by element. A schedule also picks how the
# reduced axis is swept (resident in one tile, streamed, or split across blocks
# with a merging launch), the grid (plain or swizzled), hints and the precision
# of products. The generator turns any schedule into kernels; a sketch
# contributes parameters, never code.
module cuTileExt

using Tempest
using CUDACore
import cuTile as ct
using BFloat16s: BFloat16
using Republic: @republic
using Tempest: CUDATarget
@republic republic=false inherit=:public using Tempest.IR
@republic republic=false inherit=:public import Tempest.Implementations

# Defaults measured on GB10 (48 SMs, 24 MiB L2) by the earlier cuTile schedules.
const RESIDENT_ROWS = 4096       # columns up to this many rows stay in one tile
const STREAMED_ROWS = 1024       # rows per step of a streamed column
const STREAMED_ROW_TILE = 128    # kept rows per tile of a streamed row reduction
const STREAMED_ROW_COLUMNS = 32  # columns per step of a streamed row reduction
const PACKED_ROWS = 128          # columns this short share a tile
const PACKED_ELEMENTS = 512      # values per packed tile
const MINIMUM_BLOCKS = 128
const SPLIT_KEPT_BLOCKS = (32, 64) # kept tiles below which a streamed axis splits
const SPLIT_TARGET_BLOCKS = 256
const FULL_TILE_ELEMENTS = 1024
const POINTWISE_ELEMENTS = 1024

"""
A tile axis: a group of the domain's axes, collapsed, tiled, and reduced (swept)
or kept. A kept axis's `chunk` is how many of its tiles one block covers, one
after another.
"""
struct TileAxis
    axes::Vector{Int}
    extent::Int
    tile::Int
    reduced::Bool
    chunk::Int
end

TileAxis(axes, extent, tile, reduced) = TileAxis(axes, extent, tile, reduced, 1)
with_chunk(ax::TileAxis, chunk) = TileAxis(ax.axes, ax.extent, ax.tile, ax.reduced, chunk)

struct TileSchedule
    sketch::Symbol            # :pointwise, :resident, :streamed, :split, :full
    tiling::Vector{TileAxis}
    splits::Int               # blocks sharing one kept tile
    occupancy::Union{Nothing, Int}
    swizzle::Int              # grouping of the first two kept axes on the grid; 0 for a plain grid
    precision::Symbol         # of products: :full or :tf32
end

# The two-axis sketches over groups A and B, `reduced` naming the reduced one
# (1 or 2), both (3) or neither (0).
TileSchedule(sketch, A, B, extent, reduced, tile, splits, occupancy) = TileSchedule(
    sketch,
    [TileAxis(collect(A), extent[1], tile[1], reduced in (1, 3)), TileAxis(collect(B), extent[2], tile[2], reduced in (2, 3))],
    splits, occupancy, 0, :full,
)

extent(s::TileSchedule) = Tuple(ax.extent for ax in s.tiling)
tiles(s::TileSchedule) = Tuple(ax.tile for ax in s.tiling)
groups(s::TileSchedule) = Tuple(ax.axes for ax in s.tiling)
swept(s::TileSchedule) = findall(ax -> ax.reduced, s.tiling)
function code(s::TileSchedule)
    r = swept(s)
    isempty(r) ? 0 : length(r) == length(s.tiling) ? 3 : sweep_axis(s)
end

"The swept tile axis a schedule tiles; any other it sweeps has tile one, an index per step."
function sweep_axis(s::TileSchedule)
    r = swept(s)
    r[something(findfirst(q -> s.tiling[q].tile > 1, r), 1)]
end

# The steps of a sweep: every tile of the swept axes, unless the schedule
# reduces every axis.
sweeps(s::TileSchedule) = (r = swept(s); length(r) == length(s.tiling) ? 1 :
    prod(q -> cld(s.tiling[q].extent, s.tiling[q].tile), r; init = 1))

"""
The block indices at step `t` of the sweep, an Int32 expression from one: the
swept axis the schedule tiles moves fastest, then each other swept axis.
"""
function step_indices(s::TileSchedule, indices, t)
    ix = Any[indices...]
    p = sweep_axis(s)
    others = filter(!=(p), swept(s))
    isempty(others) && (ix[p] = t; return Tuple(ix))
    counts = map(q -> cld(s.tiling[q].extent, s.tiling[q].tile), [p; others])
    rest = :($t - Int32(1))
    for (q, n) in zip([p; others], counts)
        ix[q] = :(rem($rest, Int32($n)) + Int32(1))
        rest = :(div($rest, Int32($n)))
    end
    Tuple(ix)
end
kept_tiles(s::TileSchedule) = prod((cld(ax.extent, ax.tile) for ax in s.tiling if !ax.reduced); init = 1)

function Base.show(io::IO, s::TileSchedule)
    if s.sketch === :product
        m, k, n = s.tiling[1:3]
        print(io, "product ", m.extent, "×", n.extent, " summing ", k.extent)
        turns = [ax.extent for ax in s.tiling[4:end] if ax.reduced]
        isempty(turns) || print(io, " and ", join(turns, "×"), " in turn")
        batch = [ax.extent for ax in s.tiling[4:end] if !ax.reduced]
        isempty(batch) || print(io, ", batch ", join(batch, "×"))
        print(io, ", tile ", m.tile, "×", n.tile, "×", k.tile, ", ",
            s.precision === :tf32 ? "TF32" : s.precision === :tf32x3 ? "3×TF32" : "full precision")
        s.splits > 1 && print(io, ", summing split ", s.splits, " ways")
        return
    end
    print(io, s.sketch, " ", join(extent(s), "×"), ", tile ", join(tiles(s), "×"))
    code(s) in (1, 2) && print(io, ", reducing axis ", code(s))
    s.splits > 1 && print(io, ", split ", s.splits, " ways")
    for ax in s.tiling
        ax.chunk > 1 && print(io, ", ", ax.chunk, " tiles per block")
    end
    s.occupancy === nothing || print(io, ", occupancy ", s.occupancy)
end

# Which axes of the group's domain collapse into A and B, and which one the
# schedule sweeps: the group's primary axes (those of its first fold), one run
# of axes wherever they lie. A row reduction's kept axes after the swept ones,
# C, are bound by the grid; nothing when the primary axes are not a run.
function collapse_axes(g::GroupView)
    r = length(g.domain)
    R = sort!(collect(g.primary_axes))
    if g.kind === :pointwise
        return (1:min(r, 1), 2:r, 0, 1:0)
    elseif R == collect(1:r)
        return (1:min(r, 1), 2:r, 3, 1:0)
    elseif R == collect(1:length(R))
        return (1:length(R), (length(R) + 1):r, 1, 1:0)
    elseif R == collect(first(R):last(R))
        return (1:(first(R) - 1), R, 2, (last(R) + 1):r)
    end
    nothing
end

"""
A load or store: a value's buffer as a strided view over collapsed groups of
axes. A load may add an offset computed per block (`dynamic`): any formula of
the axes the grid binds, below `reach`.
"""
struct View
    value::Int
    offset::Int
    strides::Tuple{Vararg{Int}}
    spans::Tuple{Vararg{Bool}} # whether the view extends along each group
    dynamic::Union{Nothing, Affine}
    reach::Int
end

View(value, offset, strides, spans) = View(value, offset, strides, spans, nothing, 0)

# The view of value `v` read at `index` over the groups of axes; nothing when the
# address does not chain within a group, depends on an axis outside the
# groups, or is not affine. The groups numbered in `scalar` are bound by the
# grid, one index per block: the address may depend on them through any
# formula, which becomes the view's offset per block, as does the part of the
# address over such a group that does not chain.
function view_of(g::GroupView, v, index, groups...; scalar = ())
    domain = Domain(g.domain)
    e = address(g.values[v], index, domain)
    rest = [t for t in e.terms if !(t.first isa Var)]
    e = Affine(e.constant, Tuple(t for t in e.terms if t.first isa Var))
    for i in scalar
        s = group_stride(e, groups[i], g.domain)
        s !== nothing && s >= 0 && continue
        append!(rest, [t for t in e.terms if t.first.n in groups[i]])
        e = Affine(e.constant, Tuple(t for t in e.terms if !(t.first.n in groups[i])))
    end
    dynamic = nothing
    reach = 0
    if !isempty(rest)
        dynamic = Affine(0, Tuple(rest))
        bound = reduce(vcat, (groups[i] for i in scalar); init = Int[])
        isempty(indirect_values(dynamic)) || return nothing
        all(k -> k in bound || g.domain[k] == 1 || !mentions(dynamic, k), eachindex(g.domain)) || return nothing
        lo, hi = affine_range(dynamic, domain)
        lo >= 0 || return nothing
        reach = hi + 1
    end
    for k in 1:length(g.domain)
        g.domain[k] > 1 && coefficient(e, k) != 0 && !any(grp -> k in grp, groups) && return nothing
    end
    strides = map(grp -> group_stride(e, grp, g.domain), groups)
    any(s -> s === nothing || s < 0, strides) && return nothing
    View(v, e.constant, Tuple(strides), Tuple(map(!=(0), strides)), dynamic, reach)
end

const TILE_TYPES = Union{Float32, Float16, BFloat16, Bool, Int32}

# Tile types, and tuples of them: a fold's element of several components.
tile_type(T) = T <: TILE_TYPES || T <: Tuple && all(tile_type, fieldtypes(T))

# Positions computed from a formula without gathers.
positional(t::Ix) = isempty(indirect_values(t.expr)) && term_eltype(t, nothing) <: TILE_TYPES

# Whether the generator can express a group: tile-typed values, supported
# folds, and positions without gathers; its loads are checked with a schedule.
function expressible(g::GroupView)
    collapse_axes(g) === nothing && return false
    for m in members(g)
        val = g.values[m]
        val.eltype <: TILE_TYPES || return false
        m in g.reductions && !tile_fold(val.fold, term_eltype(g.terms[m], g)) && return false
        for t in subterms((g.terms[m],))
            t isa Recomputed && return false
            t isa Ix && !positional(t) && return false
            t isa Apply && !tile_type(t.eltype) && return false
            t isa Lit && t.value isa Float64 && return false
            t isa Load && !(g.values[t.value].eltype <: TILE_TYPES) && return false
        end
    end
    true
end

function resident_tile(rows, columns)
    r = nextpow(2, rows)
    r > PACKED_ROWS && return (r, 1)
    packed = max(1, PACKED_ELEMENTS ÷ r)
    spread = max(1, prevpow(2, max(1, columns ÷ MINIMUM_BLOCKS)))
    (r, min(packed, spread))
end

function streamed_row_tile(rows)
    fitting = prevpow(2, max(1, rows ÷ SPLIT_KEPT_BLOCKS[2]))
    min(STREAMED_ROW_TILE, nextpow(2, rows), max(32, fitting))
end

function split_count(extent, tile_extent, kept)
    reduced = cld(extent, tile_extent)
    reduced >= 4 || return 1
    min(prevpow(2, reduced), nextpow(2, cld(SPLIT_TARGET_BLOCKS, kept)))
end

"""
The default schedule of a group whose domain collapses into A and B, with the
kept axes C bound by the grid: the tiles of the collapsed domain, A's within
its own extent.
"""
function default_schedule(g::GroupView, A, B, C, reduced)
    n(axes) = prod(k -> g.domain[k], axes; init = 1)
    t = default_tiles((n(A) * n(C), n(B)), reduced)
    tiling = [TileAxis(collect(A), n(A), min(t.tile[1], nextpow(2, n(A))), reduced in (1, 3)),
        TileAxis(collect(B), n(B), t.tile[2], reduced in (2, 3))]
    isempty(C) || push!(tiling, TileAxis(collect(C), n(C), 1, false))
    TileSchedule(t.sketch, tiling, t.splits, t.occupancy, 0, :full)
end

# The addresses of the group's views: every load that does not gather, also in
# the folds it recomputes, and every value it stores.
function view_addresses(g::GroupView)
    found = Affine[]
    function visit(t, extents)
        if t isa Load
            gathers(t) || push!(found, address(g.values[t.value], t.index, Domain(extents)))
        elseif t isa Apply
            foreach(a -> visit(a, extents), t.args)
        elseif t isa Recomputed
            visit(t.body, (extents..., t.extents...))
        end
    end
    for m in members(g)
        visit(g.terms[m], g.domain)
        g.values[m].stored && push!(found, address(g.values[m], g.placement[m], Domain(g.domain)))
    end
    found
end

"""
A schedule whose kept tile axes are cut where a view does not chain from one
axis to the next, or a fold over other axes than the swept ones begins or
ends. Each kept tile axis keeps its tile on its longest piece, among those
every view reads affinely; the other pieces are bound by the grid, one index
per block, where a view's formula is an offset per block. Then a fold over
part of a collapsed axis, or a value broadcast along part of it, is
generated.
"""
function refined(c)
    g, s = c.g, c.s
    addresses = view_addresses(g)
    folds = [g.values[r].reduced for r in partial_folds(c)]
    tiling = TileAxis[]
    for ax in s.tiling
        ax.reduced && (push!(tiling, ax); continue)
        found, main = pieces(ax.axes, addresses, g.domain, folds)
        for (i, piece) in enumerate(found)
            n = prod(k -> g.domain[k], piece; init = 1)
            push!(tiling, TileAxis(piece, n, i == main ? min(ax.tile, nextpow(2, n)) : 1, false))
        end
    end
    TileSchedule(s.sketch, tiling, s.splits, s.occupancy, s.swizzle, s.precision)
end

"""
    pieces(axes, addresses, domain, folds = ()) -> (pieces, main)

A run of axes cut where an address does not chain from one axis to the next,
or the axes of a fold in `folds` begin or end; and the longest piece every
address reads affinely, or zero when none does.
"""
function pieces(axes, addresses, domain, folds = ())
    affine(k) = all(e -> coefficient(e, k) !== nothing, addresses)
    chains(k, l) = affine(k) && affine(l) &&
        all(e -> coefficient(e, l) == coefficient(e, k) * domain[k], addresses)
    joined(k, l) = chains(k, l) && all(S -> (k in S) == (l in S), folds)
    found = Vector{Int}[]
    for k in axes
        !isempty(found) && joined(last(last(found)), k) ? push!(last(found), k) : push!(found, [k])
    end
    n(piece) = prod(k -> domain[k], piece; init = 1)
    eligible = [i for (i, piece) in enumerate(found) if all(affine, piece)]
    (found, isempty(eligible) ? 0 : eligible[argmax(map(i -> n(found[i]), eligible))])
end

# Whether the generator can load every load of the group under a schedule, and
# each fold over other axes than the swept ones reduces whole tile axes.
fits(c) = all(l -> loadable(c, l), (l for m in members(c.g) for l in loads(c.g.terms[m]))) &&
    all(r -> fold_tiles(c, r) !== nothing, partial_folds(c))

function default_tiles(extent, reduced)
    a, b = extent
    if reduced == 0
        ta = min(nextpow(2, a), POINTWISE_ELEMENTS)
        tb = max(1, min(nextpow(2, b), POINTWISE_ELEMENTS ÷ ta))
        return (sketch = :pointwise, tile = (ta, tb), splits = 1, occupancy = nothing)
    elseif reduced == 3
        ta = min(FULL_TILE_ELEMENTS, nextpow(2, a))
        tb = min(max(1, FULL_TILE_ELEMENTS ÷ ta), nextpow(2, b))
        return (sketch = :full, tile = (ta, tb), splits = 1, occupancy = 8)
    elseif reduced == 1
        if a <= RESIDENT_ROWS
            tile = resident_tile(a, b)
            values = prod(tile)
            occupancy = values <= PACKED_ELEMENTS ? 8 : values < RESIDENT_ROWS ? 4 : nothing
            return (sketch = :resident, tile, splits = 1, occupancy)
        end
        tile = (STREAMED_ROWS, 1)
        kept = b
        if kept < SPLIT_KEPT_BLOCKS[1]
            splits = split_count(a, tile[1], kept)
            while kept * splits < MINIMUM_BLOCKS && tile[1] > 256
                tile = (tile[1] ÷ 2, 1)
                splits = split_count(a, tile[1], kept)
            end
            splits > 1 && return (sketch = :split, tile, splits, occupancy = 8)
        end
        return (sketch = :streamed, tile, splits = 1, occupancy = nothing)
    else
        tile = (streamed_row_tile(a), min(STREAMED_ROW_COLUMNS, nextpow(2, b)))
        kept = cld(a, tile[1])
        if kept < SPLIT_KEPT_BLOCKS[2]
            tile = (min(32, nextpow(2, a)), 32)
            kept = cld(a, tile[1])
            splits = split_count(b, tile[2], kept)
            splits > 1 && return (sketch = :split, tile, splits, occupancy = 8)
            tile = (streamed_row_tile(a), min(STREAMED_ROW_COLUMNS, nextpow(2, b)))
        end
        return (sketch = cld(b, tile[2]) == 1 ? :resident : :streamed, tile, splits = 1, occupancy = nothing)
    end
end

struct Claim
    schedule::TileSchedule
end

# Blocks aim for this many when a fold over other than the swept axis keeps the
# swept axis: each block covers several kept tiles, so its partial states stay
# small.
const PARTIAL_BLOCKS = 256

recomputes_folds(::Tempest.Compiler.cuTileGenerator, target) = target isa CUDATarget

function claims(::Tempest.Compiler.cuTileGenerator, target::CUDATarget, g::GroupView)
    g.kind in (:pointwise, :reduction) || return Claim[]
    product = product_claim(target, g)
    product === nothing || return [product]
    expressible(g) || return Claim[]
    A, B, reduced, C = collapse_axes(g)
    maximum(axes -> prod(k -> g.domain[k], axes; init = 1), (A, B, C)) < typemax(Int32) ÷ 2 || return Claim[]
    s = default_schedule(g, A, B, C, reduced)
    c = Context(g, s)
    if !fits(c)
        s = refined(c)
        c = Context(g, s)
        fits(c) || return Claim[]
    end
    partial = partial_folds(c)
    # A resident kernel with folds that keep the swept axis covers several kept
    # tiles per block; each such fold reduces every kept axis.
    if s.sketch === :resident && any(r -> !(sweep_axis(s) in fold_tiles(c, r)), partial)
        kept = findfirst(ax -> !ax.reduced, s.tiling)
        if kept !== nothing && all(r -> kept in fold_tiles(c, r), partial)
            chunk = max(1, cld(cld(s.tiling[kept].extent, s.tiling[kept].tile), PARTIAL_BLOCKS))
            tiling = copy(s.tiling)
            tiling[kept] = with_chunk(tiling[kept], chunk)
            s = TileSchedule(s.sketch, tiling, s.splits, s.occupancy, s.swizzle, s.precision)
        end
    end
    [Claim(s)]
end

# The positions at which a kernel of schedule `s` computes term `t`: every
# position of the tiles it moves along, once per tile of the kept axes it does
# not (every block computes it for its tile), and once per block along the
# swept ones (before the sweep).
positions(s::TileSchedule, t) = prod(s.tiling; init = 1.0) do ax
    tiles = cld(ax.extent, ax.tile)
    any(a -> mentions(t, a), ax.axes) ? tiles * ax.tile : ax.reduced && s.sketch !== :full ? 1 : tiles
end

# The multiply-adds of the folds that `roots` recompute, at the positions of
# their bodies over the tiling the fold extends (see `fold_context`), and of
# the folds inside them.
recomputed_flops(c, roots) = sum(subterms(roots); init = 0.0) do t
    t isa Recomputed || return 0.0
    e = fold_context(c, t)
    2 * positions(e.s, t.body) + recomputed_flops(e, (t.body,))
end

# A schedule's launches: each of its kernels, moving the bytes of its
# arguments over the blocks of its grid. The first computes the group's
# products and the folds it recomputes.
function estimate(::Tempest.Compiler.cuTileGenerator, claim::Claim, target, g::GroupView)
    s = claim.schedule
    c = Context(g, s)
    workspaces = Dict{Any, Workspace}()
    kernels = schedule_kernels(c, workspaces)
    roots = [g.terms[m] for m in members(g) if haskey(g.terms, m)]
    recomputed = recomputed_flops(c, roots)
    flops, factor = if s.sketch === :product
        m, k, n = (ax.extent for ax in s.tiling[1:3])
        b = prod((ax.extent for ax in s.tiling[4:end]); init = 1) # the batch, and M and N axes on the grid
        r = first(g.reductions)
        t = first(product_factors(g.values[r].fold, g.terms[r]))
        2.0 * m * n * k * b * length(g.reductions), term_eltype(t isa Apply && t.f isa Cast{Float32} ? only(t.args) : t, g)
    else
        0.0, Float32
    end
    element(a) = a.buffer isa Int ? g.values[a.buffer].eltype : workspaces[a.buffer].eltype
    map(enumerate(kernels)) do (i, kernel)
        bytes = sum(a -> prod(a.sizes; init = 1) * sizeof(element(a)), kernel.args; init = 0)
        blocks = prod(kernel.grid; init = 1)
        i == 1 ? Work(bytes; flops, recomputed, factor, blocks, split = s.splits > 1) : Work(bytes; blocks)
    end
end

include("emit.jl")
include("cache.jl")
include("build.jl")
include("contract.jl")
include("attention.jl")

__init__() = register!(ct, Tempest.Compiler.cuTileGenerator(),
    Tempest.Operations.attention_forward => forward, Tempest.Operations.attention_backward => backward)

include("precompile.jl")

end
