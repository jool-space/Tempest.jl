# cuTile: one generator for every group a sketch can schedule.
#
# A schedule collapses the group's domain into tile axes: each a group of the
# domain's axes, with an extent, a tile size, and whether it is reduced. Every
# load and store must be a strided view of its buffer over those groups (a
# formula affine in the group's variables, with coefficients that chain within
# each group). A schedule also picks how the reduced axis is swept (resident in
# one tile, streamed, or split across blocks with a merging launch), the grid
# (plain or swizzled), hints and the precision of products. The generator turns
# any schedule into kernels; a sketch contributes parameters, never code.
module CuTileExt

using Tempest
using CUDACore
import cuTile as ct
import Tempest: Affine, Group, Plan, Term, Load, Lit, Ix, Apply, Stat, CUDATarget

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
    isempty(r) ? 0 : length(r) == 2 ? 3 : only(r)
end

sweeps(s::TileSchedule) = (r = swept(s); length(r) == 1 ? cld(s.tiling[only(r)].extent, s.tiling[only(r)].tile) : 1)
kept_tiles(s::TileSchedule) = prod((cld(ax.extent, ax.tile) for ax in s.tiling if !ax.reduced); init = 1)

function Base.show(io::IO, s::TileSchedule)
    if s.sketch === :product
        m, k, n = s.tiling[1:3]
        print(io, "product ", m.extent, "×", n.extent, " summing ", k.extent)
        length(s.tiling) > 3 && print(io, ", batch ", s.tiling[4].extent)
        print(io, ", tile ", m.tile, "×", n.tile, "×", k.tile, ", ",
            s.precision === :tf32 ? "TF32" : s.precision === :tf32x3 ? "3×TF32" : "full precision")
        s.splits > 1 && print(io, ", summing split ", s.splits, " ways")
        return
    end
    e, t = extent(s), tiles(s)
    print(io, s.sketch, " ", e[1], "×", e[2], ", tile ", t[1], "×", t[2])
    code(s) in (1, 2) && print(io, ", reducing axis ", code(s))
    s.splits > 1 && print(io, ", split ", s.splits, " ways")
    for ax in s.tiling
        ax.chunk > 1 && print(io, ", ", ax.chunk, " tiles per block")
    end
    s.occupancy === nothing || print(io, ", occupancy ", s.occupancy)
end

# Which axes of the group's domain collapse into A and B, and which one the
# schedule sweeps: the group's primary axes (those of its first fold).
function collapse_axes(graph, g::Group)
    r = length(g.domain)
    R = collect(Tempest.primary_axes(graph, g))
    if g.kind === :pointwise
        return (1:min(r, 1), 2:r, 0)
    elseif R == collect(1:r)
        return (1:min(r, 1), 2:r, 3)
    elseif R == collect(1:length(R))
        return (1:length(R), (length(R) + 1):r, 1)
    elseif R == collect((r - length(R) + 1):r)
        return (1:(r - length(R)), (r - length(R) + 1):r, 2)
    end
    nothing
end

"A load or store: a value's buffer as a strided view over collapsed groups of axes."
struct View
    value::Int
    offset::Int
    strides::Tuple{Vararg{Int}}
    spans::Tuple{Vararg{Bool}} # whether the view extends along each group
end

# The stride along one collapsed group of axes, or nothing when the address's
# coefficients do not chain over the group.
function group_stride(e::Affine, axes, domain)
    stride = nothing
    p = 1
    for k in axes
        n = domain[k]
        c = Tempest.coefficient(e, k)
        c === nothing && return nothing
        if n > 1
            c % p == 0 || return nothing
            s = c ÷ p
            stride === nothing ? (stride = s) : (stride == s || return nothing)
        end
        p *= n
    end
    something(stride, 0)
end

# The view of value `v` read at `index` over the groups of axes; nothing when the
# address is not affine, does not chain within a group, or depends on an axis
# outside the groups.
function view_of(graph, v, index, g, groups...)
    e = Tempest.address(graph, v, index, Tempest.Domain(g.domain))
    Tempest.is_affine(e) || return nothing
    for k in 1:length(g.domain)
        g.domain[k] > 1 && Tempest.coefficient(e, k) != 0 && !any(grp -> k in grp, groups) && return nothing
    end
    strides = map(grp -> group_stride(e, grp, g.domain), groups)
    any(s -> s === nothing || s < 0, strides) && return nothing
    View(v, e.constant, Tuple(strides), Tuple(map(!=(0), strides)))
end

const FOLDS = Union{Tempest.SumFold, Tempest.MaxFold, Tempest.MinFold, Tempest.ProdFold, Tempest.LogSumExpFold}
const TILE_TYPES = Union{Float32, Float16, Core.BFloat16, Bool, Int32}

# Whether the generator can express a group: tile-typed values, supported
# folds, no positions or gathers, and every load and store a strided view.
function expressible(graph, stored, g::Group)
    collapse_axes(graph, g) === nothing && return false
    for m in Tempest.members(g)
        val = graph.values[m]
        val.eltype <: TILE_TYPES || return false
        m in g.reductions && !(val.definition.fold isa FOLDS) && return false
        for t in Tempest.subterms((g.terms[m],))
            t isa Ix && return false
            t isa Apply && !(t.eltype <: TILE_TYPES) && return false
            t isa Lit && t.value isa Float64 && return false
            t isa Load && !(graph.values[t.value].eltype <: TILE_TYPES) && return false
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

"The default schedule of a group over the collapsed domain `extent`."
default_schedule(g::Group, A, B, extent, reduced) =
    (s -> TileSchedule(s.sketch, A, B, extent, reduced, s.tile, s.splits, s.occupancy))(default_tiles(extent, reduced))

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

function Tempest.claims(::Tempest.CuTileGenerator, target, p::Plan, g::Group)
    g.kind in (:pointwise, :reduction) || return Claim[]
    product = product_claim(target, p, g)
    product === nothing || return [product]
    expressible(p.graph, p.stored, g) || return Claim[]
    A, B, reduced = collapse_axes(p.graph, g)
    extent = (prod(k -> g.domain[k], A; init = 1), prod(k -> g.domain[k], B; init = 1))
    max(extent...) < typemax(Int32) ÷ 2 || return Claim[]
    for m in Tempest.members(g), l in Tempest.loads(g.terms[m])
        view_of(p.graph, l.value, l.index, g, A, B) === nothing && return Claim[]
    end
    s = default_schedule(g, A, B, extent, reduced)
    c = Context(p.graph, p.stored, g, s)
    partial = partial_folds(c)
    any(r -> fold_tiles(c, r) === nothing, partial) && return Claim[]
    # A resident kernel with folds that keep the swept axis covers several kept
    # tiles per block; each such fold reduces every kept axis.
    if s.sketch === :resident && any(r -> !(only(swept(s)) in fold_tiles(c, r)), partial)
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

include("cutile/emit.jl")
include("cutile/build.jl")
include("cutile/contract.jl")
include("cutile/attention.jl")

function __init__()
    push!(Tempest.EXTRA_IMPLEMENTATIONS, Tempest.CuTileGenerator())
    push!(Tempest.EXTRA_IMPLEMENTATIONS, CuTileAttention())
end

end
