# cuTile: one generator for every group a sketch can schedule.
#
# A group's domain collapses into two axes, A and B: each load or store must be
# a strided view of its buffer in those axes (a formula affine in the group's
# variables, with coefficients that chain within each collapsed group of axes).
# A schedule picks the tile, which collapsed axis is reduced, how the reduced
# axis is swept (resident in one tile, streamed, or split across blocks with a
# finishing launch), and hints. The generator turns any schedule into kernels;
# a sketch contributes parameters, never code.
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

struct TileSchedule
    sketch::Symbol            # :pointwise, :resident, :streamed, :split, :full
    extent::Tuple{Int, Int}   # the collapsed domain
    reduced::Int              # 0 none, 1 or 2 one axis, 3 both
    tile::Tuple{Int, Int}
    splits::Int               # blocks sharing one kept tile
    chunk::Int                # kept tiles one block sweeps, one after another
    occupancy::Union{Nothing, Int}
end

TileSchedule(sketch, extent, reduced, tile, splits, occupancy) =
    TileSchedule(sketch, extent, reduced, tile, splits, 1, occupancy)
with_chunk(s::TileSchedule, chunk) = TileSchedule(s.sketch, s.extent, s.reduced, s.tile, s.splits, chunk, s.occupancy)

function Base.show(io::IO, s::TileSchedule)
    print(io, s.sketch, " ", s.extent[1], "×", s.extent[2], ", tile ", s.tile[1], "×", s.tile[2])
    s.reduced in (1, 2) && print(io, ", reducing axis ", s.reduced)
    s.splits > 1 && print(io, ", split ", s.splits, " ways")
    s.chunk > 1 && print(io, ", ", s.chunk, " kept tiles per block")
    s.occupancy === nothing || print(io, ", occupancy ", s.occupancy)
end

sweeps(s::TileSchedule) = s.reduced in (1, 2) ? cld(s.extent[s.reduced], s.tile[s.reduced]) : 1
kept_tiles(s::TileSchedule) = s.reduced in (1, 2) ? cld(s.extent[3 - s.reduced], s.tile[3 - s.reduced]) : 1

# Which axes of the group's domain collapse into A and B, and which one the
# schedule sweeps: the group's primary axes.
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

"A load or store: a value's buffer as a strided view over the collapsed axes."
struct View
    value::Int
    offset::Int
    strides::Tuple{Int, Int}
    spans::Tuple{Bool, Bool} # whether the view extends along A and B
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

function view_of(graph, v, index, g, A, B)
    e = Tempest.address(graph, v, index, Tempest.Domain(g.domain))
    Tempest.is_affine(e) || return nothing
    sa = group_stride(e, A, g.domain)
    sb = group_stride(e, B, g.domain)
    (sa === nothing || sb === nothing || sa < 0 || sb < 0) && return nothing
    View(v, e.constant, (sa, sb), (sa != 0, sb != 0))
end

const FOLDS = Union{Tempest.SumFold, Tempest.MaxFold, Tempest.MinFold, Tempest.ProdFold, Tempest.LogSumExpFold}
const TILE_TYPES = Union{Float32, Float16, Core.BFloat16, Bool, Int32}

# Whether the generator can express a group: tile-typed values, supported
# folds, no positions or gathers, and every load and store a strided view.
function expressible(graph, stored, g::Group)
    axes = collapse_axes(graph, g)
    axes === nothing && return false
    for m in Tempest.members(g)
        val = graph.values[m]
        val.eltype <: TILE_TYPES || return false
        if m in g.reductions
            val.definition.fold isa FOLDS || return false
            collapsed_reduce(axes[1], axes[2], val.definition.reduced) === nothing && return false
        end
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
function default_schedule(g::Group, extent, reduced)
    a, b = extent
    if reduced == 0
        ta = min(nextpow(2, a), POINTWISE_ELEMENTS)
        tb = max(1, min(nextpow(2, b), POINTWISE_ELEMENTS ÷ ta))
        return TileSchedule(:pointwise, extent, 0, (ta, tb), 1, nothing)
    elseif reduced == 3
        ta = min(FULL_TILE_ELEMENTS, nextpow(2, a))
        tb = min(max(1, FULL_TILE_ELEMENTS ÷ ta), nextpow(2, b))
        return TileSchedule(:full, extent, 3, (ta, tb), 1, 8)
    elseif reduced == 1
        if a <= RESIDENT_ROWS
            tile = resident_tile(a, b)
            values = prod(tile)
            occupancy = values <= PACKED_ELEMENTS ? 8 : values < RESIDENT_ROWS ? 4 : nothing
            return TileSchedule(:resident, extent, 1, tile, 1, occupancy)
        end
        tile = (STREAMED_ROWS, 1)
        kept = b
        if kept < SPLIT_KEPT_BLOCKS[1]
            splits = split_count(a, tile[1], kept)
            while kept * splits < MINIMUM_BLOCKS && tile[1] > 256
                tile = (tile[1] ÷ 2, 1)
                splits = split_count(a, tile[1], kept)
            end
            splits > 1 && return TileSchedule(:split, extent, 1, tile, splits, 8)
        end
        return TileSchedule(:streamed, extent, 1, tile, 1, nothing)
    else
        tile = (streamed_row_tile(a), min(STREAMED_ROW_COLUMNS, nextpow(2, b)))
        kept = cld(a, tile[1])
        if kept < SPLIT_KEPT_BLOCKS[2]
            tile = (min(32, nextpow(2, a)), 32)
            kept = cld(a, tile[1])
            splits = split_count(b, tile[2], kept)
            splits > 1 && return TileSchedule(:split, extent, 2, tile, splits, 8)
            tile = (streamed_row_tile(a), min(STREAMED_ROW_COLUMNS, nextpow(2, b)))
        end
        return TileSchedule(cld(b, tile[2]) == 1 ? :resident : :streamed, extent, 2, tile, 1, nothing)
    end
end

struct Claim
    schedule::TileSchedule
    axes::Tuple
end

# The collapsed axes a fold reduces (1, 2 or both), or nothing when its axes
# are not whole collapsed groups.
function collapsed_reduce(A, B, axes)
    S = Set(axes)
    # An empty group of axes (B of a vector) is reduced by any fold.
    a = issubset(A, S)
    b = issubset(B, S)
    S == union(Set(a ? A : Int[]), Set(b ? B : Int[])) || return nothing
    a && b ? (1, 2) : a ? 1 : b ? 2 : nothing
end

# Blocks aim for this many when a fold over other than the swept axis keeps an
# axis: each block sweeps several kept tiles, so its partial states stay small.
const PARTIAL_BLOCKS = 256

function Tempest.claims(::Tempest.CuTileGenerator, p::Plan, g::Group)
    g.kind in (:pointwise, :reduction) || return Claim[]
    expressible(p.graph, p.stored, g) || return Claim[]
    A, B, reduced = collapse_axes(p.graph, g)
    extent = (prod(k -> g.domain[k], A; init = 1), prod(k -> g.domain[k], B; init = 1))
    max(extent...) < typemax(Int32) ÷ 2 || return Claim[]
    for m in Tempest.members(g), l in Tempest.loads(g.terms[m])
        view_of(p.graph, l.value, l.index, g, A, B) === nothing && return Claim[]
    end
    s = default_schedule(g, extent, reduced)
    keeping = any(r -> !Tempest.primary(p.graph, g, r) &&
        collapsed_reduce(A, B, Tempest.fold_axes(p.graph, r)) != (1, 2), g.reductions)
    s.sketch === :resident && keeping && (s = with_chunk(s, max(1, cld(kept_tiles(s), PARTIAL_BLOCKS))))
    [Claim(s, (A, B))]
end

include("cutile/emit.jl")
include("cutile/build.jl")

function __init__()
    push!(Tempest.EXTRA_IMPLEMENTATIONS, Tempest.CuTileGenerator())
end

end
