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
        length(s.tiling) > 3 && print(io, ", batch ", join((ax.extent for ax in s.tiling[4:end]), "×"))
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
function view_of(graph, v, index, g, groups...; scalar = ())
    domain = Tempest.Domain(g.domain)
    e = Tempest.address(graph, v, index, domain)
    rest = [t for t in e.terms if !(t.first isa Tempest.Var)]
    e = Affine(e.constant, Tuple(t for t in e.terms if t.first isa Tempest.Var))
    for i in scalar
        s = Tempest.group_stride(e, groups[i], g.domain)
        s !== nothing && s >= 0 && continue
        append!(rest, [t for t in e.terms if t.first.n in groups[i]])
        e = Affine(e.constant, Tuple(t for t in e.terms if !(t.first.n in groups[i])))
    end
    dynamic = nothing
    reach = 0
    if !isempty(rest)
        dynamic = Affine(0, Tuple(rest))
        bound = reduce(vcat, (groups[i] for i in scalar); init = Int[])
        isempty(Tempest.indirect_values(dynamic)) || return nothing
        all(k -> k in bound || g.domain[k] == 1 || !Tempest.mentions(dynamic, k), eachindex(g.domain)) || return nothing
        lo, hi = Tempest.affine_range(dynamic, domain)
        lo >= 0 || return nothing
        reach = hi + 1
    end
    for k in 1:length(g.domain)
        g.domain[k] > 1 && Tempest.coefficient(e, k) != 0 && !any(grp -> k in grp, groups) && return nothing
    end
    strides = map(grp -> Tempest.group_stride(e, grp, g.domain), groups)
    any(s -> s === nothing || s < 0, strides) && return nothing
    View(v, e.constant, Tuple(strides), Tuple(map(!=(0), strides)), dynamic, reach)
end

const TILE_TYPES = Union{Float32, Float16, Core.BFloat16, Bool, Int32}

# Tile types, and tuples of them: a fold's element of several components.
tile_type(T) = T <: TILE_TYPES || T <: Tuple && all(tile_type, fieldtypes(T))

# Positions computed from a formula without gathers.
positional(t::Ix) = isempty(Tempest.indirect_values(t.expr)) && Tempest.term_eltype(t, nothing) <: TILE_TYPES

# Whether the generator can express a group: tile-typed values, supported
# folds, and positions without gathers; its loads are checked with a schedule.
function expressible(graph, stored, g::Group)
    collapse_axes(graph, g) === nothing && return false
    for m in Tempest.members(g)
        val = graph.values[m]
        val.eltype <: TILE_TYPES || return false
        m in g.reductions && !tile_fold(val.definition.fold, Tempest.term_eltype(g.terms[m], graph)) && return false
        for t in Tempest.subterms((g.terms[m],))
            t isa Tempest.Fold && return false
            t isa Ix && !positional(t) && return false
            t isa Apply && !tile_type(t.eltype) && return false
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

Tempest.recomputes_folds(::Tempest.cuTileGenerator, target) = target isa CUDATarget

function Tempest.claims(::Tempest.cuTileGenerator, target::CUDATarget, p::Plan, g::Group)
    g.kind in (:pointwise, :reduction) || return Claim[]
    product = product_claim(target, p, g)
    product === nothing || return [product]
    expressible(p.graph, p.stored, g) || return Claim[]
    A, B, reduced = collapse_axes(p.graph, g)
    extent = (prod(k -> g.domain[k], A; init = 1), prod(k -> g.domain[k], B; init = 1))
    max(extent...) < typemax(Int32) ÷ 2 || return Claim[]
    s = default_schedule(g, A, B, extent, reduced)
    c = Context(p.graph, p.stored, g, s)
    for m in Tempest.members(g), l in Tempest.loads(g.terms[m])
        loadable(c, l) || return Claim[]
    end
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

# Product rates the generator's schedules reach, measured on known devices
# against cuBLAS (the scratchpad's rates.jl): full Float32 on CUDA cores, TF32
# and half precision on tensor cores. Other devices scale GB10's by SM count.
const PRODUCT_RATES = Dict("NVIDIA GB10" => (full = 2.6e12, tf32 = 25.0e12, half = 70.0e12))

function product_rate(target::CUDATarget, T)
    device = CUDACore.device()
    r = get(PRODUCT_RATES, CUDACore.name(device)) do
        scale = CUDACore.attribute(device, CUDACore.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) / 48
        map(x -> scale * x, PRODUCT_RATES["NVIDIA GB10"])
    end
    T <: Tempest.HalfFloat ? r.half : target.precision === :full ? r.full : target.precision === :tf32 ? r.tf32 : r.tf32 / 3
end

# The blocks of a schedule's first launch: one per kept tile (or chunk of
# them) and split, and one per tile of a fully reduced domain.
first_blocks(s::TileSchedule) = s.splits * prod(s.tiling; init = 1) do ax
    tiles = cld(ax.extent, ax.tile)
    ax.reduced ? (s.sketch === :full ? tiles : 1) : cld(tiles, ax.chunk)
end

# A schedule's launches: the main kernel over its blocks, products at the
# generator's rate, and a merge of partial states when blocks share a tile.
function Tempest.estimate(::Tempest.cuTileGenerator, claim::Claim, target, p::Plan, g::Group)
    s = claim.schedule
    blocks = first_blocks(s)
    bytes = Tempest.group_bytes(p, g)
    work = if s.sketch === :product
        m, k, n = (ax.extent for ax in s.tiling[1:3])
        b = prod((ax.extent for ax in s.tiling[4:end]); init = 1) # the batch, and M and N axes on the grid
        r = first(g.reductions)
        t = first(Tempest.product_factors(p.graph.values[r].definition.fold, g.terms[r]))
        factor = t isa Apply && t.f isa Tempest.Cast{Float32} ? only(t.args) : t
        rate = product_rate(target, Tempest.term_eltype(factor, p.graph))
        [Tempest.Work(bytes; flops = 2.0 * m * n * k * b * length(g.reductions), rate, blocks)]
    else
        [Tempest.Work(bytes; blocks)]
    end
    if s.splits > 1 || s.sketch === :full
        kept = prod((ax.extent for ax in s.tiling if !ax.reduced); init = 1)
        push!(work, Tempest.Work(2.0 * 4 * kept * blocks * length(g.reductions); blocks = max(1, kept / 256)))
    end
    work
end

include("emit.jl")
include("cache.jl")
include("build.jl")
include("contract.jl")
include("attention.jl")

__init__() = Tempest.register!(ct, Tempest.cuTileGenerator(), cuTileAttention())

include("precompile.jl")

end
