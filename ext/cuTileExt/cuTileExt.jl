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
@republic republic=false inherit=:public using Tempest.Compiler.IR
@republic republic=false inherit=:public import Tempest.Compiler.Implement
using Tempest.Compiler: Plan, group_views

# cuTile's generator, `Tempest.cuTileGenerator`: one implementation for every
# group a sketch can schedule, which can be made once this extension loads.
Implement.loaded(::Type{cuTileGenerator}) = true

in_place(::cuTileGenerator, claim) = true

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
    code(s) in (1, 2) && print(io, s.sketch === :scan ? ", scanning axis " : ", reducing axis ", code(s))
    s.splits > 1 && print(io, ", split ", s.splits, " ways")
    for ax in s.tiling
        ax.chunk > 1 && print(io, ", ", ax.chunk, " tiles per block")
    end
    s.occupancy === nothing || print(io, ", occupancy ", s.occupancy)
end

# Which axes of the group's domain collapse into A and B, and which one the
# schedule sweeps: the group's primary axes (those of its first fold not by
# key), one run of axes wherever they lie, or none where its folds are by key.
# A row reduction's kept axes after the swept ones, C, are bound by the grid;
# nothing when the primary axes are not a run.
function collapse_axes(g::GroupView)
    r = length(g.domain)
    R = sort!(collect(g.primary_axes))
    if g.kind === :pointwise || isempty(R)
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
the axes the grid binds, below `reach`. A group the grid binds may run
backward through the buffer: the view counts its positions from the end
(`reversed`, their number, zero for a group read forward).
"""
struct View
    value::Int
    offset::Int
    strides::Tuple{Vararg{Int}}
    spans::Tuple{Vararg{Bool}} # whether the view extends along each group
    dynamic::Union{Nothing, Affine}
    reach::Int
    reversed::Tuple{Vararg{Int}}
end

View(value, offset, strides, spans) = View(value, offset, strides, spans, nothing, 0, map(_ -> 0, strides))

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
    reversed = zeros(Int, length(groups))
    for i in scalar
        s = group_stride(e, groups[i], g.domain)
        s !== nothing && s >= 0 && continue
        if s !== nothing
            # Counted from the end, forward: a reversed axis, a recurrence's
            # derivative stepping back through the steps.
            n = prod(k -> g.domain[k], groups[i]; init = 1)
            e = Affine(e.constant + s * (n - 1), Tuple(t.first.n in groups[i] ? t.first => -t.second : t for t in e.terms))
            reversed[i] = n
            continue
        end
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
        # A formula that falls below zero, a position counted back from the
        # end as a reversed axis's, takes the address's constant with it.
        if lo < 0 && lo + e.constant >= 0
            lo, hi = lo + e.constant, hi + e.constant
            dynamic = Affine(e.constant, dynamic.terms)
            e = Affine(0, e.terms)
        end
        lo >= 0 || return nothing
        reach = hi + 1
    end
    for k in 1:length(g.domain)
        g.domain[k] > 1 && coefficient(e, k) != 0 && !any(grp -> k in grp, groups) && return nothing
    end
    strides = map(grp -> group_stride(e, grp, g.domain), groups)
    any(s -> s === nothing || s < 0, strides) && return nothing
    e.constant >= 0 || return nothing
    View(v, e.constant, Tuple(strides), Tuple(map(!=(0), strides)), dynamic, reach, Tuple(reversed))
end

const TILE_TYPES = Union{Float32, Float16, BFloat16, Bool, Int32, Int64}

# Tile types, and tuples of them: a fold's element of several components.
tile_type(T) = T <: TILE_TYPES || T <: Tuple && all(tile_type, fieldtypes(T))

# Positions computed from a formula without gathers.
positional(t::Ix) = isempty(indirect_values(t.expr)) && term_eltype(t, nothing) <: TILE_TYPES

# A fold by key the generator realizes: a sum of 32-bit floats or integers,
# which adds atomically, or a write, which stores; its positions Int32 lanes,
# and the contents it starts from a constant or a value it copies, laid out
# as its own.
function keyed_expressible(g::GroupView, r)
    val = g.values[r]
    val.fold isa SumFold && val.eltype in (Float32, Int32) || val.fold isa Overwrite || return false
    storage_length(val) < typemax(Int32) || return false
    t = val.into
    t === nothing || t isa Lit || t isa Load && copied(g, r, t)
end

function copied(g::GroupView, r, t::Load)
    val, from = g.values[r], g.values[t.value]
    d = Domain(val.dims)
    storage_length(from) == storage_length(val) && from.eltype === val.eltype &&
        address(from, t.index, d) == address(val, ntuple(k -> Affine(0, (Var(k) => 1,)), length(val.dims)), d)
end

# Whether the generator can express a group: tile-typed values, supported
# folds, scalar functions it lowers, and positions without gathers; its loads
# are checked with a schedule.
function expressible(g::GroupView)
    collapse_axes(g) === nothing && return false
    for m in members(g)
        val = g.values[m]
        val.eltype <: TILE_TYPES || return false
        m in g.reductions && !tile_fold(val.fold, term_eltype(g.terms[m], g)) && return false
        val.keyed && !keyed_expressible(g, m) && return false
        for t in subterms((g.terms[m],))
            t isa Ix && !positional(t) && return false
            t isa Apply && !tile_type(t.eltype) && return false
            t isa Lit && t.value isa Float64 && return false
            t isa Load && !(g.values[t.value].eltype <: TILE_TYPES) && return false
            t isa Apply && !lowered(t, g) && return false
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
        g.values[m].stored && !g.values[m].keyed && push!(found, address(g.values[m], g.placement[m], Domain(g.domain)))
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

# Whether the generator can load every load of the group under a schedule,
# and the indices the keys of its folds by key read; and each fold over other
# axes than the swept ones reduces whole tile axes. The loads of a fold the
# group recomputes are over the tiling it extends (`recomputing_fits`).
fits(c) = recomputing(c.g) ? recomputing_fits(c) :
    all(l -> loadable(c, l), (l for m in members(c.g) for l in loads(c.g.terms[m]))) &&
    all(a -> loadable(c, Load(a.value, a.index); index = true), key_atoms(c.g)) &&
    all(r -> fold_tiles(c, r) !== nothing, partial_folds(c))

key_atoms(g::GroupView) = [a for r in g.reductions if g.values[r].keyed for e in g.placement[r] for a in indirect_atoms(e)]

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

recomputes_folds(::cuTileGenerator, target) = target isa CUDATarget

# A scan's group sweeps its scanned axes as one tile axis, the scans' linear
# order, a block per tile of the kept axes (see `scan_kernel`). Its tiles
# group the elements and pad with the identity, which needs associativity
# and an identity: the builtin folds have them and a `GenericFold` claims
# them. An `InSequence` fold, Base's `accumulate` by a function of the
# user's, claims neither, and stays on the floor, a thread per line; so does
# an exclusive scan, which only derivatives make, until a tile's prefixes
# can shift by a lane. A long scanned axis with few kept tiles has few
# blocks: it could split as the `:split` sketch does, a launch folding each
# chunk to its total, a scan of the totals giving each chunk its carry, and
# a launch scanning the chunks again from it, where Futhark and CUB scan in
# one pass by a look-back between blocks, which needs flags cuTile does not
# have.
const SCAN_ELEMENTS = 1024
const SCAN_KEPT_TILE = 64

function scan_claims(g::GroupView)
    R = sort!(collect(g.primary_axes))
    R == collect(first(R):last(R)) || return Claim[]
    for r in g.reductions
        v = g.values[r]
        v.keyed && return Claim[]
        v.fold isa InSequence && return Claim[]
        v.scan === nothing && v.reduced != g.primary_axes && return Claim[]
        v.scan !== nothing && v.scan.exclusive && return Claim[]
    end
    all(m -> g.values[m].dims == g.domain, g.outputs) || return Claim[]
    expressible(g) && !recomputing(g) || return Claim[]
    n(axes) = prod(k -> g.domain[k], axes; init = 1)
    maximum(n, (R, 1:(first(R) - 1), (last(R) + 1):length(g.domain))) < typemax(Int32) ÷ 2 || return Claim[]
    c = Context(g, scan_schedule(g, R))
    if !fits(c)
        c = Context(g, refined(c))
        fits(c) || return Claim[]
    end
    [Claim(c.s)]
end

# The scanned axes one tile axis, swept; the axes before them, which vary
# fastest in memory, a kept tile axis, and those after them bound by the grid,
# or the axes after them kept when none come before. The kept tile is small
# enough to give the grid blocks, and the swept tile fills the rest of a
# tile's elements.
function scan_schedule(g::GroupView, R)
    n(axes) = prod(k -> g.domain[k], axes; init = 1)
    before, after = collect(1:(first(R) - 1)), collect((last(R) + 1):length(g.domain))
    kept = isempty(before) ? after : before
    tk = min(nextpow(2, n(kept)), prevpow(2, max(1, n(kept) ÷ MINIMUM_BLOCKS)), SCAN_KEPT_TILE)
    swept = TileAxis(R, n(R), min(nextpow(2, n(R)), SCAN_ELEMENTS ÷ tk), true)
    tiling = isempty(before) ? [swept, TileAxis(after, n(after), tk, false)] :
        [TileAxis(before, n(before), tk, false), swept]
    isempty(before) || isempty(after) || push!(tiling, TileAxis(after, n(after), 1, false))
    TileSchedule(:scan, tiling, 1, nothing, 0, :full)
end

# A fold whose term recomputes a product, as the log-sum-exp over keys of the
# scores a training step's backward recomputes, does a product's work at each
# step of its sweep: its kept tile takes a product's columns and its steps a
# product's depth, so the fold it recomputes is a product of tiles (see
# `fold_context`). The generator claims it under each such tile, and the cover
# measures them.
const RECOMPUTING_TILES = ((32, 64), (64, 64), (32, 128), (64, 128))

recomputing(g::GroupView) = any(m -> any(t -> t isa Recomputed, subterms((g.terms[m],))), members(g))

function recomputing_claims(target, g::GroupView, A, B, C, reduced)
    reduced in (1, 2) || return Claim[]
    n(axes) = prod(k -> g.domain[k], axes; init = 1)
    swept, kept = reduced == 1 ? (A, B) : (B, A)
    found = Claim[]
    for (ts, tk) in unique((min(ts, nextpow(2, n(swept))), min(tk, nextpow(2, n(kept)))) for (ts, tk) in RECOMPUTING_TILES)
        axes = [TileAxis(collect(swept), n(swept), ts, true), TileAxis(collect(kept), n(kept), tk, false)]
        tiling = reduced == 1 ? axes : reverse(axes)
        isempty(C) || push!(tiling, TileAxis(collect(C), n(C), 1, false))
        c = Context(g, TileSchedule(ts >= n(swept) ? :resident : :streamed, tiling, 1, nothing, 0, target.precision))
        recomputing_fits(c) || (c = Context(g, refined(c)); recomputing_fits(c)) || continue
        push!(found, Claim(c.s))
    end
    found
end

# Whether the generator can lower every term under a schedule, the folds it
# recomputes over the tiling they extend, and the indices the keys of its folds
# by key read; and each fold over other axes than the swept ones reduces whole
# tile axes.
recomputing_fits(c) = all(m -> terms_expressible(c, (c.g.terms[m],); stats = true), members(c.g)) &&
    all(a -> loadable(c, Load(a.value, a.index); index = true), key_atoms(c.g)) &&
    all(r -> fold_tiles(c, r) !== nothing, partial_folds(c))

# Candidates: of a kernel that is not a product's, its product folds, the
# most that split the axes alike, apart from its other folds, and apart from
# each other but where they share a read or a fold recomputed. Products the
# generator groups by a value they read in parts, chunks of a sequence say,
# stream as one kernel while one fold among them, a product by a constant, is
# not one; and one product's kernel steps each of its sums in every block, so
# products that share nothing gain nothing there and hold many sums at once.
function candidates(::cuTileGenerator, target::CUDATarget, p::Plan)
    found = []
    for g in group_views(p)
        g.kind === :reduction && product_axes(g) === nothing || continue
        alike = Dict{Any, Vector{Int}}()
        for r in g.reductions
            axes = product_axes(g, r)
            axes === nothing || push!(get!(() -> Int[], alike, min(axes, (axes[2], axes[1], axes[3], axes[4]))), r)
        end
        isempty(alike) && continue
        parts = Tuple{Vector{Int}, Vector{Term}}[]
        for r in argmax(length, values(alike))
            shared = Term[t for t in subterms((g.terms[r],); nested = true) if t isa Union{Load, Recomputed}]
            sharing = [k for (k, (_, ts)) in enumerate(parts) if any(t -> t in ts, shared)]
            joined = (vcat([r], (parts[k][1] for k in sharing)...), vcat(shared, (parts[k][2] for k in sharing)...))
            deleteat!(parts, sharing)
            push!(parts, joined)
        end
        sets = sort!([sort!(first(x)) for x in parts])
        length(sets) > 1 ? push!(found, sets) : length(only(sets)) > 1 && push!(found, only(sets))
    end
    found
end

# A recurrence: a block per tile of the axes along which no step reads what
# it computes at another position, stepping through the steps with the
# state's tile on chip, as flash-linear-attention's chunk_fwd_h holds its
# state; each step computes its members over the other axes whole, their
# products the kernel's recomputed folds (see `recurrence_kernel`). The tiled
# axis's tile is searched; each member's tile stays within
# `RECURRENCE_STATE` lanes, past which a Float32 state crowds the registers
# (flash-linear-attention's state at 128×16 took 0.51 ms, 128×32 0.79 ms and
# 128×64 13 ms).
const RECURRENCE_TILES = (16, 32, 64)
const RECURRENCE_STATE = 2048

function recurrence_claims(target, g::GroupView)
    (; axis, parallel, inner) = recurrence_axes(g)
    D = g.domain
    spans(m) = [k for k in eachindex(D) if any(e -> mentions(e, k), g.placement[m])]
    tiled = isempty(parallel) ? 0 : first(parallel)
    found = Claim[]
    for tile in (tiled == 0 ? (1,) : RECURRENCE_TILES)
        t = tiled == 0 ? 1 : min(tile, nextpow(2, D[tiled]))
        lanes(m) = prod(k -> k == tiled ? t : k in inner ? nextpow(2, D[k]) : 1, spans(m); init = 1)
        all(m -> lanes(m) <= RECURRENCE_STATE, g.outputs) || continue
        tiling = map(eachindex(D)) do k
            k == axis ? TileAxis([k], D[k], 1, true) : k in inner ? TileAxis([k], D[k], nextpow(2, D[k]), false) :
                TileAxis([k], D[k], k == tiled ? t : 1, false)
        end
        c = Context(g, TileSchedule(:recurrence, tiling, 1, nothing, 0, target.precision))
        all(m -> g.values[m].eltype <: TILE_TYPES && (!g.values[m].stored || output_view(c, m) !== nothing), g.outputs) &&
            terms_expressible(c, [g.terms[m] for m in g.outputs]) || continue
        any(cl -> cl.schedule.tiling == c.s.tiling, found) || push!(found, Claim(c.s))
    end
    found
end

function claims(::cuTileGenerator, target::CUDATarget, g::GroupView)
    g.kind === :scan && return scan_claims(g)
    g.kind === :recurrence && return recurrence_claims(target, g)
    g.kind in (:pointwise, :reduction) || return Claim[]
    products = product_claims(target, g)
    isempty(products) || return products
    expressible(g) || return Claim[]
    A, B, reduced, C = collapse_axes(g)
    maximum(axes -> prod(k -> g.domain[k], axes; init = 1), (A, B, C)) < typemax(Int32) ÷ 2 || return Claim[]
    recomputing(g) && return recomputing_claims(target, g, A, B, C, reduced)
    s = default_schedule(g, A, B, C, reduced)
    c = Context(g, s)
    if !fits(c)
        s = refined(c)
        c = Context(g, s)
        fits(c) || return Claim[]
    end
    # A pointwise kernel writes its outputs and scatters its folds by key; a
    # group with another fold, one over no axes say, is not one.
    s.sketch === :pointwise && any(r -> !g.values[r].keyed, g.reductions) && return Claim[]
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

"""
The variants of a product claim, its other depths and its split (see
`product_variants`); of a matrix-vector product's, its longer steps along the
sum (see `vector_product_variants`).
"""
variants(::cuTileGenerator, target::CUDATarget, g::GroupView, c::Claim) =
    [product_variants(target, g, c); vector_product_variants(g, c)]

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

# The scalar functions `roots` apply, each with the positions it is applied
# at: those of its term over the tiling, and inside recomputed folds over the
# tiling the fold extends. A tuple is no work; a product's own multiply is
# priced as flops, not here; a shifted sum's weight is an exponential.
function applied!(counts, c, roots; inner = counts)
    for t in subterms(roots)
        if t isa Recomputed
            # A recomputed product's multiply is its flops, priced as such; its
            # factors are applied, inside the fold's sweep.
            e = fold_context(c, t)
            body = t.body
            bilinear = c.g.values[t.value].fold isa SumFold && body isa Apply && body.f === (*) && length(body.args) == 2
            applied!(inner, e, bilinear ? map(x -> product_factor(e, x), body.args) : (body,); inner)
        elseif t isa Apply && t.f !== tuple && !(t.f isa Weight)
            counts[t.f] = get(counts, t.f, 0.0) + positions(c.s, t)
        end
    end
    counts
end

# The functions a schedule's kernels apply, streamed, and those applied inside
# a product's sweep: the factors of its sums, without the multiply that the
# tensor cores do, and the exponential of each element an exponential fold
# weighs. Any other schedule streams everything.
function applied(c, roots)
    sorted(counts) = sort!(collect(counts); by = p -> string(p.first))
    streaming, swept, inner = Dict{Any, Float64}(), Dict{Any, Float64}(), Dict{Any, Float64}()
    if c.s.sketch !== :product
        applied!(streaming, c, roots; inner)
        return (sorted(streaming), Pair{Any, Float64}[], sorted(inner))
    end
    factors = Term[]
    for r in c.g.reductions
        sums = fold_sums(c, r)
        if sums === nothing
            push!(factors, c.g.terms[r])
        else
            for (term, product) in sums
                product === nothing ? (term.f isa Weight || push!(factors, term)) :
                    append!(factors, [product_factor(c, product[1]), product_factor(c, product[2])])
            end
        end
        f = fold_of(c, r)
        if f isa Relative{Exponential}
            y = first(shifted_parts(c, c.g.terms[r]))
            push!(factors, y)
            swept[exp] = get(swept, exp, 0.0) + positions(c.s, y)
        end
    end
    applied!(swept, c, unique(factors); inner)
    applied!(streaming, c, [c.g.terms[m] for m in c.g.outputs]; inner)
    # A product that recomputes a fold sweeps the fold's tiles, which hide
    # a stall differently from a product's: all its swept work is inner.
    any(t -> t isa Recomputed, subterms(unique(factors))) && (mergewith!(+, inner, swept); empty!(swept))
    (sorted(streaming), sorted(swept), sorted(inner))
end

# A schedule's launches: each of its kernels, moving the bytes of its
# arguments over the blocks of its grid. The first after the folds by key's
# initial states computes the group's products and the folds it recomputes.
# A fold by key that starts from contents in storage is priced in place, as
# the plan's bytes are: no launch copies the contents, and a kernel moves the
# elements it combines, a sum reading them too.
function estimate(::cuTileGenerator, claim::Claim, target, g::GroupView)
    s = claim.schedule
    c = Context(g, s)
    workspaces = Dict{Any, Workspace}()
    kernels = schedule_kernels(c, workspaces; held = [r for r in keyed_folds(c) if g.values[r].into isa Load])
    roots = [g.terms[m] for m in members(g) if haskey(g.terms, m)]
    # A fold whose mask skips steps of its sweep (`live_steps`) does its sums,
    # and the folds it recomputes, and loads what it reads along the sweep, in
    # the share of the steps it takes; folds alike in their masks share what
    # they recompute.
    live = Dict(r => live_fraction(c, r) for r in g.reductions)
    classes = Dict{String, Vector{Int}}()
    foreach(r -> push!(get!(classes, string(mask_condition(c, r)), Int[]), r), g.reductions)
    recomputed = recomputed_flops(c, [g.terms[m] for m in members(g) if haskey(g.terms, m) && !(m in g.reductions)]) +
        sum(rs -> live[first(rs)] * recomputed_flops(c, [g.terms[r] for r in rs]), values(classes); init = 0.0)
    swept_axes = [a for ax in s.tiling if ax.reduced for a in ax.axes]
    function share(v)
        f = 0.0
        for m in members(g), t in (haskey(g.terms, m) ? loads(g.terms[m]) : Load[])
            t.value == v || continue
            f = max(f, m in g.reductions && any(a -> mentions(t, a), swept_axes) ? live[m] : 1.0)
        end
        f == 0.0 ? 1.0 : f
    end
    flops, factor = if s.sketch === :product
        m, k, n = (ax.extent for ax in s.tiling[1:3])
        b = prod((ax.extent for ax in s.tiling[4:end]); init = 1) # the batch, and M and N axes on the grid
        r = g.reductions[findfirst(r -> product_factors(g.values[r].fold, g.terms[r]) !== nothing, g.reductions)]
        2.0 * m * n * k * b * sum(r -> live[r], g.reductions), product_eltype(c, product_factors(g.values[r].fold, g.terms[r]))
    else
        0.0, Float32
    end
    # A load is read again by every tile along a kept axis its index does not
    # mention, and by every block a split sweep gives a part of the swept axes
    # when it mentions none of them: a value is read as often as its most
    # read load is. A product's tiles reuse what they read through the cache,
    # by design, and are priced on their line.
    kept_tiles = [(ax.axes, cld(ax.extent, ax.tile)) for ax in s.tiling if !ax.reduced]
    function times_read(v)
        s.sketch === :product && return 1
        n = 1
        for m in members(g), t in (haskey(g.terms, m) ? loads(g.terms[m]) : Load[])
            t.value == v || continue
            along = prod((k for (axes, k) in kept_tiles if !any(a -> mentions(t, a), axes)); init = 1)
            split = s.splits > 1 && !any(a -> mentions(t, a), swept_axes) ? s.splits : 1
            n = max(n, along * split)
        end
        n
    end
    element(a) = a.buffer isa Int ? g.values[a.buffer].eltype : workspaces[a.buffer].eltype
    streaming, swept, inner = applied(c, roots)
    main = findfirst(k -> k.name != INITIAL_STATES, kernels)
    combined(r) = (g.values[r].fold isa Overwrite ? 1 : 2) * prod(k -> g.domain[k], g.values[r].reduced; init = 1)
    elements(k, a) = k.name != INITIAL_STATES && a.buffer isa Int && g.values[a.buffer].keyed ? combined(a.buffer) : reachable(a)
    map(enumerate(kernels)) do (i, kernel)
        moved(a) = elements(kernel, a) * sizeof(element(a)) * (i == main && a.buffer isa Int ? share(a.buffer) : 1.0)
        bytes = sum(moved, kernel.args; init = 0.0)
        reread = i == main ? sum(a -> a.buffer isa Int ? moved(a) * (times_read(a.buffer) - 1) : 0.0, kernel.args; init = 0.0) : 0.0
        blocks = prod(kernel.grid; init = 1)
        i == main ? Work(bytes; reread, flops, recomputed, factor, blocks, split = s.splits > 1, applied = streaming, swept, inner) : Work(bytes; blocks)
    end
end

include("emit.jl")
include("cache.jl")
include("mask.jl")
include("build.jl")
include("contract.jl")
include("lowers.jl")

include("precompile.jl")

end
