# Emitting tile programs from a group and its schedule.

"A kernel argument: a buffer as a strided view."
struct ArgSpec
    buffer::Any # a value id, or a key into the kernel's workspace
    offset::Int
    sizes::Tuple{Vararg{Int}}
    strides::Tuple{Vararg{Int}}
end

# A view keeps only the tile axes it spans, in the order of their strides, so
# loads and stores have the shape and order a hand-written kernel would give
# them: the unit-stride axis first. Tiles are permuted and reshaped to the
# kernel's tile axes in registers (a transposed operand is loaded in storage
# order, then permuted).
compact(spans, xs) = Tuple(x for (s, x) in zip(spans, xs) if s)
storage_order(v) = sortperm(collect(compact(v.spans, v.strides)))

# A view's offset per block, if it has one, is one more axis of unit stride
# after the others, loaded at that offset with extent one.
function spec(v, extents)
    sizes, strides = v.dynamic === nothing ? ((), ()) : ((v.reach,), (1,))
    any(v.spans) || return ArgSpec(v.value, v.offset, (1, sizes...), (1, strides...))
    order = storage_order(v)
    ArgSpec(v.value, v.offset, (compact(v.spans, extents)[order]..., sizes...), (compact(v.spans, v.strides)[order]..., strides...))
end

# `at` is the one-based offset per block of a view that has one.
function load_nd(sym, v, tiles, indices, at = nothing)
    full = map((s, t) -> s ? t : 1, v.spans, tiles)
    extra, one = v.dynamic === nothing ? ((), ()) : ((at,), (1,))
    any(v.spans) || return :(reshape($(ct.load)($sym; index = (Int32(1), $(extra...)), shape = (1, $(one...))), $full))
    order = storage_order(v)
    shape, index = compact(v.spans, tiles), compact(v.spans, indices)
    loaded = :($(ct.load)($sym; index = ($(index[order]...), $(extra...)), shape = $((shape[order]..., one...)),
        padding_mode = $(ct.PaddingMode.Zero)))
    isempty(one) || (loaded = :(reshape($loaded, $(shape[order]))))
    order == 1:length(order) || (loaded = :(permutedims($loaded, $(Tuple(invperm(order))))))
    shape == full ? loaded : :(reshape($loaded, $full))
end

function store_nd(sym, v, tiles, indices, value)
    full = map((s, t) -> s ? t : 1, v.spans, tiles)
    tile = :($(ct.broadcast_to)($value, $full))
    any(v.spans) || return :($(ct.store)($sym; index = (Int32(1),), tile = reshape($tile, (1,))))
    order = storage_order(v)
    shape, index = compact(v.spans, tiles), compact(v.spans, indices)
    shape == full || (tile = :(reshape($tile, $shape)))
    order == 1:length(order) || (tile = :(permutedims($tile, $(Tuple(order)))))
    :($(ct.store)($sym; index = ($(index[order]...),), tile = $tile))
end

struct Workspace
    key::Any
    eltype::DataType
    length::Int
end

mutable struct TileKernel
    name::String
    args::Vector{ArgSpec}
    symbols::Dict{ArgSpec, Symbol}
    body::Vector{Any}
    grid::Tuple{Vararg{Int}}
    occupancy::Union{Nothing, Int}
end

TileKernel(name, grid, occupancy) = TileKernel(name, ArgSpec[], Dict{ArgSpec, Symbol}(), Any[], grid, occupancy)

function argument!(k::TileKernel, spec::ArgSpec)
    get!(k.symbols, spec) do
        push!(k.args, spec)
        Symbol(:arg, length(k.args))
    end
end

function source(k::TileKernel)
    :(function ($(map(i -> Symbol(:arg, i), eachindex(k.args))...),)
        $(k.body...)
        return
    end)
end

# Everything a kernel of a group needs to know while emitting.
struct Context
    graph::Any
    stored::Vector{Bool}
    g::Group
    s::TileSchedule
end

tile(c::Context) = tiles(c.s)
groups(c::Context) = groups(c.s)
reduced_dims(c::Context) = code(c.s) == 3 ? (1, 2) : only(swept(c.s))

value_spec(c::Context, v::View) = spec(v, extent(c.s))

# Tile axes of extent one per tile are bound by the grid (or a loop): one index
# per block, so a load may depend on them through any formula.
scalar_groups(c::Context) = Tuple(i for (i, ax) in enumerate(c.s.tiling) if ax.tile == 1)
load_view(c::Context, v, index) = view_of(c.graph, v, index, c.g, groups(c)...; scalar = scalar_groups(c))

# Positions. Domain variable `n` at the tile of block indices `indices`: the
# lanes of its tile axis, their collapsed index decoded into the axes of the
# axis's group; an Int32 tile over the tile axes, zero outside the tiling.
function variable_tile(c::Context, n, indices)
    ones = ntuple(_ -> 1, length(c.s.tiling))
    i = findfirst(ax -> n in ax.axes, c.s.tiling)
    i === nothing && return :($(ct.broadcast_to)($(ct.Tile)(Int32(0)), $ones))
    ax = c.s.tiling[i]
    lanes = :(reshape($(ct.arange)($(ax.tile)), $(Base.setindex(ones, ax.tile, i))))
    decode(c, ax, n, :($lanes .+ (($(indices[i]) - Int32(1)) * Int32($(ax.tile)) - Int32(1))); tile = true)
end

# Variable `n` of a tile axis's group from the axis's collapsed index `x`.
function decode(c::Context, ax::TileAxis, n, x; tile)
    q = findfirst(==(n), ax.axes)
    below = prod(k -> c.g.domain[k], ax.axes[1:(q - 1)]; init = 1)
    if below > 1
        x = tile ? :(div.($x, Int32($below))) : :(div($x, Int32($below)))
    end
    if q < length(ax.axes)
        x = tile ? :(rem.($x, Int32($(c.g.domain[n])))) : :(rem($x, Int32($(c.g.domain[n]))))
    end
    x
end

index_tile(c::Context, e::Affine, indices) =
    foldl(e.terms; init = :($(ct.broadcast_to)($(ct.Tile)(Int32($(e.constant))), $(ntuple(_ -> 1, length(c.s.tiling)))))) do x, (atom, k)
        :($x .+ Int32($k) .* $(index_tile(c, atom, indices)))
    end
index_tile(c::Context, a::Tempest.Var, indices) = variable_tile(c, a.n, indices)
index_tile(c::Context, a::Tempest.Div, indices) = :(fld.($(index_tile(c, a.e, indices)), Int32($(a.d))))
index_tile(c::Context, a::Tempest.Mod, indices) = :(mod.($(index_tile(c, a.e, indices)), Int32($(a.d))))

# A formula of axes bound by the grid, as a scalar per block.
block_scalar(c::Context, e::Affine, indices) =
    foldl(e.terms; init = :(Int32($(e.constant)))) do x, (atom, k)
        :($x + Int32($k) * $(block_scalar(c, atom, indices)))
    end
function block_scalar(c::Context, a::Tempest.Var, indices)
    i = findfirst(ax -> a.n in ax.axes, c.s.tiling)
    i === nothing && return :(Int32(0))
    decode(c, c.s.tiling[i], a.n, :($(indices[i]) - Int32(1)); tile = false)
end
block_scalar(c::Context, a::Tempest.Div, indices) = :(fld($(block_scalar(c, a.e, indices)), Int32($(a.d))))
block_scalar(c::Context, a::Tempest.Mod, indices) = :(mod($(block_scalar(c, a.e, indices)), Int32($(a.d))))

offset_at(c::Context, v::View, indices) = v.dynamic === nothing ? nothing : :(Int32(1) + $(block_scalar(c, v.dynamic, indices)))

# The workspace views of split schedules (partials, statistics) stay two-axis.
function load_expr(sym, spans, ta, tb, ia, ib)
    shape = (spans[1] ? ta : 1, spans[2] ? tb : 1)
    index = (spans[1] ? ia : :(Int32(1)), spans[2] ? ib : :(Int32(1)))
    :($(ct.load)($sym; index = ($(index...),), shape = $shape, padding_mode = $(ct.PaddingMode.Zero)))
end

store_expr(sym, spans, ta, tb, ia, ib, value) = :($(ct.store)(
    $sym;
    index = ($(spans[1] ? ia : :(Int32(1))), $(spans[2] ? ib : :(Int32(1)))),
    tile = $(ct.broadcast_to)($value, $((spans[1] ? ta : 1, spans[2] ? tb : 1))),
))

# Statements for `roots`, each subterm once, at the tile of block indices
# `indices` (one per tile axis). `names` maps terms already in scope (loads
# hoisted out of a loop, reductions) to their symbols.
function emit_tiles!(k::TileKernel, statements, c::Context, roots, names, indices...)
    for t in Tempest.subterms(roots; known = t -> haskey(names, t))
        if t isa Lit
            names[t] = t.value
            continue
        end
        name = gensym(:t)
        expr = if t isa Load
            v = load_view(c, t.value, t.index)
            load_nd(argument!(k, value_spec(c, v)), v, tile(c), indices, offset_at(c, v, indices))
        elseif t isa Tempest.Fold
            emit_fold!(k, statements, c, t, indices)
        elseif t isa Ix
            x = index_tile(c, t.expr, indices)
            T = Tempest.term_eltype(t, c.graph)
            T === Int32 ? x : :(convert($(ct.Tile){$T}, $x))
        elseif t isa Apply
            args = map(a -> names[a], t.args)
            t.f isa Tempest.Cast ? :(convert($(ct.Tile){$(t.eltype)}, $(only(args)))) : :($(t.f).($(args...)))
        else
            error("no tile lowering for $(typeof(t))")
        end
        push!(statements, :($name = $expr))
        names[t] = name
    end
    names
end

fold_of(c::Context, r) = c.graph.values[r].definition.fold
accumulator(c::Context, r) = Tempest.fold_types(fold_of(c, r), Tempest.term_eltype(c.g.terms[r], c.graph)).accumulator
state_width(f) = f isa Tempest.LogSumExpFold ? 2 : 1

state_symbols(r, f) = [Symbol(:acc_, r, :_, i) for i in 1:state_width(f)]

function init_state(f, A, shape)
    f isa Tempest.SumFold && return [:(fill($(zero(A)), $shape))]
    f isa Tempest.MaxFold && return [:(fill($(typemin(A)), $shape))]
    f isa Tempest.MinFold && return [:(fill($(typemax(A)), $shape))]
    f isa Tempest.ProdFold && return [:(fill($(one(A)), $shape))]
    [:(fill($(A(-Inf)), $shape)), :(fill($(zero(A)), $shape))]
end

const SHIFT = Tempest.lse_shift

# Accumulate the elements of tile `x` where `valid`, elementwise.
function accumulate_elements!(st, f, A, acc, x, valid)
    if f isa Tempest.SumFold
        push!(st, :($(acc[1]) = $(acc[1]) .+ ifelse.($valid, $x, $(zero(A)))))
    elseif f isa Tempest.MaxFold
        push!(st, :($(acc[1]) = max.($(acc[1]), ifelse.($valid, $x, $(typemin(A))))))
    elseif f isa Tempest.MinFold
        push!(st, :($(acc[1]) = min.($(acc[1]), ifelse.($valid, $x, $(typemax(A))))))
    elseif f isa Tempest.ProdFold
        push!(st, :($(acc[1]) = $(acc[1]) .* ifelse.($valid, $x, $(one(A)))))
    else
        m, s = acc
        top, sh = gensym(:top), gensym(:shift)
        push!(st, :($top = max.($m, ifelse.($valid, $x, $(A(-Inf))))))
        push!(st, :($sh = $SHIFT.($top)))
        push!(st, :($s = $s .* exp.($m .- $sh) .+ ifelse.($valid, exp.($x .- $sh), $(zero(A)))))
        push!(st, :($m = $top))
    end
end

# Merge partial states (unobserved) where `valid`, elementwise.
function accumulate_states!(st, f, A, acc, parts, valid)
    f isa Tempest.LogSumExpFold || return accumulate_elements!(st, f, A, acc, only(parts), valid)
    m, s = acc
    pm, ps = parts
    top, sh = gensym(:top), gensym(:shift)
    push!(st, :($top = max.($m, ifelse.($valid, $pm, $(A(-Inf))))))
    push!(st, :($sh = $SHIFT.($top)))
    push!(st, :($s = $s .* exp.($m .- $sh) .+ ifelse.($valid, $ps .* exp.($pm .- $sh), $(zero(A)))))
    push!(st, :($m = $top))
end

# The accumulators reduced along `dims`: a state, still unobserved.
function reduce_state!(st, f, A, acc, dims)
    names = [gensym(:state) for _ in acc]
    if f isa Tempest.SumFold
        push!(st, :($(names[1]) = sum($(acc[1]); dims = $dims)))
    elseif f isa Tempest.MaxFold
        push!(st, :($(names[1]) = maximum($(acc[1]); dims = $dims)))
    elseif f isa Tempest.MinFold
        push!(st, :($(names[1]) = minimum($(acc[1]); dims = $dims)))
    elseif f isa Tempest.ProdFold
        push!(st, :($(names[1]) = prod($(acc[1]); dims = $dims)))
    else
        m, s = acc
        push!(st, :($(names[1]) = maximum($m; dims = $dims)))
        push!(st, :($(names[2]) = sum($s .* exp.($m .- $SHIFT.($(names[1]))); dims = $dims)))
    end
    names
end

# A whole tile reduced directly where `valid`: no accumulators, and for a
# logsumexp one exponential per element (maximum, then a shifted sum).
function direct_state!(st, f, A, x, valid, dims)
    names = [gensym(:state) for _ in 1:state_width(f)]
    if f isa Tempest.SumFold
        push!(st, :($(names[1]) = sum(ifelse.($valid, $x, $(zero(A))); dims = $dims)))
    elseif f isa Tempest.MaxFold
        push!(st, :($(names[1]) = maximum(ifelse.($valid, $x, $(typemin(A))); dims = $dims)))
    elseif f isa Tempest.MinFold
        push!(st, :($(names[1]) = minimum(ifelse.($valid, $x, $(typemax(A))); dims = $dims)))
    elseif f isa Tempest.ProdFold
        push!(st, :($(names[1]) = prod(ifelse.($valid, $x, $(one(A))); dims = $dims)))
    else
        push!(st, :($(names[1]) = maximum(ifelse.($valid, $x, $(A(-Inf))); dims = $dims)))
        push!(st, :($(names[2]) = sum(ifelse.($valid, exp.($x .- $SHIFT.($(names[1]))), $(zero(A))); dims = $dims)))
    end
    names
end

# Merge partial state `parts` into accumulators of the same shape.
function combine_states!(st, f, A, acc, parts)
    if f isa Tempest.SumFold
        push!(st, :($(acc[1]) = $(acc[1]) .+ $(parts[1])))
    elseif f isa Tempest.MaxFold
        push!(st, :($(acc[1]) = max.($(acc[1]), $(parts[1]))))
    elseif f isa Tempest.MinFold
        push!(st, :($(acc[1]) = min.($(acc[1]), $(parts[1]))))
    elseif f isa Tempest.ProdFold
        push!(st, :($(acc[1]) = $(acc[1]) .* $(parts[1])))
    else
        m, s = acc
        pm, ps = parts
        top, sh = gensym(:top), gensym(:shift)
        push!(st, :($top = max.($m, $pm)))
        push!(st, :($sh = $SHIFT.($top)))
        push!(st, :($s = $s .* exp.($m .- $sh) .+ $ps .* exp.($pm .- $sh)))
        push!(st, :($m = $top))
    end
end

function observe!(st, f, A, T, state)
    name = gensym(:stat)
    value = f isa Tempest.LogSumExpFold ? :($SHIFT.($(state[1])) .+ log.($(state[2]))) : state[1]
    push!(st, :($name = $(A === T ? value : :(convert($(ct.Tile){$T}, $value)))))
    name
end

function as_accumulator(st, x, A, T, shape)
    name = gensym(:x)
    converted = A === T ? x : :(convert($(ct.Tile){$A}, $x))
    push!(st, :($name = $(ct.broadcast_to)($converted, $shape)))
    name
end

# A mask of the lanes of tile `t` along the reduced axis inside the domain.
function valid_mask!(st, c::Context, t)
    p = only(swept(c.s))
    T, E = c.s.tiling[p].tile, c.s.tiling[p].extent
    shape = ntuple(i -> i == p ? T : 1, length(c.s.tiling))
    name = gensym(:valid)
    push!(st, :($name = reshape($(ct.arange)($T), $shape) .<= Int32($E) - ($t - Int32(1)) * Int32($T)))
    name
end

# A mask of the tile's lanes inside the domain, along every tile axis.
function tile_mask!(st, c::Context, indices)
    n = length(c.s.tiling)
    terms = map(1:n) do i
        T, E = c.s.tiling[i].tile, c.s.tiling[i].extent
        shape = ntuple(q -> q == i ? T : 1, n)
        :(reshape($(ct.arange)($T), $shape) .<= Int32($E) - ($(indices[i]) - Int32(1)) * Int32($T))
    end
    name = gensym(:valid)
    push!(st, :($name = $(foldl((a, b) -> :($a .& $b), terms))))
    name
end

# Folds over the swept axes complete in a block (or by merging splits) before
# what reads them, in phases; folds over other axes complete by merging
# partial states.
primary_folds(c::Context) = [r for r in c.g.reductions if Tempest.primary(c.graph, c.g, r)]
partial_folds(c::Context) = [r for r in c.g.reductions if !Tempest.primary(c.graph, c.g, r)]

function reduction_phases(c::Context)
    folds = primary_folds(c)
    phases = sort!(unique(c.g.phase[r] for r in folds))
    [[r for r in folds if c.g.phase[r] == p] for p in phases]
end

"The tile axes fold `r` reduces, or nothing when its axes are not whole tile axes."
function fold_tiles(c::Context, r)
    S = Tempest.fold_axes(c.graph, r)
    tiles = [i for (i, ax) in enumerate(c.s.tiling) if issubset(ax.axes, S)]
    covered = reduce(vcat, (c.s.tiling[i].axes for i in tiles); init = Int[])
    all(k -> k in covered || c.g.domain[k] == 1, S) ? tiles : nothing
end

kept_tiles_of(c::Context, r) = [i for i in eachindex(c.s.tiling) if !(i in fold_tiles(c, r))]
fold_tile_shape(c::Context, r) = ntuple(i -> i in fold_tiles(c, r) ? 1 : tile(c)[i], length(c.s.tiling))

# A fold's value as a view over the tiling: its own buffer when stored, else a
# workspace of its shape (a fold that later launches read).
function fold_spec!(c::Context, r, workspaces)
    v = output_view(c, r)
    spec = value_spec(c, v)
    c.stored[r] && return (v, spec)
    key = (:fold, r)
    workspaces[key] = Workspace(key, c.graph.values[r].eltype, Tempest.storage_length(c.graph.values[r]))
    (v, ArgSpec(key, spec.offset, spec.sizes, spec.strides))
end

# Load the values of folds `rs` at the tile of block indices `indices`.
function load_folds!(k, st, c::Context, rs, names, indices, workspaces)
    for r in rs
        v, spec = fold_spec!(c, r, workspaces)
        index = ntuple(i -> i in fold_tiles(c, r) ? :(Int32(1)) : indices[i], length(c.s.tiling))
        name = gensym(:stat)
        push!(st, :($name = $(load_nd(argument!(k, spec), v, tile(c), index))))
        names[Stat(r)] = name
    end
end

# Partial states. A fold a block does not complete leaves its partial state per
# slot in a workspace laid out as (the extents of the tile axes it keeps...,
# slots); `merge_kernel` completes it.
partial_key(r, i) = (:partials, r, i)

function partial_spec!(c::Context, r, i, slots, workspaces)
    sizes = (Tuple(c.s.tiling[q].extent for q in kept_tiles_of(c, r))..., slots)
    strides = Tuple(prod(sizes[1:(d - 1)]; init = 1) for d in eachindex(sizes))
    key = partial_key(r, i)
    workspaces[key] = Workspace(key, accumulator(c, r), prod(sizes))
    ArgSpec(key, 0, sizes, strides)
end

# Store fold `r`'s partial state (a tile with its reduced tile axes at one) at
# block indices `indices` along the axes it keeps, in `slot`.
function store_partial!(k, st, c::Context, r, state, indices, slot, slots, workspaces)
    kept = kept_tiles_of(c, r)
    shape = (Tuple(tile(c)[q] for q in kept)..., 1)
    index = (Tuple(indices[q] for q in kept)..., slot)
    for (i, s) in enumerate(state)
        sym = argument!(k, partial_spec!(c, r, i, slots, workspaces))
        value = :(reshape($(ct.broadcast_to)($s, $(fold_tile_shape(c, r))), $shape))
        push!(st, :($(ct.store)($sym; index = ($(index...),), tile = $value)))
    end
end

# The partial state of each fold in `rs` over the tile at `indices`.
function tile_states!(k, st, c::Context, rs, names, indices)
    states = Dict{Int, Vector{Symbol}}()
    isempty(rs) && return states
    emit_tiles!(k, st, c, [c.g.terms[r] for r in rs], names, indices...)
    valid = tile_mask!(st, c, indices)
    for r in rs
        f, A = fold_of(c, r), accumulator(c, r)
        x = as_accumulator(st, names[c.g.terms[r]], A, Tempest.term_eltype(c.g.terms[r], c.graph), tile(c))
        states[r] = direct_state!(st, f, A, x, valid, Tuple(fold_tiles(c, r)))
    end
    states
end

# A linear slot from block indices along tile axes `axes`, whose counts are
# `counts`; the number of slots.
function slot_expr(indices, axes, counts)
    isempty(axes) && return (:(Int32(1)), 1)
    expr = :(Int32(0))
    stride = 1
    for (q, n) in zip(axes, counts)
        expr = :($expr + ($(indices[q]) - Int32(1)) * Int32($stride))
        stride *= n
    end
    (:($expr + Int32(1)), stride)
end

statistic_outputs(c::Context) = [m for m in c.g.outputs if c.graph.values[m].dims != c.g.domain]
full_outputs(c::Context) = [m for m in c.g.outputs if c.graph.values[m].dims == c.g.domain]

output_view(c::Context, m) = view_of(c.graph, m, Tempest.placement(c.graph, c.g, m), c.g, groups(c)...)

# Store values the group computes (outputs, or reductions as `Stat`s) at the
# tile of block indices `indices`.
function store_values!(k, st, c, vs, names, indices...)
    for m in vs
        v = output_view(c, m)
        value = m in c.g.reductions ? names[Stat(m)] : names[c.g.terms[m]]
        push!(st, store_nd(argument!(k, value_spec(c, v)), v, tile(c), indices, value))
    end
end

store_outputs!(k, st, c, ms, names, indices...) = store_values!(k, st, c, ms, names, indices...)

# The one product rule of the fold algebra: a sum over tile axis `p` whose
# contribution is `x * y`, where x and y both span `p` and each spans one
# other tiled axis, different ones, accumulates as a matrix product of their
# tiles, `muladd` on tensor cores. Returns the factors and their other axes,
# or nothing.
function bilinear_over(c::Context, f, t::Term, p)
    f isa Tempest.SumFold && t isa Apply && t.f === (*) && length(t.args) == 2 || return nothing
    x, y = t.args
    sx, sy = term_spans(c, x), term_spans(c, y)
    (sx === nothing || sy === nothing) && return nothing
    sx[p] && sy[p] || return nothing
    tiled(spans) = [q for q in eachindex(spans) if spans[q] && q != p && c.s.tiling[q].tile > 1]
    ix, iy = tiled(sx), tiled(sy)
    length(ix) == 1 && length(iy) == 1 && ix != iy || return nothing
    (x, y, only(ix), only(iy))
end

function bilinear(c::Context, r)
    sw = swept(c.s)
    length(sw) == 1 || return nothing
    bilinear_over(c, fold_of(c, r), c.g.terms[r], only(sw))
end

# The tile axes a term's loads, positions and recomputed folds span, or
# nothing if it reads reductions.
function term_spans(c::Context, t::Term)
    spans = fill(false, length(c.s.tiling))
    for s in Tempest.subterms((t,))
        s isa Stat && return nothing
        if s isa Union{Ix, Tempest.Fold}
            for (i, ax) in enumerate(c.s.tiling)
                any(k -> Tempest.mentions(s, k), ax.axes) && (spans[i] = true)
            end
        end
        s isa Load || continue
        v = load_view(c, s.value, s.index)
        v === nothing && return nothing
        spans .|= collect(v.spans)
    end
    spans
end

# A half-precision factor converted to Float32 enters the matrix product as
# it is: tensor cores accumulate its products in Float32 exactly.
function product_factor(c::Context, t::Term)
    t isa Apply && t.f isa Tempest.Cast{Float32} || return t
    inner = only(t.args)
    Tempest.term_eltype(inner, c.graph) <: Tempest.HalfFloat ? inner : t
end

# A tile whose lanes lie along tile axes `u` and `w` only, as a `(u, w)`
# matrix; and back.
function matrix_tile(c::Context, x, u, w)
    tu, tw = tile(c)[u], tile(c)[w]
    full = ntuple(q -> q == u ? tu : q == w ? tw : 1, length(c.s.tiling))
    m = :(reshape($(ct.broadcast_to)($x, $full), $(u < w ? (tu, tw) : (tw, tu))))
    u < w ? m : :(permutedims($m, (2, 1)))
end

function from_matrix(c::Context, m, u, w)
    full = ntuple(q -> q == u ? tile(c)[u] : q == w ? tile(c)[w] : 1, length(c.s.tiling))
    :(reshape($(u < w ? m : :(permutedims($m, (2, 1)))), $full))
end

# Accumulate step `t` of a product over tile axis `p` into `acc`.
function accumulate_product!(st, c::Context, names, (a, b, i, j), acc, t, p = only(swept(c.s)))
    tk = tile(c)[p]
    fa, fb = product_factor(c, a), product_factor(c, b)
    xa, xb = gensym(:a), gensym(:b)
    push!(st, :($xa = $(matrix_tile(c, names[fa], i, p))))
    push!(st, :($xb = $(matrix_tile(c, names[fb], p, j))))
    # Padded steps of the swept axis read zeros from loads, but a computed
    # factor need not vanish there.
    E = c.s.tiling[p].extent
    if E % tk != 0 && !(fa isa Load && fb isa Load)
        push!(st, :($xa = ifelse.(reshape($(ct.arange)($tk), (1, $tk)) .<= Int32($E) - ($t - Int32(1)) * Int32($tk), $xa,
            $(zero(Tempest.term_eltype(fa, c.graph))))))
    end
    single = Tempest.term_eltype(fa, c.graph) === Float32
    if c.s.precision === :tf32x3 && single
        # Each factor as a TF32 high part and a TF32 remainder; the product of
        # the remainders is below Float32 rounding and is dropped.
        TF = :($(ct.Tile){$(ct.TFloat32)})
        F = :($(ct.Tile){Float32})
        ah, al, bh, bl = gensym(:ah), gensym(:al), gensym(:bh), gensym(:bl)
        push!(st, :($ah = convert($TF, $xa)), :($al = convert($TF, $xa .- convert($F, $ah))))
        push!(st, :($bh = convert($TF, $xb)), :($bl = convert($TF, $xb .- convert($F, $bh))))
        push!(st, :($acc = muladd($al, $bh, $acc)), :($acc = muladd($ah, $bl, $acc)), :($acc = muladd($ah, $bh, $acc)))
        return
    end
    if c.s.precision === :tf32 && single
        push!(st, :($xa = convert($(ct.Tile){$(ct.TFloat32)}, $xa)))
        push!(st, :($xb = convert($(ct.Tile){$(ct.TFloat32)}, $xb)))
    end
    push!(st, :($acc = muladd($xa, $xb, $acc)))
end

# The accumulator of a product, as a state over the tile axes.
product_state(c::Context, (a, b, i, j), acc) = [from_matrix(c, acc, i, j)]

# Recomputed folds. A fold the kernel recomputes where it reads it extends the
# tiling by its reduced axis, swept in steps: a product's steps when its body
# is bilinear in that axis, the kernel's own steps when its body recomputes
# folds in turn (as much work per step as the kernel's sweep), `INNER_TILE`
# lanes otherwise, and one step when it fits. A bilinear body accumulates as a
# product of tiles; any other folds elementwise.
const INNER_TILE = 128

function fold_context(c::Context, t::Tempest.Fold; lanes = INNER_TILE)
    base = length(c.g.domain)
    inner = [TileAxis([base + j], e, min(nextpow(2, e), lanes), true) for (j, e) in enumerate(t.extents)]
    s = c.s
    e = Context(c.graph, c.stored, Group(c.g.kind, (c.g.domain..., t.extents...)),
        TileSchedule(s.sketch, [s.tiling; inner], s.splits, s.occupancy, s.swizzle, s.precision, s.online))
    lanes == INNER_TILE || return e
    f = c.graph.values[t.value].definition.fold
    if bilinear_over(e, f, t.body, length(s.tiling) + 1) !== nothing
        return fold_context(c, t; lanes = product_step(s.precision, Tempest.term_eltype(product_factor(e, t.body.args[1]), c.graph)))
    end
    sw = swept(s)
    any(x -> x isa Tempest.Fold, Tempest.subterms((t.body,))) && !isempty(sw) && s.tiling[first(sw)].tile != INNER_TILE || return e
    fold_context(c, t; lanes = s.tiling[first(sw)].tile)
end

# The shape of a term's tile: the tile along each axis it spans, one elsewhere.
spanned_shape(c::Context, t::Term) =
    ntuple(q -> any(k -> Tempest.mentions(t, k), c.s.tiling[q].axes) ? tile(c)[q] : 1, length(c.s.tiling))

function fold_expressible(c::Context, t::Tempest.Fold)
    val = c.graph.values[t.value]
    val.definition.fold isa FOLDS && val.eltype <: TILE_TYPES && length(t.extents) == 1 || return false
    terms_expressible(fold_context(c, t), (t.body,))
end

# Whether the generator can lower every subterm of `roots`.
function terms_expressible(c::Context, roots; stats = false)
    for s in Tempest.subterms(roots)
        s isa Stat && !stats && return false
        s isa Tempest.Fold && !fold_expressible(c, s) && return false
        s isa Ix && !positional(s) && return false
        s isa Apply && !(s.eltype <: TILE_TYPES) && return false
        s isa Lit && s.value isa Float64 && return false
        s isa Load && load_view(c, s.value, s.index) === nothing && return false
    end
    true
end

# A recomputed fold at block indices `indices`, as a tile over the tiling.
function emit_fold!(k, st, c::Context, t::Tempest.Fold, indices)
    e = fold_context(c, t)
    n = length(c.s.tiling)
    p = n + 1
    f = c.graph.values[t.value].definition.fold
    A = Tempest.fold_types(f, Tempest.term_eltype(t.body, c.graph)).accumulator
    T = c.graph.values[t.value].eltype
    tp, E = e.s.tiling[p].tile, e.s.tiling[p].extent
    steps = cld(E, tp)
    j = gensym(:j)
    at = steps == 1 ? :(Int32(1)) : j
    loop = steps == 1 ? st : Any[]
    product = bilinear_over(e, f, t.body, p)
    if product !== nothing
        x, y, u, w = product
        acc = gensym(:acc)
        push!(st, :($acc = fill(0.0f0, ($(tile(e)[u]), $(tile(e)[w])))))
        names = emit_tiles!(k, loop, e, unique([product_factor(e, x), product_factor(e, y)]), Dict{Term, Any}(), indices..., at)
        accumulate_product!(loop, e, names, product, acc, at, p)
        steps == 1 || push!(st, :(for $j in Int32(1):Int32($steps)
            $(loop...)
        end))
        stat = gensym(:stat)
        push!(st, :($stat = $(from_matrix(c, T === Float32 ? acc : :(convert($(ct.Tile){$T}, $acc)), u, w))))
        return stat
    end
    shape = Base.setindex(spanned_shape(e, t.body), tp, p)
    lanes = ntuple(q -> q == p ? tp : 1, p)
    names = emit_tiles!(k, loop, e, [t.body], Dict{Term, Any}(), indices..., at)
    valid = gensym(:valid)
    push!(loop, :($valid = reshape($(ct.arange)($tp), $lanes) .<= Int32($E) - ($at - Int32(1)) * Int32($tp)))
    x = as_accumulator(loop, names[t.body], A, Tempest.term_eltype(t.body, c.graph), shape)
    if steps == 1
        state = direct_state!(st, f, A, x, valid, p)
    else
        acc = [gensym(:acc) for _ in 1:state_width(f)]
        foreach((a, init) -> push!(st, :($a = $init)), acc, init_state(f, A, shape))
        accumulate_elements!(loop, f, A, acc, x, valid)
        push!(st, :(for $j in Int32(1):Int32($steps)
            $(loop...)
        end))
        state = reduce_state!(st, f, A, acc, p)
    end
    stat = observe!(st, f, A, T, state)
    result = gensym(:stat)
    push!(st, :($result = reshape($stat, $(shape[1:n]))))
    result
end
