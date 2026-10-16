# Emitting tile programs from a group and its schedule.

"A kernel argument: a buffer as a strided view."
struct ArgSpec
    buffer::Any # a value id, or a key into the kernel's workspace
    offset::Int
    sizes::Tuple{Vararg{Int}}
    strides::Tuple{Vararg{Int}}
end

# A view keeps only the tile axes it spans, so loads and stores have the shape
# a hand-written kernel would give them; tiles are reshaped to the kernel's
# tile axes in registers.
compact(spans, xs) = Tuple(x for (s, x) in zip(spans, xs) if s)

function spec(v, extents)
    any(v.spans) || return ArgSpec(v.value, v.offset, (1,), (1,))
    ArgSpec(v.value, v.offset, compact(v.spans, extents), compact(v.spans, v.strides))
end

function load_nd(sym, spans, tiles, indices)
    full = map((s, t) -> s ? t : 1, spans, tiles)
    any(spans) || return :(reshape($(ct.load)($sym; index = (Int32(1),), shape = (1,)), $full))
    shape, index = compact(spans, tiles), compact(spans, indices)
    loaded = :($(ct.load)($sym; index = ($(index...),), shape = $shape, padding_mode = $(ct.PaddingMode.Zero)))
    shape == full ? loaded : :(reshape($loaded, $full))
end

function store_nd(sym, spans, tiles, indices, value)
    full = map((s, t) -> s ? t : 1, spans, tiles)
    shape, index = any(spans) ? (compact(spans, tiles), compact(spans, indices)) : ((1,), (:(Int32(1)),))
    tile = :($(ct.broadcast_to)($value, $full))
    shape == full || (tile = :(reshape($tile, $shape)))
    :($(ct.store)($sym; index = ($(index...),), tile = $tile))
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
    for t in Tempest.subterms(roots)
        haskey(names, t) && continue
        if t isa Lit
            names[t] = t.value
            continue
        end
        name = gensym(:t)
        expr = if t isa Load
            v = view_of(c.graph, t.value, t.index, c.g, groups(c)...)
            load_nd(argument!(k, value_spec(c, v)), v.spans, tile(c), indices)
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

function full_mask!(st, c::Context, ia, ib)
    ta, tb = tile(c)
    a, b = extent(c.s)
    name = gensym(:valid)
    push!(st, :($name = (reshape($(ct.arange)($ta), ($ta, 1)) .<= Int32($a) - ($ia - Int32(1)) * Int32($ta)) .&
        (reshape($(ct.arange)($tb), (1, $tb)) .<= Int32($b) - ($ib - Int32(1)) * Int32($tb))))
    name
end

# Where a reduction or a reduction-shaped output lives in storage, as a view
# of the collapsed kept axis.
function statistic_spec(c::Context, r, buffer)
    a, b = extent(c.s)
    code(c.s) == 1 && return ArgSpec(buffer, 0, (1, b), (1, 1))
    code(c.s) == 2 && return ArgSpec(buffer, 0, (a, 1), (1, 1))
    ArgSpec(buffer, 0, (1, 1), (1, 1))
end

statistic_spans(c::Context) = code(c.s) == 1 ? (false, true) : code(c.s) == 2 ? (true, false) : (false, false)

function statistic_buffer(c::Context, r, workspaces)
    c.stored[r] && return r
    key = (:statistic, r)
    a, b = extent(c.s)
    n = code(c.s) == 1 ? b : code(c.s) == 2 ? a : 1
    workspaces[key] = Workspace(key, c.graph.values[r].eltype, n)
    key
end

function reduction_phases(g::Group)
    phases = sort!(unique(g.phase[r] for r in g.reductions))
    [[r for r in g.reductions if g.phase[r] == p] for p in phases]
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
        push!(st, store_nd(argument!(k, value_spec(c, v)), v.spans, tile(c), indices, value))
    end
end

store_outputs!(k, st, c, ms, names, indices...) = store_values!(k, st, c, ms, names, indices...)

# The one product rule of the fold algebra: a sum whose contribution is `x * y`,
# where x and y each span the swept axis and a different one of two kept tiled
# axes (around it in tile order), accumulates as a matrix product of their
# tiles, `muladd` on tensor cores. Returns the factors in matrix order and the
# kept axes, or nothing.
function bilinear(c::Context, r)
    fold_of(c, r) isa Tempest.SumFold || return nothing
    t = c.g.terms[r]
    t isa Apply && t.f === (*) && length(t.args) == 2 || return nothing
    sw = swept(c.s)
    length(sw) == 1 || return nothing
    p = only(sw)
    x, y = t.args
    sx, sy = term_spans(c, x), term_spans(c, y)
    (sx === nothing || sy === nothing) && return nothing
    sx[p] && sy[p] || return nothing
    kept = [i for i in eachindex(sx) if i != p && c.s.tiling[i].tile > 1]
    length(kept) == 2 || return nothing
    i, j = kept
    i < p < j || return nothing
    sx[i] && !sx[j] && sy[j] && !sy[i] && return (x, y, i, j)
    sy[i] && !sy[j] && sx[j] && !sx[i] && return (y, x, i, j)
    nothing
end

# The tile axes a term's loads span, or nothing if it reads positions or
# reductions.
function term_spans(c::Context, t::Term)
    spans = fill(false, length(c.s.tiling))
    for s in Tempest.subterms((t,))
        s isa Union{Ix, Stat} && return nothing
        s isa Load || continue
        v = view_of(c.graph, s.value, s.index, c.g, groups(c)...)
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

function accumulate_product!(st, c::Context, names, (a, b, i, j), acc, t)
    shape = tile(c)
    p = only(swept(c.s))
    tm, tk, tn = shape[i], shape[p], shape[j]
    n = length(shape)
    fa, fb = product_factor(c, a), product_factor(c, b)
    xa, xb = gensym(:a), gensym(:b)
    push!(st, :($xa = reshape($(ct.broadcast_to)($(names[fa]), $(ntuple(k -> k == i ? tm : k == p ? tk : 1, n))), ($tm, $tk))))
    push!(st, :($xb = reshape($(ct.broadcast_to)($(names[fb]), $(ntuple(k -> k == p ? tk : k == j ? tn : 1, n))), ($tk, $tn))))
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
product_state(c::Context, (a, b, i, j), acc) =
    [:(reshape($acc, $(ntuple(k -> k == only(swept(c.s)) ? 1 : tile(c)[k], length(c.s.tiling)))))]
