# Emitting tile programs from a group and its schedule.

"A kernel argument: a buffer as a two-dimensional strided view."
struct ArgSpec
    buffer::Any # a value id, or a key into the kernel's workspace
    offset::Int
    sizes::Tuple{Int, Int}
    strides::Tuple{Int, Int}
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
    axes::Tuple
end

tile(c::Context) = c.s.tile
reduced_dims(c::Context) = c.s.reduced == 3 ? (1, 2) : c.s.reduced

value_spec(c::Context, v::View) = ArgSpec(
    v.value,
    v.offset,
    (v.spans[1] ? c.s.extent[1] : 1, v.spans[2] ? c.s.extent[2] : 1),
    (v.spans[1] ? v.strides[1] : 1, v.spans[2] ? v.strides[2] : 1),
)

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

# Statements for `roots`, each subterm once. `names` maps terms already in
# scope (loads hoisted out of a loop, reductions) to their symbols.
function emit_tiles!(k::TileKernel, statements, c::Context, roots, names, ia, ib)
    ta, tb = tile(c)
    A, B = c.axes
    for t in Tempest.subterms(roots)
        haskey(names, t) && continue
        if t isa Lit
            names[t] = t.value
            continue
        end
        name = gensym(:t)
        expr = if t isa Load
            v = view_of(c.graph, t.value, t.index, c.g, A, B)
            load_expr(argument!(k, value_spec(c, v)), v.spans, ta, tb, ia, ib)
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
    ta, tb = tile(c)
    a, b = c.s.extent
    name = gensym(:valid)
    if c.s.reduced == 1
        push!(st, :($name = reshape($(ct.arange)($ta), ($ta, 1)) .<= Int32($a) - ($t - Int32(1)) * Int32($ta)))
    else
        push!(st, :($name = reshape($(ct.arange)($tb), (1, $tb)) .<= Int32($b) - ($t - Int32(1)) * Int32($tb)))
    end
    name
end

function full_mask!(st, c::Context, ia, ib)
    ta, tb = tile(c)
    a, b = c.s.extent
    name = gensym(:valid)
    push!(st, :($name = (reshape($(ct.arange)($ta), ($ta, 1)) .<= Int32($a) - ($ia - Int32(1)) * Int32($ta)) .&
        (reshape($(ct.arange)($tb), (1, $tb)) .<= Int32($b) - ($ib - Int32(1)) * Int32($tb))))
    name
end

# Where a reduction or a reduction-shaped output lives in storage, as a view
# of the collapsed kept axis.
function statistic_spec(c::Context, r, buffer)
    a, b = c.s.extent
    c.s.reduced == 1 && return ArgSpec(buffer, 0, (1, b), (1, 1))
    c.s.reduced == 2 && return ArgSpec(buffer, 0, (a, 1), (1, 1))
    ArgSpec(buffer, 0, (1, 1), (1, 1))
end

statistic_spans(c::Context) = c.s.reduced == 1 ? (false, true) : c.s.reduced == 2 ? (true, false) : (false, false)

function statistic_buffer(c::Context, r, workspaces)
    c.stored[r] && return r
    key = (:statistic, r)
    a, b = c.s.extent
    n = c.s.reduced == 1 ? b : c.s.reduced == 2 ? a : 1
    workspaces[key] = Workspace(key, c.graph.values[r].eltype, n)
    key
end

function reduction_phases(g::Group)
    phases = sort!(unique(g.phase[r] for r in g.reductions))
    [[r for r in g.reductions if g.phase[r] == p] for p in phases]
end

statistic_outputs(c::Context) = [m for m in c.g.outputs if c.graph.values[m].dims != c.g.domain]
full_outputs(c::Context) = [m for m in c.g.outputs if c.graph.values[m].dims == c.g.domain]

output_view(c::Context, m) =
    view_of(c.graph, m, Tempest.placement(c.graph, c.g, m), c.g, c.axes[1], c.axes[2])

# Store full-domain outputs of the tile at (ia, ib).
function store_outputs!(k, st, c, ms, names, ia, ib)
    ta, tb = tile(c)
    for m in ms
        v = output_view(c, m)
        push!(st, store_expr(argument!(k, value_spec(c, v)), v.spans, ta, tb, ia, ib, names[c.g.terms[m]]))
    end
end
