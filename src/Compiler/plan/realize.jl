# How a value read on another device than its own gets there, chosen for each
# value once the cut has placed it (see `realizations`): copied there, a
# crossing of it; made again there from what it reads there; or made in
# pieces, a range of its last axis on each of several devices, each piece
# copied to the others, which put the pieces together. A realization is a
# type with three methods: `link_load`, what it moves over each device's
# link in each direction, of which the lighter wins; `realize!`, what it adds
# to the placed graph on a device that reads the value, other than as its own
# device's value; and `home!`, what it adds for its own device. Another is
# added here, with its methods, and to the union.

"A value copied to each other device that reads it, from its own: a crossing."
struct Copied end

"""
A value made again on each other device that reads it, from what it reads
there, by its own definition, so that each device computes it by the same
arithmetic in the same order.
"""
struct Remade end

"""
An elementwise value made in pieces: the range `cuts[q]:(cuts[q + 1] - 1)` of
its last axis made on device `owners[q]`, by the value's own definition over
the range, from the ranges of what it reads; each piece copied to each other
device that reads the value, which reads the pieces put together, a select
of each by the block of positions it holds. Each element is computed on one
device, so that every device holds the same bits by construction, and each
device's link carries its share of the terms and of the pieces rather than
every term.
"""
struct Pieced
    cuts::Vector{Int}
    owners::Vector{Int}
end

const Realization = Union{Copied, Remade, Pieced}

# A placed graph being realized: the graph, the devices the cut gave its
# values, the realization of each, and the values realizing them made, each
# once by its key: what each value is on each device that reads it other
# than as its own device's value, each piece of a value made in pieces and
# each piece copied to a device, and each value made over a range.
struct Realizing
    graph::IndexGraph
    devices::Vector{Int}
    realized::Vector{Realization}
    # Whether each value is elementwise with terms on several devices: made
    # over a range wherever a value made over the range reads it.
    chained::BitVector
    made::Dict{Any, Int}
end

Realizing(graph, devices, realized, chained) = Realizing(graph, devices, realized, chained, Dict{Any, Int}())

# The value `key` names, made by `f()` the first time it is asked for.
function made!(f, p::Realizing, key)
    haskey(p.made, key) && return p.made[key]
    p.made[key] = f()
end

"""
    across!(p, b, r, here) -> id

What value `r` is on device `here` in the graph `b` builds, other than as
its own device's value is: made there once, as `r` is realized.
"""
across!(p::Realizing, b, r, here) = made!(() -> realize!(p.realized[r], p, b, r, here), p, (:there, r, here))

function realize!(::Copied, p::Realizing, b, r, here)
    source = p.graph.values[r]
    add!(b, Crossed(Read(b.ids[r], identity_index(source.dims))), source.dims, source.eltype, source.scalar)
end

function realize!(::Remade, p::Realizing, b, r, here)
    source = p.graph.values[r]
    add!(b, remap(source.definition, rerouted!(p, b, operands(source.definition), here)), source.dims, source.eltype, source.scalar)
end

# A value made in pieces, on a device that reads it: the pieces put together,
# each the one made there or a crossing of it, selected by the positions of
# the last axis it holds, from the last piece back, as a concatenation is.
# Each read of a piece wraps around its range, so that it stays inside the
# piece where the piece is not selected.
function realize!(pieced::Pieced, p::Realizing, b, r, here)
    val = p.graph.values[r]
    d = Domain(val.dims)
    axis = length(val.dims)
    whole = identity_index(val.dims)
    function placed(q)
        piece = piece!(p, b, r, pieced, q)
        dims = b.values[piece].dims
        held = pieced.owners[q] == here ? piece :
            made!(() -> add!(b, Crossed(Read(piece, identity_index(dims))), dims, val.eltype), p, (:copied_piece, r, q, here))
        range = piece_range(pieced, q)
        kept = ntuple(var, axis - 1)
        Read(held, length(range) == 1 ? kept : (kept..., fmod(add(var(axis), -first(range), d), length(range), d)))
    end
    selected = placed(length(pieced.owners))
    for q in (length(pieced.owners) - 1):-1:1
        mask = add!(b, Pointwise(<=, (IndexOperand(var(axis), Int32), Literal(Int32(last(piece_range(pieced, q)))))), val.dims, Bool)
        selected = Read(add!(b, Pointwise(ifelse, (Read(mask, whole), placed(q), selected)), val.dims, val.eltype), whole)
    end
    selected.value
end

"""
    home!(realization, p, b, id) -> id or nothing

What value `id` is on its own device in the graph `b` builds: its own
definition, reading what each value it reads is there, or nothing where it
reads nothing of another device; made in pieces, the pieces put together
there, as on every device that reads it.
"""
function home!(::Union{Copied, Remade}, p::Realizing, b, id)
    v = p.graph.values[id]
    here = p.devices[id]
    all(r -> p.devices[r] == here, (r for o in operands(v.definition) for r in read_values(o))) && return nothing
    add!(b, remap(v.definition, rerouted!(p, b, operands(v.definition), here)), v.dims, v.eltype, v.scalar)
end

home!(::Pieced, p::Realizing, b, id) = across!(p, b, id, p.devices[id])

# The new ids of the values operands `os` read, read on device `here`: each
# value on another device as it is there.
function rerouted!(p::Realizing, b, os, here)
    through = Dict{Int, Int}()
    for o in os, u in read_values(o)
        p.devices[u] == here || (through[u] = across!(p, b, u, here))
    end
    Rerouted(b.ids, through)
end

piece_range(pieced::Pieced, q) = pieced.cuts[q]:(pieced.cuts[q + 1] - 1)

# Piece `q` of value `r`, made in pieces, on its owner: the value's definition
# over the piece's range, from the ranges of what it reads there, made once,
# a value of the backward where `r` is one.
function piece!(p::Realizing, b, r, pieced::Pieced, q)
    made!(p, (:piece, r, q)) do
        dims, at = range_positions(p.graph.values[r].dims, piece_range(pieced, q))
        on(b, pieced.owners[q], p.graph.backward[r]) do
            restricted!(p, b, r, pieced.owners[q], at, Domain(dims))
        end
    end
end

# `f()` with the values added on `device`, of the backward where `backward`.
function on(f, b, device, backward)
    before = (b.on_device[], b.in_backward[])
    b.on_device[], b.in_backward[] = device, backward
    try
        f()
    finally
        b.on_device[], b.in_backward[] = before
    end
end

# Elementwise value `u` on device `here` at `index` over `d`: its definition
# with each operand read at its own index composed with `index`, a value
# over `d`, made once for each device and index.
function restricted!(p::Realizing, b, u, here, index, d::Domain)
    made!(p, (:over, u, here, index, d.extents)) do
        val = p.graph.values[u]
        def = val.definition::Pointwise
        add!(b, Pointwise(def.f, map(x -> over!(p, b, composed(x, index, d), here, d), def.operands)), d.extents, val.eltype)
    end
end

# An operand of a value made over a range, read over `d` on device `here`,
# its formulas the range's: a value made there over the same range, where the
# value it reads is made over a range wherever it is read so (see
# `over_range`), and otherwise the value as it is there, read at the
# formulas.
function over!(p::Realizing, b, x::Read, here, d::Domain)
    over_range(p, x.value) && return Read(restricted!(p, b, x.value, here, x.index, d), identity_index(d.extents))
    remap(x, rerouted!(p, b, (x,), here))
end

# Whether value `u`, read by a value made over a range, is made over the
# range too: where it is elementwise and made again, or its terms lie on
# several devices, as the partial sums of a sum made in pieces are, which
# each piece adds up over its own range from the ranges of the terms, however
# the cut placed them.
over_range(p::Realizing, u) = p.graph.values[u].definition isa Pointwise && (p.realized[u] isa Remade || p.chained[u])
over!(p::Realizing, b, x::Operand, here, d::Domain) = remap(x, rerouted!(p, b, (x,), here))

"""
    realizations(graph, devices, landing) -> Realizing

How each value of `graph`, placed on `devices`, its results pinned by
`landing` (see `result_devices`), gets to each other device that reads it,
the values before it first:

  - a value made from nothing another holds (see `made_anywhere`) is made
    again, where a copy would move it on every call;
  - an elementwise value no larger than any value it reads, read directly
    on another device, by a value there or a result pinned there, is
    realized as the candidate whose heaviest device and direction carries
    fewer bytes (see `link_load`), of equal ones the one fewer crossings
    deep, then the one of fewer crossings, then the first of copied, made
    again and made in pieces: a sum of what each of two devices computed,
    read on both, is made again on each from the other's term; over more
    devices it is made in pieces, each device's link carrying its share of
    the terms and of the pieces rather than every term; and a sum of two
    devices' terms read on its own device and on a third that holds neither
    is copied to the third once rather than its terms to it twice;
  - an elementwise value no other device reads directly is rebuilt with a
    remade reader where its terms lie on several devices and its own device
    reads it, as the sum of the first two of three terms is when the sum of
    all three is remade: no load of its own decides it, since what it is on
    another device is part of what its reader is there;
  - every other value is copied.

Its terms are what it is computed from through elementwise values, down to
values that are not, those made anywhere left out. A value is made in
pieces over the devices its terms lie on and the devices that read it, as
many pieces as those or as its last axis holds positions, whichever is
fewer, in equal ranges of its last axis but for rounding, so that each piece
is one span of its storage; a value of no axes never is. The pieces, their
crossings and what they read are values of the placed graph and not of
`graph`: none is realized again.
"""
function realizations(graph::IndexGraph, devices, landing)
    values = graph.values
    anywhere = made_anywhere(graph)
    reads = direct_reads(graph, devices, landing)
    realized = Vector{Realization}(undef, length(values))
    p = Realizing(graph, devices, realized, falses(length(values)))
    # The devices each value's terms lie on, and how many crossings deep it is
    # where it is made.
    terms = Vector{BitSet}(undef, length(values))
    depths = zeros(Int, length(values))
    for (v, val) in enumerate(values)
        depths[v] = operand_arrivals!(crossings(), p, depths, v, devices[v])
        if anywhere[v]
            terms[v] = BitSet()
            realized[v] = Remade()
        elseif elementwise(graph, v)
            terms[v] = mapreduce(r -> terms[r], union, (r for o in operands(val.definition) for r in read_values(o)); init = BitSet())
            p.chained[v] = length(terms[v]) > 1
            elsewhere = filter(d -> d != devices[v], keys(reads[v]))
            if isempty(elsewhere)
                realized[v] = rebuilt_with_reader(v, devices, terms, reads)
            else
                realized[v], depths[v] = lightest(p, depths, v, reads[v], union(terms[v], keys(reads[v])))
            end
        else
            terms[v] = BitSet(devices[v])
            realized[v] = Copied()
        end
    end
    p
end

# Whether value `v` is elementwise and no larger than any value it reads, so
# that it can be made again wherever what it reads is, or over a range.
function elementwise(graph, v)
    val = graph.values[v]
    val.definition isa Pointwise || return false
    all(r -> value_bytes(graph.values[r]) >= value_bytes(val), (r for o in operands(val.definition) for r in read_values(o)))
end

# An elementwise value no other device reads directly: remade where its terms
# lie on several devices and its own device reads it.
rebuilt_with_reader(v, devices, terms, reads) = length(terms[v]) > 1 && haskey(reads[v], devices[v]) ? Remade() : Copied()

# Of the candidates for value `v`, read directly on the devices `reads` keys
# with the ranges of its last axis there, its terms and readers on the
# devices `touched`, the one whose load is lightest, of equal ones the first;
# and how many crossings deep it is where it is made.
function lightest(p::Realizing, depths, v, reads, touched)
    candidates = Realization[Copied(), Remade()]
    pieced = pieces(p.graph.values[v], touched)
    pieced === nothing || push!(candidates, pieced)
    loads = [link_load(c, p, depths, v, reads) for c in candidates]
    k = argmin(k -> (heaviest(loads[k]), loads[k].deep, loads[k].crossings), eachindex(loads))
    (candidates[k], made_depth(candidates[k], depths, v, loads[k]))
end

# How many crossings deep a value is where it is made: on its own device as
# it was found, and in pieces, on their owners, one fewer than the farthest
# reader holds it.
made_depth(::Union{Copied, Remade}, depths, v, load) = depths[v]
made_depth(::Pieced, depths, v, load) = load.deep - 1

# Value `val` in pieces over `devices`, by label: as many as there are
# devices or positions of its last axis, whichever is fewer, the first
# devices taken, in equal ranges but for rounding; nothing where that is one
# piece, or the value has no axes.
function pieces(val::Value, devices)
    isempty(val.dims) && return nothing
    k = min(length(devices), last(val.dims))
    k > 1 || return nothing
    Pieced(round.(Int, range(0, last(val.dims); length = k + 1)), sort!(collect(devices))[1:k])
end

"""
What a candidate realization of a value moves between devices: the bytes
over each device's link in each direction, `(device, :out)` and `(device,
:in)`, how many crossings deep the farthest reader holds the value, and how
many crossings it makes.
"""
struct LinkLoad
    bytes::Dict{Tuple{Int, Symbol}, Int}
    deep::Int
    crossings::Int
end

heaviest(l::LinkLoad) = maximum(values(l.bytes); init = 0)

# The crossings a candidate makes, each of a value from a device to another,
# with the range of the value's last axis it copies.
crossings() = Dict{Tuple{Int, Int, Int}, UnitRange{Int}}()

# The load of the crossings `crossed` holds, `deep` crossings deep.
function load_of(p::Realizing, crossed, deep)
    bytes = Dict{Tuple{Int, Symbol}, Int}()
    for ((u, from, to), range) in crossed
        b = span_bytes(p.graph.values[u], range)
        bytes[(from, :out)] = get(bytes, (from, :out), 0) + b
        bytes[(to, :in)] = get(bytes, (to, :in), 0) + b
    end
    LinkLoad(bytes, deep, length(crossed))
end

"""
    link_load(realization, p, depths, v, reads) -> LinkLoad

What realizing value `v` that way moves, read directly on the devices
`reads` keys, the ranges of its last axis read there: copied, it is
computed on its own device from what it reads there and crossed to each
other device, the range read there; made again, it is computed on each
other device from what it reads there, and on its own where its own device
reads it; made in pieces, each piece is computed on its owner from the
ranges of what it reads there and crossed whole to each other device that
reads the value. What a value reads on a device it is not on arrives as
that value is realized (see `arrive!`). The ranges are the reads' own, as
the crossings will be narrowed to them (see `narrowed_crossings`).
"""
function link_load(::Copied, p::Realizing, depths, v, reads)
    crossed = crossings()
    home = p.devices[v]
    operand_arrivals!(crossed, p, depths, v, home)
    for d in keys(reads)
        d == home || reach!(crossed, (v, home, d), reads[d])
    end
    load_of(p, crossed, depths[v] + 1)
end

function link_load(::Remade, p::Realizing, depths, v, reads)
    crossed = crossings()
    deep = maximum(d -> operand_arrivals!(crossed, p, depths, v, d), keys(reads))
    load_of(p, crossed, deep)
end

function link_load(pieced::Pieced, p::Realizing, depths, v, reads)
    crossed = crossings()
    deep = 0
    for (q, owner) in enumerate(pieced.owners)
        dims, at = range_positions(p.graph.values[v].dims, piece_range(pieced, q))
        made = over_arrivals!(crossed, p, depths, v, owner, at, Domain(dims))
        for d in keys(reads)
            d == owner || reach!(crossed, (v, owner, d), piece_range(pieced, q))
        end
        deep = max(deep, made + 1)
    end
    load_of(p, crossed, deep)
end

# The crossings value `v` made on device `here` makes to read what it reads
# there, into `crossed`; how many crossings deep it is there.
function operand_arrivals!(crossed, p::Realizing, depths, v, here)
    val = p.graph.values[v]
    maximum((arrive!(crossed, p, depths, u, here, o, d) for (o, d) in zip(operands(val.definition), operand_domains(val))
        for u in read_values(o)); init = 0)
end

# The same of elementwise value `v` read at `index` over `d`: of its operands
# read at their indices composed with `index`.
function range_arrivals!(crossed, p::Realizing, depths, v, here, index, d)
    val = p.graph.values[v]
    maximum((arrive!(crossed, p, depths, u, here, composed(o, index, d), d) for o in operands(val.definition)
        for u in read_values(o)); init = 0)
end

# The same of elementwise value `v` made over a range, `index` over `d`:
# what it reads made over the range in turn, where it is (see `over_range`).
function over_arrivals!(crossed, p::Realizing, depths, v, here, index, d)
    val = p.graph.values[v]
    maximum((over_arrive!(crossed, p, depths, u, here, composed(o, index, d), d) for o in operands(val.definition)
        for u in read_values(o)); init = 0)
end

over_arrive!(crossed, p::Realizing, depths, u, here, o, d) = over_range(p, u) && o isa Read && o.value == u ?
    over_arrivals!(crossed, p, depths, u, here, o.index, d) : arrive!(crossed, p, depths, u, here, o, d)

"""
    arrive!(crossed, p, depths, u, here, o, d) -> crossings deep

The crossings a read `o` over domain `d` on device `here` makes of value
`u`, which it reads at its own index or gathers through, into `crossed`;
how many crossings deep `u` is there. A value made again is read as what it
is made from, on `here` or on its own device alike, at the read's index
composed with its own reads': what crosses for it is what crosses for its
terms, which the candidates for a value made from it differ by. A value
made in pieces is read as its pieces put together, each piece made
elsewhere crossing whole. Any other value is read in place on its own
device, and copied, elsewhere, crosses the range of its last axis the read
reaches, the whole where the read gathers through it.
"""
arrive!(crossed, p::Realizing, depths, u, here, o, d) = arrive!(p.realized[u], crossed, p, depths, u, here, o, d)

function arrive!(::Copied, crossed, p::Realizing, depths, u, here, o, d)
    p.devices[u] == here && return depths[u]
    reach!(crossed, (u, p.devices[u], here), read_range(p.graph, u, o, d))
    depths[u] + 1
end

function arrive!(::Remade, crossed, p::Realizing, depths, u, here, o, d)
    p.graph.values[u].definition isa Pointwise && o isa Read && o.value == u || return operand_arrivals!(crossed, p, depths, u, here)
    range_arrivals!(crossed, p, depths, u, here, o.index, d)
end

function arrive!(pieced::Pieced, crossed, p::Realizing, depths, u, here, o, d)
    for (q, owner) in enumerate(pieced.owners)
        owner == here || reach!(crossed, (u, owner, here), piece_range(pieced, q))
    end
    depths[u] + any(!=(here), pieced.owners)
end

# Operand `x` of an elementwise value read at `index` over `d`: read over `d`.
composed(x::Read, index, d) = Read(x.value, map(e -> substitute(e, index, d), x.index))
composed(x::IndexOperand{T}, index, d) where {T} = IndexOperand{T}(substitute(x.expr, index, d))
composed(x::Literal, index, d) = x

# The range of `u`'s last axis a read `o` over `d` reaches: of its index, where
# `o` reads `u`, and all of it where `o` gathers through `u`.
read_range(graph, u, o::Read, d) = o.value == u ? read_range(graph.values[u], o, d) : whole_range(graph.values[u])
read_range(graph, u, o::Operand, d) = whole_range(graph.values[u])

# A crossing, as `key` names it, of `range` of a value's last axis, into
# `crossed`: one per key, of the ranges united.
reach!(crossed, key, range) = (crossed[key] = haskey(crossed, key) ? united(crossed[key], range) : range)

united(a, b) = min(first(a), first(b)):max(last(a), last(b))

# The devices that read each value directly, by a value there or a result
# pinned there, each with the range of the value's last axis read there; a
# result pinned to no device reads it on its own.
function direct_reads(graph::IndexGraph, devices, landing)
    reads = [Dict{Int, UnitRange{Int}}() for _ in graph.values]
    function reached!(o, d, device)
        for r in read_values(o)
            reach!(reads[r], device, read_range(graph, r, o, d))
        end
    end
    for (u, val) in enumerate(graph.values)
        foreach((o, d) -> reached!(o, d, devices[u]), operands(val.definition), operand_domains(val))
        val.definition isa Carried && foreach(r -> reach!(reads[r], devices[u], whole_range(graph.values[r])), read_values(val.definition.from))
    end
    for (k, o) in enumerate(graph.results), r in read_values(o)
        reach!(reads[r], landing[k] == 0 ? devices[r] : landing[k], read_range(graph, r, o, Domain(graph.result_dims[k])))
    end
    reads
end
