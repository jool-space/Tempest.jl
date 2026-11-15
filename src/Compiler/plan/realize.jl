# How a value read on another device than its own gets there, chosen for each
# value once the cut has placed it (see `realizations`): copied there, a
# crossing of it, or made again there from what it reads there. A
# realization is a type with two methods: `link_load`, what it moves over
# each device's link in each direction, of which the lighter wins, and
# `realize!`, what it adds to the placed graph on a device that reads the
# value. Another is added here, with its methods, and to the union.

"A value copied to each other device that reads it, from its own: a crossing."
struct Copied end

"""
A value made again on each other device that reads it, from what it reads
there, by its own definition, so that each device computes it by the same
arithmetic in the same order.
"""
struct Remade end

const Realization = Union{Copied, Remade}

# A placed graph being realized: the graph, the devices the cut gave its
# values, the realization of each, and what each value is on each other
# device that reads it, made once there.
struct Realizing
    graph::IndexGraph
    devices::Vector{Int}
    realized::Vector{Realization}
    there::Dict{Tuple{Int, Int}, Int}
end

Realizing(graph, devices, realized) = Realizing(graph, devices, realized, Dict{Tuple{Int, Int}, Int}())

"What value `r` is on device `here`, not its own, in the graph `b` builds: made there once, as `r` is realized."
function across!(p::Realizing, b, r, here)
    haskey(p.there, (r, here)) && return p.there[(r, here)]
    p.there[(r, here)] = realize!(p.realized[r], p, b, r, here)
end

function realize!(::Copied, p::Realizing, b, r, here)
    source = p.graph.values[r]
    add!(b, Crossed(Read(b.ids[r], identity_index(source.dims))), source.dims, source.eltype, source.scalar)
end

function realize!(::Remade, p::Realizing, b, r, here)
    source = p.graph.values[r]
    through = Dict{Int, Int}()
    for o in operands(source.definition), u in read_values(o)
        p.devices[u] == here || (through[u] = across!(p, b, u, here))
    end
    add!(b, remap(source.definition, Rerouted(b.ids, through)), source.dims, source.eltype, source.scalar)
end

"""
    realizations(graph, devices, landing) -> Vector{Realization}

How each value of `graph`, placed on `devices`, its results pinned by
`landing` (see `result_devices`), gets to each other device that reads it,
the values before it first:

  - a value made from nothing another holds (see `made_anywhere`) is made
    again, where a copy would move it on every call;
  - an elementwise value no larger than any value it reads, read directly
    on another device, by a value there or a result pinned there, is
    realized as the candidate whose heaviest device and direction carries
    fewer bytes (see `link_load`), of equal ones the one fewer crossings
    deep, then the one of fewer crossings, then copied: a sum of what each
    device computed, read on each, is made again on each from the others'
    terms, and a sum of two devices' terms read on its own device and on a
    third that holds none is copied to the third once rather than its terms
    to it twice;
  - an elementwise value no other device reads directly is rebuilt with a
    remade reader where its terms lie on several devices and its own device
    reads it, as the sum of the first two of three terms is when the sum of
    all three is remade: no load of its own decides it, since what it is on
    another device is part of what its reader is there;
  - every other value is copied.

Its terms are what it is computed from through elementwise values, down to
values that are not, those made anywhere left out.
"""
function realizations(graph::IndexGraph, devices, landing)
    values = graph.values
    anywhere = made_anywhere(graph)
    reads = direct_reads(graph, devices, landing)
    realized = Vector{Realization}(undef, length(values))
    p = Realizing(graph, devices, realized)
    # The devices each value's terms lie on, and how many crossings deep it is
    # on its own device.
    terms = Vector{BitSet}(undef, length(values))
    depths = zeros(Int, length(values))
    for (v, val) in enumerate(values)
        depths[v] = operand_arrivals!(Dict{Tuple{Int, Int}, UnitRange{Int}}(), p, depths, v, devices[v])
        if anywhere[v]
            terms[v] = BitSet()
            realized[v] = Remade()
        elseif elementwise(graph, v)
            terms[v] = mapreduce(r -> terms[r], union, (r for o in operands(val.definition) for r in read_values(o)); init = BitSet())
            elsewhere = filter(d -> d != devices[v], keys(reads[v]))
            realized[v] = isempty(elsewhere) ? rebuilt_with_reader(v, devices, terms, reads) : lighter(p, depths, v, reads[v])
        else
            terms[v] = BitSet(devices[v])
            realized[v] = Copied()
        end
    end
    realized
end

# Whether value `v` is elementwise and no larger than any value it reads, so
# that it can be made again wherever what it reads is.
function elementwise(graph, v)
    val = graph.values[v]
    val.definition isa Pointwise || return false
    all(r -> value_bytes(graph.values[r]) >= value_bytes(val), (r for o in operands(val.definition) for r in read_values(o)))
end

# An elementwise value no other device reads directly: remade where its terms
# lie on several devices and its own device reads it.
rebuilt_with_reader(v, devices, terms, reads) = length(terms[v]) > 1 && haskey(reads[v], devices[v]) ? Remade() : Copied()

# Of the candidates for value `v`, read directly on the devices `reads` keys
# with the ranges of its last axis there, the one whose load is lighter, of
# equal ones copied.
function lighter(p::Realizing, depths, v, reads)
    copied, remade = link_load(Copied(), p, depths, v, reads), link_load(Remade(), p, depths, v, reads)
    lighter_load(remade, copied) ? Remade() : Copied()
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

# Whether load `a` is lighter than `b`: its heaviest device and direction,
# then its depth, then its crossings.
lighter_load(a::LinkLoad, b::LinkLoad) = (heaviest(a), a.deep, a.crossings) < (heaviest(b), b.deep, b.crossings)

# The load of the crossings `crossed` holds, each of a value to a device, of
# the range of the value's last axis it copies, `deep` crossings deep.
function load_of(p::Realizing, crossed, deep)
    bytes = Dict{Tuple{Int, Symbol}, Int}()
    for ((u, to), range) in crossed
        b = span_bytes(p.graph.values[u], range)
        from = p.devices[u]
        bytes[(from, :out)] = get(bytes, (from, :out), 0) + b
        bytes[(to, :in)] = get(bytes, (to, :in), 0) + b
    end
    LinkLoad(bytes, deep, length(crossed))
end

"""
    link_load(realization, p, depths, v, reads) -> LinkLoad

What realizing value `v` that way moves, read directly on the devices
`reads` keys, the ranges of its last axis read there: copied, it is computed on its
own device from what it reads there and crossed to each other device, the
range read there; made again, it is computed on each other device from
what it reads there, and on its own where its own device reads it. What a
value reads on a device it is not on arrives as that value is realized: a
crossing of the range it reaches, or what that value reads in turn (see
`arrive!`). The ranges are the reads' own, as the crossings will be
narrowed to them (see `narrowed_crossings`).
"""
function link_load(::Copied, p::Realizing, depths, v, reads)
    crossed = Dict{Tuple{Int, Int}, UnitRange{Int}}()
    home = p.devices[v]
    operand_arrivals!(crossed, p, depths, v, home)
    elsewhere = [d for d in keys(reads) if d != home]
    for d in elsewhere
        reach!(crossed, (v, d), reads[d])
    end
    load_of(p, crossed, depths[v] + 1)
end

function link_load(::Remade, p::Realizing, depths, v, reads)
    crossed = Dict{Tuple{Int, Int}, UnitRange{Int}}()
    deep = maximum(d -> operand_arrivals!(crossed, p, depths, v, d), keys(reads))
    load_of(p, crossed, deep)
end

# The crossings value `v` made on device `here` makes to read what it reads
# there, into `crossed`; how many crossings deep it is there.
function operand_arrivals!(crossed, p::Realizing, depths, v, here)
    val = p.graph.values[v]
    maximum((arrive!(crossed, p, depths, u, here, o, d) for (o, d) in zip(operands(val.definition), operand_domains(val))
        for u in read_values(o)); init = 0)
end

# The crossings a read `o` over domain `d` on device `here` makes of value
# `u`, which it reads at its own index or gathers through, into `crossed`;
# how many crossings deep `u` is there. A value made again is read as what it
# is made from, on `here` or on its own device alike, at the read's index
# composed with its own reads': what crosses for it is what crosses for its
# terms, which the candidates for a value made from it differ by. Any other
# value is read in place on its own device, and copied, elsewhere, crosses
# the range of its last axis the read reaches, the whole where the read
# gathers through it.
function arrive!(crossed, p::Realizing, depths, u, here, o, d)
    remade = p.realized[u] isa Remade
    p.devices[u] == here && !remade && return depths[u]
    remade || return (reach!(crossed, (u, here), read_range(p.graph, u, o, d)); depths[u] + 1)
    val = p.graph.values[u]
    at = o isa Read && o.value == u ? o.index : identity_index(val.dims)
    domain = o isa Read && o.value == u ? d : Domain(val.dims)
    maximum((arrive!(crossed, p, depths, w, here, composed(x, at, domain), domain)
        for x in operands(val.definition) for w in read_values(x)); init = 0)
end

# Operand `x` of an elementwise value read at `index` over `d`: read over `d`.
composed(x::Read, index, d) = Read(x.value, map(e -> substitute(e, index, d), x.index))
composed(x::IndexOperand{T}, index, d) where {T} = IndexOperand{T}(substitute(x.expr, index, d))
composed(x::Literal, index, d) = x

# The range of `u`'s last axis a read `o` over `d` reaches: of its index, where
# `o` reads `u`, and all of it where `o` gathers through `u`.
read_range(graph, u, o::Read, d) = o.value == u ? read_range(graph.values[u], o, d) : whole_range(graph.values[u])
read_range(graph, u, o::Operand, d) = whole_range(graph.values[u])

# A crossing of value `u` to a device, as `key` names it, of `range` of its
# last axis, into `crossed`: one per value and device, of the ranges united.
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
