# How a value read on another device than its own gets there, chosen once the
# cut has placed the values (see `realizations`). The unit is a chain: an
# elementwise value read directly on another device, its frontier, with the
# elementwise values through which what it is computed from reaches it, its
# interior (see `Chain`). A chain is realized as a whole: copied, each of
# its values computed where the cut placed it and copied where another
# device reads it; made again, computed on each device that reads its
# frontier; or made in pieces, a range of the frontier's last axis computed
# on each of several devices, each piece copied to the others, which put the
# pieces together. Wherever a chain is made again or in pieces, over the
# whole or a range, its values are made there from what they read outside
# it, its terms, as the terms are realized there. A realization is a type with two methods: `link_load`,
# what it moves over each device's link in each direction, of which the
# lighter wins, and `realize!`, what it adds to the placed graph on a device
# that reads the frontier. Another is added here, with its methods, and to
# the union.

"""
A chain computed where the cut placed its values, each value read on
another device copied there, a crossing: its frontier to each device that
reads it, and a value of its interior to the device of the value of the
chain that reads it. A value of no chain is copied too.
"""
struct Copied end

"""
A chain made again on each device that reads its frontier, from what it
reads there, by its own definitions, so that each device computes it by the
same arithmetic in the same order.
"""
struct Remade end

"""
A chain made in pieces: the range `cuts[q]:(cuts[q + 1] - 1)` of its
frontier's last axis made on device `owners[q]`, each value of the chain
over the positions the range reads of it, from the ranges of its terms
there; each piece copied to each other device that reads the frontier,
which reads the pieces put together, a select of each by the block of
positions it holds. Each element is computed on one device, so that every
device holds the same bits by construction, and each device's link carries
its share of the terms and of the pieces rather than every term.
"""
struct Pieced
    cuts::Vector{Int}
    owners::Vector{Int}
end

const Realization = Union{Copied, Remade, Pieced}

"""
A chain of the placed graph: its frontier, an elementwise value read
directly on another device than its own or read by another chain, and its
interior, the elementwise values whose terms lie on several devices through
which the frontier's terms reach it, each read by the values of the chain
alone, each at an index, and otherwise only on its own device. `readers`
holds the devices that read the frontier directly, each with the range of
its last axis read there, `terms` the devices its terms lie on, and
`values` the values of the chain, in the graph's order.
"""
struct Chain
    frontier::Int
    readers::Dict{Int, UnitRange{Int}}
    terms::BitSet
    values::Vector{Int}
end

# The index of a read and the extents of the domain it is read over.
const Positions = Tuple{Tuple{Vararg{Affine}}, Tuple{Vararg{Int}}}

# A placed graph being realized: the graph, the devices the cut gave its
# values, which are made from nothing another holds, the chains, the chain
# each value is of (zero for none), which values of a chain are read on
# their own device by values of none or by results, and what each value of
# a chain reads outside it, each operand at the formulas over the value's own
# domain its reads compose to; the realization of each chain and how many
# crossings deep each value is where it is made; and the values realizing
# them, each made once: what each value is on each device that reads it,
# each value of a chain made at the positions of a read on a device, and each
# piece's copy on each device.
struct Realizing
    graph::IndexGraph
    devices::Vector{Int}
    anywhere::BitVector
    chains::Vector{Chain}
    chain_of::Vector{Int}
    homed::BitVector
    reads::Vector{Vector{Operand}}
    realized::Vector{Realization}
    depths::Vector{Int}
    there::Dict{Tuple{Int, Int}, Int}
    over::Dict{Tuple{Int, Int, Positions}, Int}
    copies::Dict{Tuple{Int, Int}, Int}
end

# The value `key` names in `memo`, made by `f()` the first time it is asked for.
function made!(f, memo, key)
    haskey(memo, key) && return memo[key]
    memo[key] = f()
end

frontier(p::Realizing, u) = p.chain_of[u] != 0 && p.chains[p.chain_of[u]].frontier == u

# Whether value `u` is of its chain's interior, as value `v` of a chain is.
interior(p::Realizing, u, v) = p.chain_of[u] == p.chain_of[v] && !frontier(p, u)

# Whether value `u` is computed where the cut placed it: of no chain, or of a
# chain copied.
placed(p::Realizing, u) = p.chain_of[u] == 0 || p.realized[p.chain_of[u]] isa Copied

"""
    value_on!(p, b, u, here) -> id

What value `u` is on device `here` in the graph `b` builds: a value
computed where it was placed, itself on its own device; a value made
anywhere, made again there by its own definition; a chain's frontier, as
the chain is realized; a value of the interior of a chain made again or in
pieces, made there whole as the chain is; any other value copied, a
crossing. Made once for each value and device.
"""
function value_on!(p::Realizing, b, u, here)
    val = p.graph.values[u]
    placed(p, u) || frontier(p, u) || return made_over!(p, b, u, here, identity_index(val.dims), Domain(val.dims))
    placed(p, u) && p.devices[u] == here && return b.ids[u]
    made!(p.there, (u, here)) do
        p.anywhere[u] && return add!(b, remap(val.definition, rerouted!(p, b, operands(val.definition), here)), val.dims, val.eltype, val.scalar)
        frontier(p, u) && return realize!(p.realized[p.chain_of[u]], p, b, p.chains[p.chain_of[u]], here)
        crossed!(b, b.ids[u], val)
    end
end

# A crossing of new value `id`, of the extents and type of `val`.
crossed!(b, id, val::Value) = add!(b, Crossed(Read(id, identity_index(val.dims))), val.dims, val.eltype, val.scalar)

# The new ids of the values operands `os` read, read on device `here`: each
# value not computed there where it was placed as it is there (see
# `value_on!`).
function rerouted!(p::Realizing, b, os, here)
    through = Dict{Int, Int}()
    for o in os, u in read_values(o)
        placed(p, u) && p.devices[u] == here || (through[u] = value_on!(p, b, u, here))
    end
    Rerouted(b.ids, through)
end

# A chain copied, on a device that reads its frontier other than its own.
realize!(::Copied, p::Realizing, b, c::Chain, here) = crossed!(b, b.ids[c.frontier], p.graph.values[c.frontier])

function realize!(::Remade, p::Realizing, b, c::Chain, here)
    val = p.graph.values[c.frontier]
    made_over!(p, b, c.frontier, here, identity_index(val.dims), Domain(val.dims))
end

# A chain made in pieces, on a device that reads its frontier: the pieces put
# together, each the one made there or a crossing of it, selected by the
# positions of the last axis it holds, from the last piece back, as a
# concatenation is. Each read of a piece wraps around its range, so that it
# stays inside the piece where the piece is not selected.
function realize!(pieced::Pieced, p::Realizing, b, c::Chain, here)
    val = p.graph.values[c.frontier]
    d = Domain(val.dims)
    axis = length(val.dims)
    whole = identity_index(val.dims)
    function held(q)
        piece = piece!(p, b, c, pieced, q)
        there = pieced.owners[q] == here ? piece : made!(() -> crossed!(b, piece, b.values[piece]), p.copies, (piece, here))
        range = piece_range(pieced, q)
        kept = ntuple(var, axis - 1)
        Read(there, length(range) == 1 ? kept : (kept..., fmod(add(var(axis), -first(range), d), length(range), d)))
    end
    selected = held(length(pieced.owners))
    for q in (length(pieced.owners) - 1):-1:1
        mask = add!(b, Pointwise(<=, (IndexOperand(var(axis), Int32), Literal(Int32(last(piece_range(pieced, q)))))), val.dims, Bool)
        selected = Read(add!(b, Pointwise(ifelse, (Read(mask, whole), held(q), selected)), val.dims, val.eltype), whole)
    end
    selected.value
end

piece_range(pieced::Pieced, q) = pieced.cuts[q]:(pieced.cuts[q + 1] - 1)

# Piece `q` of a chain made in pieces, on its owner: the frontier over the
# piece's range, a value of the backward where the frontier is one.
function piece!(p::Realizing, b, c::Chain, pieced::Pieced, q)
    dims, at = range_positions(p.graph.values[c.frontier].dims, piece_range(pieced, q))
    on(b, pieced.owners[q], p.graph.backward[c.frontier]) do
        made_over!(p, b, c.frontier, pieced.owners[q], at, Domain(dims))
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

"""
    made_over!(p, b, u, here, index, d) -> id

Value `u` of a chain made on device `here` at `index` over `d`: its
definition with each operand read at its own index composed with `index`, a
value over `d`; a value of the chain's interior it reads made there at the
composed index in turn, and anything else read as it is there (see
`value_on!`). Made once for each device and index.
"""
function made_over!(p::Realizing, b, u, here, index, d::Domain)
    made!(p.over, (u, here, (index, d.extents))) do
        val = p.graph.values[u]
        def = val.definition::Pointwise
        on(b, here, b.in_backward[]) do
            add!(b, Pointwise(def.f, map(x -> made_operand!(p, b, u, composed(x, index, d), here, d), def.operands)), d.extents, val.eltype, val.scalar)
        end
    end
end

# An operand of value `u` of a chain made over `d` on device `here`, its
# formulas composed: a read of a value of the chain's interior, that value
# made there at the read's formulas; anything else read as it is there.
function made_operand!(p::Realizing, b, u, x::Read, here, d::Domain)
    interior(p, x.value, u) || return remap(x, rerouted!(p, b, (x,), here))
    Read(made_over!(p, b, x.value, here, x.index, d), identity_index(d.extents))
end

made_operand!(p::Realizing, b, u, x::Operand, here, d::Domain) = remap(x, rerouted!(p, b, (x,), here))

"""
    home!(p, b, id) -> id, nothing or zero

What value `id` is on its own device in the graph `b` builds, at its place
in the graph's order: a value computed where it was placed by its own
definition, reading what each value it reads is there, or nothing where it
reads only such values on its own device; a value of a chain made again or
in pieces made there as the chain is, and a frontier made in pieces that
nothing reads there its pieces, on their owners, and nothing of its own
there (zero).
"""
function home!(p::Realizing, b, id)
    here = p.devices[id]
    c = p.chain_of[id]
    placed(p, id) || p.homed[id] && return value_on!(p, b, id, here)
    placed(p, id) || return home!(p.realized[c], p, b, p.chains[c])
    v = p.graph.values[id]
    all(u -> placed(p, u) && p.devices[u] == here, (u for o in operands(v.definition) for u in read_values(o))) && return nothing
    add!(b, remap(v.definition, rerouted!(p, b, operands(v.definition), here)), v.dims, v.eltype, v.scalar)
end

# A chain's frontier made in pieces that nothing reads on its own device, at
# its place: its pieces made on their owners.
function home!(pieced::Pieced, p::Realizing, b, c::Chain)
    foreach(q -> piece!(p, b, c, pieced, q), length(pieced.owners):-1:1)
    0
end

"""
    made_at_home(p) -> BitVector

Whether each value has a place of its own in the graph's order (see
`home!`): every value computed where it was placed; a value of a chain made
again or in pieces where a value of no chain or a result reads it on its
own device; and a frontier made in pieces, the pieces made there. Wherever
else a chain's values are read, they are made as the chain is realized
there.
"""
made_at_home(p::Realizing) = BitVector(placed(p, u) || p.homed[u] || frontier(p, u) && p.realized[p.chain_of[u]] isa Pieced
    for u in eachindex(p.graph.values))

"""
    realizations(graph, devices, landing) -> Realizing

How each value of `graph`, placed on `devices`, its results pinned by
`landing` (see `result_devices`), gets to each other device that reads it:

  - a value made from nothing another holds (see `made_anywhere`) is made
    again, where a copy would move it on every call;
  - a chain (see `Chain`), the chains it reads first, is realized as the
    candidate whose heaviest device and direction carries fewer bytes (see
    `link_load`), of equal ones the one fewer crossings deep, then the one
    of fewer crossings, then the first of copied, made again and made in
    pieces: a sum of what each of two devices computed, read on both, is
    made again on each from the other's term; over more devices it is made
    in pieces, each device's link carrying its share of the terms and of the
    pieces rather than every term; and a sum of two devices' terms read on
    its own device and on a third that holds neither is copied to the third
    once rather than its terms to it twice;
  - every other value is copied.

A chain is made in pieces over the devices its terms lie on and the devices
that read it, as many pieces as those or as its last axis holds positions,
whichever is fewer, in equal ranges of its last axis but for rounding, so
that each piece is one span of its storage; a value of no axes never is.
"""
function realizations(graph::IndexGraph, devices, landing)
    values = graph.values
    anywhere = made_anywhere(graph)
    chains, chain_of, homed = chains_of(graph, devices, landing, anywhere)
    p = Realizing(graph, devices, anywhere, chains, chain_of, homed, [Operand[] for _ in values], Vector{Realization}(undef, length(chains)),
        zeros(Int, length(values)), Dict{Tuple{Int, Int}, Int}(), Dict{Tuple{Int, Int, Positions}, Int}(), Dict{Tuple{Int, Int}, Int}())
    for (v, val) in enumerate(values)
        c = chain_of[v]
        c == 0 || (p.reads[v] = outside_reads(p, v))
        anywhere[v] && continue
        p.depths[v] = operand_arrivals!(crossings(), p, v, devices[v])
        frontier(p, v) && ((p.realized[c], p.depths[v]) = lightest(p, chains[c]))
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

"""
    chains_of(graph, devices, landing, anywhere) -> (chains, chain_of, homed)

The chains of the placed graph, the readers of each value before it: an
elementwise value whose terms lie on several devices, read by the values of
one chain, each at an index, and otherwise only on its own device, is of
that chain's interior; any other elementwise value read directly on another
device, or by a chain's values, is the frontier of a chain of its own, which
those values read as a term. A value's terms are what it is computed from
through elementwise values, down to values that are not, those made
anywhere left out. And whether each value of a chain is read on its own
device by a value of none or a result.
"""
function chains_of(graph::IndexGraph, devices, landing, anywhere)
    values = graph.values
    terms = Vector{BitSet}(undef, length(values))
    for (v, val) in enumerate(values)
        if anywhere[v]
            terms[v] = BitSet()
        elseif elementwise(graph, v)
            terms[v] = mapreduce(r -> terms[r], union, (r for o in operands(val.definition) for r in read_values(o)); init = BitSet())
        else
            terms[v] = BitSet(devices[v])
        end
    end
    reads = direct_reads(graph, devices, landing)
    # Each value's readers, each with whether it reads the value at an index
    # alone; a state reads what it is carried from, and a result on the device
    # it lands on, a reader of no chain (zero).
    users = [Tuple{Int, Bool}[] for _ in values]
    for (u, val) in enumerate(values)
        for o in operands(val.definition), r in read_values(o)
            push!(users[r], (u, o isa Read && o.value == r))
        end
        val.definition isa Carried && foreach(r -> push!(users[r], (u, false)), read_values(val.definition.from))
    end
    results = [Int[] for _ in values]
    for (k, o) in enumerate(graph.results), r in read_values(o)
        push!(results[r], landing[k] == 0 ? devices[r] : landing[k])
    end
    chains = Chain[]
    chain_of = zeros(Int, length(values))
    homed = falses(length(values))
    for v in length(values):-1:1
        !anywhere[v] && elementwise(graph, v) || continue
        outer = unique!([chain_of[u] for (u, _) in users[v] if chain_of[u] != 0])
        # Read only at an index by values of chains, and otherwise on its own device.
        inside = all(((u, indexed),) -> chain_of[u] == 0 ? devices[u] == devices[v] : indexed, users[v]) && all(==(devices[v]), results[v])
        if length(terms[v]) > 1 && length(outer) == 1 && inside
            chain_of[v] = only(outer)
        elseif !isempty(outer) || any(!=(devices[v]), keys(reads[v]))
            push!(chains, Chain(v, reads[v], terms[v], Int[]))
            chain_of[v] = length(chains)
        else
            continue
        end
        pushfirst!(chains[chain_of[v]].values, v)
        homed[v] = any(((u, _),) -> chain_of[u] == 0 && devices[u] == devices[v], users[v]) || devices[v] in results[v]
    end
    chains, chain_of, homed
end

# What value `v` of a chain reads outside it, each operand at the formulas
# over `v`'s domain its reads compose to: its own operands that read no value
# of the chain's interior, and what each value of the interior it reads reads
# outside it, composed with the read's formulas.
function outside_reads(p::Realizing, v)
    val = p.graph.values[v]
    d = Domain(val.dims)
    found = Operand[]
    for x in operands(val.definition)
        if x isa Read && interior(p, x.value, v)
            append!(found, (composed(o, x.index, d) for o in p.reads[x.value]))
        elseif !isempty(read_values(x))
            push!(found, x)
        end
    end
    unique!(found)
end

# Of the candidates for chain `c`, the one whose load is lightest, of equal
# ones the first; and how many crossings deep its frontier is where it is
# made.
function lightest(p::Realizing, c::Chain)
    candidates = Realization[Copied(), Remade()]
    pieced = pieces(p.graph.values[c.frontier], union(c.terms, keys(c.readers)))
    pieced === nothing || push!(candidates, pieced)
    loads = [link_load(r, p, c) for r in candidates]
    k = argmin(k -> (heaviest(loads[k]), loads[k].deep, loads[k].crossings), eachindex(loads))
    (candidates[k], made_depth(candidates[k], p, c, loads[k]))
end

# How many crossings deep a chain's frontier is where it is made: on its own
# device as the cut placed it, and in pieces, on their owners, one fewer than
# the farthest reader holds it.
made_depth(::Union{Copied, Remade}, p::Realizing, c::Chain, load) = p.depths[c.frontier]

made_depth(::Pieced, p::Realizing, c::Chain, load) = load.deep - 1

# A frontier of value `val` in pieces over `devices`, by label: as many as
# there are devices or positions of its last axis, whichever is fewer, the
# first devices taken, in equal ranges but for rounding; nothing where that
# is one piece, or the value has no axes.
function pieces(val::Value, devices)
    isempty(val.dims) && return nothing
    k = min(length(devices), last(val.dims))
    k > 1 || return nothing
    Pieced(round.(Int, range(0, last(val.dims); length = k + 1)), sort!(collect(devices))[1:k])
end

"""
What a candidate realization of a chain moves between devices: the bytes
over each device's link in each direction, `(device, :out)` and `(device,
:in)`, how many crossings deep the farthest reader holds the frontier, and
how many crossings it makes.
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
    link_load(realization, p, c) -> LinkLoad

What realizing chain `c` that way moves: copied, each value of the chain is
computed where the cut placed it, from what it reads there, and the
frontier crossed to each other device that reads it, the range read there;
made again, it is made on each device that reads
the frontier; made in pieces, each piece is made on its owner and crossed
whole to each other device that reads the frontier. Wherever the chain is
made, what crosses is what its values read outside it, at the ranges of the
terms the positions made reach (see `made_arrivals!`), as the crossings
will be narrowed to them (see `narrowed_crossings`).
"""
function link_load(::Copied, p::Realizing, c::Chain)
    crossed = crossings()
    foreach(u -> operand_arrivals!(crossed, p, u, p.devices[u]), c.values)
    home = p.devices[c.frontier]
    for d in keys(c.readers)
        d == home || reach!(crossed, (c.frontier, home, d), c.readers[d])
    end
    load_of(p, crossed, p.depths[c.frontier] + 1)
end

function link_load(::Remade, p::Realizing, c::Chain)
    crossed = crossings()
    val = p.graph.values[c.frontier]
    deep = maximum(d -> made_arrivals!(crossed, p, c.frontier, d, identity_index(val.dims), Domain(val.dims)), keys(c.readers))
    load_of(p, crossed, deep)
end

function link_load(pieced::Pieced, p::Realizing, c::Chain)
    crossed = crossings()
    deep = 0
    for (q, owner) in enumerate(pieced.owners)
        dims, at = range_positions(p.graph.values[c.frontier].dims, piece_range(pieced, q))
        made = made_arrivals!(crossed, p, c.frontier, owner, at, Domain(dims))
        for d in keys(c.readers)
            d == owner || reach!(crossed, (c.frontier, owner, d), piece_range(pieced, q))
        end
        deep = max(deep, made + 1)
    end
    load_of(p, crossed, deep)
end

# The crossings value `u` of a chain made on device `here` at `index` over
# `d` makes to read what it reads outside the chain there, into `crossed`;
# how many crossings deep it is there.
function made_arrivals!(crossed, p::Realizing, u, here, index, d)
    maximum((arrive!(crossed, p, w, here, composed(o, index, d), d) for o in p.reads[u] for w in read_values(o)); init = 0)
end

# The same of value `v` computed on device `here` from its operands, a value
# of a chain's interior it reads where the cut placed it.
function operand_arrivals!(crossed, p::Realizing, v, here)
    val = p.graph.values[v]
    maximum((arrive!(crossed, p, u, here, o, d) for (o, d) in zip(operands(val.definition), operand_domains(val))
        for u in read_values(o)); init = 0)
end

"""
    arrive!(crossed, p, u, here, o, d) -> crossings deep

The crossings a read `o` over domain `d` on device `here` makes of value
`u`, which it reads at its own index or gathers through, into `crossed`;
how many crossings deep `u` is there. A value made anywhere is made there
of nothing that crosses. A chain's frontier is read as its chain is
realized: made again, as the chain made there at what the read reaches; in
pieces, as the pieces put together, each piece made elsewhere crossing
whole; copied, as any other value, read in place where the cut placed it
and elsewhere crossing the range of its last axis the read reaches, the
whole where the read gathers through it.
"""
function arrive!(crossed, p::Realizing, u, here, o, d)
    p.anywhere[u] && return 0
    frontier(p, u) && return arrive!(p.realized[p.chain_of[u]], crossed, p, u, here, o, d)
    arrive!(Copied(), crossed, p, u, here, o, d)
end

function arrive!(::Copied, crossed, p::Realizing, u, here, o, d)
    p.devices[u] == here && return p.depths[u]
    reach!(crossed, (u, p.devices[u], here), read_range(p.graph, u, o, d))
    p.depths[u] + 1
end

function arrive!(::Remade, crossed, p::Realizing, u, here, o, d)
    o isa Read && o.value == u && return made_arrivals!(crossed, p, u, here, o.index, d)
    val = p.graph.values[u]
    made_arrivals!(crossed, p, u, here, identity_index(val.dims), Domain(val.dims))
end

function arrive!(pieced::Pieced, crossed, p::Realizing, u, here, o, d)
    for (q, owner) in enumerate(pieced.owners)
        owner == here || reach!(crossed, (u, owner, here), piece_range(pieced, q))
    end
    p.depths[u] + any(!=(here), pieced.owners)
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
