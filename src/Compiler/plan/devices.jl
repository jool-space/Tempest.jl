# Where each value lives. An argument lives where the caller's array is, a
# fact read of it as its extents are; every other value's device is planned
# (`assigned_devices`), and a value read on another device than its own gets
# there as it is realized (see plan/realize.jl): copied, a crossing between
# the two (`Crossed`), so that every other read joins values on one device
# and a kernel, which computes what it reads in place or loads it, stays on
# one; or made again there from what it reads there, where that loads the
# devices' links less: a sum of what several devices computed, which each of
# them reads, is computed on each from the others' terms, one copy each way
# and none of the sum after it. A crossing copies only the range of its
# source's last axis that its readers read, one span of its storage (see
# `narrowed_crossings`): a slice read across devices is copied, not what it
# is a slice of. Devices are labels: the first device the arguments are on is
# 1, the next 2, so that a plan and its decisions do not depend on which of a
# machine's devices hold the arrays.

"""
    with_devices(graph, arguments; pins = []) -> IndexGraph

`graph` with a device for each value, `arguments[k]` the label of the `k`th
argument's (see `assigned_devices`), and each result `k` of a pair `k =>
device` in `pins` on that device: a result that replaces an argument, written
where the argument lives, and a gradient, returned where its argument lives.
A value read on another device than its own is realized there as
`realizations` says: made there again from what it reads there, by the same
definition, so that each device computes it by the same arithmetic in the
same order, or copied there, a crossing, one per value and device it is read
on, shared by every reader there, made just before the first of them, a
result's after the values, and copying only what they read of it (see
`narrowed_crossings`).
"""
function with_devices(graph::IndexGraph, arguments; pins = Pair{Int, Int}[])
    landing = result_devices(graph, pins)
    devices = assigned_devices(graph, arguments, landing)
    placed = IndexGraph(graph.values, graph.arguments, graph.results, graph.result_dims, graph.result_scalar, graph.result_eltype,
        graph.rewrites, graph.backward, devices)
    p = realizations(graph, devices, landing)
    rebuilt, ids = rebuild((b, id) -> home!(p, b, id), placed, made_at_home(p))
    narrowed_crossings(landed(rebuilt, ids, graph.results, landing, devices, (b, r, here) -> value_on!(p, b, r, here)))
end

# The device each result is pinned to, by `pins`, zero for one that is not.
# A result pinned to two is refused: it cannot be returned on both.
function result_devices(graph, pins)
    landing = zeros(Int, length(graph.results))
    for (k, device) in pins
        landing[k] == 0 || landing[k] == device ||
            throw(ArgumentError("result $k must be on device $device and on device $(landing[k]): it replaces an argument or is a gradient of one on each"))
        landing[k] = device
    end
    landing
end

# The rebuilt graph with each pinned result on its device: reading through a
# crossing what it reads on another, and a result that is not a value's read
# whole made a value there, as storing would make it (see `result_copy`) on no
# device in particular. The crossings follow the values, in the results'
# order, made by `across!` on a builder of the rebuilt graph's values.
function landed(rebuilt, ids, results, landing, devices, across!)
    b = Builder(copy(rebuilt.values), ids, Pair{Int, String}[], copy(rebuilt.backward), Ref(false), copy(rebuilt.devices), Ref(1))
    placed = map(enumerate(results)) do (k, o)
        landing[k] == 0 && return remap(o, ids)
        b.on_device[] = landing[k]
        through = Dict{Int, Int}()
        for r in read_values(o)
            devices[r] == landing[k] && continue
            # A crossing belongs to the backward where what it copies does.
            b.in_backward[] = rebuilt.backward[ids[r]]
            through[r] = across!(b, r, landing[k])
        end
        r = remap(o, Rerouted(ids, through))
        r isa Read && return r
        b.in_backward[] = false
        held, index = result_copy(rebuilt, k, r)
        Read(add!(b, held.definition, held.dims, held.eltype, held.scalar), index)
    end
    IndexGraph(b.values, rebuilt.arguments, Operand[placed...], rebuilt.result_dims, rebuilt.result_scalar, rebuilt.result_eltype,
        rebuilt.rewrites, b.backward, b.devices)
end

"""
    narrowed_crossings(graph) -> IndexGraph

`graph`, placed, with each crossing copying only the range of its source's
last axis that the values and results reading it read, where that is less
than all of it: one span of the source's storage (see `storage_span`), its
other axes whole, so that a value reading a slice of a value on another
device has the slice copied rather than the whole. A crossing a formula
gathers through, or a recurrence's state is carried from, is copied whole.
Each value keeps its id: the crossing's extents and what it reads change,
and its readers read it at their positions less where its range starts.
"""
function narrowed_crossings(graph::IndexGraph)
    spans = crossing_spans(graph)
    isempty(spans) && return graph
    narrowed = Narrowed(spans)
    values = map(enumerate(graph.values)) do (v, val)
        haskey(spans, v) || return Value(remap(val.definition, narrowed), val.dims, val.eltype, val.scalar)
        dims, at = range_positions(val.dims, spans[v])
        source = val.definition.operand
        Value(Crossed(Read(source.value, map(e -> substitute(e, at, Domain(dims)), source.index))), dims, val.eltype, val.scalar)
    end
    IndexGraph(values, graph.arguments, Operand[remap(o, narrowed) for o in graph.results], graph.result_dims, graph.result_scalar,
        graph.result_eltype, graph.rewrites, graph.backward, graph.devices)
end

# The extents of `range` of the last axis of a value of extents `dims`, the
# axis dropped where the range holds one position, and the position of the
# whole that each of its positions is.
function range_positions(dims, range)
    kept = dims[1:(end - 1)]
    extents = length(range) == 1 ? kept : (kept..., length(range))
    along = length(range) == 1 ? constant(first(range)) : add(var(length(dims)), first(range), Domain(extents))
    (extents, (ntuple(var, length(dims) - 1)..., along))
end

# The range of its last axis each crossing's readers read, where that is less
# than all of it: of each read, the range of its last formula over the domain
# it is read over, a value's operand's or a result's. A value nothing reads,
# as a value made in pieces is not on its own device, where the pieces are
# put together instead, reads nothing a crossing need copy.
function crossing_spans(graph::IndexGraph)
    values = graph.values
    live = live_values(graph)
    read = Dict{Int, UnitRange{Int}}()
    whole = Set{Int}()
    function reached!(o::Read, d::Domain)
        union!(whole, gathered_values(o))
        crossing(graph, o.value) && reach!(read, o.value, read_range(values[o.value], o, d))
    end
    reached!(o::Operand, d::Domain) = union!(whole, gathered_values(o))
    for (v, val) in enumerate(values)
        live[v] || continue
        foreach(reached!, operands(val.definition), operand_domains(val))
        val.definition isa Carried && union!(whole, read_values(val.definition.from))
    end
    foreach((o, dims) -> reached!(o, Domain(dims)), graph.results, graph.result_dims)
    Dict{Int, UnitRange{Int}}(c => span for (c, span) in read if !(c in whole) && span != whole_range(values[c]))
end

# The range of value `v`'s last axis a read `o` of it over domain `d`
# reaches, within its extent: by the range of the read's last formula, of
# which a crossing of `v` copies no more. A value of no axes is read whole.
function read_range(v::Value, o::Read, d::Domain)
    isempty(v.dims) && return whole_range(v)
    lo, hi = affine_range(last(o.index), d)
    max(lo, 0):min(hi, last(v.dims) - 1)
end

whole_range(v::Value) = isempty(v.dims) ? (0:0) : (0:(last(v.dims) - 1))

# The bytes of `range` of value `v`'s last axis, its other axes whole.
span_bytes(v::Value, range) = isempty(v.dims) ? value_bytes(v) : value_bytes(v) ÷ last(v.dims) * length(range)

# The crossings of a graph narrowed to the ranges `spans` gives, as their
# readers read them: at their positions less where each range starts, the
# last axis dropped where it holds one position. Every other value keeps its id.
struct Narrowed
    spans::Dict{Int, UnitRange{Int}}
end

Base.getindex(::Narrowed, v::Int) = v

remap(o::Read, n::Narrowed) = haskey(n.spans, o.value) ? Read(o.value, within(o.index, n.spans[o.value])) : o

function within(index, span)
    length(span) == 1 && return index[1:(end - 1)]
    e = last(index)
    (index[1:(end - 1)]..., Affine(e.constant - first(span), e.terms))
end

# The new ids of a reader's operands: a value it reads across devices, its
# crossing's.
struct Rerouted
    ids::Vector{Int}
    through::Dict{Int, Int}
end

Base.getindex(r::Rerouted, v::Int) = get(r.through, v, r.ids[v])

"""
    assigned_devices(graph, arguments, landing) -> Vector{Int}

The device of each value of `graph`, by label: an argument's is
`arguments[k]`, and the rest are placed by a minimum cut of bytes, so that as
little as can be is copied between devices. Each value is a net joining it
to what reads it, of its bytes, cut where they are not all on one device: a
value read on another device is copied there once, however many read it
there, and in either direction, which Lawler's construction makes exact for
two devices. A result pinned to a device (`landing[k]`, zero where it is
not) reads what it reads there, a vertex of its own on that device: the cut
decides whether the value is computed there or copied there, the result's
operands never pinned. A recurrence's cycle is one vertex, its values
computed step by step together on one device. A value made from nothing
another holds is no net (see `made_anywhere`): made again on each device
that reads it, it is on its first reader's. Of minimum cuts alike, the one
with most on the lower device. Over more than two devices, each device is
cut against those after it in turn, an approximation: a value one cut leaves
on a device and two later devices read is copied twice and charged once.
"""
function assigned_devices(graph::IndexGraph, arguments, landing)
    values = graph.values
    n = length(values)
    # A program on one device has nothing to place.
    all(==(1), arguments) && all(<=(1), landing) && return ones(Int, n)
    home = recurrence_vertices(graph)
    anywhere = made_anywhere(graph)
    # Result `k` pinned is vertex `n + k`.
    users = value_readers(graph)
    landers = [Int[] for _ in values]
    for (k, o) in enumerate(graph.results), r in read_values(o)
        landing[k] == 0 || push!(landers[r], k)
    end
    vertices = [[v for v in 1:n if home[v] == v && !anywhere[v]]; [n + k for k in eachindex(landing) if landing[k] != 0]]
    required = [[v.definition isa Input ? arguments[v.definition.argument] : 0 for v in values]; landing]
    last_device = maximum(required; init = 1)
    devices = zeros(Int, length(required))
    nets = [(value_bytes(values[v]), unique!(sort!([home[v]; [home[u] for u in users[v]]; [n + k for k in landers[v]]])))
        for v in 1:n if !anywhere[v]]
    for d in 1:(last_device - 1)
        cut_device!(devices, vertices, required, nets, d)
    end
    for v in vertices
        devices[v] == 0 && (devices[v] = last_device)
    end
    placed = [devices[home[v]] for v in 1:n]
    # Readers first: a value made anywhere may be read by another.
    for v in n:-1:1
        anywhere[v] && (placed[v] = first_reader_device(placed, users[v], landing, landers[v]))
    end
    placed
end

# Where a value made anywhere is: on its first reader's device, a value's or
# else a pinned result's, or on 1 where nothing reads it.
function first_reader_device(placed, users, landing, landers)
    isempty(users) || return placed[first(users)]
    isempty(landers) || return landing[first(landers)]
    1
end

# Places on device `d` the vertices not placed yet that a minimum cut puts on
# its side, against the devices after it: those required on `d` are joined to
# the source and those required on a later one to the sink, with infinite
# capacity, and each net is a pair of vertices of its own, `inside` and
# `outside`, an edge of its bytes between them, every vertex of the net
# joined to `inside` and from `outside` with infinite capacity, so that a
# cut separating any two of its vertices crosses that edge, and only it.
function cut_device!(devices, vertices, required, nets, d)
    n = length(devices)
    source, sink = n + 2length(nets) + 1, n + 2length(nets) + 2
    flow = Flow(sink)
    for v in vertices
        devices[v] == 0 && required[v] != 0 || continue
        required[v] == d ? edge!(flow, source, v, Inf) : edge!(flow, v, sink, Inf)
    end
    for (k, (bytes, net)) in enumerate(nets)
        open = filter(v -> devices[v] == 0, net)
        length(open) > 1 || continue
        inside, outside = n + 2k - 1, n + 2k
        edge!(flow, inside, outside, Float64(bytes))
        for v in open
            edge!(flow, v, inside, Inf)
            edge!(flow, outside, v, Inf)
        end
    end
    maxflow!(flow, source, sink)
    later = reaching(flow, sink)
    for v in vertices
        devices[v] == 0 && !later[v] && (devices[v] = d)
    end
end

# The vertex each value is placed as: the first value of its recurrence's
# cycle for each value of the cycle, which the cycle's states read back from
# a value after them and so compute step by step together, and itself
# otherwise. Cycles that share a value are one vertex.
function recurrence_vertices(graph)
    home = collect(eachindex(graph.values))
    root(v) = home[v] == v ? v : root(home[v])
    for cycle in recurrences(graph)
        members = [cycle.states; cycle.from; cycle.values]
        head = minimum(root, members)
        foreach(m -> home[root(m)] = head, members)
    end
    map(root, home)
end

# The values reading each value, in order: through its operands, the values
# their positions gather from too.
function value_readers(graph)
    users = [Int[] for _ in graph.values]
    for (u, v) in enumerate(graph.values), o in operands(v.definition), r in read_values(o)
        isempty(users[r]) || last(users[r]) != u || continue
        push!(users[r], u)
    end
    users
end

"""
    made_anywhere(graph) -> BitVector

Whether each value of `graph` is made from nothing another value holds: a
constant, a function of positions and literals, or a function of such
values, as a mask of a range is the conjunction of two comparisons of
positions. It is made again on each device that reads it, where a copy
across would move it on every call: a constant is put on each device that
reads it once, before the first call. An argument is the caller's, on its
device, and a recurrence's state reads what it is carried from, which
`operands` leaves out.
"""
function made_anywhere(graph::IndexGraph)
    anywhere = falses(length(graph.values))
    for (v, value) in enumerate(graph.values)
        d = value.definition
        d isa Union{Input, Carried, Crossed} && continue
        anywhere[v] = all(r -> anywhere[r], (r for o in operands(d) for r in read_values(o)))
    end
    anywhere
end

"Whether value `v` of `graph` is read across devices: a crossing's copy."
crossing(graph, v) = graph.values[v].definition isa Crossed
