# Where each value lives. An argument lives where the caller's array is, a
# fact read of it as its extents are; every other value's device is planned
# (`assigned_devices`), and a value read on another device than its own is
# read through a crossing, a copy between the two (`Crossed`), so that every
# other read joins values on one device and a kernel, which computes what it
# reads in place or loads it, stays on one. A value is made again on each
# device that reads it, rather than copied there, where that copies no more
# than what it is made from and spares a hop (see `remade_values`): a sum of
# what several devices computed, which each of them reads, is computed on
# each from the others' terms, one copy each way and none of the sum after
# it. Devices are labels: the first
# device the arguments are on is 1, the next 2, so that a plan and its
# decisions do not depend on which of a machine's devices hold the arrays.

"""
    with_devices(graph, arguments; pins = []) -> IndexGraph

`graph` with a device for each value, `arguments[k]` the label of the `k`th
argument's (see `assigned_devices`), and each result `k` of a pair `k =>
device` in `pins` on that device: a result that replaces an argument, written
where the argument lives, and a gradient, returned where its argument lives.
A value read on another device than its own is made there again where
`remade_values` says so, from what it reads there, its own operands or what
they are there in turn, by the same definition, so that each device computes
it by the same arithmetic in the same order; a crossing is made wherever
else a value is read on another device, one per value and device it is read
on, shared by every reader there, made just before the first of them, a
result's after the values.
"""
function with_devices(graph::IndexGraph, arguments; pins = Pair{Int, Int}[])
    landing = result_devices(graph, pins)
    devices = assigned_devices(graph, arguments, landing)
    placed = IndexGraph(graph.values, graph.arguments, graph.results, graph.result_dims, graph.result_scalar, graph.result_eltype,
        graph.rewrites, graph.backward, devices)
    remade = remade_values(graph, devices, landing)
    # What value `r` is on device `here`, not its own: made there again, or a
    # crossing, a copy of it.
    there = Dict{Tuple{Int, Int}, Int}()
    function across!(b, r, here)
        haskey(there, (r, here)) && return there[(r, here)]
        source = graph.values[r]
        if remade[r]
            through = Dict{Int, Int}()
            for o in operands(source.definition), u in read_values(o)
                devices[u] == here || (through[u] = across!(b, u, here))
            end
            made = add!(b, remap(source.definition, Rerouted(b.ids, through)), source.dims, source.eltype, source.scalar)
        else
            made = add!(b, Crossed(Read(b.ids[r], identity_index(source.dims))), source.dims, source.eltype, source.scalar)
        end
        there[(r, here)] = made
    end
    rebuilt, ids = rebuild(placed) do b, id
        v = graph.values[id]
        here = devices[id]
        through = Dict{Int, Int}()
        for o in operands(v.definition), r in read_values(o)
            devices[r] == here || (through[r] = across!(b, r, here))
        end
        isempty(through) && return nothing
        add!(b, remap(v.definition, Rerouted(b.ids, through)), v.dims, v.eltype, v.scalar)
    end
    landed(rebuilt, ids, graph.results, landing, devices, across!)
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
    remade_values(graph, devices, landing) -> BitVector

Whether each value of `graph`, placed on `devices`, is made again on each
other device that reads it, from what it reads there, rather than copied
there: a value made from nothing another holds (see `made_anywhere`), and an
elementwise value no larger than any value it reads, whose terms lie on
several devices and which its own device reads too. Its terms are what it is
computed from through such values, down to values that are not, those made
anywhere left out. Made again on a device that holds a term, it copies there
the others, no more than it would copy of itself, and is read where it is
made rather than a hop later, after it is computed and copied: a sum of what
several devices computed, read on each, is computed on each from the
others' terms, and every device holds it at once. Terms all on one device
are joined there once, and their result copied.
"""
function remade_values(graph::IndexGraph, devices, landing)
    values = graph.values
    n = length(values)
    # The devices each value's terms lie on, the values before it first.
    terms = Vector{BitSet}(undef, n)
    remade = falses(n)
    elementwise = falses(n)
    for v in 1:n
        d = values[v].definition
        reads = [r for o in operands(d) for r in read_values(o)]
        if made_anywhere(d)
            terms[v] = BitSet()
            remade[v] = true
        elseif d isa Pointwise && all(r -> value_bytes(values[r]) >= value_bytes(values[v]), reads)
            elementwise[v] = true
            terms[v] = mapreduce(r -> terms[r], union, reads; init = BitSet())
        else
            terms[v] = BitSet(devices[v])
        end
    end
    # Whether each value is read on its own device: by a value there, or a
    # result there or on no device in particular.
    home = falses(n)
    for (u, v) in enumerate(values), o in operands(v.definition), r in read_values(o)
        devices[u] == devices[r] && (home[r] = true)
    end
    for (k, o) in enumerate(graph.results), r in read_values(o)
        landing[k] in (0, devices[r]) && (home[r] = true)
    end
    remade .|= elementwise .& (length.(terms) .> 1) .& home
    remade
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
    anywhere = [made_anywhere(v.definition) for v in values]
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
    for v in 1:n
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
    made_anywhere(definition) -> Bool

Whether a value is made from nothing another value holds: a constant, or a
function of positions and literals. It is made again on each device that
reads it, where a copy across would move it on every call: a constant is put
on each device that reads it once, before the first call. An argument is the
caller's, on its device, and a recurrence's state reads what it is carried
from, which `operands` leaves out.
"""
made_anywhere(d::Definition) = all(o -> isempty(read_values(o)), operands(d))
made_anywhere(::Union{Input, Carried, Crossed}) = false

"Whether value `v` of `graph` is read across devices: a crossing's copy."
crossing(graph, v) = graph.values[v].definition isa Crossed
