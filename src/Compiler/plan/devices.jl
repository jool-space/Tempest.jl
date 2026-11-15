# Where each value lives. An argument lives where the caller's array is, a
# fact read of it as its extents are; every other value's device is planned
# (`assigned_devices`), and a value read on another device than its own is
# read through a crossing, a copy between the two (`Crossed`), so that every
# other read joins values on one device and a kernel, which computes what it
# reads in place or loads it, stays on one. Devices are labels: the first
# device the arguments are on is 1, the next 2, so that a plan and its
# decisions do not depend on which of a machine's devices hold the arrays.

"""
    with_devices(graph, arguments; pins = Dict()) -> IndexGraph

`graph` with a device for each value, `arguments[k]` the label of the `k`th
argument's (see `assigned_devices`), each value in `pins` on the device it
names; and a crossing wherever a value is read on another device than its
own: one value per value and device it is read on, shared by every reader
there, made just before the first of them.
"""
function with_devices(graph::IndexGraph, arguments; pins = Dict{Int, Int}())
    devices = assigned_devices(graph, arguments, pins)
    placed = IndexGraph(graph.values, graph.arguments, graph.results, graph.result_dims, graph.result_scalar, graph.result_eltype,
        graph.rewrites, graph.backward, devices)
    crossed = Dict{Tuple{Int, Int}, Int}()
    first(rebuild(placed) do b, id
        v = graph.values[id]
        here = devices[id]
        through = Dict{Int, Int}()
        for o in operands(v.definition), r in read_values(o)
            devices[r] == here && continue
            through[r] = get!(crossed, (r, here)) do
                source = graph.values[r]
                add!(b, Crossed(Read(b.ids[r], identity_index(source.dims))), source.dims, source.eltype, source.scalar)
            end
        end
        isempty(through) && return nothing
        add!(b, remap(v.definition, Rerouted(b.ids, through)), v.dims, v.eltype, v.scalar)
    end)
end

# The new ids of a reader's operands: a value it reads across devices, its
# crossing's.
struct Rerouted
    ids::Vector{Int}
    through::Dict{Int, Int}
end

Base.getindex(r::Rerouted, v::Int) = get(r.through, v, r.ids[v])

"""
    assigned_devices(graph, arguments, pins) -> Vector{Int}

The device of each value of `graph`, by label: an argument's is
`arguments[k]`, and a value in `pins` is on the device it names. Any other
value is on the device of what it reads, and where that is on several, of
the value it reads with the most bytes, the latest of those alike; one that
reads nothing is on device 1. A first rule, local to each value: where a
model's layers are on two devices, it copies across both an activation and
the values computed from it that the next layer reads, where a minimum cut
of bytes would copy the activation alone.
"""
function assigned_devices(graph::IndexGraph, arguments, pins)
    devices = ones(Int, length(graph.values))
    for (id, v) in enumerate(graph.values)
        d = v.definition
        devices[id] = haskey(pins, id) ? pins[id] : d isa Input ? arguments[d.argument] : read_device(graph, devices, id)
    end
    devices
end

function read_device(graph, devices, id)
    read = [r for o in operands(graph.values[id].definition) for r in read_values(o)]
    isempty(read) && return 1
    devices[argmax(r -> (value_bytes(graph.values[r]), r), read)]
end

"Whether value `v` of `graph` is read across devices: a crossing's copy."
crossing(graph, v) = graph.values[v].definition isa Crossed
