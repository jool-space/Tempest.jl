# What a launch costs: the device's rates, each package's line for products of
# a class and its memory, and the seconds of a launch's work on them, after
# XLA's GPU performance model. The rates are the device's as it reports them
# (`device_rates`); lines, memories and the price of each scalar function are
# measured on first need (see calibrate.jl) and kept in the rates file beside
# the tuning record.

"""
A device's rates for estimates: seconds per launch in a captured graph, bytes
per second from memory for a launch that moves at most `cache` bytes and for
one that moves more, the blocks that fill the device, and the relative margin
within which two measurements tie.
"""
struct Rates
    launch::Float64
    cached::Float64
    bandwidth::Float64
    cache::Float64
    blocks::Int
    noise::Float64
end

"""
A package's products of one class: a launch of `flops`, `recomputed` of them
in folds it recomputes inside the products, takes `latency + flops / rate`
seconds; when its blocks split the sums among them, the products take `split`
and the recomputed flops `recompute`. A borrowed line is another package's,
for a package that claims no reference product.
"""
struct Line
    latency::Float64
    rate::Float64
    split::Float64
    recompute::Float64
    borrowed::Bool
end

"""
A package's memory: a launch that moves `bytes` within the cache takes
`latency + bytes / bandwidth` seconds, and one whose blocks sweep `steps`
tiles one after another at least `latency + steps * step`, a step waiting
for what it loads.
"""
struct Memory
    latency::Float64
    bandwidth::Float64
    step::Float64
end

# A launch of fewer blocks than the device runs at once computes on that
# share of the device, a block on a multiprocessor. In memory, what its
# blocks reach is what its steps bring in, where it states them: a block
# alone waits on a step's loads at a time, and blocks side by side add until
# the memory's bandwidth (see `stepping`), a matrix-vector product of 10
# blocks of 32 rows stepping 512 along its sum reaching four fifths of it,
# of 5 blocks of 128 stepping 128, three fifths, where on that share of the
# device they had taken 3.5 and 5 times their time. A launch that states no
# steps moves its bytes on the share as well: a product's sweep, priced by
# its line, and the floor's items, each sweeping its fold a position at a
# time.
# Memory time and compute time overlap all but a little: the combination is
# the larger plus a share of the smaller (XLA's GPU performance model uses
# 0.05 of the smaller, chosen empirically).
const OVERLAP = 0.05

function seconds(w::Work, r::Rates, memory::Memory, line, price = (regime, f) -> 0.0)
    share = min(1.0, w.blocks / r.blocks)
    reach = w.steps > 1 ? 1.0 : share
    # What blocks read again, other blocks along the grid read just before:
    # it comes from the cache, at the rate of a launch whose bytes fit it.
    # So do partial results one launch passes to the next, where they fit.
    passed = w.passed / (w.passed <= r.cache ? r.cached : r.bandwidth)
    memory_seconds = memory.latency + (w.bytes / (w.bytes <= r.cache ? memory.bandwidth : r.bandwidth) + w.reread / r.cached + passed) / reach
    # Elementwise work: each function at its positions, at the seconds per
    # position measured for it on the device, in the regime it is applied in.
    cost(regime, pairs) = sum(((f, n),) -> n * price(regime, f), pairs; init = 0.0)
    elementwise = (cost(:streamed, w.applied) + cost(:swept, w.swept) + cost(:inner, w.inner)) / share
    products = w.flops + w.recomputed == 0 ? 0.0 :
        line.latency + (w.split ? w.flops / line.split + w.recomputed / line.recompute : (w.flops + w.recomputed) / line.rate) / share
    compute = products + elementwise
    # A sweep of many steps waits on each step's loads and on the chain of
    # functions it computes on them, one step after another, while the
    # blocks sweep side by side: its time is the steps', however few bytes
    # each step moves (see `Work`).
    stepping = w.steps <= 1 ? 0.0 : memory.latency + w.steps * memory.step + cost(:stepped, w.stepped)
    longest = max(memory_seconds, compute, stepping)
    # A fold a block recomputes waits on its own loads, step by step, before
    # the block goes on (see `Work`): its steps add to the rest.
    refolding = w.refold_steps * memory.step
    longest + OVERLAP * (memory_seconds + compute + stepping - longest) + refolding
end

"The class of products of `T` factors: a half type by name, Float32 by the target's precision."
product_class(target::CUDATarget, T) = T <: HalfFloat ? nameof(T) : target.precision
product_class(target, T) = nameof(T)

# The package an implementation is priced as, asked as the protocol is (see
# `options`).
package_name(implementation) = Base.invokelatest(package_of, implementation)

const NOISE = 0.05

# A target that cannot measure, or a decider that must not, has nominal
# rates: no latency, one line for every product (see `line!`).
rates(::Target) = Rates(0.0, 2.0e10, 2.0e10, Inf, 1, NOISE)

value_bytes(v::Value) = sizeof(v.eltype) * storage_length(v)

# A fold by key that starts from contents in storage is priced in place, as
# XLA prices a dynamic-update-slice and a scatter (`HloCostAnalysis`): it moves
# the elements it combines, a write writing them and a sum reading them too,
# and none of the contents it starts from.
starts_from_storage(graph, v) = keyed(graph, v) && graph.values[v].definition.into isa Read

function combined_bytes(val::Value)
    d = val.definition
    (d.fold isa Overwrite ? 1 : 2) * sizeof(val.eltype) * prod(d.iteration::Tuple{Vararg{Int}}; init = 1)
end

# A recurrence's floor kernel: an item per position of its parallel axes,
# loading what each member's term reads at every point it computes the
# member, a recomputed fold's body at each of its own positions too. The
# bytes are counted in floating point: folds recomputed one inside another
# multiply their extents, six products of 1024 columns past what an Int
# holds, and a count wrapped below zero priced the kernel under any other.
function recurrence_floor_estimate(p::Plan, g::Group)
    bytes(t::Load) = Float64(sizeof(p.graph.values[t.value].eltype))
    bytes(t::Recomputed) = prod(Float64, t.extents; init = 1.0) * bytes(t.body)
    bytes(t::Apply) = sum(bytes, t.args; init = 0.0)
    bytes(t::Term) = 0.0
    (; axis, parallel) = recurrence_axes(p.graph, g)
    places = recurrence_places(p.graph, g.outputs).places
    points(m) = prod(k -> k == axis || any(e -> mentions(e, k), places[m]) ? g.domain[k] : 1, eachindex(g.domain); init = 1)
    items = prod(k -> g.domain[k], parallel; init = 1)
    [Work(sum(m -> points(m) * bytes(g.terms[m]), g.outputs; init = 0.0) + group_bytes(p, g); blocks = items / 256)]
end

"Bytes a group moves: what it reads from storage once, and what it writes."
function group_bytes(p::Plan, g::Group)
    graph = p.graph
    reads = group_reads(graph, g)
    loaded(a) = any(m -> haskey(g.terms, m) && a in stored_reads(g.terms[m]) || a in key_reads(graph, m), members(g))
    for v in members(g)
        starts_from_storage(graph, v) || continue
        a = graph.values[v].definition.into.value
        loaded(a) || delete!(reads, a)
    end
    written(v) = !p.stored[v] ? 0 : starts_from_storage(graph, v) ? combined_bytes(graph.values[v]) : value_bytes(graph.values[v])
    found = GroupLoads(graph, g)
    Float64(sum(v -> read_bytes(p, g, found, v), reads; init = 0) + sum(written, members(g); init = 0))
end

"""
What `read_bytes` asks of a group's terms, found once for every value the
group reads: whether every member has a term, the values a fold by key's
key reads, those an index gathers from, and each value's loads. Asked of
each value afresh, a sum of many values, as a gradient of an input many
branches read, walked its whole term once for each.
"""
struct GroupLoads
    termed::Bool
    keys::Set{Int}
    gathered::Set{Int}
    loading::Dict{Int, Vector{Load}}
end

function GroupLoads(graph, g::Group)
    termed = all(m -> haskey(g.terms, m), members(g))
    keys = Set{Int}(v for m in members(g) for v in key_reads(graph, m))
    gathered, loading = Set{Int}(), Dict{Int, Vector{Load}}()
    termed || return GroupLoads(termed, keys, gathered, loading)
    terms = [g.terms[m] for m in members(g)]
    for s in subterms(terms; nested = true)
        s isa Load && foreach(e -> union!(gathered, indirect_values(e)), s.index)
        s isa Ix && union!(gathered, indirect_values(s.expr))
    end
    for t in terms, l in loads(t)
        push!(get!(() -> Load[], loading, l.value), l)
    end
    GroupLoads(termed, keys, gathered, loading)
end

# The bytes of value `v` group `g` reads, by what it loads (`found`): where
# its loads reach, a step's slice of a value along the steps say, when the
# group only loads it; whole when the group gathers from it or reads it
# otherwise.
function read_bytes(p::Plan, g::Group, found::GroupLoads, v)
    val = p.graph.values[v]
    found.termed && !(v in found.keys) || return value_bytes(val)
    v in found.gathered && return value_bytes(val)
    ts = get(found.loading, v, Load[])
    isempty(ts) && return value_bytes(val)
    sizeof(val.eltype) * min(storage_length(val), reached_elements(ts, val.dims, Domain(g.domain); run = sector_elements(val.eltype)))
end

# The floor reads each value a member loads at every point of the domain,
# one item a thread; a fold's or a scan's item sweeps the axes it folds
# along. A value a member loads at several positions of a point, a stencil's
# neighbours, is read from memory once: its other loads find it in the
# cache, where neighbouring items brought it. Where the target splits a
# reduction's items (`split`, see `Split`), the lanes of their parts share
# the device, and each phase of folds is a launch more.
floor_estimate(p::Plan, g::Group, split = (items, positions, adjacent) -> nothing) = floor_estimate(g.kind, p, g, split)
floor_estimate(::RecurrenceKind, p::Plan, g::Group, split) = recurrence_floor_estimate(p, g)

function floor_estimate(::Union{ReductionKind, ScanKind, PointwiseKind}, p::Plan, g::Group, split)
    points = prod(g.domain; init = 1)
    loaded(m) = unique(l.value for l in loads(g.terms[m]))
    per_point = sum(m -> sum(v -> sizeof(p.graph.values[v].eltype), loaded(m); init = 0), members(g); init = 0)
    bytes = Float64(points * per_point) + group_bytes(p, g)
    # A group of maps has no primary axes: its items are its points.
    items = prod(kept_dims(g.domain, primary_axes(p.graph, g)); init = 1)
    s = group_split(p.graph, g, split)
    s === nothing && return [Work(bytes; blocks = items / 256)]
    blocks = items * s.parts * s.lanes / 256
    phases = length(unique(g.phase[r] for r in g.reductions if primary(p.graph, g, r)))
    [Work(bytes; blocks); [Work(0.0; blocks) for _ in 1:phases]]
end
