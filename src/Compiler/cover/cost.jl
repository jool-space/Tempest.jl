# What a launch costs: the device's rates, each package's line for products of
# a class and its memory, and the seconds of a launch's work on them, after
# XLA's GPU performance model. Rates, lines, memories and the price of each
# scalar function are measured on first need (see calibrate.jl) and kept in
# the rates file beside the tuning record.

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

"A package's memory: a launch that moves `bytes` within the cache takes `latency + bytes / bandwidth` seconds."
struct Memory
    latency::Float64
    bandwidth::Float64
end

# A launch of fewer blocks than the device runs at once moves its bytes and
# computes its products on that share of the device.
# Memory time and compute time overlap all but a little: the combination is
# the larger plus a share of the smaller (XLA's GPU performance model uses
# 0.05 of the smaller, chosen empirically).
const OVERLAP = 0.05

function seconds(w::Work, r::Rates, memory::Memory, line, price = (regime, f) -> 0.0)
    share = min(1.0, w.blocks / r.blocks)
    # What blocks read again, other blocks along the grid read just before:
    # it comes from the cache, at the rate of a launch whose bytes fit it.
    memory_seconds = memory.latency + (w.bytes / (w.bytes <= r.cache ? memory.bandwidth : r.bandwidth) + w.reread / r.cached) / share
    # Elementwise work: each function at its positions, at the seconds per
    # position measured for it on the device, in the regime it is applied in.
    cost(regime, pairs) = sum(((f, n),) -> n * price(regime, f), pairs; init = 0.0)
    elementwise = (cost(:streamed, w.applied) + cost(:swept, w.swept) + cost(:inner, w.inner)) / share
    products = w.flops + w.recomputed == 0 ? 0.0 :
        line.latency + (w.split ? w.flops / line.split + w.recomputed / line.recompute : (w.flops + w.recomputed) / line.rate) / share
    compute = products + elementwise
    max(memory_seconds, compute) + OVERLAP * min(memory_seconds, compute)
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
    (d.fold isa Overwrite ? 1 : 2) * sizeof(val.eltype) * prod(d.iteration; init = 1)
end

# A recurrence's floor kernel: an item per position of its parallel axes,
# loading what each term reads at every point of the domain, a recomputed
# fold's body at each of its own positions too.
function recurrence_floor_estimate(p::Plan, g::Group)
    bytes(t::Load) = sizeof(p.graph.values[t.value].eltype)
    bytes(t::Recomputed) = prod(t.extents; init = 1) * bytes(t.body)
    bytes(t::Apply) = sum(bytes, t.args; init = 0)
    bytes(t::Term) = 0
    points = prod(g.domain; init = 1)
    items = prod(k -> g.domain[k], recurrence_axes(p.graph, g).parallel; init = 1)
    [Work(Float64(points * sum(m -> bytes(g.terms[m]), g.outputs)) + group_bytes(p, g); blocks = items / 256)]
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
    Float64(sum(v -> value_bytes(graph.values[v]), reads; init = 0) + sum(written, members(g); init = 0))
end

# The floor reads every load at every point of the domain, one item a thread;
# a fold's or a scan's item sweeps the axes it folds along. Where the target
# splits a reduction's items (`split`, see `Split`), the lanes of their parts
# share the device, and each phase of folds is a launch more.
function floor_estimate(p::Plan, g::Group, split = (items, positions, adjacent) -> nothing)
    g.kind === :recurrence && return recurrence_floor_estimate(p, g)
    points = prod(g.domain; init = 1)
    per_point = sum(m -> sum(l -> sizeof(p.graph.values[l.value].eltype), loads(g.terms[m]); init = 0), members(g); init = 0)
    bytes = Float64(points * per_point) + group_bytes(p, g)
    items = g.kind in (:reduction, :scan) ? prod(kept_dims(g.domain, primary_axes(p.graph, g)); init = 1) : points
    s = group_split(p.graph, g, split)
    s === nothing && return [Work(bytes; blocks = items / 256)]
    blocks = items * s.parts * s.lanes / 256
    phases = length(unique(g.phase[r] for r in g.reductions if primary(p.graph, g, r)))
    [Work(bytes; blocks); [Work(0.0; blocks) for _ in 1:phases]]
end
