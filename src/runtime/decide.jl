# Choosing how each group runs, and which groups there are: the cover.
#
# Every implementation's claims on a group are estimated, as launches with the
# bytes they move, the flops they compute and the blocks they keep busy.
# Implementations also propose candidates of their own, sets of values grown
# from anchors that they accept; a candidate replaces the generator's groups
# when the cover with it is cheaper. Estimates within each other's error are
# close calls: they are measured at preparation and kept in the tuning record,
# and a measurement overrides the estimate only beyond the device's noise
# margin, so the choice is deterministic given the record.

"One launch's work: bytes moved, flops at the rate the launch reaches, and its blocks."
struct Work
    bytes::Float64
    flops::Float64
    rate::Float64
    blocks::Float64
end

Work(bytes; flops = 0.0, rate = Inf, blocks = Inf) = Work(bytes, flops, rate, blocks)

"""
A device's rates for estimates: memory bandwidth in bytes per second, seconds
per launch in a captured graph, the blocks that fill the device, and the
relative margin within which two measurements tie.
"""
struct Rates
    bandwidth::Float64
    launch::Float64
    blocks::Int
    noise::Float64
end

seconds(w::Work, r::Rates) = r.launch + max(w.bytes / r.bandwidth, w.flops / w.rate) / min(1.0, w.blocks / r.blocks)
seconds(ws::AbstractVector{Work}, r::Rates) = sum(w -> seconds(w, r), ws; init = 0.0)

# Two estimates closer than this factor are a close call.
const ESTIMATE_ERROR = 1.5

rates(::CPUTarget) = Rates(2.0e10, 0.0, 1, 0.05)
device_name(::CPUTarget) = "cpu"
"The target's numerics, as part of the tuning record's key."
numerics(::Target) = :default
numerics(target::CUDATarget) = target.online ? target.precision : Symbol(target.precision, "-two-pass")

"""
    estimate(implementation, claim, target, plan, group) -> Vector{Work}

The launches of a claim. By default one launch moving the group's bytes.
"""
estimate(::Implementation, claim, target, p, g) = [Work(group_bytes(p, g))]

"""
    candidates(implementation, target, plan) -> Vector{Vector{Int}}

Sets of values the implementation proposes to realize as one group each, in
place of the generator's groups: grown from anchors in `plan` (a product, a
call) and accepted by the implementation. None by default.
"""
candidates(::Implementation, target, p) = Vector{Int}[]

"""
    grow(graph, anchor, accept) -> Vector{Int}

A candidate grown from `anchor` along its readers: each step adds the first
reader of the set, in value order, with which `accept(set)` holds.
"""
function grow(graph, anchor, accept)
    users = readers(graph)
    set = [anchor]
    while true
        next = sort!(unique(u for v in set for u in users[v] if !(u in set)))
        k = findfirst(u -> accept([set; u]), next)
        k === nothing && return set
        push!(set, next[k])
    end
end

"A claim's name in the tuning record, without value numbers."
record_name(implementation::Implementation, claim) = string(nameof(typeof(implementation)), " ", claim)

"Whether claims on `target` can be timed at preparation; `time_launch` times one."
measurable(::Target) = false
function time_launch end

value_bytes(v::Value) = sizeof(v.eltype) * storage_length(v)

"Bytes a group moves: what it reads from storage once, and what it writes."
function group_bytes(p::Plan, g::Group)
    reads = sum(v -> value_bytes(p.graph.values[v]), group_reads(p.graph, g); init = 0)
    writes = sum(v -> p.stored[v] ? value_bytes(p.graph.values[v]) : 0, members(g); init = 0)
    Float64(reads + writes)
end

# The floor reads every load at every point of the domain, one item a thread.
function floor_estimate(p::Plan, g::Group)
    g.kind === :opaque && return [Work(group_bytes(p, g))]
    points = prod(g.domain; init = 1)
    per_point = sum(m -> sum(l -> sizeof(p.graph.values[l.value].eltype), loads(g.terms[m]); init = 0), members(g); init = 0)
    items = g.kind === :reduction ? prod(kept_dims(g.domain, primary_axes(p.graph, g)); init = 1) : points
    [Work(Float64(points * per_point) + group_bytes(p, g); blocks = items / 256)]
end

"A way to realize a group: an implementation's claim, or the floor's (`nothing`)."
struct Choice
    implementation::Union{Nothing, Implementation}
    claim::Any
    estimate::Float64
    key::String
end

"What was chosen for a group, how, and the time measured for it, if any."
struct Decision
    choice::Choice
    options::Vector{Choice}
    how::Symbol # :only, :estimated or :measured
    measured::Union{Nothing, Float64}
end

function Base.show(io::IO, d::Decision)
    print(io, "estimated ", round(1.0e6 * d.choice.estimate; sigdigits = 3), " μs")
    d.measured === nothing || print(io, ", measured ", round(1.0e6 * d.measured; sigdigits = 3), " μs")
    d.how === :measured && print(io, "; chosen by measurement among ", length(d.options))
    d.how === :estimated && length(d.options) > 1 && print(io, "; chosen by estimate among ", length(d.options))
end

mutable struct Decider
    target::Any
    graph::IndexGraph
    measure::Bool
    rates::Rates
    record::Dict{String, Float64}
    record_path::String
    buffers::Vector{Any}
    built::Dict{Tuple{Any, String}, Any}
    decided::Dict{Any, Decision}
end

function Decider(target, graph; measure = true)
    path = record_path(target)
    Decider(target, graph, measure && measurable(target), rates(target), load_record(path), path,
        Any[], Dict{Tuple{Any, String}, Any}(), Dict{Any, Decision}())
end

# A group within one preparation: its kind, members, terms and what it stores.
group_identity(p::Plan, g::Group) = (g.kind, g.domain, Tuple(g.reductions), Tuple(g.outputs),
    Tuple(get(g.terms, m, nothing) for m in members(g)), Tuple(p.stored[m] for m in members(g)))

# A group across programs: value numbers in order of appearance, with each
# value's type and shape.
function canonical(p::Plan, g::Group)
    io = IOBuffer()
    print(io, g.kind, " ", g.domain)
    for m in members(g)
        v = p.graph.values[m]
        what = m in g.reductions ? string(fold_name(v.definition.fold), v.definition.reduced) : "value"
        print(io, "; %", m, " ", what, " ", v.eltype, v.dims, p.stored[m] ? " stored" : "", " = ", get(g.terms, m, "opaque"))
    end
    numbers = Dict{String, Int}()
    text = replace(String(take!(io)), r"%\d+" => s -> string("%", get!(numbers, s, length(numbers) + 1)))
    shapes = map(sort!(collect(numbers); by = last)) do (s, k)
        v = p.graph.values[parse(Int, s[2:end])]
        string("%", k, " ", v.eltype, v.dims)
    end
    string(text, "; ", join(shapes, ", "))
end

function choice(d::Decider, p::Plan, g::Group, implementation, claim)
    work = implementation === nothing ? floor_estimate(p, g) : estimate(implementation, claim, d.target, p, g)
    name = implementation === nothing ? "floor" : record_name(implementation, claim)
    key = replace(string(numerics(d.target), " ", name, " | ", canonical(p, g)), r"[\t\n]" => " ")
    Choice(implementation, claim, seconds(work, d.rates), key)
end

"Every claim on a group, cheapest estimate first; the floor when none claims it."
function options(d::Decider, p::Plan, g::Group)
    os = Choice[choice(d, p, g, i, c) for i in implementations(d.target) for c in claims(i, d.target, p, g)]
    isempty(os) && push!(os, choice(d, p, g, nothing, nothing))
    sort!(os; by = o -> o.estimate) # stable: ties keep the implementations' order
end

close_call(a, b) = max(a, b) <= ESTIMATE_ERROR * min(a, b)

"Choose how a group runs: the cheapest estimate, unless measurements of the close calls disagree beyond the noise."
function decide(d::Decider, p::Plan, g::Group)
    get!(d.decided, group_identity(p, g)) do
        os = options(d, p, g)
        best = first(os)
        length(os) == 1 && return Decision(best, os, :only, get(d.record, best.key, nothing))
        contenders = [o for o in os if close_call(o.estimate, best.estimate)]
        length(contenders) == 1 && return Decision(best, os, :estimated, get(d.record, best.key, nothing))
        d.measure && foreach(o -> measure!(d, p, g, o), contenders)
        times = [get(d.record, o.key, nothing) for o in contenders]
        any(isnothing, times) && return Decision(best, os, :estimated, nothing)
        k = argmin(times)
        k = times[k] < (1 - d.rates.noise) * times[1] ? k : 1
        Decision(contenders[k], os, :measured, times[k])
    end
end

decisions(d::Decider, p::Plan) = [g.kind === :opaque ? nothing : decide(d, p, g) for g in p.groups]

# Whether plan `b` is cheaper than plan `a`, comparing the groups they do not
# share: by estimate, or by measurement beyond the noise when every one of them
# is measured. Close calls are measured.
function cheaper(d::Decider, b::Plan, a::Plan)
    ia = Set(group_identity(a, g) for g in a.groups)
    ib = Set(group_identity(b, g) for g in b.groups)
    only_a = [(a, g) for g in a.groups if !(group_identity(a, g) in ib) && g.kind !== :opaque]
    only_b = [(b, g) for g in b.groups if !(group_identity(b, g) in ia) && g.kind !== :opaque]
    decided(side) = [(p, g, decide(d, p, g)) for (p, g) in side]
    da, db = decided(only_a), decided(only_b)
    ea = sum(x -> x[3].choice.estimate, da; init = 0.0)
    eb = sum(x -> x[3].choice.estimate, db; init = 0.0)
    if d.measure && close_call(ea, eb)
        foreach(((p, g, x),) -> measure!(d, p, g, x.choice), [da; db])
    end
    ta = [get(d.record, x[3].choice.key, nothing) for x in da]
    tb = [get(d.record, x[3].choice.key, nothing) for x in db]
    if !any(isnothing, ta) && !any(isnothing, tb)
        ma, mb = sum(ta; init = 0.0), sum(tb; init = 0.0)
        abs(ma - mb) > d.rates.noise * min(ma, mb) && return mb < ma
    end
    eb < ea
end

"""
    cover(decider) -> (plan, accepted)

The generator's groups, with the candidates implementations propose accepted
greedily: by estimated gain over the generator's plan, then by their values,
each when its proposer claims the group it forms and the plan with it is
cheaper than without.
"""
function cover(d::Decider)
    base, keep = generator_plan(d)
    proposals = Tuple{Vector{Int}, Implementation}[]
    for i in implementations(d.target), set in candidates(i, d.target, base)
        any(q -> q[1] == set, proposals) || push!(proposals, (set, i))
    end
    recompute_folds = recomputes_folds(d.target)
    trial(sets) = try
        plan(d.graph; fixed = sets, recompute_folds, keep)
    catch e
        e isa NotAGroup || rethrow()
        nothing
    end
    gain(p) = sum(x -> decide(d, base, x).choice.estimate, filter(g -> g.kind !== :opaque, base.groups); init = 0.0) -
        sum(x -> decide(d, p, x).choice.estimate, filter(g -> g.kind !== :opaque, p.groups); init = 0.0)
    ranked = map(proposals) do (set, i)
        p = trial([set])
        (p === nothing || !proposer_claims(d, p, set, i)) ? nothing : (gain(p), set, i)
    end
    ranked = sort!(filter(!isnothing, ranked); by = r -> (-r[1], r[2]))
    current, accepted = base, Pair{Vector{Int}, Implementation}[]
    for (_, set, i) in ranked
        p = trial([first.(accepted); [set]])
        p === nothing && continue
        proposer_claims(d, p, set, i) && cheaper(d, p, current) || continue
        current = p
        push!(accepted, set => i)
    end
    current, accepted
end

recomputes_folds(target) = any(i -> recomputes_folds(i, target), implementations(target))

recomputed_in(g::Group; nested = true) =
    Set(t.value for m in members(g) if haskey(g.terms, m) for t in subterms((g.terms[m],); nested) if t isa Fold)

# The generator's plan, folds recomputed where they are read when an
# implementation recomputes them, and the folds kept stored: the outermost ones
# recomputed in a group no implementation claims (folds they recompute in turn
# may still be recomputed in their kernels).
function generator_plan(d::Decider)
    recompute_folds = recomputes_folds(d.target)
    keep = Set{Int}()
    while true
        p = plan(d.graph; recompute_folds, keep)
        unclaimed = [g for g in p.groups if !isempty(recomputed_in(g)) &&
                     all(i -> isempty(claims(i, d.target, p, g)), implementations(d.target))]
        isempty(unclaimed) && return p, keep
        foreach(g -> union!(keep, recomputed_in(g; nested = false)), unclaimed)
    end
end

function proposer_claims(d::Decider, p::Plan, set, implementation)
    g = p.groups[p.owner[first(set)]]
    !isempty(claims(implementation, d.target, p, g))
end

# Buffers for a group's reads and stored members, shared by every plan the
# decider considers (they number values alike), zeroed so that measuring reads
# nothing undefined.
function buffers!(d::Decider, p::Plan, g::Group)
    n = length(p.graph.values)
    length(d.buffers) < n && append!(d.buffers, fill(nothing, n - length(d.buffers)))
    for v in [collect(group_reads(p.graph, g)); filter(m -> p.stored[m], members(g))]
        d.buffers[v] === nothing || continue
        val = p.graph.values[v]
        def = val.definition
        d.buffers[v] = def isa ConstantArray ? upload(d.target, vec(collect(def.data))) :
            zeroed!(allocate(d.target, val.eltype, storage_length(val)))
    end
    d.buffers
end

zeroed!(x) = (fill!(reinterpret(UInt8, x), 0x00); x)

# Build a choice on the decider's buffers, once.
function build_choice!(d::Decider, p::Plan, g::Group, o::Choice)
    get!(d.built, (group_identity(p, g), o.key)) do
        buffers = buffers!(d, p, g)
        o.implementation === nothing ? build_floor(d.target, p, g, buffers) :
            build_claim(o.implementation, o.claim, d.target, p, g, buffers)
    end
end

# A group's kernels: its choice, or the next claim when a build fails, and
# the floor last.
function build_group(d::Decider, p::Plan, g::Group, x, buffers)
    g.kind === :opaque && return build_opaque(d.target, p, g, buffers)
    for o in [x.choice; filter(!=(x.choice), x.options)]
        o.implementation === nothing && break
        built = try
            build_choice!(d, p, g, o)
        catch err
            @debug "claim failed to build; trying the next" o.implementation exception = err
            nothing
        end
        built === nothing || return built
    end
    build_floor(d.target, p, g, buffers)
end

function measure!(d::Decider, p::Plan, g::Group, o::Choice)
    haskey(d.record, o.key) && return
    launch, _ = build_choice!(d, p, g, o)
    t = time_launch(d.target, launch)
    d.record[o.key] = t
    save_record(d.record_path, o.key, t)
end

## The tuning record: measured seconds by claim, one file per device.

"""
    Tempest.cache_directory()

Where Tempest keeps what it measures: `ENV["TEMPEST_CACHE"]`, or `tempest` in
the first depot.
"""
cache_directory() = get(ENV, "TEMPEST_CACHE", joinpath(first(DEPOT_PATH), "tempest"))

record_path(target) = joinpath(cache_directory(), "tuning", replace(device_name(target), r"[^A-Za-z0-9._-]" => "_") * ".tsv")

function load_record(path)
    record = Dict{String, Float64}()
    isfile(path) || return record
    for line in eachline(path)
        key, value = rsplit(line, '\t'; limit = 2)
        record[key] = parse(Float64, value)
    end
    record
end

function save_record(path, key, seconds)
    mkpath(dirname(path))
    open(io -> println(io, key, '\t', seconds), path, "a")
end
