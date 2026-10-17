# Choosing how each group runs, and which groups there are: the cover.
#
# Every implementation's claims on a group are estimated, as launches with the
# bytes they move, the flops they compute and the blocks they keep busy.
# Implementations also propose candidates of their own, sets of values grown
# from anchors that they accept, and claim calls. A candidate replaces the
# generator's groups, and a claimed call its reference expanded and fused into
# its neighbours, when the cover with it is cheaper. Estimates within each
# other's error are close calls: they are measured at preparation and kept in
# the tuning record, and a measurement overrides the estimate only beyond the
# device's noise margin, so the choice is deterministic given the record.

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
    candidates(implementation, target, plan) -> Vector

What the implementation proposes in place of the generator's groups: sets of
values to realize as one group each, grown from anchors in `plan` (a product)
and accepted by the implementation, or `ClaimedCall`s of computations it
recognizes as calls it claims. None by default.
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

"""
A claim on a call's group made through `claims_call`; the implementation's
`build_call` builds it.
"""
struct CallClaim
    f::Any
    kwargs::NamedTuple
end

Base.show(io::IO, c::CallClaim) = print(io, "call of ", c.f, isempty(c.kwargs) ? "" : string(c.kwargs))

"The call an opaque group computes, or nothing."
function called(graph, g::Group)
    g.kind === :opaque || return nothing
    op = graph.values[first(g.outputs)].definition.node.op
    op isa Call ? op : nothing
end

# Every group has a choice but opaque work other than a call, which only the
# floor runs.
chosen(p::Plan, g::Group) = g.kind !== :opaque || called(p.graph, g) !== nothing

# The `(dims, eltype)` of each argument of a call.
call_inputs(graph, operands, shapes) = map((o, (dims, _)) -> (dims, graph.values[o.value].eltype), operands, shapes)

# An implementation's claims on a group, and on a call's group its claim
# through `claims_call`, when it applies every epilogue the group absorbed.
function group_claims(i::Implementation, target, p::Plan, g::Group)
    cs = claims(i, target, p, g)
    op = called(p.graph, g)
    op === nothing && return cs
    d = p.graph.values[first(g.outputs)].definition
    claims_call(i, target, op.f, op.kwargs, call_inputs(p.graph, d.operands, d.shapes)) || return cs
    taken = epilogue_results(i, op.f, op.kwargs)
    all(o -> p.graph.values[o].definition.node.op.output in taken, keys(g.epilogues)) || return cs
    Any[cs...; CallClaim(op.f, op.kwargs)]
end

"Every claim on a group, cheapest estimate first; the floor when none claims it."
function options(d::Decider, p::Plan, g::Group)
    os = Choice[choice(d, p, g, i, c) for i in implementations(d.target) for c in group_claims(i, d.target, p, g)]
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

decisions(d::Decider, p::Plan) = [chosen(p, g) ? decide(d, p, g) : nothing for g in p.groups]

"""
How a plan compares with the one before it on the groups they do not share:
the time of each side, measured when every one of those groups is and the
sums differ beyond the noise, else estimated.
"""
struct Comparison
    old::Float64
    new::Float64
    how::Symbol # :estimated or :measured
end

cheaper(c::Comparison) = c.new < c.old

# Compare plan `b` with plan `a`. Close calls are measured.
function compare(d::Decider, b::Plan, a::Plan)
    ia = Set(group_identity(a, g) for g in a.groups)
    ib = Set(group_identity(b, g) for g in b.groups)
    only_a = [(a, g) for g in a.groups if !(group_identity(a, g) in ib) && chosen(a, g)]
    only_b = [(b, g) for g in b.groups if !(group_identity(b, g) in ia) && chosen(b, g)]
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
        abs(ma - mb) > d.rates.noise * min(ma, mb) && return Comparison(ma, mb, :measured)
    end
    Comparison(ea, eb, :estimated)
end

"""
    ClaimedCall(site, epilogues)

A call claimed in place of its reference expanded: the call's place in the
graph, and the results after which the claimant applies an epilogue. An implementation may propose
one among its `candidates` for a computation it recognizes (see `call_site`).
"""
struct ClaimedCall
    site::CallSite
    epilogues::Tuple{Vararg{Int}}
end

"""
    call_site(graph, f, kwargs, values, arguments, results) -> CallSite

A computation written out in `graph` as a call of `f` with keywords `kwargs`,
for an implementation that recognizes it: `values` compute it from the values
`arguments`, read whole, and the values `results` hold what `f` returns, in
order.
"""
function call_site(graph, f, kwargs, values, arguments, results)
    whole(v) = Read(v, identity_index(graph.values[v].dims))
    structure = length(results) == 1 ? 1 : Tuple(eachindex(results))
    nodes = map(enumerate(results)) do (k, r)
        v = graph.values[r]
        Node(Call(f, kwargs, k, -first(results), structure), (), v.dims, v.eltype, v.scalar)
    end
    shapes = Tuple(map(a -> (graph.values[a].dims, graph.values[a].scalar), arguments))
    CallSite(collect(values), Tuple(map(whole, arguments)), shapes, nodes, map(whole, results))
end

"""
    claimed(graph, calls) -> IndexGraph

The graph with each claimed call computing its own results: opaque values that
read its arguments, in place of the values its reference expanded to, which
nothing then reads.
"""
function claimed(graph::IndexGraph, calls)
    isempty(calls) && return graph
    values = copy(graph.values)
    for c in calls
        site = c.site
        for (k, (n, r)) in enumerate(zip(site.results, site.reads))
            v = values[r.value]
            values[r.value] = Value(Opaque(n, site.operands, site.shapes, k in c.epilogues, r.index), v.dims, v.eltype, v.scalar)
        end
    end
    IndexGraph(values, graph.arguments, graph.results, graph.result_dims, graph.result_scalar, graph.result_eltype, graph.calls)
end

# The calls an implementation claims, with the epilogues it applies.
function call_proposals(d::Decider, i::Implementation)
    graph = d.graph
    proposals = ClaimedCall[]
    for site in graph.calls
        op = first(site.results).op
        claims_call(i, d.target, op.f, op.kwargs, call_inputs(graph, site.operands, site.shapes)) || continue
        epilogues = Tuple(k for k in eachindex(site.results) if k in epilogue_results(i, op.f, op.kwargs))
        push!(proposals, ClaimedCall(site, epilogues))
    end
    proposals
end

"A proposal the cover compared with its plan at the time, and whether it took it."
struct Considered
    proposal::Any # a set of values, or a `ClaimedCall`
    implementation::Implementation
    comparison::Comparison
    accepted::Bool
end

"""
    cover(decider) -> (plan, considered)

The generator's groups, with the candidates implementations propose and the
calls they claim accepted greedily: by estimated gain over the generator's
plan, then by their values, each when its proposer claims the group it forms
and the plan with it is cheaper than without. Every proposal compared is
returned, taken or not.
"""
function cover(d::Decider)
    base, keep = generator_plan(d)
    proposals = Tuple{Any, Implementation}[]
    for i in implementations(d.target), q in [call_proposals(d, i); candidates(i, d.target, base)]
        any(x -> x[1] == q, proposals) || push!(proposals, (q, i))
    end
    recompute_folds = recomputes_folds(d.target)
    function trial(qs)
        sets = Vector{Int}[q for q in qs if q isa Vector{Int}]
        calls = ClaimedCall[q for q in qs if q isa ClaimedCall]
        # A value belongs to one proposal.
        allunique(Iterators.flatten([sets; [c.site.values for c in calls]])) || return nothing
        try
            plan(claimed(d.graph, calls); fixed = sets, recompute_folds, keep)
        catch e
            e isa NotAGroup || rethrow()
            nothing
        end
    end
    total(p) = sum(g -> decide(d, p, g).choice.estimate, filter(g -> chosen(p, g), p.groups); init = 0.0)
    ranked = map(proposals) do (q, i)
        p = trial([q])
        (p === nothing || !proposer_claims(d, p, q, i)) ? nothing : (total(base) - total(p), q, i)
    end
    order(q) = q isa ClaimedCall ? (1, [minimum(q.site.values); q.epilogues...]) : (0, q)
    ranked = sort!(filter(!isnothing, ranked); by = r -> (-r[1], order(r[2])))
    current, considered = base, Considered[]
    for (_, q, i) in ranked
        p = trial(Any[[c.proposal for c in considered if c.accepted]; [q]])
        p === nothing && continue
        proposer_claims(d, p, q, i) || continue
        c = compare(d, p, current)
        push!(considered, Considered(q, i, c, cheaper(c)))
        cheaper(c) && (current = p)
    end
    current, considered
end

function explain(io::IO, p::Plan, c::Considered)
    q, t, graph = c.proposal, c.comparison, p.graph
    name = nameof(typeof(c.implementation))
    μs(x) = string(round(1.0e6 * x; sigdigits = 3), " μs")
    times = c.accepted ? (t.new, t.old) : (t.old, t.new)
    why = string(" (the kernels that differ: ", μs(times[1]), " against ", μs(times[2]), ", ", t.how, ")")
    if q isa Vector{Int}
        c.accepted && println(io, "candidate ", join(map(v -> "%$v", q), " "), " from ", name, " replaces the generator's grouping", why)
        return
    end
    site = q.site
    f = first(site.results).op.f
    v = first(site.reads).value
    epilogue = c.accepted && !isempty(p.groups[p.owner[v]].epilogues)
    claim = string(name, "'s claim", epilogue ? " with an epilogue" : "")
    instead = any(s -> s === site, graph.calls) ? "its reference expanded" : "the computation as written"
    print(io, "call of ", f, " (", join(map(r -> "%$(r.value)", site.reads), " "), "): ")
    println(io, c.accepted ? string(claim, " replaces ", instead) : string(instead, " is kept over ", claim), why)
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

function proposer_claims(d::Decider, p::Plan, q, implementation)
    v = q isa ClaimedCall ? first(q.site.reads).value : first(q)
    p.owner[v] == 0 && return false
    !isempty(group_claims(implementation, d.target, p, p.groups[p.owner[v]]))
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
        o.claim isa CallClaim ? build_call(o.implementation, p, g, buffers, d.target) :
        build_claim(o.implementation, o.claim, d.target, p, g, buffers)
    end
end

# A group's kernels: its choice, or the next claim when a build fails, and
# the floor last; a call's group has no floor. Opaque work other than calls
# runs on the host.
function build_group(d::Decider, p::Plan, g::Group, x, buffers)
    x === nothing && return opaque_launch(p, g, buffers)
    failure = nothing
    for o in [x.choice; filter(!=(x.choice), x.options)]
        o.implementation === nothing && break
        built = try
            build_choice!(d, p, g, o)
        catch err
            @debug "claim failed to build; trying the next" o.implementation exception = err
            failure = err
            nothing
        end
        built === nothing || return built
    end
    op = called(p.graph, g)
    op === nothing || error("no claim on the call of $(op.f) built: $(sprint(showerror, failure))")
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
