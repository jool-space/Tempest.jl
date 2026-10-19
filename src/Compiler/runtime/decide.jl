# Choosing how each group runs, and which groups there are: the cover.
#
# Every implementation's claims on a group are estimated, as launches with the
# bytes they move, the flops they compute and the blocks they keep busy. The
# device's rates are measured on first use, and each package's products are
# priced on a line fitted to two reference products it runs, both kept in a
# rates file beside the record.
# Implementations also propose candidates of their own, sets of values grown
# from anchors that they accept, and claim calls. A candidate replaces the
# generator's groups, and a claimed call its reference expanded and fused into
# its neighbours, when the cover with it is cheaper. Estimates within each
# other's error are close calls: they are measured at preparation and kept in
# the tuning record, and a measurement overrides the estimate only beyond the
# device's noise margin, so the choice is deterministic given the record.

"""
A device's rates for estimates: seconds per launch in a captured graph, bytes
per second for a launch that moves at most `cache` bytes (which a replayed
launch finds in the cache) and for one that moves more, the blocks that fill
the device, and the relative margin within which two measurements tie.
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
A package's products of one class: a launch of `flops` takes `latency + flops
/ rate` seconds. A borrowed line is another package's, for a package that
claims no reference product.
"""
struct Line
    latency::Float64
    rate::Float64
    borrowed::Bool
end

seconds(w::Work, r::Rates, line) =
    max(r.launch + w.bytes / (w.bytes <= r.cache ? r.cached : r.bandwidth) / min(1.0, w.blocks / r.blocks),
        w.flops == 0 ? 0.0 : line.latency + w.flops / line.rate)

"The class of products of `T` factors: a half type by name, Float32 by the target's precision."
product_class(target::CUDATarget, T) = T <: HalfFloat ? nameof(T) : target.precision
product_class(target, T) = nameof(T)

"The name of the package that registered `implementation`."
package_of(implementation) = nameof(first(m for (m, is) in REGISTERED if implementation in is))

const NOISE = 0.05

# The CPU has no product claims: its lines are one nominal line (see `line!`).
rates(::CPUTarget) = Rates(0.0, 2.0e10, 2.0e10, Inf, 1, NOISE)

"Measure the device's rates on `target` (see CUDAExt)."
function calibrate end

# Two estimates closer than this factor are a close call.
const ESTIMATE_ERROR = 1.5

device_name(::CPUTarget) = "cpu"
"The target's numerics, as part of the tuning record's key."
numerics(::Target) = :default
numerics(target::CUDATarget) = target.online ? target.precision : Symbol(target.precision, "-two-pass")

"Whether a target permits the rewrites of sums that change rounding (see `rewrite`)."
permits_rewrites(::Target) = true
permits_rewrites(target::CUDATarget) = target.online

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

"What `explain` calls an implementation: its package for operation pairs, else its type."
label(i::Implementation) = string(nameof(typeof(i)))
label(k::Kernels) = string(nameof(k.package))

"A way to realize a group: an implementation's claim, or the floor's (`nothing`)."
struct Choice
    implementation::Union{Nothing, Implementation}
    claim::Any
    estimate::Float64
    key::String
    borrowed::Bool # priced on another package's line, which bounds a time only from below
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
    measure::Symbol
    rates::Rates
    calibration::Dict{String, Float64}
    calibration_path::Union{Nothing, String}
    lines::Dict{Tuple{Symbol, Symbol}, Line}
    record::Dict{String, Float64}
    record_path::Union{Nothing, String}
    buffers::Vector{Any}
    built::Dict{String, Any}
    # The choice for each group alike (by `canonical`): its rank among the
    # options, how it was made, and the time measured; and each group's decision.
    decided::Dict{String, Tuple{Int, Symbol, Union{Nothing, Float64}}}
    instances::Dict{Any, Decision}
end

function Decider(target, graph; measure = :close)
    path, calibration_path = record_path(target), rates_path(target)
    calibration = load_record(calibration_path)
    Decider(target, graph, measurable(target) ? level(measure) : :none, device_rates!(target, calibration, calibration_path),
        calibration, calibration_path, Dict{Tuple{Symbol, Symbol}, Line}(), load_record(path), path,
        Any[], Dict{String, Any}(), Dict{String, Tuple{Int, Symbol, Union{Nothing, Float64}}}(), Dict{Any, Decision}())
end

level(measure::Bool) = measure ? :close : :none
level(measure::Symbol) = measure

# A group within one preparation: its kind, members, terms and what it stores.
group_identity(p::Plan, g::Group) = (g.kind, g.domain, Tuple(g.reductions), Tuple(g.outputs),
    Tuple(get(g.terms, m, nothing) for m in members(g)), Tuple(p.stored[m] for m in members(g)))

# An opaque value without a term, as its operation computes it: the
# operation, its keywords, and the values it reads.
computed(d::Opaque) = string(op_name(d.node.op), op_detail(d.node.op), d.node.op isa Call ? string(" ", d.node.op.kwargs) : "",
    " of ", join(map(o -> "%$(o.value)", reads(d)), " "))

# A group across programs: value numbers in order of appearance, with each
# value's type and shape.
function canonical(p::Plan, g::Group)
    io = IOBuffer()
    print(io, g.kind, " ", g.domain)
    for m in members(g)
        v = p.graph.values[m]
        what = m in g.reductions ? string(fold_name(v.definition.fold), v.definition.reduced) : "value"
        print(io, "; %", m, " ", what, " ", v.eltype, v.dims, p.stored[m] ? " stored" : "", " = ", haskey(g.terms, m) ? g.terms[m] : computed(v.definition))
    end
    numbers = Dict{String, Int}()
    text = replace(String(take!(io)), r"%\d+" => s -> string("%", get!(numbers, s, length(numbers) + 1)))
    shapes = map(sort!(collect(numbers); by = last)) do (s, k)
        v = p.graph.values[parse(Int, s[2:end])]
        string("%", k, " ", v.eltype, v.dims)
    end
    string(text, "; ", join(shapes, ", "))
end

function choice(d::Decider, p::Plan, g::Group, view, implementation, claim)
    work = implementation === nothing ? floor_estimate(p, g) : estimate(implementation, claim, d.target, view)
    name = implementation === nothing ? "floor" : record_name(implementation, claim)
    key = replace(string(numerics(d.target), " ", name, " | ", canonical(p, g)), r"[\t\n]" => " ")
    lines = [w.flops == 0 ? nothing : line!(d, package_of(implementation), w.factor) for w in work]
    Choice(implementation, claim, sum(map((w, l) -> seconds(w, d.rates, l), work, lines); init = 0.0), key,
        any(l -> l !== nothing && l.borrowed, lines))
end

"The call an opaque group computes, or nothing."
function called(graph, g::Group)
    g.kind === :opaque || return nothing
    op = graph.values[first(g.outputs)].definition.node.op
    op isa Call ? op : nothing
end

# Every group has a choice but opaque work other than a call, which only the
# floor runs.
chosen(p::Plan, g::Group) = g.kind !== :opaque || called(p.graph, g) !== nothing

# The `Spec` of each argument of a call.
call_specs(graph, operands, shapes) = map((o, (dims, _)) -> Spec(graph.values[o.value].eltype, dims), operands, shapes)

# An implementation's claims on a group, and on a call's group its claims
# through `call_claims`, those that apply every epilogue the group absorbed.
function group_claims(i::Implementation, target, p::Plan, g::Group, view = group_view(p, g))
    cs = claims(i, target, view)
    op = called(p.graph, g)
    op === nothing && return cs
    d = p.graph.values[first(g.outputs)].definition
    taken = [p.graph.values[o].definition.node.op.output for o in keys(g.epilogues)]
    calls = call_claims(i, target, op, call_specs(p.graph, d.operands, d.shapes))
    Any[cs...; [c for c in calls if all(in(c.absorbs), taken)]...]
end

"""
Every claim on a group, cheapest estimate first; the floor when none claims
it, except for a call, which only implementations run.
"""
function options(d::Decider, p::Plan, g::Group)
    view = group_view(p, g)
    os = Choice[choice(d, p, g, view, i, c) for i in implementations(d.target) for c in group_claims(i, d.target, p, g, view)]
    if isempty(os)
        op = called(p.graph, g)
        op === nothing || throw(ArgumentError("no implementation on $(d.target) runs $(op.f), an operation without a reference"))
        push!(os, choice(d, p, g, view, nothing, nothing))
    end
    sort!(os; by = o -> o.estimate) # stable: ties keep the implementations' order
end

close_call(a, b) = max(a, b) <= ESTIMATE_ERROR * min(a, b)

"""
Choose how a group runs: the cheapest estimate, unless measurements of the
close calls disagree beyond the noise. Groups alike choose once, by rank among
their options, which each group's own claims realize: a claim holds the ids of
its group's values, so one taken from another group would build on its buffers.
"""
function decide(d::Decider, p::Plan, g::Group)
    get!(d.instances, group_identity(p, g)) do
        os = options(d, p, g)
        k, how, measured = get!(() -> choose(d, p, g, os), d.decided, canonical(p, g))
        Decision(os[k], os, how, measured)
    end
end

# The rank of the choice among options `os`, sorted by estimate; how it was
# made; and its measured time. A rank and not a claim, because claims hold
# value ids.
function choose(d::Decider, p::Plan, g::Group, os)
    best = first(os)
    if length(os) == 1
        d.measure === :all && measure!(d, p, g, best)
        return (1, :only, get(d.record, best.key, nothing))
    end
    # Every option under `:all`; otherwise those within the estimate's error of
    # the best, and every option when the best rests on a borrowed line, which
    # bounds a time only from below.
    ranks = d.measure === :all || best.borrowed ? collect(eachindex(os)) :
        [k for (k, o) in enumerate(os) if o.borrowed || close_call(o.estimate, best.estimate)]
    length(ranks) == 1 && return (1, :estimated, get(d.record, best.key, nothing))
    d.measure === :none || foreach(k -> measure!(d, p, g, os[k]), ranks)
    times = [get(d.record, os[k].key, nothing) for k in ranks]
    any(isnothing, times) && return (1, :estimated, nothing)
    k = argmin(times)
    k = times[k] < (1 - d.rates.noise) * times[1] ? k : 1
    (ranks[k], :measured, times[k])
end

decisions(d::Decider, p::Plan) = [chosen(p, g) ? decide(d, p, g) : nothing for g in p.groups]

"""
How a plan compares with the one before it on the groups they do not share:
the time of each side, measured when every one of those groups is and the
sums differ beyond the noise, else estimated. `sides` holds each side's
groups, with their plans and decisions, as `(old, new)`.
"""
struct Comparison
    old::Float64
    new::Float64
    how::Symbol # :estimated or :measured
    sides::Tuple{Vector, Vector}
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
    if d.measure === :all || (d.measure === :close && close_call(ea, eb))
        foreach(((p, g, x),) -> measure!(d, p, g, x.choice), [da; db])
    end
    ta = [get(d.record, x[3].choice.key, nothing) for x in da]
    tb = [get(d.record, x[3].choice.key, nothing) for x in db]
    if !any(isnothing, ta) && !any(isnothing, tb)
        ma, mb = sum(ta; init = 0.0), sum(tb; init = 0.0)
        abs(ma - mb) > d.rates.noise * min(ma, mb) && return Comparison(ma, mb, :measured, (da, db))
    end
    Comparison(ea, eb, :estimated, (da, db))
end

"""
    ClaimedCall(site, epilogues)

A call claimed in place of its reference expanded (an opaque call, in place of
its claim without epilogues): the call's place in the graph, and the results
after which the claimant applies an epilogue. An implementation may propose
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
            # The read over the node's shape, over its domain.
            index = narrow(r, narrowing(n.dims, wide_axes(n.dims))).index
            values[r.value] = Value(Opaque(n, site.operands, site.shapes, k in c.epilogues, index), v.dims, v.eltype, v.scalar)
        end
    end
    IndexGraph(values, graph.arguments, graph.results, graph.result_dims, graph.result_scalar, graph.result_eltype, graph.calls, graph.rewrites)
end

# The calls an implementation claims, with the epilogues each claim applies. An
# opaque call is proposed only with epilogues: without, its group is in every plan.
function call_proposals(d::Decider, i::Implementation)
    graph = d.graph
    proposals = ClaimedCall[]
    for site in graph.calls
        op = first(site.results).op
        for c in call_claims(i, d.target, op, call_specs(graph, site.operands, site.shapes))
            epilogues = Tuple(k for k in eachindex(site.results) if k in c.absorbs)
            opaque(op.f) && isempty(epilogues) && continue
            q = ClaimedCall(site, epilogues)
            q in proposals || push!(proposals, q)
        end
    end
    proposals
end

"A proposal the cover compared with its plan at the time, and whether it took it."
struct Considered
    proposal::Any # a set of values, a `ClaimedCall`, or a `Join`
    implementation::Union{Nothing, Implementation} # nothing for the planner's own
    comparison::Comparison
    accepted::Bool
end

"""
What a plan's neighbourhoods need, found once per plan: the values each
group owns, where the planner forms each group (a fixed set's index, a fold's
seed or a pointwise seed), the groups that read each value, and the values
that read each.
"""
struct Around
    owned::Vector{Vector{Int}}
    places::Vector{Union{Nothing, Tuple{Symbol, Int}}}
    reading::Vector{Vector{Int}}
    followers::Vector{Vector{Int}}
end

function Around(p::Plan, sets)
    owned = [Int[] for _ in p.groups]
    for (v, k) in enumerate(p.owner)
        k == 0 || push!(owned[k], v)
    end
    function place(j)
        g = p.groups[j]
        i = findfirst(s -> p.owner[first(s)] == j, sets)
        i === nothing || return (:fixed, i)
        g.kind === :reduction ? (:fold, first(g.reductions)) : g.kind === :pointwise ? (:pointwise, minimum(owned[j])) : nothing
    end
    reading = [Int[] for _ in p.graph.values]
    for (j, g) in enumerate(p.groups), r in group_reads(p.graph, g)
        push!(reading[r], j)
    end
    followers = [Int[] for _ in p.graph.values]
    for (id, v) in enumerate(p.graph.values), o in operands(v.definition), r in read_values(o)
        push!(followers[r], id)
    end
    Around(owned, map(place, eachindex(p.groups)), reading, followers)
end

# The groups of `p` that a change to value `v` could change: its own; those
# that could grow into it, owning or reading what it loads; and those that
# read it, directly or through values not stored.
function neighbours(p::Plan, a::Around, v)
    v > length(p.owner) && return Int[]
    found = p.owner[v] == 0 ? Int[] : [p.owner[v]]
    pending, seen = [v], Set{Int}()
    while !isempty(pending)
        for o in operands(p.graph.values[pop!(pending)].definition), r in read_values(o)
            r in seen && continue
            push!(seen, r)
            if p.stored[r] || p.owner[r] != 0
                p.owner[r] == 0 || push!(found, p.owner[r])
                append!(found, a.reading[r])
            else
                push!(pending, r)
            end
        end
    end
    pending, seen = [v], Set{Int}()
    while !isempty(pending)
        for w in a.followers[pop!(pending)]
            w in seen && continue
            push!(seen, w)
            p.owner[w] == 0 || push!(found, p.owner[w])
            p.stored[w] || push!(pending, w)
        end
    end
    found
end

"A stored value joined into the group of a kernel that reads it: a proposal of the planner's own."
struct Join
    value::Int
    group::Vector{Int}
end

"The values a proposal fixes as one group, or nothing for a call."
fixes(q::Vector{Int}) = q
fixes(q::Join) = q.group
fixes(q::ClaimedCall) = nothing

"The values a proposal touches."
touches(q) = q isa ClaimedCall ? [q.site.values; [r.value for r in q.site.reads]] : fixes(q)

"""
A proposal's place in a plan, by what it touches: the groups around its
values, in the order the planner forms them, and its values among them,
numbered as `canonical` numbers one group's. Proposals with one key in the
generator's plan are the same edit in identical layers.
"""
function proposal_key(p::Plan, a::Around, q)
    around = Set{Int}()
    foreach(v -> union!(around, neighbours(p, a, v)), touches(q))
    io = IOBuffer()
    for j in sort!(collect(around); by = j -> something(a.places[j], (:opaque, minimum(a.owned[j]))))
        g = p.groups[j]
        print(io, "| ", g.kind, " ", g.domain)
        for m in members(g)
            v = p.graph.values[m]
            what = m in g.reductions ? string(fold_name(v.definition.fold), v.definition.reduced) : "value"
            print(io, "; %", m, " ", what, " ", v.eltype, v.dims, p.stored[m] ? " stored" : "", " = ", haskey(g.terms, m) ? g.terms[m] : computed(v.definition))
        end
    end
    q isa Join && print(io, " || join %", q.value)
    q isa ClaimedCall && print(io, " || call of ", first(q.site.results).op.f, " ", first(q.site.results).op.kwargs, " absorbing ", q.epilogues)
    print(io, " || ", join(map(v -> "%$v", touches(q)), " "))
    numbers = Dict{String, Int}()
    text = replace(String(take!(io)), r"%\d+" => s -> string("%", get!(numbers, s, length(numbers) + 1)))
    shapes = map(sort!(collect(numbers); by = last)) do (s, k)
        v = parse(Int, s[2:end])
        string("%", k, " ", p.graph.values[v].eltype, p.graph.values[v].dims, p.stored[v] ? " stored" : "")
    end
    string(text, "; ", join(shapes, ", "))
end

# Every stored value of a kernel that another kernel reads, joined into the
# reader's group: a choice the planner's rules make, offered to the cover.
function joins(p::Plan)
    a = Around(p, Vector{Int}[])
    found = Join[]
    for (k, g) in enumerate(p.groups)
        g.kind in (:reduction, :pointwise) || continue
        for v in sort!(collect(group_reads(p.graph, g)))
            j = p.owner[v]
            j == 0 || j == k || p.groups[j].kind === :opaque || push!(found, Join(v, sort!([a.owned[k]; v])))
        end
    end
    found
end

"""
    cover(decider) -> (plan, considered)

The generator's groups, with the proposals implementations make, the calls
they claim, and the planner's own joins accepted greedily by estimated gain
over the generator's plan, then by their values, each when its proposer
claims the group it forms (any claim, or the floor, for a join). Proposals
alike in the generator's plan (`proposal_key`) are one edit
in identical layers: priced once, and taken together. A proposal is priced
again in the current plan when an edit taken already changed a group around
it. Each trial plans the whole graph, and is compared with the plan it
changes on the groups that differ. Every proposal compared is returned, taken
or not.
"""
function cover(d::Decider)
    base, keep = generator_plan(d)
    proposals = Tuple{Any, Union{Nothing, Implementation}}[]
    for i in implementations(d.target), q in [call_proposals(d, i); candidates(i, d.target, base)]
        any(x -> x[1] == q, proposals) || push!(proposals, (q, i))
    end
    for q in joins(base)
        any(x -> fixes(x[1]) == q.group, proposals) || push!(proposals, (q, nothing))
    end
    arounds = IdDict{Plan, Around}()
    around(p, taken) = get!(() -> Around(p, Vector{Int}[fixes(x) for x in taken if !(x isa ClaimedCall)]), arounds, p)
    function trial(current, taken, qs)
        all = Any[taken; qs]
        sets = Vector{Int}[fixes(x) for x in all if !(x isa ClaimedCall)]
        calls = ClaimedCall[x for x in all if x isa ClaimedCall]
        # A value belongs to one proposal.
        allunique(Iterators.flatten([sets; [c.site.values for c in calls]])) || return nothing
        try
            first(claimed_plan(d, claimed(d.graph, calls); fixed = sets, keep))
        catch e
            e isa NotAGroup || rethrow()
            nothing
        end
    end
    # A proposal's comparison with the plan it is priced in, and the
    # implementation that claims its group: by key, once.
    priced = Dict{String, Any}()
    price(current, taken, q, i, key) = get!(priced, key) do
        p = trial(current, taken, Any[q])
        p === nothing && return nothing
        j = i === nothing ? nothing : claimant(d, p, q, i)
        i === nothing || j !== nothing || return nothing
        (compare(d, p, current), j)
    end
    a = around(base, Any[])
    ranked = Tuple{Float64, Any, Union{Nothing, Implementation}, String}[]
    for (q, i) in proposals
        k = proposal_key(base, a, q)
        x = price(base, Any[], q, i, k)
        x === nothing || push!(ranked, (x[1].old - x[1].new, q, i, k))
    end
    order(q) = q isa ClaimedCall ? (1, [minimum(q.site.values); q.epilogues...]) : (0, fixes(q))
    sort!(ranked; by = r -> (-r[1], order(r[2])))
    near(q) = Set(j for v in touches(q) for j in neighbours(base, a, v))
    current, considered, taken, done = base, Considered[], Any[], Set{Int}()
    kept = Set(group_identity(base, g) for g in base.groups)
    for (n, (_, _, _, k)) in enumerate(ranked)
        n in done && continue
        class = [m for m in n:length(ranked) if !(m in done) && ranked[m][4] == k]
        union!(done, class)
        # Those around which an edit taken already changed a group are priced
        # again in the current plan, by the first of them; the rest keep
        # their price.
        moved(m) = any(j -> !(group_identity(base, base.groups[j]) in kept), near(ranked[m][2]))
        taking = Considered[]
        for part in (filter(!moved, class), filter(moved, class))
            isempty(part) && continue
            _, q, i, key = ranked[first(part)]
            x = moved(first(part)) ? price(current, taken, q, i, proposal_key(current, around(current, taken), q)) : priced[key]
            x === nothing && continue
            c, j = x
            cs = [Considered(ranked[m][2], j, c, cheaper(c)) for m in part]
            cheaper(c) ? append!(taking, cs) : append!(considered, cs)
        end
        isempty(taking) && continue
        p = trial(current, taken, Any[c.proposal for c in taking])
        if p === nothing
            # Proposals alike that share values are taken one at a time, each
            # compared with the plan as it then is; one that no longer forms
            # a group is not compared.
            for t in taking
                p = trial(current, taken, Any[t.proposal])
                p === nothing && continue
                c = compare(d, p, current)
                push!(considered, Considered(t.proposal, t.implementation, c, cheaper(c)))
                cheaper(c) && (current = p; push!(taken, t.proposal))
            end
        else
            current = p
            append!(considered, taking)
            append!(taken, [c.proposal for c in taking])
        end
        kept = Set(group_identity(current, g) for g in current.groups)
    end
    current, considered
end

function explain(io::IO, p::Plan, c::Considered)
    q, t, graph = c.proposal, c.comparison, p.graph
    μs(x) = string(round(1.0e6 * x; sigdigits = 3), " μs")
    times = c.accepted ? (t.new, t.old) : (t.old, t.new)
    why = string(" (the kernels that differ: ", μs(times[1]), " against ", μs(times[2]), ", ", t.how, ")")
    if q isa Join
        c.accepted && println(io, "%", q.value, " joins the kernel of %", first(v for v in q.group if v != q.value), ", which reads it", why)
        return
    end
    name = label(c.implementation)
    if q isa Vector{Int}
        c.accepted && println(io, "candidate ", join(map(v -> "%$v", q), " "), " from ", name, " replaces the generator's grouping", why)
        return
    end
    site = q.site
    f = first(site.results).op.f
    v = first(site.reads).value
    epilogue = c.accepted && !isempty(p.groups[p.owner[v]].epilogues)
    claim = string(name, "'s kernel", epilogue ? " with an epilogue" : "")
    instead = opaque(f) ? "its kernel without an epilogue" :
        any(s -> s === site, graph.calls) ? "its reference expanded" : "the computation as written"
    print(io, "call of ", f, " (", join(map(r -> "%$(r.value)", site.reads), " "), "): ")
    println(io, c.accepted ? string(claim, " replaces ", instead) : string(instead, " is kept over ", claim), why)
end

# Whether any implementation the target uses recomputes folds where it reads them.
any_recomputes_folds(target) = any(i -> recomputes_folds(i, target), implementations(target))

recomputed_in(g; nested = true) =
    Set(t.value for m in members(g) if haskey(g.terms, m) for t in subterms((g.terms[m],); nested) if t isa Recomputed)

# A plan of `graph`, folds recomputed where they are read when an
# implementation recomputes them, and the folds kept stored: those in `keep`,
# and the outermost ones recomputed in a group no implementation claims (folds
# they recompute in turn may still be recomputed in their kernels). The
# generator's plan is the plan of the program as written; the cover's trials
# fix proposals, which leave other folds to recompute.
generator_plan(d::Decider) = claimed_plan(d, d.graph)

function claimed_plan(d::Decider, graph; fixed = Vector{Int}[], keep = Set{Int}())
    recompute_folds = any_recomputes_folds(d.target)
    keep = Set{Int}(keep)
    has_claim(view) = any(i -> !isempty(claims(i, d.target, view)), implementations(d.target))
    while true
        p = plan(graph; fixed, recompute_folds, keep)
        unclaimed = [g for g in p.groups if !isempty(recomputed_in(g)) && !has_claim(group_view(p, g))]
        isempty(unclaimed) && return p, keep
        foreach(g -> union!(keep, recomputed_in(g; nested = false)), unclaimed)
    end
end

# The implementation that claims the group a proposal of `i` forms: `i`, or for
# a call another of its package's, as when a written-out matcher proposes what
# the package's kernels run.
function claimant(d::Decider, p::Plan, q, i)
    v = q isa ClaimedCall ? first(q.site.reads).value : first(q)
    p.owner[v] == 0 && return nothing
    g = p.groups[p.owner[v]]
    for j in (q isa ClaimedCall ? registered_with(i) : (i,))
        isempty(group_claims(j, d.target, p, g)) || return j
    end
    nothing
end

"The implementations registered under the package of `i`, in order."
registered_with(i::Implementation) = first(is for (_, is) in REGISTERED if any(j -> j === i, is))

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

# Build a choice on `buffers`.
build_choice(target, p::Plan, g::Group, o::Choice, buffers) =
    o.implementation === nothing ? build_floor(target, p, g, buffers) :
    o.claim isa KernelClaim ? build_kernel(target, p, g, o.claim, buffers) :
    build_claim(o.implementation, o.claim, target, group_view(p, g), buffers)

# Build a choice on the decider's buffers, once, to measure it: its key names
# the group alike, so the first of them builds for all.
build_choice!(d::Decider, p::Plan, g::Group, o::Choice) =
    get!(() -> build_choice(d.target, p, g, o, buffers!(d, p, g)), d.built, o.key)

# A group's kernels: its choice, or the next claim when a build fails, and
# the floor last; a call's group has no floor. Opaque work other than calls,
# a scatter, runs on the floor.
function build_group(d::Decider, p::Plan, g::Group, x, buffers)
    x === nothing && return build_floor(d.target, p, g, buffers)
    failure = nothing
    for o in [x.choice; filter(!=(x.choice), x.options)]
        o.implementation === nothing && break
        built = try
            build_choice(d.target, p, g, o, buffers)
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
    # A claim that fails to build loses every comparison in this preparation;
    # the record keeps only times.
    launch = try
        first(build_choice!(d, p, g, o))
    catch err
        @debug "claim failed to build; it loses" o.implementation exception = err
        d.record[o.key] = Inf
        return
    end
    t = time_launch(d.target, launch)
    d.record[o.key] = t
    save_record(d.record_path, o.key, t)
end

"""
The line of `package` for products of `T` factors on the decider's target,
fitted on first need and kept in the rates file. A package that claims no
reference product borrows the class's fastest line.
"""
function line!(d::Decider, package, T)
    class = product_class(d.target, T)
    get!(d.lines, (package, class)) do
        measurable(d.target) || return Line(0.0, 1.0e10, false)
        key(p, what) = string(p, " ", class, " ", what)
        own(p) = Line(d.calibration[key(p, "latency")], d.calibration[key(p, "rate")], false)
        haskey(d.calibration, key(package, "rate")) || fit!(d, T)
        haskey(d.calibration, key(package, "rate")) && return own(package)
        fastest = argmax(l -> l.rate, [own(nameof(m)) for m in used_packages(d.target) if haskey(d.calibration, key(nameof(m), "rate"))])
        Line(fastest.latency, fastest.rate, true)
    end
end

const REFERENCE_SIZES = (256, 2048)

"""
Fit the line of every package on the decider's target that has none for
products of `T` factors: the fastest of the package's claims on `a * b` at
each reference size, timed as claims are, and the line through the two times.
"""
function fit!(d::Decider, T)
    class = product_class(d.target, T)
    unfitted = [m for m in used_packages(d.target) if !haskey(d.calibration, string(nameof(m), " ", class, " rate"))]
    times = map(REFERENCE_SIZES) do n
        graph = lower(trace(*, zeros(T, n, n), zeros(T, n, n)))
        reference = Decider(d.target, graph; measure = :none)
        p = plan(graph)
        g = only(p.groups)
        view = group_view(p, g)
        fastest = Dict{Symbol, Float64}()
        for (m, is) in REGISTERED, i in is, c in (m in unfitted ? claims(i, d.target, view) : ())
            t = time_launch(d.target, first(build_claim(i, c, d.target, view, buffers!(reference, p, g))))
            fastest[nameof(m)] = min(get(fastest, nameof(m), Inf), t)
        end
        fastest
    end
    small, large = 2.0 .* REFERENCE_SIZES .^ 3
    for (package, t) in times[1]
        rate = (large - small) / (times[2][package] - t)
        keep!(d.calibration, d.calibration_path, string(package, " ", class, " latency"), t - small / rate)
        keep!(d.calibration, d.calibration_path, string(package, " ", class, " rate"), rate)
    end
end

"Fit the lines of every package on `target` for Float32, Float16 and BFloat16 factors now, rather than on first need."
function calibrate!(target)
    d = Decider(target, lower(trace(identity, zeros(Float32, 1))); measure = :none)
    foreach(T -> foreach(m -> line!(d, nameof(m), T), used_packages(target)), (Float32, Float16, BFloat16))
end

## The tuning record: measured seconds by claim, one file per device.

using Scratch: @get_scratch!

# Tempest's `tuning` scratch space in the depot. Its path is found when Tempest
# loads, so it is not part of the precompiled image; while Tempest precompiles
# there is none, and its workload records nothing unless an environment
# variable names a directory.
const TUNING = Ref{Union{Nothing, String}}(nothing)

__init__() = (TUNING[] = @get_scratch!("tuning"); nothing)

"""
    Tempest.Compiler.tuning_directory() -> directory or nothing

Where Tempest keeps what it measures: `tuning` in `ENV["TEMPEST_CACHE"]` when
set, otherwise Tempest's `tuning` scratch space.
"""
function tuning_directory()
    dir = get(ENV, "TEMPEST_CACHE", nothing)
    dir === nothing ? TUNING[] : joinpath(dir, "tuning")
end

device_file(target, suffix) = (dir = tuning_directory(); dir === nothing ? nothing :
    joinpath(dir, replace(device_name(target), r"[^A-Za-z0-9._-]" => "_") * suffix))
record_path(target) = device_file(target, ".tsv")
rates_path(target) = device_file(target, ".rates.tsv")

function load_record(path)
    record = Dict{String, Float64}()
    path !== nothing && isfile(path) || return record
    for line in eachline(path)
        key, value = rsplit(line, '\t'; limit = 2)
        record[key] = parse(Float64, value)
    end
    record
end

function save_record(path, key, seconds)
    path === nothing && return
    mkpath(dirname(path))
    open(io -> println(io, key, '\t', seconds), path, "a")
end

keep!(calibration, path, key, value) = (calibration[key] = value; save_record(path, key, value))

# The device's rates from its rates file, measured and kept there first when
# the file has none. A target that cannot measure has fixed rates.
function device_rates!(target, calibration, path)
    measurable(target) || return rates(target)
    haskey(calibration, "launch") || foreach(((k, v),) -> keep!(calibration, path, string(k), v), pairs(calibrate(target)))
    c = calibration
    Rates(c["launch"], c["cached"], c["bandwidth"], c["cache"], Int(c["blocks"]), NOISE)
end

