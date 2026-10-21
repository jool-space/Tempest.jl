# How a group runs. Every implementation's claims on a group are estimated,
# as launches with the bytes they move, the flops they compute and the blocks
# they keep busy, and so is the floor's. Estimates within each other's error
# are close calls: they are measured at preparation and kept in the tuning
# record, and a measurement overrides the estimate only beyond the device's
# noise margin, so the choice is deterministic given the record. An
# implementation may stage its claims: the variants it offers of the claim
# that measured best are tried next, until the best has none new. Groups alike
# choose once.

# Two estimates closer than this factor are a close call.
const ESTIMATE_ERROR = 1.5

"What `explain` calls an implementation: its type."
label(i::Implementation) = string(nameof(typeof(i)))

"A way to realize a group: an implementation's claim, or the floor's (`nothing`)."
struct Choice
    implementation::Union{Nothing, Implementation}
    claim::Any
    estimate::Float64
    key::String
    borrowed::Bool # priced on another package's line, which bounds a time only from below
    order::Int # the claim's place among its implementation's claims on the group, its own ranking of them; a variant's is past its parent's
    variants::Vector{Any} # the claims its implementation offers once it measures best (see `Tempest.Implement.variants`)
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
    memories::Dict{Symbol, Memory}
    # Seconds per position of each scalar function on the device, measured
    # once, in each regime: streamed, swept in a product of a class, swept in
    # a product of a class that recomputes a fold.
    applied::Dict{Tuple{Symbol, Symbol, Any}, Float64}
    record::Dict{String, Float64}
    record_path::Union{Nothing, String}
    # Folds pinned to be recomputed where they are read, in every plan.
    pinned::Set{Int}
    buffers::Vector{Any}
    built::Dict{String, Any}
    # The choice for each group alike (by `canonical`): its rank among the
    # options, how it was made, and the time measured; and each group's decision.
    decided::Dict{String, Tuple{Int, Symbol, Union{Nothing, Float64}}}
    instances::Dict{Any, Decision}
    # Each group's options (by `group_identity`), found before its decision
    # when its kernels are built ahead (see `prebuild!`).
    options::Dict{Any, Vector{Choice}}
end

function Decider(target, graph; measure = :close, recompute = ())
    dry = level(measure) === :dry
    path, calibration_path = dry ? (nothing, nothing) : (record_path(target), rates_path(target))
    calibration = load_record(calibration_path)
    r = dry ? rates(target) : device_rates!(target, calibration, calibration_path)
    Decider(target, graph, dry ? :dry : measurable(target) ? level(measure) : :none, r, calibration, calibration_path, Dict{Tuple{Symbol, Symbol}, Line}(), Dict{Symbol, Memory}(), Dict{Tuple{Symbol, Symbol, Any}, Float64}(), load_record(path), path,
        Set{Int}(recompute), Any[], Dict{String, Any}(), Dict{String, Tuple{Int, Symbol, Union{Nothing, Float64}}}(), Dict{Any, Decision}(),
        Dict{Any, Vector{Choice}}())
end

# What is measured: nothing (`:none`), the close calls (`:close`) or every
# option (`:all`); `:dry` touches no device at all, pricing claims on nominal
# rates and reading no record, for a preparation that builds nothing.
level(measure::Bool) = measure ? :close : :none
level(measure::Symbol) = measure

"Whether the decider may time claims and fit rates on its target: not when dry."
measures(d::Decider) = d.measure !== :dry && measurable(d.target)

# A group within one preparation: its kind, members, terms and what it stores.
# One concrete type, so that the sets and tables of identities compile once.
struct GroupIdentity
    kind::Symbol
    domain::Tuple{Vararg{Int}}
    reductions::Vector{Int}
    outputs::Vector{Int}
    terms::Vector{Any}
    stored::Vector{Bool}
end

Base.:(==)(a::GroupIdentity, b::GroupIdentity) = all(f -> getfield(a, f) == getfield(b, f), fieldnames(GroupIdentity))
Base.hash(i::GroupIdentity, h::UInt) = foldl((h, f) -> hash(getfield(i, f), h), fieldnames(GroupIdentity); init = h)

group_identity(p::Plan, g::Group) = GroupIdentity(g.kind, g.domain, copy(g.reductions), copy(g.outputs),
    Any[get(g.terms, m, nothing) for m in members(g)], Bool[p.stored[m] for m in members(g)])

# An opaque value without a term, as its operation computes it: the
# operation and the values it reads.
computed(d::Opaque) = string(op_name(d.node.op), op_detail(d.node.op), " of ", join(map(o -> "%$(o.value)", reads(d)), " "))

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

# A choice's name in the tuning record: the target's numerics, the claim's
# record name and the group's canonical form.
function choice_key(d::Decider, p::Plan, g::Group, implementation, claim)
    name = implementation === nothing ? "floor" : Base.invokelatest(record_name, implementation, claim)
    replace(string(numerics(d.target), " ", name, " | ", canonical(p, g)), r"[\t\n]" => " ")
end

function choice(d::Decider, p::Plan, g::Group, view, implementation, claim, order = 0)
    work = implementation === nothing ? floor_estimate(p, g) : Base.invokelatest(estimate, implementation, claim, d.target, view)
    key = choice_key(d, p, g, implementation, claim)
    # The floor moves bytes at the device's rates.
    memory = implementation === nothing ? Memory(d.rates.launch, d.rates.cached) : memory!(d, package_of(implementation))
    lines = [w.flops + w.recomputed == 0 ? nothing : line!(d, package_of(implementation), w.factor) for w in work]
    price(w) = (regime, f) -> price!(d, regime, product_class(d.target, w.factor), f)
    Choice(implementation, claim, sum(map((w, l) -> seconds(w, d.rates, memory, l, price(w)), work, lines); init = 0.0), key,
        any(l -> l !== nothing && l.borrowed, lines), order,
        implementation === nothing ? Any[] : Any[Base.invokelatest(variants, implementation, d.target, view, claim)...])
end

# Every group has a choice but opaque work, a scatter, which only the floor runs.
chosen(p::Plan, g::Group) = g.kind !== :opaque

"Every claim on a group, cheapest estimate first; the floor when none claims it."
function options(d::Decider, p::Plan, g::Group)
    view = group_view(p, g)
    # The protocol is called in the latest world here and wherever the
    # compiler calls it, as a callback is: kernel packages add methods to it
    # as they load, and were a call to bind the one method a package saw
    # while it precompiled, the cover's code that package cached would be
    # void in every session where another package loaded too.
    os = Choice[choice(d, p, g, view, i, c, k) for i in implementations(d.target) for (k, c) in enumerate(Base.invokelatest(claims, i, d.target, view))]
    isempty(os) && push!(os, choice(d, p, g, view, nothing, nothing))
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
        os = options!(d, p, g)
        # Searched already, or alike a group that was: the search grows the
        # options from the record and measures nothing.
        search!(d, [(p, g, os)])
        k, how, measured = get!(() -> choose(d, os), d.decided, canonical(p, g))
        Decision(os[k], os, how, measured)
    end
end

# The rank of the choice among options `os`, sorted by estimate; how it was
# made; and its measured time. A rank and not a claim, because claims hold
# value ids.
"A group's options, found once by its identity (see `prebuild!`)."
options!(d::Decider, p::Plan, g::Group) = get!(() -> options(d, p, g), d.options, group_identity(p, g))

# The ranks among `os`, sorted by estimate, of the options a decision
# measures: every option under `:all`; otherwise those within the estimate's
# error of the best, every option when the best rests on a borrowed line,
# which bounds a time only from below, and every claim of the best's
# implementation when it made several: alternatives it offers for
# measurement, which the estimate cannot rank. None when the best stands
# alone, unless it has variants: then it is measured to seed their search.
function measured_ranks(d::Decider, os)
    best = first(os)
    alone = isempty(best.variants)
    length(os) == 1 && return d.measure === :all || !alone ? [1] : Int[]
    searched(o) = o.implementation !== nothing && o.implementation === best.implementation
    ranks = d.measure === :all || best.borrowed ? collect(eachindex(os)) :
        [k for (k, o) in enumerate(os) if o.borrowed || close_call(o.estimate, best.estimate) || searched(o)]
    length(ranks) == 1 && alone ? Int[] : ranks
end

"""
Search how `p`'s groups run, all of them at once: each stage's kernels compile
side by side and are timed one at a time, for every group, rather than group
by group (see `search!` on groups); each decision then finds its options
searched. Groups decided already search nothing here.
"""
function search!(d::Decider, p::Plan)
    d.measure in (:close, :all) || return nothing
    price_all!(d, p)
    search!(d, [(p, g, options!(d, p, g)) for g in p.groups if chosen(p, g) && !haskey(d.decided, canonical(p, g))])
    nothing
end

"""
Search the options of each of `groups`, `(plan, group, options)` each, in
stages, all groups in each stage: the options a decision measures (see
`measured_ranks`) are built side by side and timed one at a time; then the
variants an implementation offers of each group's best measured option, and
of each within the noise of it (of every measured option, under `:all`); and
so on, until no group has a variant left to try. Each group's options grow by
what was tried, sorted by estimate, so that groups alike, searched again from
the record, grow alike. Under `:none`, the record alone is searched: what it
holds times is expanded, and nothing is measured.
"""
function search!(d::Decider, groups)
    while true
        pending = [(p, g, os[k]) for (p, g, os) in groups for k in measured_ranks(d, os) if !haskey(d.record, os[k].key)]
        if !isempty(pending) && d.measure in (:close, :all)
            build_unmeasured!(d, pending)
            foreach(((p, g, o),) -> measure!(d, p, g, o), pending)
        end
        any(map(((p, g, os),) -> expand!(d, p, g, os), groups)) || break
    end
end

# Grow a group's options by the variants of its best measured option, and of
# each within the noise of it (of every measured option under `:all`), that
# were not tried; whether any were added. A variant's order is past its
# parent's: it is never an implementation's first claim.
function expand!(d::Decider, p::Plan, g::Group, os::Vector{Choice})
    timed = [(o, d.record[o.key]) for o in os if o.implementation !== nothing && haskey(d.record, o.key)]
    isempty(timed) && return false
    best = minimum(last, timed)
    isfinite(best) || return false
    from = [o for (o, t) in timed if d.measure === :all || t <= best / (1 - d.rates.noise)]
    view = group_view(p, g)
    keys = Set(o.key for o in os)
    added = false
    for o in from, c in o.variants
        key = choice_key(d, p, g, o.implementation, c)
        key in keys && continue
        push!(keys, key)
        push!(os, choice(d, p, g, view, o.implementation, c, o.order + 1))
        added = true
    end
    added && sort!(os; by = o -> o.estimate)
    added
end

# Every scalar function the claims on `p`'s groups apply, in every regime and
# class they apply it in, priced together where the rates file lacks it (see
# `fit_applied_all!`): the references are built side by side rather than two
# at a time on first need. A function the table does not name is priced as
# addition, as `price!` prices it.
function price_all!(d::Decider, p::Plan)
    measures(d) || return nothing
    needs = Set{Tuple{Symbol, Symbol, Any}}()
    named(f) = findfirst(((_, g),) -> g == f, APPLIED) === nothing ? (+) : f
    for g in p.groups
        chosen(p, g) || continue
        view = group_view(p, g)
        for i in implementations(d.target), c in Base.invokelatest(claims, i, d.target, view), w in Base.invokelatest(estimate, i, c, d.target, view)
            class = product_class(d.target, w.factor)
            foreach(((f, _),) -> push!(needs, (:streamed, :any, named(f))), w.applied)
            foreach(((f, _),) -> push!(needs, (:swept, class, named(f))), w.swept)
            foreach(((f, _),) -> push!(needs, (:inner, class, named(f))), w.inner)
        end
    end
    key((regime, class, f)) = join(("applied", regime, class, APPLIED[findfirst(((_, g),) -> g == f, APPLIED)][1]), " ")
    unfitted = [n for n in needs if !haskey(d.applied, n) && !haskey(d.calibration, key(n))]
    isempty(unfitted) || fit_applied_all!(d, unfitted)
    nothing
end

# The choice among options `os`, sorted by estimate and searched (see
# `search!`): the rank of the best estimate, unless a measurement beats its
# time beyond the noise, among every option the record holds a time for.
function choose(d::Decider, os)
    best = first(os)
    length(os) == 1 && return (1, :only, get(d.record, best.key, nothing))
    isempty(measured_ranks(d, os)) && return (1, :estimated, get(d.record, best.key, nothing))
    ranks = [k for (k, o) in enumerate(os) if haskey(d.record, o.key)]
    # The best estimate unmeasured, each implementation stands by its first
    # claim: the estimate ranks implementations, not an implementation's own
    # alternatives.
    1 in ranks || return (findfirst(o -> o.order <= 1, os), :estimated, nothing)
    times = [d.record[os[k].key] for k in ranks]
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
        build_unmeasured!(d, [(p, g, x.choice) for (p, g, x) in [da; db]])
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

# Buffers for a group's reads and stored members, shared by every plan the
# decider considers (they number values alike), filled so that measuring reads
# nothing undefined.
function buffers!(d::Decider, p::Plan, g::Group)
    n = length(p.graph.values)
    length(d.buffers) < n && append!(d.buffers, fill(nothing, n - length(d.buffers)))
    for v in [collect(group_reads(p.graph, g)); filter(m -> p.stored[m], members(g))]
        d.buffers[v] === nothing || continue
        val = p.graph.values[v]
        def = val.definition
        d.buffers[v] = def isa ConstantArray ? upload(d.target, vec(collect(def.data))) :
            filled!(allocate(d.target, val.eltype, storage_length(val)))
    end
    d.buffers
end

# A kernel timed on a buffer reads it as data, and as an index if it is an
# integer: zero is outside every axis and one is inside any, so integers are
# ones and the rest zeros.
filled!(x) = eltype(x) <: Integer ? fill!(x, one(eltype(x))) : (fill!(reinterpret(UInt8, x), 0x00); x)

# Build a choice on `buffers`.
build_choice(target, p::Plan, g::Group, o::Choice, buffers) =
    o.implementation === nothing ? build_floor(target, p, g, buffers) :
    Base.invokelatest(build_claim, o.implementation, o.claim, target, group_view(p, g), buffers)

# Build a choice on the decider's buffers, once, to measure it: its key names
# the group alike, so the first of them builds for all.
build_choice!(d::Decider, p::Plan, g::Group, o::Choice) =
    get!(() -> build_choice(d.target, p, g, o, buffers!(d, p, g)), d.built, o.key)

# A group's kernels: its choice, or the next claim when a build fails, and
# the floor last. Opaque work, a scatter, runs on the floor.
function build_group(d::Decider, p::Plan, g::Group, x, buffers)
    x === nothing && return build_floor(d.target, p, g, buffers)
    for o in [x.choice; filter(!=(x.choice), x.options)]
        o.implementation === nothing && break
        built = try
            build_choice(d.target, p, g, o, buffers)
        catch err
            @warn "a claim failed to build; the next option runs the kernel" implementation = label(o.implementation) claim = o.claim exception = err maxlog = 1 _id = Symbol(o.key)
            nothing
        end
        built === nothing || return built
    end
    build_floor(d.target, p, g, buffers)
end

# Build an option to measure it. A claim that fails to build loses every
# comparison in this preparation; the record keeps only times.
function build_option!(d::Decider, p::Plan, g::Group, o::Choice)
    try
        first(build_choice!(d, p, g, o))
    catch err
        @debug "claim failed to build; it loses" o.implementation exception = err
        d.record[o.key] = Inf
        nothing
    end
end

# Build the options with no time yet, `(plan, group, option)` each,
# concurrently, so that their kernels compile side by side before they are
# timed one at a time, alone on the device.
function build_unmeasured!(d::Decider, options)
    unmeasured = unique(x -> x[3].key, filter(x -> !haskey(d.record, x[3].key), options))
    concurrently(((p, g, o),) -> build_option!(d, p, g, o), unmeasured, d.target)
end

function measure!(d::Decider, p::Plan, g::Group, o::Choice)
    haskey(d.record, o.key) && return
    launch = build_option!(d, p, g, o)
    launch === nothing && return
    t = time_launch(d.target, launch)
    d.record[o.key] = t
    save_record(d.record_path, o.key, t)
end
