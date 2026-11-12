# How a group runs. Every implementation's claims on a group are estimated,
# as launches with the bytes they move, the flops they compute and the blocks
# they keep busy, and so is the floor's. Estimates within each other's error
# are close calls: they are measured at preparation and kept in the tuning
# record, and a measurement overrides the estimate only beyond the device's
# noise margin, so the choice is deterministic given the record. An
# implementation may stage its claims: the variants it offers of the claim
# that measured best are tried next, until the best has none new. Groups alike
# choose once. A claim that fails to build is no option once it has: the
# decider keeps it, and the cover plans again without it.

# Two estimates closer than this factor are a close call.
const ESTIMATE_ERROR = 1.5

"What `explain` calls what runs a group: an implementation, by its type, or the floor (`nothing`)."
label(i::Implementation) = string(nameof(typeof(i)))
label(::Nothing) = "floor"

"A way to realize a group: an implementation's claim, or the floor's (`nothing`)."
struct Choice
    implementation::Union{Nothing, Implementation}
    claim::Any
    estimate::Float64
    key::String
    borrowed::Bool # priced on another package's line, which bounds a time only from below
    order::Int # the claim's place among its implementation's claims on the group, its own ranking of them; a variant's is past its parent's
    variants::Vector{Any} # the claims its implementation offers once it measures best (see `Tempest.Compiler.Implement.variants`)
end

"""
What an implementation says of its claim on a group: the work the claim is
estimated at, and the claims it offers once this one measures best (see
`Tempest.Compiler.Implement.variants`). Of the claim, the target and the group
alone, and so found side by side (see `side_by_side`).
"""
struct Said
    work::Vector{Work}
    variants::Vector{Any}
end

said(target, implementation, claim, view) = Said(Base.invokelatest(estimate, implementation, claim, target, view),
    Any[Base.invokelatest(variants, implementation, target, view, claim)...])

"""
What the implementations offer for a group: each claim on it, in the
implementations' order, with its place among its implementation's claims and
what its implementation says of it (`Said`); the claims priced as choices,
once the rates they apply are fitted (see `price_all!`); and each variant of
an option priced as the search reaches it, with its key, by the option it
varies and its place among that option's variants. Claims, estimates and
prices are of the group, the target and the rates, not of the record: a
cover that runs again on a record written since, settling, finds them as
they were found, and grows each group's options from the record again (see
`search!`).
"""
struct Offered
    said::Vector{Tuple{Implementation, Any, Int, Said}}
    claims::Vector{Choice}
    keys::Dict{Tuple{String, Int}, String}
    variants::Dict{Tuple{String, Int}, Choice}
end

"An option that failed to build: what `explain` calls what runs it (see `label`), the group (by `canonical`), and why, in a line."
struct Failure
    implementation::String
    group::String
    reason::String
end

"""
What was chosen for a group, how, and the time measured for it, if any; and
the claims on it that failed to build, which are no options. A choice made
`:instead` of one that failed to build as the program was prepared is the
next option that built, the floor last.
"""
struct Decision
    choice::Choice
    options::Vector{Choice}
    how::Symbol # :only, :estimated, :measured or :instead
    measured::Union{Nothing, Float64}
    failed::Vector{Failure}
end

function Base.show(io::IO, d::Decision)
    print(io, "estimated ", round(1.0e6 * d.choice.estimate; sigdigits = 3), " μs")
    d.measured === nothing || print(io, ", measured ", round(1.0e6 * d.measured; sigdigits = 3), " μs")
    d.how === :measured && print(io, "; chosen by measurement among ", length(d.options))
    d.how === :estimated && length(d.options) > 1 && print(io, "; chosen by estimate among ", length(d.options))
    d.how === :instead && print(io, "; in place of the choice, which failed to build")
end

"""
What an executable's launches share and nothing else uses, by
implementation: what `resources` made for each, as the first of its launches
is built. A decider has its own, for the claims it builds to time, which it
times one at a time; a recurrence's step's decider shares its preparation's.
"""
struct Resources
    target::Any
    made::Dict{Any, Any}
    lock::ReentrantLock
end

Resources(target) = Resources(target, Dict{Any, Any}(), ReentrantLock())

mutable struct Decider
    target::Any
    # Whether the target times kernels, asked once: the target's package
    # defines it, so code inferred before it loads would be compiled again.
    measurable::Bool
    graph::IndexGraph
    measure::Symbol
    rates::Rates
    calibration::Noted
    calibration_path::Union{Nothing, String}
    lines::Dict{Tuple{Symbol, Symbol}, Line}
    memories::Dict{Symbol, Memory}
    # Seconds per position of each scalar function on the device, measured
    # once, in each regime: streamed, swept in a product of a class, swept in
    # a product of a class that recomputes a fold.
    applied::Dict{Tuple{Symbol, Symbol, Any}, Float64}
    record::Noted
    record_path::Union{Nothing, String}
    # Folds pinned to be recomputed where they are read, in every plan, and
    # folds the cover keeps stored.
    pinned::Set{Int}
    kept::Set{Int}
    # Sums pinned to ride the steps of a recurrence, in every plan.
    rides::Set{Int}
    buffers::Vector{Any}
    built::Dict{String, Any}
    # What the claims it builds to time share (see `Resources`).
    resources::Resources
    # The choice for each group alike (by `canonical`): its rank among the
    # options, how it was made, and the time measured; and each group's decision.
    decided::Dict{String, Tuple{Int, Symbol, Union{Nothing, Float64}}}
    instances::Dict{Any, Decision}
    # Each group's options (by `group_identity`) as this cover searches
    # them, grown from the record; and what the implementations offer for
    # each, found once in the preparation (see `Offered`).
    options::Dict{Any, Vector{Choice}}
    offered::Dict{Any, Offered}
    # The claims that failed to build, by key.
    failed::Dict{String, Failure}
    # Each recurrence's step planned (by `group_identity`; see `stepped_plan`).
    stepped::Dict{Any, Any}
    # How many times the graph's kernels run: a recurrence's step's run once
    # per step (see `plan_step`).
    steps::Int
    # The options a decider measuring close calls would have timed where
    # this one, measuring nothing, decided without a time, by key; shared,
    # as the tables are, with a recurrence's step's decider.
    wanted::Set{String}
    # What times a launch, `(target, launch) -> seconds`: `timed`, or the
    # timer `decide!` is given, which a recurrence's step's decider shares.
    timer::Any
end

# A decider reads the rates file and the record as they are when it is made,
# and prices on the rates `calibration` holds: by default those fitted on this
# machine over those shipped for its device (see `known_rates`).
function Decider(target, graph; measure = measuring(), recompute = (), keep = (), ride = (), steps = 1, calibration = known_rates(target, measure))
    dry = level(measure) === :dry
    path, calibration_path = dry ? (nothing, nothing) : (record_path(target), rates_path(target))
    r = dry ? rates(target) : target_rates(target)
    how = dry ? :dry : measurable(target) ? level(measure) : :none
    decider(target, measurable(target), graph, how, r, calibration, calibration_path, Noted(load_record(path)), path;
        recompute, keep, ride, steps)
end

"""
A decider of a recurrence's step, planned within a preparation that decider
`parent` decides: of its target, measuring as it does, on its rates, and
reading its record and rates file, so that the step's decisions rest on what
the preparation's do, noting the options it wanted with the preparation's,
and building the claims it times with the preparation's resources.
"""
Decider(parent::Decider, graph::IndexGraph; steps) =
    decider(parent.target, parent.measurable, graph, parent.measure, parent.rates, parent.calibration, parent.calibration_path, parent.record, parent.record_path;
        steps, wanted = parent.wanted, timer = parent.timer, resources = parent.resources)

decider(target, measurable, graph, measure, rates, calibration, calibration_path, record, record_path; recompute = (), keep = (), ride = (), steps = 1,
        wanted = Set{String}(), timer = timed, resources = Resources(target)) =
    Decider(target, measurable, graph, measure, rates, calibration, calibration_path, Dict{Tuple{Symbol, Symbol}, Line}(), Dict{Symbol, Memory}(), Dict{Tuple{Symbol, Symbol, Any}, Float64}(), record, record_path,
        Set{Int}(recompute), Set{Int}(keep), Set{Int}(ride), Any[], Dict{String, Any}(), resources, Dict{String, Tuple{Int, Symbol, Union{Nothing, Float64}}}(), Dict{Any, Decision}(),
        Dict{Any, Vector{Choice}}(), Dict{Any, Offered}(), Dict{String, Failure}(), Dict{Any, Any}(), steps, wanted, timer)

# Forget every decision and each group's options as searched, so that the
# cover plans again from what the decider knows now: the claims that failed
# to build, and the times it measured, which it does not measure again. What
# the implementations offer for each group (see `Offered`) is not the
# record's, and is kept.
function forget_decisions!(d::Decider)
    empty!(d.decided)
    empty!(d.instances)
    empty!(d.options)
    empty!(d.wanted)
    d
end

# Keep an option on a group that failed to build: a claim, or the floor,
# which a decider builds to time it as it builds a claim, and which fails
# where the decider's buffers for it cannot be allocated (see `buffers!`).
fail!(d::Decider, p::Plan, g::Group, o::Choice, err) = d.failed[o.key] = Failure(label(o.implementation)::String, canonical(p, g), reason(err))

# Why a build failed, in a line: the first of its message's that does not
# only announce the lines after it, as "kernel compilation produced 2
# errors:" does.
function reason(err)
    lines = filter(!isempty, strip.(split(sprint(showerror, err), '\n')))
    isempty(lines) && return string(nameof(typeof(err)))
    String(lines[something(findfirst(l -> !endswith(l, ':'), lines), 1)])
end

"The claims on a group that failed to build."
function failures(d::Decider, p::Plan, g::Group)
    isempty(d.failed) && return Failure[]
    c = canonical(p, g)
    sort!([f for f in values(d.failed) if f.group == c]; by = f -> (f.implementation, f.reason))
end

# What is measured: nothing (`:none`), the close calls (`:close`) or every
# option (`:all`); `:dry` touches no device at all, pricing claims on nominal
# rates and reading no record, for a preparation that builds nothing.
level(measure::Bool) = measure ? :close : :none
level(measure::Symbol) = measure

# `prepare`'s and `compile`'s default of what is measured: nothing, unless the
# environment's TEMPEST_MEASURE says otherwise, so that a preparation gives
# the plan the estimates pick, and what the record holds, and nothing after;
# measuring is asked for. The tests measure nothing unless a test asks, since
# a time taken beside other jobs on one device decides a close call by
# chance, and a test of the plan would then pass by chance.
function measuring()
    m = Symbol(get(ENV, "TEMPEST_MEASURE", "none"))
    m in (:none, :close, :all) || throw(ArgumentError("TEMPEST_MEASURE is none, close or all, not $m"))
    m
end

# `compile`'s default of when a preparation that measures times its claims:
# before the first result (`:before`), unless the environment's TEMPEST_SEARCH
# says after it (`:after`). Where nothing is measured it means nothing.
function searching()
    s = Symbol(get(ENV, "TEMPEST_SEARCH", "before"))
    s in (:before, :after) || throw(ArgumentError("TEMPEST_SEARCH is before or after, not $s"))
    s
end

"Whether the decider may time claims and fit rates on its target: not when dry."
measures(d::Decider) = d.measure !== :dry && d.measurable

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

group_identity(p::Plan, g::Group) = GroupIdentity(kind_name(g.kind), g.domain, copy(g.reductions), copy(g.outputs),
    Any[get(g.terms, m, nothing) for m in members(g)], Bool[p.stored[m] for m in members(g)])

# A fold in a group's key: its fold and axes, and a scan's direction.
fold_key(d::Reduction) = string(fold_name(d.fold), d.reduced)
fold_key(d::Scan) = string(fold_name(d.fold), d.axes, d.reverse ? " reversed" : "", d.exclusive ? " exclusive" : "")
fold_key(d::Keyed) = string(fold_name(d.fold), " by key ", d.key)

# A group as the plan numbers its values: its members in order, each with
# its term.
function numbered_text(p::Plan, g::Group)
    io = IOBuffer()
    print(io, g.kind, " ", g.domain)
    for m in members(g)
        v = p.graph.values[m]
        what = m in g.reductions ? fold_key(v.definition) : "value"
        print(io, "; %", m, " ", what, " ", v.eltype, v.dims, p.stored[m] ? " stored" : "", " = ", g.terms[m])
    end
    String(take!(io))
end

"The values a group mentions, in the order its text first mentions them: the order `canonical` numbers them in."
appearance(p::Plan, g::Group) = unique!([parse(Int, m.match[2:end]) for m in eachmatch(r"%\d+", numbered_text(p, g))])

# A group across programs: value numbers in order of appearance, with each
# value's type and shape.
function canonical(p::Plan, g::Group)
    numbers = Dict{String, Int}()
    text = replace(numbered_text(p, g), r"%\d+" => s -> string("%", get!(numbers, s, length(numbers) + 1)))
    shapes = map(sort!(collect(numbers); by = last)) do (s, k)
        v = p.graph.values[parse(Int, s[2:end])]
        string("%", k, " ", v.eltype, v.dims)
    end
    string(text, "; ", join(shapes, ", "))
end

# A choice's name in the tuning record: the target's numerics, the claim's
# record name and the group's canonical form. A recurrence on the floor runs
# a step at a time where it can, and the run is named by its step's kernels
# too, the choices their decisions made (see `stepped_plan`): its time is
# theirs, and a step decided otherwise is another run.
function choice_key(d::Decider, p::Plan, g::Group, implementation, claim)
    function floor()
        sp = g.kind isa RecurrenceKind ? stepped_plan(d, p, g) : nothing
        sp === nothing && return "floor"
        string("stepped ", content_key([x === nothing ? "" : x.choice.key for x in sp.decisions]))
    end
    name = implementation === nothing ? floor() : Base.invokelatest(record_name, implementation, claim)
    replace(string(numerics(d.target), " ", name, " | ", canonical(p, g)), r"[\t\n]" => " ")
end

# The floor is split as the target splits it, except by a dry decider, which
# touches no device and prices it unsplit.
floor_splits(d::Decider) = d.measure === :dry ? (items, positions, adjacent) -> nothing :
    (items, positions, adjacent) -> floor_split(d.target, items, positions, adjacent)

function choice(d::Decider, p::Plan, g::Group, view, implementation, claim, order = 0,
        told = implementation === nothing ? Said(floor_estimate(p, g, floor_splits(d)), Any[]) : said(d.target, implementation, claim, view))
    work = told.work
    key = choice_key(d, p, g, implementation, claim)
    memory = memory!(d, implementation === nothing ? :floor : package_name(implementation))
    lines = [w.flops + w.recomputed == 0 ? nothing : line!(d, package_name(implementation), w.factor) for w in work]
    price(w) = (regime, @nospecialize(f)) -> price!(d, regime, product_class(d.target, w.factor), f)
    # A recurrence on the floor runs a step at a time where it can (see
    # `build_group`), at its steps' price.
    stepped = implementation === nothing && g.kind isa RecurrenceKind ? stepped_seconds(d, p, g) : nothing
    seconds_of = stepped === nothing ? sum(map((w, l) -> seconds(w, d.rates, memory, l, price(w)), work, lines); init = 0.0) : stepped
    o = Choice(implementation, claim, seconds_of, key,
        any(l -> l !== nothing && l.borrowed, lines), order, told.variants)
    stepped === nothing ? o : run_whole(d, p, g, o)
end

"""
A recurrence's floor `o`, run a step at a time, priced at its run timed
whole where the record holds that time, as a decider measuring close calls
takes it here: pricing the run planned its step and timed the step's
kernels already, and the run is their launches. Its steps' price (see
`stepped_seconds`) stands where no time does.

A kernel's time in the record is taken alone, its data in memory (see
`time_launch`), so that it is one time wherever the kernel runs. A step
runs among the steps around it, each finding in the cache what the one
before brought, the weights it reads again, and its kernels' times alone
overstate the run: the neural ODE's gradient's recurrence, 100 steps of 13
kernels, priced so at 11.43 ms where it runs in 7.30. The run timed whole
is one time too, the group's as it runs, as a recurrence's single kernel's
is, and its key names the step's kernels (see `choice_key`).
"""
function run_whole(d::Decider, p::Plan, g::Group, o::Choice)
    d.measure in (:close, :all) && measure!(d, p, g, o)
    t = get(d.record, o.key, Inf)
    isfinite(t) || return o
    Choice(o.implementation, o.claim, t, o.key, o.borrowed, o.order, o.variants)
end

"""
Every claim on a group but those that failed to build, cheapest estimate
first; the floor when none is left, and beside them for a map whose kernel runs
once per step of a recurrence: there each is measured (see
`measured_ranks`), as the floor compiles once per shape and a step's kernel
runs as many times as there are steps. A stencil's step, which cuTile reads
through a gather at each shifted position, took 4.5 μs in cuTile and 2.2 on
the floor.
"""
function options(d::Decider, p::Plan, g::Group)
    # A recurrence's options rest on its step planned, which names and
    # prices its floor, and which the decisions reached keep (see `reached`).
    g.kind isa RecurrenceKind && stepped_plan(d, p, g)
    # A recurrence with sums riding its steps runs them only a step at a time.
    riders = g.kind isa RecurrenceKind && !isempty(g.reductions)
    os = riders ? Choice[] : filter(o -> !haskey(d.failed, o.key), priced!(d, p, g))
    (isempty(os) || d.steps > 1 && g.kind isa PointwiseKind) && push!(os, choice(d, p, g, group_view(p, g), nothing, nothing))
    sort!(os; by = o -> o.estimate) # stable: ties keep the implementations' order
end

"What the implementations offer for a group (see `Offered`), found once by its identity."
offered!(d::Decider, p::Plan, g::Group) = get!(() -> offered(d, p, g), d.offered, group_identity(p, g))

"""
What the implementations offer for each of `groups`, `(plan, group)` each,
found for those it was not found for yet: side by side on the threads there
are, as a group's claims and estimates are of the group, the target and the
implementations alone (see `side_by_side`), and kept here, by the caller.
"""
function offer!(d::Decider, groups)
    fresh = Dict{GroupIdentity, Tuple{Plan, Group}}()
    for (p, g) in groups
        id = group_identity(p, g)
        haskey(d.offered, id) || haskey(fresh, id) || (fresh[id] = (p, g))
    end
    ids = collect(keys(fresh))
    for (id, found) in zip(ids, side_by_side(id -> offered(d, fresh[id]...), ids, d.target))
        d.offered[id] = found
    end
    nothing
end

function offered(d::Decider, p::Plan, g::Group)
    tally!(:options)
    view = group_view(p, g)
    # The protocol is called in the latest world here and wherever the
    # compiler calls it, as a callback is: kernel packages add methods to it
    # as they load, and were a call to bind the one method a package saw
    # while it precompiled, the cover's code that package cached would be
    # void in every session where another package loaded too.
    told = Tuple{Implementation, Any, Int, Said}[(i, c, k, said(d.target, i, c, view))
        for i in implementations(d.target) for (k, c) in enumerate(Base.invokelatest(claims, i, d.target, view))]
    Offered(told, Choice[], Dict{Tuple{String, Int}, String}(), Dict{Tuple{String, Int}, Choice}())
end

# A group's claims priced as choices, once.
function priced!(d::Decider, p::Plan, g::Group)
    found = offered!(d, p, g)
    if length(found.claims) < length(found.said)
        view = group_view(p, g)
        append!(found.claims, [choice(d, p, g, view, i, c, k, told) for (i, c, k, told) in found.said])
    end
    found.claims
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
        os = searched_options!(d, p, g)
        k, how, measured = get!(() -> choose(d, os), d.decided, canonical(p, g))
        Decision(os[k], os, how, measured, failures(d, p, g))
    end
end

# The rank of the choice among options `os`, sorted by estimate; how it was
# made; and its measured time. A rank and not a claim, because claims hold
# value ids.
"A group's options, found once by its identity."
options!(d::Decider, p::Plan, g::Group) = get!(() -> options(d, p, g), d.options, group_identity(p, g))

# A group's options, searched (see `search!`): searched already, or alike a
# group that was, the search grows them from the record and measures
# nothing. Claims that failed to build as they were measured are no options:
# the rest are found and searched again. The floor, the option of last
# resort, stays one where it failed so: its time is then infinite (see
# `build_option!`), and it loses every measurement, a plan that needs it
# every comparison.
function searched_options!(d::Decider, p::Plan, g::Group)
    os = options!(d, p, g)
    search!(d, [(p, g, os)])
    while any(o -> o.implementation !== nothing && haskey(d.failed, o.key), os)
        os = d.options[group_identity(p, g)] = options(d, p, g)
        search!(d, [(p, g, os)])
    end
    os
end

# The ranks among `os`, sorted by estimate, of the options a decision
# measures: every option under `:all`; otherwise those within the estimate's
# error of the best, every option when the best rests on a borrowed line,
# which bounds a time only from below, and every claim of the best's
# implementation once the record refutes the best's estimate (see
# `refuted`). An implementation's alternatives the estimate cannot rank, its
# output tiles at one depth, are estimated alike and so within its error;
# one estimated past it is ranked by it until a time shows the estimate
# wrong: the GPT step's products split four ways, estimated 3.0 to 4.4 times
# the best tile, measured 14 times and never fastest. None when the best
# stands alone, unless it has variants: then it is measured to seed their
# search. Where the kernels run once per step, the floor and the best claim
# are measured beside each other (see `options`).
function measured_ranks(d::Decider, os)
    best = first(os)
    alone = isempty(best.variants)
    length(os) == 1 && return d.measure === :all || !alone ? [1] : Int[]
    searched(o) = o.implementation !== nothing && o.implementation === best.implementation && refuted(d, best)
    claimed = findfirst(o -> o.implementation !== nothing, os)
    floor = d.steps > 1 && any(o -> o.implementation === nothing, os)
    beside(k, o) = floor && (o.implementation === nothing || k == claimed)
    ranks = d.measure === :all || best.borrowed ? collect(eachindex(os)) :
        [k for (k, o) in enumerate(os) if o.borrowed || close_call(o.estimate, best.estimate) || searched(o) || beside(k, o)]
    length(ranks) == 1 && alone ? Int[] : ranks
end

"""
Search how `p`'s groups run, all of them at once: each stage's kernels compile
side by side and are timed one at a time, for every group, rather than group
by group (see `search!` on groups); each decision then finds its options
searched. Groups decided already search nothing here. What the
implementations offer for the groups is found first, side by side (see
`offer!`), measuring or not.
"""
function search!(d::Decider, p::Plan)
    offer!(d, [(p, g) for g in p.groups])
    d.measure in (:close, :all) || return nothing
    price_all!(d, p)
    search!(d, [(p, g, options!(d, p, g)) for g in p.groups if !haskey(d.decided, canonical(p, g))])
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
        elseif d.measure === :none
            foreach(((_, _, o),) -> push!(d.wanted, o.key), pending)
        end
        grown = [growth(d, p, g, os) for (p, g, os) in groups]
        price_variants!(d, groups, grown)
        any(map(((p, g, os), new) -> expand!(d, p, g, os, new), groups, grown)) || break
    end
end

# The variants a group's options grow by: those of its best measured option,
# and of each within the noise of it (of every measured option under `:all`),
# that were not tried and did not fail to build, each `(option, place,
# claim)` by the option it varies and its place among that option's variants.
function growth(d::Decider, p::Plan, g::Group, os::Vector{Choice})
    grown = Tuple{Choice, Int, Any}[]
    timed = [(o, d.record[o.key]) for o in os if o.implementation !== nothing && haskey(d.record, o.key)]
    isempty(timed) && return grown
    best = minimum(last, timed)
    isfinite(best) || return grown
    from = [o for (o, t) in timed if d.measure === :all || t <= best / (1 - d.rates.noise)]
    found = offered!(d, p, g)
    keys = Set(o.key for o in os)
    for o in from, (j, c) in enumerate(o.variants)
        key = get!(() -> choice_key(d, p, g, o.implementation, c), found.keys, (o.key, j))
        (key in keys || haskey(d.failed, key)) && continue
        push!(keys, key)
        push!(grown, (o, j, c))
    end
    grown
end

# Each variant `grown` holds for `groups` that was not priced before, priced
# and kept with what the implementations offer for its group (see
# `Offered`): what its implementation says of it found side by side (see
# `Said`), then priced here, in the groups' order and each group's, as the
# rates it applies are fitted on first need.
function price_variants!(d::Decider, groups, grown)
    seen = Set{Tuple{GroupIdentity, Tuple{String, Int}}}()
    unpriced = Tuple{Plan, Group, Choice, Int, Any}[]
    for ((p, g, _), new) in zip(groups, grown), (o, j, c) in new
        at = (group_identity(p, g), (o.key, j))
        haskey(offered!(d, p, g).variants, last(at)) || at in seen || (push!(seen, at); push!(unpriced, (p, g, o, j, c)))
    end
    told = side_by_side(((p, g, o, _, c),) -> said(d.target, o.implementation, c, group_view(p, g)), unpriced, d.target)
    for ((p, g, o, j, c), t) in zip(unpriced, told)
        offered!(d, p, g).variants[(o.key, j)] = choice(d, p, g, group_view(p, g), o.implementation, c, o.order + 1, t)
    end
end

# Grow a group's options by the variants `grown` found for it (see `growth`),
# priced (see `price_variants!`); whether any were added. Groups alike share
# their options, and a variant one of them added is not added again. A
# variant's order is past its parent's: it is never an implementation's
# first claim.
function expand!(d::Decider, p::Plan, g::Group, os::Vector{Choice}, grown)
    found = offered!(d, p, g)
    keys = Set(o.key for o in os)
    added = false
    for (o, j, _) in grown
        at = (o.key, j)
        found.keys[at] in keys && continue
        push!(keys, found.keys[at])
        push!(os, found.variants[at])
        added = true
    end
    added && sort!(os; by = o -> o.estimate)
    added
end

# Every scalar function the claims on `p`'s groups apply, in every regime and
# class they apply it in, priced together where the rates file lacks it (see
# `fit_applied_all!`): the references are built side by side rather than two
# at a time on first need. A function the table does not name is priced as
# addition, as `price!` prices it. The claims and their work are those each
# group's options are priced from (see `Offered`), found once.
function price_all!(d::Decider, p::Plan)
    measures(d) || return nothing
    needs = Set{Tuple{Symbol, Symbol, Any}}()
    named(@nospecialize(f)) = applied_function(f)
    for g in p.groups, (_, _, _, told) in offered!(d, p, g).said, w in told.work
        class = product_class(d.target, w.factor)
        foreach(((f, _),) -> push!(needs, (:streamed, :any, named(f))), w.applied)
        foreach(((f, _),) -> push!(needs, (:swept, class, named(f))), w.swept)
        foreach(((f, _),) -> push!(needs, (:inner, class, named(f))), w.inner)
        foreach(((f, _),) -> push!(needs, (:stepped, :any, named(f))), w.stepped)
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

decisions(d::Decider, p::Plan) = [decide(d, p, g) for g in p.groups]

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

# Each plan's group identities, and their set, kept in `known` across
# comparisons: the cover compares one plan with each of its trials.
identities!(known, p::Plan) = get!(known, p) do
    ids = [group_identity(p, g) for g in p.groups]
    (ids, Set(ids))
end::Tuple{Vector{GroupIdentity}, Set{GroupIdentity}}

# Compare plan `b` with plan `a`. Close calls are measured, and so is a
# comparison where the record already refutes an estimate on either side, a
# kernel measured past its estimate's error: an estimate that misprices one
# kernel decides nothing. The log-sum-exp of GPT-2's attention recomputed in
# the two backward kernels that read it was estimated at 3.5 ms against 5.9
# for the kernels of the plan that stores it, when those two had been
# measured at 40 and 13 ms. A side measured is timed only until it has lost
# (see `measure_sides!`), and a side that lost so is decided against by what
# is known of it (see `stopped`); otherwise the measurements decide where
# they differ beyond the noise, and the estimates where not.
function compare(d::Decider, b::Plan, a::Plan, known = IdDict{Plan, Any}())
    tally!(:compared)
    ids_a, ia = identities!(known, a)
    ids_b, ib = identities!(known, b)
    only_a = [(a, g) for (g, id) in zip(a.groups, ids_a) if !(id in ib)]
    only_b = [(b, g) for (g, id) in zip(b.groups, ids_b) if !(id in ia)]
    decided(side) = [(p, g, decide(d, p, g)) for (p, g) in side]
    da, db = decided(only_a), decided(only_b)
    ea = sum(x -> x[3].choice.estimate, da; init = 0.0)
    eb = sum(x -> x[3].choice.estimate, db; init = 0.0)
    close = close_call(ea, eb)
    measured = d.measure === :all || d.measure in (:close, :none) && (close || any(x -> refuted(d, x[3].choice), [da; db]))
    measured && measure_sides!(d, da, db, close)
    ta = [get(d.record, x[3].choice.key, nothing) for x in da]
    tb = [get(d.record, x[3].choice.key, nothing) for x in db]
    if !any(isnothing, ta) && !any(isnothing, tb)
        ma, mb = sum(ta; init = 0.0), sum(tb; init = 0.0)
        past(max(ma, mb), min(ma, mb), d.rates.noise) && return Comparison(ma, mb, :measured, (da, db))
    elseif measured
        c = stopped(d, da, db, close)
        c === nothing || return c
    end
    Comparison(ea, eb, :estimated, (da, db))
end

# Whether time `t` passes time `bound` beyond the noise, as one measurement
# beats another.
past(t, bound, noise) = t > (1 + noise) * bound

# What is known of the time a side of a comparison takes, `(plan, group,
# decision)` each: each group's time where the record holds one, and its
# estimate where not. The least it can take: each group's time where the
# record holds one, and where not, in a close call nothing, its estimate
# ranking nothing there, and otherwise its estimate at the least its error
# allows.
known_time(d::Decider, side) = sum(x -> get(d.record, x[3].choice.key, x[3].choice.estimate), side; init = 0.0)
least_time(d::Decider, side, close) = sum(x -> get(d.record, x[3].choice.key, close ? 0.0 : x[3].choice.estimate / ESTIMATE_ERROR), side; init = 0.0)

"""
Time the two sides of a comparison, `(plan, group, decision)` each, as far
as the comparison needs them (under `:all`, whole): the side known to be
faster first, whole, its time the bound of the other's; then the other's
groups, the dearest first, until the least it can take (see `least_time`)
passes that bound beyond the noise. It has lost then, and is timed no
further. A decider measuring nothing notes, as wanted, what it would have
timed.

A launch begun cannot be stopped, and one is timed whole, as many times as
`time_launch` takes it, so that the record holds only times taken alike: the
bound falls between launches, as whether a group's next is begun. Before its
first, a group's estimate is all that is known of it. In a close call
(`close`) that ranks nothing, and a side stops only where the groups it
has timed pass the bound, which whole it would pass too. Elsewhere the
comparison is measured because the record refuted an estimate on a side,
and a group whose own estimate would put its side past the bound even at
the least its error allows is not begun, as the cover leaves a call that
far apart to the estimates wherever none is refuted. Qwen3.5-4B's layer
output riding its delta rule's steps is a floor kernel of one item, priced
at 1.5e6 s against the 0.017 the plan that does not ride it was measured
at: timed whole, it ran for longer than the preparation lasts. In a close
call the estimates of the LSTM's steps' maps run up to 3.3 times their
times, and a side stopped by its estimates there lost a comparison whose
times it won, 16.2 μs against 17.2.
"""
function measure_sides!(d::Decider, da, db, close)
    d.measure === :none || build_unmeasured!(d, [(p, g, x.choice) for (p, g, x) in [da; db]])
    if d.measure === :all
        foreach(x -> measure_side!(d, x), [da; db])
        return
    end
    ahead, behind = known_time(d, da) <= known_time(d, db) ? (da, db) : (db, da)
    foreach(x -> measure_side!(d, x), ahead)
    bound = known_time(d, ahead)
    for x in sort(filter(x -> !haskey(d.record, x[3].choice.key), behind); by = x -> x[3].choice.estimate, rev = true)
        if past(least_time(d, behind, close), bound, d.rates.noise)
            tally!(:stopped)
            break
        end
        measure_side!(d, x)
    end
end

# Time the choice of a group on one side of a comparison, `(plan, group,
# decision)`, or note it wanted by a decider measuring nothing.
function measure_side!(d::Decider, (p, g, x))
    d.measure === :none || return measure!(d, p, g, x.choice)
    haskey(d.record, x.choice.key) || push!(d.wanted, x.choice.key)
    nothing
end

# A comparison one side of which was timed no further once it had lost (see
# `measure_sides!`): the other's time is in the record, and its own least
# (see `least_time`) passes that beyond the noise. It is decided against by
# what is known of each side, measured where its groups timed alone pass the
# other's time, else estimated; nothing where no side lost so.
function stopped(d::Decider, da, db, close)
    whole(side) = all(x -> haskey(d.record, x[3].choice.key), side)
    recorded(side) = sum(x -> get(d.record, x[3].choice.key, 0.0), side; init = 0.0)
    for (side, other) in ((da, db), (db, da))
        whole(other) && !whole(side) || continue
        bound = recorded(other)
        past(least_time(d, side, close), bound, d.rates.noise) || continue
        how = past(recorded(side), bound, d.rates.noise) ? :measured : :estimated
        return Comparison(known_time(d, da), known_time(d, db), how, (da, db))
    end
    nothing
end

# Whether the record holds a time for a choice beyond its estimate's error.
refuted(d::Decider, o::Choice) = haskey(d.record, o.key) && !close_call(d.record[o.key], o.estimate)

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
            val.eltype <: Integer ? upload(d.target, held_integers(p.graph, v, val.eltype, storage_length(val))) :
            zeroed!(allocate(d.target, val.eltype, storage_length(val)))
    end
    d.buffers
end

# Inferred for no buffer type, which the target's package defines.
Base.@nospecializeinfer zeroed!(@nospecialize x) = (fill!(reinterpret(UInt8, x), 0x00); x)

# A kernel timed on a buffer reads it as data, and an integer as the program
# does. A bound compared with positions is at the far end of them, so that
# the sweep it bounds runs whole, as the estimate prices it: at one, a
# decoding step's length bounded its sweep to the first key, and the tile
# fastest there was chosen for every length. An index into an axis is
# spread over the axis, so that a gather reads and a scatter writes in as
# many places as it has elements rather than in one. Both lie in every axis
# the integer indexes; any other integer is one, inside any axis.
function held_integers(graph::IndexGraph, v, T, n)
    axes, bounds = integer_uses(graph, v)
    lo = maximum(first, axes; init = typemin(T))
    hi = minimum(last, axes; init = typemax(T))
    isempty(bounds) || return fill(T(clamp(first(bounds), lo, hi)), n)
    isempty(axes) && return fill(one(T), n)
    spread(T, lo, hi, n)
end

# The ranges of the axes integer value `u` indexes, and the integers at which
# each comparison of positions with it holds at every position.
function integer_uses(graph::IndexGraph, u)
    axes, bounds = Tuple{Int, Int}[], Int[]
    indexes!(e::Affine) = foreach(((a, _),) -> indexes!(a), e.terms)
    indexes!(a::Union{Div, Mod}) = indexes!(a.e)
    indexes!(::Var) = nothing
    function indexes!(a::Indirect)
        a.value == u && push!(axes, (a.lo, a.hi))
        foreach(indexes!, a.index)
    end
    for val in graph.values, (k, o) in enumerate(operands(val.definition))
        o isa Read && foreach(indexes!, o.index)
        o isa IndexOperand && indexes!(o.expr)
        o isa Read && o.value == u || continue
        b = holding_bound(val.definition, k, Domain(val.dims))
        b === nothing || push!(bounds, b)
    end
    axes, bounds
end

# The integer at which comparison `def`, its `k`th operand that integer and
# the other a position, holds at every position: the far end of them.
function holding_bound(def, k, d::Domain)
    def isa Pointwise && def.f in (<=, <, >=, >) && length(def.operands) == 2 && def.operands[3 - k] isa IndexOperand || return nothing
    lo, hi = affine_range(def.operands[3 - k].expr, d)
    holds(c) = all(x -> k == 1 ? def.f(c, x) : def.f(x, c), (lo, hi))
    ends = (hi, lo, hi + 1, lo - 1)
    ends[findfirst(holds, ends)]
end

# `n` integers spread over `lo:hi`, each once while there are as many as it
# holds: a stride coprime with its length, near the golden share of it, as
# Fibonacci hashing spreads keys.
function spread(T, lo, hi, n)
    m = hi - lo + 1
    s = max(1, round(Int, 0.618 * m))
    while gcd(s, m) != 1
        s += 1
    end
    [T(lo + mod((k - 1) * s, m)) for k in 1:n]
end

# An implementation's resources, made once: the launches are built side by
# side (`concurrently`), and making one may wait for the device.
resources!(r::Resources, i) = @lock r.lock get!(() -> Base.invokelatest(resources, i, r.target), r.made, i)

# Build a choice on `buffers`, with the executable's resources `r`.
build_choice(target, p::Plan, g::Group, o::Choice, buffers, r::Resources) =
    o.implementation === nothing ? build_floor(target, p, g, buffers) :
    Base.invokelatest(build_claim, o.implementation, o.claim, target, group_view(p, g), buffers, resources!(r, o.implementation))

# Build a choice on the decider's buffers and resources, once, to measure it:
# its key names the group alike, so the first of them builds for all. A
# recurrence on the floor is measured as it runs, a step at a time where it can.
build_choice!(d::Decider, p::Plan, g::Group, o::Choice) = get!(d.built, o.key) do
    buffers = buffers!(d, p, g)
    stepped = o.implementation === nothing && g.kind isa RecurrenceKind ? build_stepped(d, p, g, buffers, d.resources) : nothing
    stepped === nothing ? build_choice(d.target, p, g, o, buffers, d.resources) : stepped
end

"""
A group's kernels, built on `buffers` with the executable's resources `r`,
and the decision that runs them: its choice, or, when that fails to build,
the next option that builds, the floor last, a decision made `:instead`.
The decider keeps each claim that fails (see `fail!`), so that the cover
plans the program again without it (see `prepare`).
"""
function build_group(d::Decider, p::Plan, g::Group, x, buffers, r::Resources)
    # A recurrence no claim takes runs a step at a time where its step is a
    # program of its own, rather than on the floor (see `build_stepped`).
    if g.kind isa RecurrenceKind && (x === nothing || x.choice.implementation === nothing)
        built = build_stepped(d, p, g, buffers, r)
        built === nothing || return (built..., x)
        isempty(g.reductions) || error("a recurrence with sums riding its steps runs only a step at a time")
    end
    (x === nothing || x.choice.implementation === nothing) && return (build_floor(d.target, p, g, buffers)..., x)
    for o in [x.choice; filter(!=(x.choice), x.options)]
        o.implementation === nothing && break
        built = try
            build_choice(d.target, p, g, o, buffers, r)
        catch err
            err isa InterruptException && rethrow()
            fail!(d, p, g, o, err)
            @warn "a claim failed to build; the next option runs the kernel, and the program is planned again without it" implementation = label(o.implementation) claim = o.claim exception = err maxlog = 1 _id = Symbol(o.key)
            nothing
        end
        built === nothing && continue
        return (built..., o === x.choice ? x : Decision(o, x.options, :instead, nothing, failures(d, p, g)))
    end
    floor = choice(d, p, g, group_view(p, g), nothing, nothing)
    (build_floor(d.target, p, g, buffers)..., Decision(floor, x.options, :instead, nothing, failures(d, p, g)))
end

# Build an option to measure it. An option that fails to build, a claim or
# the floor, loses every comparison in this preparation; the record keeps
# only times.
function build_option!(d::Decider, p::Plan, g::Group, o::Choice)
    try
        first(build_choice!(d, p, g, o))
    catch err
        @debug "claim failed to build; it loses" o.implementation exception = err
        d.record[o.key] = Inf
        fail!(d, p, g, o, err)
        nothing
    end
end

# Build the options with no time yet, `(plan, group, option)` each,
# concurrently, so that their kernels compile side by side before they are
# timed one at a time, alone on the device.
function build_unmeasured!(d::Decider, options)
    unmeasured = unique(x -> x[3].key, filter(x -> !haskey(d.record, x[3].key), options))
    phase(() -> concurrently(((p, g, o),) -> build_option!(d, p, g, o), unmeasured, d.target), :burst)
end

function measure!(d::Decider, p::Plan, g::Group, o::Choice)
    haskey(d.record, o.key) && return
    launch = build_option!(d, p, g, o)
    launch === nothing && return
    t = phase(() -> d.timer(d.target, launch), :timing)
    keep!(d.record, d.record_path, o.key, t)
    nothing
end
