# Recurrences: the states a program carries from step to step with the
# cycle of values between, the sums that can ride their steps, and the group
# of the kernel that steps one.

"""
A recurrence as the index graph has it, a cycle: its carried states, the
value each is carried from, in the states' order, and the cycle's values,
those that read a state and that a value a state is carried from reads,
both through others: what its group computes at each step.
"""
struct Cycle
    states::Vector{Int}
    from::Vector{Int}
    values::Vector{Int}
end

# A graph's recurrences. States each carried from a value reading the other,
# through others, are one recurrence's: an LSTM's hidden state and its cell,
# each step reading both, and two states assigned at once,
# `s1, s2 = f(s2), g(s1)`, though neither state's value is carried from one
# that reads it.
function recurrences(graph)
    carried = Tuple{Int, Int}[]
    for (c, v) in enumerate(graph.values)
        d = v.definition
        d isa Carried || continue
        from = d.from
        from isa Read && from.value > c || continue
        push!(carried, (c, from.value))
    end
    cycles = [recurrence_cycle(graph, [c], [f]) for (c, f) in carried]
    length(carried) <= 1 && return Cycle[Cycle([c], [f], cycle) for ((c, f), cycle) in zip(carried, cycles)]
    up = ancestors(graph)
    # Which states each state's value is carried from reads, through others.
    n = length(carried)
    reach = [[up[f][c] for (c, _) in carried] for (_, f) in carried]
    for m in 1:n, k in 1:n
        reach[k][m] || continue
        for j in 1:n
            reach[m][j] && (reach[k][j] = true)
        end
    end
    sets = Vector{Int}[]
    for k in 1:n
        any(set -> k in set, sets) && continue
        push!(sets, [j for j in 1:n if j == k || reach[k][j] && reach[j][k]])
    end
    map(sort!(sets; by = minimum)) do set
        states = [carried[k][1] for k in sort(set)]
        from = [carried[k][2] for k in sort(set)]
        Cycle(states, from, length(states) == 1 ? cycles[only(set)] : recurrence_cycle(graph, states, from))
    end
end

# The states of recurrence group `g`, its first outputs, and the values each is carried from.
recurrence_states(graph, members) = [m for m in members if graph.values[m].definition isa Carried]
carried_from(graph, c) = (graph.values[c].definition.from::Read).value

"""
A sum over a recurrence's steps of what they compute, riding them: each
step adds its part as the steps run, as a weight's gradient adds each
step's while its operands are fresh. It is a fold of the recurrence's group,
complete only after the steps.
"""
struct Summed end

"""
A value along a recurrence's steps, at its axis `axis` there, riding them:
each step writes its own, as a recurrent network's output at each step. It
is a member of the recurrence's group.
"""
struct Along
    axis::Int
end

"""
A value of a recurrence's last step alone, riding the steps: the last
writes it, what the steps leave, as the state after a loop. It is a member
of the recurrence's group that lies along none of its steps (see
`last_step`).
"""
struct AtLast end

"""
How a value meets a recurrence's steps: along them (`Along`), at their last
(`AtLast`), or, for a value riding them, summed over them (`Summed`). A union,
as a group's kinds are (see `GroupKind`): each way says by a method whether
the group takes the value as a fold or as a member (`folded`), whether the
steps can compute it so (`fits_steps`), and, to the cover, where it is
offered and how `explain` words it (`offered`, `ridden`).
"""
const Meeting = Union{Summed, Along, AtLast}

Base.show(io::IO, ::Summed) = print(io, "summed")
Base.show(io::IO, a::Along) = print(io, "along ", a.axis)
Base.show(io::IO, ::AtLast) = print(io, "at the last")

"""
Whether a value riding a recurrence's steps is one of its group's folds,
complete after the steps and read by nothing in the group, rather than a
member a step writes. Only a run a step at a time computes a recurrence's
folds (see `build_stepped`).
"""
folded(::Summed) = true
folded(::Union{Along, AtLast}) = false

"""
    fits_steps(how, graph, cycle, between, r) -> Bool

Whether the steps of a recurrence whose cycle is `cycle` compute value `r`,
through the values `between`, as `how` says it meets them. A sum adds its
parts as they come, which a sum does in any order. A member of the group has
a place among the state's axes, and lies along each of the recurrence's
lanes, the state's axes that every value of its cycle lies along, which the
steps run apart: each lane computes its part from its own. One summing across
them, as a projection of a network's output over its heads, would make of
the steps a single lane, reading its weights again at every step.
"""
fits_steps(::Summed, graph, cycle, between, r) = graph.values[r].definition.fold isa SumFold
function fits_steps(::Union{Along, AtLast}, graph, cycle, between, r)
    values = graph.values
    states = [c for c in cycle if values[c].definition isa Carried]
    lies = lying(graph, states, sort!([cycle; between; r]))
    lies === nothing && return false
    state = values[first(states)]
    lanes = [q for q in eachindex(state.dims) if q != state.definition.axis && all(c -> lies[c][q] != 0, cycle)]
    all(u -> all(q -> lies[u][q] != 0, lanes), [between; r])
end

"""
    riding(graph, live, cycle, r, reaching = reaching_from(graph, cycle)) -> (; between, how) or nothing

Value `r` riding a recurrence's steps: computed at each step from the values
of its `cycle` there, through `between`, the values `r` reads between it and
the cycle, which the steps compute with it, meeting the steps as `how` says
(see `Meeting`). Each of `between` is a map or a fold that only `r` and the
others read; nothing where `r` reads no value of the cycle, or another reads
one of them, or a state is carried from one, or a value reads the cycle at
another step than its own or the last, or along two of its axes, or a fold
sums the steps without being `r`, or the steps cannot compute `r` as it meets
them (`fits_steps`).
"""
function riding(graph, live, cycle, r, reaching = reaching_from(graph, cycle))
    values = graph.values
    d = values[r].definition
    (; inside, reaches, users, carried) = reaching
    d isa Union{Pointwise, Reduction} && !(r in inside) && reaches[r] || return nothing
    # The values `r` reads that read the cycle, through others: each a map
    # or a fold that nothing after `r` reads, or `r` rides nothing, found
    # before the walk goes on to what they read, which for a value of a
    # later layer is every layer back to the recurrence's.
    found, pending = Set{Int}(), [r]
    while !isempty(pending)
        u = pop!(pending)
        for o in operands(values[u].definition), x in read_values(o)
            reaches[x] && !(x in inside) && !(x in found) || continue
            values[x].definition isa Union{Pointwise, Reduction} && all(w -> w <= r || !live[w], users[x]) || return nothing
            push!(found, x)
            push!(pending, x)
        end
    end
    between = sort!(collect(found))
    set = Set(push!(copy(between), r))
    all(u -> all(w -> !live[w] || w in set, users[u]), between) || return nothing
    # A state reads the value it is carried from, which no reader counts.
    any(((_, f),) -> f in set, carried) && return nothing
    # How each value of the cycle and between meets the steps, from the
    # states', along them at an axis of its own; `r` meets them along an axis
    # it sums, a step's along its own, or at the last, not otherwise.
    state = values[first(c for c in cycle if values[c].definition isa Carried)]
    carry = state.definition::Carried
    steps, final = state.dims[carry.axis], carry.reverse ? 0 : state.dims[carry.axis] - 1
    meets = Dict{Int, Union{Along, AtLast}}(c => Along(values[c].definition.axis) for c in cycle if values[c].definition isa Carried)
    for u in sort!([[w for w in cycle if !haskey(meets, w)]; between])
        e = values[u].definition
        m = meeting(e, values[u].dims, meets, steps, final)
        m isa Astray && return nothing
        m === nothing && continue
        m = as_value(e, m)
        m isa Summed && return nothing
        meets[u] = m
    end
    m = meeting(d, values[r].dims, meets, steps, final)
    (m === nothing || m isa Astray) && return nothing
    how = as_value(d, m)
    fits_steps(how, graph, cycle, between, r) || return nothing
    (; between, how)
end

"""
What `riding` asks of a graph about a recurrence's cycle, whatever value
rides: the cycle's values (`inside`); each value that reads them, through
others or a state carried from one that does, as a second recurrence whose
step reads the first's (`reaches`); each value's readers; and each state
with the value it is carried from.
"""
struct Reaching
    inside::Set{Int}
    reaches::BitVector
    users::Vector{Vector{Int}}
    carried::Vector{Tuple{Int, Int}}
end

function reaching_from(graph, cycle, users = readers(graph), feeds = feeding(graph))
    values = graph.values
    inside = Set(cycle)
    carried = [(c, carried_from(graph, c)) for (c, v) in enumerate(values) if v.definition isa Carried && v.definition.from isa Read]
    state_of = Dict(f => c for (c, f) in carried)
    # Forward from the cycle, through what reads each value and the state
    # carried from it: a pass over every value until none grew took a pass
    # a recurrence after it, 1 s a plan of Qwen3.5-4B's.
    reaches = falses(length(values))
    pending = collect(cycle)
    foreach(u -> reaches[u] = true, pending)
    while !isempty(pending)
        u = pop!(pending)
        for w in [feeds[u]; get(state_of, u, Int[])]
            reaches[w] || (reaches[w] = true; push!(pending, w))
        end
    end
    Reaching(inside, reaches, users, carried)
end

# The values that read each value, through any operand, an index gathered
# by among them.
function feeding(graph)
    feeds = [Int[] for _ in graph.values]
    for (id, v) in enumerate(graph.values), o in operands(v.definition), x in read_values(o)
        push!(feeds[x], id)
    end
    feeds
end

"What `meeting` gives a value that reads a recurrence's steps otherwise than one at a time or at the last: no step computes it."
struct Astray end

# How `d` meets a recurrence's `steps`, by how each value it reads meets them
# (`meets`): along them at the axis of its iteration that reads each a step at
# a time, the whole of the steps along it; at the last, where it reads each
# at position `final`, the last step's, or a value of the last; nothing where
# it reads none; astray where it reads one otherwise, along two axes, or
# both. A map iterates over its axes, of extents `dims`, a fold over its
# iteration's.
function meeting(d, dims, meets, steps, final)
    extents = d isa Reduction ? d.iteration : dims
    n = length(extents)
    m = nothing
    for o in operands(d)
        o isa Read && haskey(meets, o.value) || continue
        j = met_at(meets[o.value], o.index, extents, steps, final)
        j === nothing && return Astray()
        m === nothing || m == j || return Astray()
        m = j
    end
    m
end

# How the value of `d`, meeting the steps as `m` over its iteration, meets
# them over its own axes: a fold along them at an axis it keeps, along them at
# that axis among those kept; one summing the axis, a sum over them.
as_value(d, m) = m
as_value(d::Reduction, m::Along) = m.axis in d.reduced ? Summed() : Along(count(j -> !(j in d.reduced), 1:m.axis))

# How a read at `index` of a value meeting the steps as `a` meets them.
met_at(::AtLast, index, extents, steps, final) = AtLast()
function met_at(a::Along, index, extents, steps, final)
    e = index[a.axis]
    e == constant(final) && return AtLast()
    i = findfirst(i -> e == var(i) && extents[i] == steps, eachindex(extents))
    i === nothing ? nothing : Along(i)
end

function recurrence_cycle(graph, cs, fs)
    values = graph.values
    lo, hi = minimum(cs), maximum(fs)
    reaches, feeds = falses(hi), falses(hi)
    foreach(c -> reaches[c] = true, cs)
    foreach(f -> feeds[f] = true, fs)
    for w in lo:hi
        reaches[w] && continue
        reaches[w] = any(o -> any(r -> r <= hi && reaches[r], read_values(o)), operands(values[w].definition))
    end
    for w in hi:-1:lo
        feeds[w] || continue
        for o in operands(values[w].definition), r in read_values(o)
            lo <= r && (feeds[r] = true)
        end
    end
    cycle = [w for w in lo:hi if reaches[w] && feeds[w]]
    all(w -> w in cs || values[w].definition isa Union{Pointwise, Reduction}, cycle) ||
        throw(ArgumentError("a recurrence whose step scans, scatters or writes is not supported yet"))
    cycle
end

"""
    recurrence_places(graph, members) -> (; domain, places)

A recurrence's group's domain and where it keeps each of `members`, the
states first, as the member's index over it. The domain is the states', then an
axis for each axis of a member along none of the state's. A value of the
cycle lies along the state's axis `q` at its own axis `r` where it reads the
state, or a value of the cycle lying along `q` at its axis `s`, with `s`
indexed by `r`'s variable: as the value the state is carried from lies along
the state's own axes, and as flash-linear-attention's `v_new`, `Ũ - W S`,
lies along the state's columns and chunks, its rows an axis of its own.
"""
function recurrence_places(graph, members)
    values = graph.values
    cs = recurrence_states(graph, members)
    fs = [carried_from(graph, c) for c in cs]
    c = first(cs)
    d = values[c].definition::Carried
    D = values[c].dims
    # The values a member riding the steps reads between it and the cycle.
    cycle = recurrence_cycle(graph, cs, fs)
    lo = minimum(cs)
    between, pending = Set{Int}(), [m for m in members if !(m in cycle)]
    while !isempty(pending)
        u = pop!(pending)
        for o in operands(values[u].definition), x in read_values(o)
            x >= lo && !(x in cycle) && !(x in members) && !(x in between) && (push!(between, x); push!(pending, x))
        end
    end
    along = lying(graph, cs, sort!([cycle; collect(between); [m for m in members if !(m in cycle)]]))
    along === nothing && throw(ArgumentError("a recurrence whose step reads a value along two of the state's axes is not supported yet"))
    domain = collect(D)
    places = Dict{Int, Tuple{Vararg{Affine}}}()
    for m in members
        (m in cs || m in fs) && (places[m] = identity_index(D); continue)
        # A value of the last step alone lies along none of them.
        along[m][d.axis] == 0 && m in cycle &&
            throw(ArgumentError("a recurrence whose step computes a value not along its steps is not supported yet"))
        places[m] = ntuple(length(values[m].dims)) do r
            q = findfirst(==(r), along[m])
            q === nothing || return var(q)
            push!(domain, values[m].dims[r])
            var(length(domain))
        end
    end
    (; domain = Tuple(domain), places)
end

# Where each of values `ws`, in the graph's order, lies along the axes of
# recurrence states `cs`: at each of the states' axes, the value's own axis
# there, 0 for none. A value lies along a state's axis `q` at its axis `r`
# where it reads, at `r`'s variable, a value lying along `q`. Nothing where
# one lies along a state's axis at two of its own.
function lying(graph, cs, ws)
    values = graph.values
    along = Dict{Int, Vector{Int}}(c => collect(eachindex(values[c].dims)) for c in cs)
    for w in ws
        w in cs && continue
        e = values[w].definition
        kept = e isa Reduction ? [k for k in eachindex(e.iteration) if !(k in e.reduced)] : collect(eachindex(values[w].dims))
        at = zeros(Int, length(values[first(cs)].dims))
        for o in operands(e)
            o isa Read && haskey(along, o.value) || continue
            for (q, s) in enumerate(along[o.value])
                s == 0 && continue
                r = findfirst(k -> o.index[s] == var(k), kept)
                r === nothing && continue
                at[q] in (0, r) || return nothing
                at[q] = r
            end
        end
        along[w] = at
    end
    along
end

# A recurrence's group, stepping along the states' axis: the states, each
# where it starts or else the value it is carried from one step back; the
# values of the cycle stored, the values the states are carried from, stored
# or not, each its state one step on, and the values riding the steps along
# them or at their last, in the graph's order. Their terms recompute the rest
# of the cycle and read what the group computes wherever this step left it;
# a value of the last step reads the step it is computed at, the last, and
# lies along none of them (see `last_step`). The sums riding its steps are
# its folds, each over its own iteration: as a fold along a scan's axes rides
# its sweep, each is complete only after the steps, and nothing in the group
# reads it. `riders` pairs each value riding the steps with how it meets
# them (see `Meeting`), which says which it is (`folded`).
function recurrence_group(graph, cx::Forming, recurrence::Cycle, riders = Pair{Int, Meeting}[])
    values = graph.values
    cs, fs, cycle = recurrence.states, recurrence.from, recurrence.values
    c = first(cs)
    d = values[c].definition
    all(k -> values[k].dims == values[c].dims && values[k].definition.axis == d.axis && values[k].definition.reverse == d.reverse, cs) ||
        throw(ArgumentError("a recurrence whose states differ in shape or direction is not supported yet"))
    all(((k, f),) -> values[f].dims == values[k].dims, zip(cs, fs)) ||
        throw(ArgumentError("a recurrence's state carried from a value of another shape is not supported yet"))
    allunique([cs; fs]) || throw(ArgumentError("a recurrence's state carried from another state is not supported yet"))
    # The states, then the rest in the graph's order, which a step computes
    # them in: an LSTM's hidden state one step on reads its cell's.
    sums = [r for (r, how) in riders if folded(how)]
    stepped = [r for (r, how) in riders if !folded(how)]
    members = [cs; sort!([[w for w in cycle if !(w in cs) && !(w in fs) && isstored(cx, w)]; fs; stepped])]
    (; domain, places) = recurrence_places(graph, members)
    g = Group(RecurrenceKind(), domain)
    inside = Stepping(cx, Set(w for w in cycle if !(w in members)))
    append!(g.outputs, members)
    whole = Domain(domain)
    memo = Dict{Any, Term}()
    foreach(c -> g.terms[c] = operand_term(graph, inside, g, values[c].definition.init, places[c], whole, memo), cs)
    for m in members[(length(cs) + 1):end]
        g.terms[m] = values[m].definition isa Reduction ? fold_term(graph, inside, g, m, places[m], whole, memo) :
            definition_term(graph, inside, g, values[m].definition, values[m], places[m], whole, memo)
    end
    # A value of the last step reads each member there, at the step it is
    # computed at.
    at = Dict(m => findfirst(==(var(d.axis)), places[m]) for m in members if any(==(var(d.axis)), places[m]))
    final = d.reverse ? 0 : domain[d.axis] - 1
    for m in members
        last_step(places[m], d.axis) && (g.terms[m] = at_the_step(g.terms[m], at, constant(final), var(d.axis)))
    end
    foreach(m -> g.phase[m] = 0, members)
    for r in sums
        d = values[r].definition::Reduction
        push!(g.reductions, r)
        g.terms[r] = operand_term(graph, inside, g, d.operand, identity_index(d.iteration), Domain(d.iteration), memo)
        g.phase[r] = 0
    end
    g
end

"""
    last_step(place, axis) -> Bool

Whether a member of a recurrence's group at `place` over its domain is a
value of the last step alone: one along none of the steps (the domain's
`axis`), which the group writes at the last of them, what the steps leave.
"""
last_step(place, axis) = !any(e -> mentions(e, axis), place)

# Term `t` with each load of a member at `from` along the steps, the member
# lying along them at its axis `at[member]`, made a load at `to`.
at_the_step(t::Term, at, from, to) = t
function at_the_step(t::Load, at, from, to)
    a = get(at, t.value, nothing)
    a === nothing || t.index[a] != from ? t : Load(t.value, Base.setindex(t.index, to, a))
end
at_the_step(t::Apply, at, from, to) = Apply(t.f, map(x -> at_the_step(x, at, from, to), t.args), t.eltype)
at_the_step(t::Recomputed, at, from, to) = Recomputed(t.value, t.index, t.extents, at_the_step(t.body, at, from, to))

"A recurrence group's axes (see `Tempest.Compiler.Implement.recurrence_axes`)."
function recurrence_axes(graph, g::Group)
    state = graph.values[first(g.outputs)]
    Implement.recurrence_axes(state.definition.axis, length(state.dims), g.domain, g.outputs, g.terms,
        recurrence_places(graph, g.outputs).places)
end
