# Recurrences: the states a program carries from step to step with the
# cycle of values between, the sums that can ride their steps, and the group
# of the kernel that steps one.

# Recurrences: their carried states, the value each is carried from, and the
# cycle between, the values that read a state and that a value a state is
# carried from reads, both through others: what its group computes at each
# step. States each carried from a value reading the other, through others,
# are one recurrence's: an LSTM's hidden state and its cell, each step
# reading both, and two states assigned at once, `s1, s2 = f(s2), g(s1)`,
# though neither state's value is carried from one that reads it.
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
    length(carried) <= 1 && return [([c], [f], cycle) for ((c, f), cycle) in zip(carried, cycles)]
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
        cs, fs = [carried[k][1] for k in sort(set)], [carried[k][2] for k in sort(set)]
        (cs, fs, length(cs) == 1 ? cycles[only(set)] : recurrence_cycle(graph, cs, fs))
    end
end

# The states of recurrence group `g`, its first outputs, and the values each is carried from.
recurrence_states(graph, members) = [m for m in members if graph.values[m].definition isa Carried]
carried_from(graph, c) = (graph.values[c].definition.from::Read).value

"""
    riding(graph, live, cycle, r) -> values or nothing

The values a sum `r` reads between it and a recurrence's `cycle`, which ride
the recurrence's steps with it: computed at each step from the cycle's
values there, the sum adding each step's part as the steps run, as a weight's
gradient adds each step's while its operands are fresh. Each is a map or a
fold that only `r` and the others read; nothing where `r` reads no value of
the cycle, or another reads one of them, or a state is carried from one, or
`r` sums an axis other than the steps.
"""
function riding(graph, live, cycle, r)
    values = graph.values
    d = values[r].definition
    d isa Reduction && d.fold isa SumFold && !(r in cycle) || return nothing
    inside = Set(cycle)
    # Each value that reads the cycle, through others or a state carried
    # from one that does, as a second recurrence whose step reads the
    # first's; and those `r` reads, through others.
    reaches, read = falses(length(values)), falses(r)
    read[r] = true
    carried = [(c, carried_from(graph, c)) for (c, v) in enumerate(values) if v.definition isa Carried && v.definition.from isa Read]
    while true
        for u in eachindex(values)
            reaches[u] && continue
            reaches[u] = u in inside || any(o -> any(x -> reaches[x], read_values(o)), operands(values[u].definition))
        end
        grown = false
        for (c, f) in carried
            !reaches[c] && reaches[f] && (reaches[c] = grown = true)
        end
        grown || break
    end
    reaches[r] || return nothing
    for u in r:-1:1
        read[u] || continue
        foreach(o -> foreach(x -> read[x] = true, read_values(o)), operands(values[u].definition))
    end
    between = [u for u in 1:(r - 1) if read[u] && reaches[u] && !(u in inside)]
    all(u -> values[u].definition isa Union{Pointwise, Reduction}, between) || return nothing
    set = Set(push!(copy(between), r))
    users = readers(graph)
    all(u -> all(w -> !live[w] || w in set, users[u]), between) || return nothing
    # A state reads the value it is carried from, which no reader counts.
    any(((_, f),) -> f in set, carried) && return nothing
    # Each value of the cycle and between at its axis along the steps, from
    # the states'; the sum adds them up where it reads them along an axis it
    # sums, not a step's alone or along another.
    along = Dict{Int, Int}(c => values[c].definition.axis for c in cycle if values[c].definition isa Carried)
    for u in sort!([[w for w in cycle if !haskey(along, w)]; between])
        e = values[u].definition
        k = steps_iteration(e, length(values[u].dims), along)
        k === nothing && return nothing
        k == 0 && continue
        if e isa Reduction
            k in e.reduced && return nothing
            k = count(j -> !(j in e.reduced), 1:k)
        end
        along[u] = k
    end
    k = steps_iteration(d, 0, along)
    k isa Int && k in d.reduced || return nothing
    between
end

# The axis of `d`'s iteration along a recurrence's steps, where it reads each
# value along them, by its axis there in `along`, a step at a time: 0 where
# it reads none, nothing where it reads one otherwise or along two. A map
# iterates over its `rank` axes, a fold over its iteration's.
function steps_iteration(d, rank, along)
    n = d isa Reduction ? length(d.iteration) : rank
    k = 0
    for o in operands(d)
        o isa Read && haskey(along, o.value) || continue
        e = o.index[along[o.value]]
        j = findfirst(i -> e == var(i), 1:n)
        j === nothing && return nothing
        k in (0, j) || return nothing
        k = j
    end
    k
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
    along = Dict{Int, Vector{Int}}(c => collect(eachindex(D)) for c in cs)
    for w in recurrence_cycle(graph, cs, fs)
        w in cs && continue
        e = values[w].definition
        kept = e isa Reduction ? [k for k in eachindex(e.iteration) if !(k in e.reduced)] : collect(eachindex(values[w].dims))
        at = zeros(Int, length(D))
        for o in operands(e)
            o isa Read && haskey(along, o.value) || continue
            for (q, s) in enumerate(along[o.value])
                s == 0 && continue
                r = findfirst(k -> o.index[s] == var(k), kept)
                r === nothing && continue
                at[q] in (0, r) || throw(ArgumentError("a recurrence whose step reads a value along two of the state's axes is not supported yet"))
                at[q] = r
            end
        end
        along[w] = at
    end
    domain = collect(D)
    places = Dict{Int, Tuple{Vararg{Affine}}}()
    for m in members
        (m in cs || m in fs) && (places[m] = identity_index(D); continue)
        along[m][d.axis] == 0 && throw(ArgumentError("a recurrence whose step computes a value not along its steps is not supported yet"))
        places[m] = ntuple(length(values[m].dims)) do r
            q = findfirst(==(r), along[m])
            q === nothing || return var(q)
            push!(domain, values[m].dims[r])
            var(length(domain))
        end
    end
    (; domain = Tuple(domain), places)
end

# A recurrence's group, stepping along the states' axis: the states, each
# stored at each step, where it starts or else the value it is carried from
# one step back; the values of the cycle stored, in order; and the values
# the states are carried from, stored or not, in the states' order, each its
# state one step on. Their terms recompute the rest of the cycle
# and read what the group computes wherever this step stored it. The sums
# riding its steps are its folds, each over its own iteration: as a fold
# along a scan's axes rides its sweep, each is complete only after the
# steps, and nothing in the group reads it.
function recurrence_group(graph, cx::Forming, cs, fs, cycle, riders = Int[])
    values = graph.values
    c = first(cs)
    d = values[c].definition
    all(k -> values[k].dims == values[c].dims && values[k].definition.axis == d.axis && values[k].definition.reverse == d.reverse, cs) ||
        throw(ArgumentError("a recurrence whose states differ in shape or direction is not supported yet"))
    all(((k, f),) -> values[f].dims == values[k].dims, zip(cs, fs)) ||
        throw(ArgumentError("a recurrence's state carried from a value of another shape is not supported yet"))
    allunique([cs; fs]) || throw(ArgumentError("a recurrence's state carried from another state is not supported yet"))
    # The states, then the rest in the graph's order, which a step computes
    # them in: an LSTM's hidden state one step on reads its cell's.
    members = [cs; sort!([[w for w in cycle if !(w in cs) && !(w in fs) && isstored(cx, w)]; fs])]
    (; domain, places) = recurrence_places(graph, members)
    g = Group(:recurrence, domain)
    inside = Stepping(cx, Set(w for w in cycle if !(w in members)))
    append!(g.outputs, members)
    whole = Domain(domain)
    memo = Dict{Any, Term}()
    foreach(c -> g.terms[c] = operand_term(graph, inside, g, values[c].definition.init, places[c], whole, memo), cs)
    for m in members[(length(cs) + 1):end]
        g.terms[m] = values[m].definition isa Reduction ? fold_term(graph, inside, g, m, places[m], whole, memo) :
            definition_term(graph, inside, g, values[m].definition, values[m], places[m], whole, memo)
    end
    foreach(m -> g.phase[m] = 0, members)
    for r in riders
        d = values[r].definition::Reduction
        push!(g.reductions, r)
        g.terms[r] = operand_term(graph, inside, g, d.operand, identity_index(d.iteration), Domain(d.iteration), memo)
        g.phase[r] = 0
    end
    g
end

"A recurrence group's axes (see `Tempest.Compiler.Implement.recurrence_axes`)."
function recurrence_axes(graph, g::Group)
    state = graph.values[first(g.outputs)]
    Implement.recurrence_axes(state.definition.axis, length(state.dims), g.domain, g.outputs, g.terms,
        recurrence_places(graph, g.outputs).places)
end
