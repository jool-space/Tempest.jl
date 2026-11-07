# A recurrence stepped: its group's cycle restricted to one step, planned as
# a program of its own, and its kernels launched once per step, pointed at
# that step's place in the recurrence's storage. A one-kernel claim keeps
# the state on chip; this runs a step too large for one kernel, a neural
# ODE's field evaluated four times an RK4 step, as the kernels of an
# unrolled step, from one plan of it. A sum riding the steps (see `riding`)
# adds each step's part where it is: the step's plan adds it to the sum, in
# the sum's storage.

"""
One step of a recurrence as a graph of its own: its arguments are the states
(the first arguments, in the group's order), the values outside the cycle it reads, each whole or one
step's slice of it, and each sum riding the steps as the steps before left
it; its results are the values the states are carried from, in the states'
order, the cycle's values stored at each step, and each sum with this step's
part added.
"""
struct StepGraph
    graph::IndexGraph
    # Each argument's value outside and the axis it is sliced along, 0 for
    # whole; `(:position, T)`, the step's position as a `T`; or `(:sum, r)`,
    # sum `r` riding the steps.
    leaves::Vector{Any}
    # Each result's value outside, stored a step at a time; the sums' results follow.
    stored::Vector{Int}
    states::Int
    sums::Vector{Int}
end

eltype_of(::IndexOperand{T}) where {T} = T

# `dims` without axis `s`.
drop_axis(dims, s) = s == 0 ? Tuple(dims) : (dims[1:(s - 1)]..., dims[(s + 1):end]...)

# The variables of a domain without axis `s`, as the domain's: each after it one down.
function renumbered(e::Affine, s, d::Domain)
    s == 0 && return e
    mentions(e, s) && return nothing
    substitute(e, [k < s ? var(k) : k == s ? constant(0) : var(k - 1) for k in 1:(length(d.extents) + 1)], d)
end

"""
    step_graph(graph, stored, g) -> StepGraph or nothing

The cycle of recurrence group `g` at one step, the values it recomputes
outside the cycle with it, and each part of the sums riding its steps; or
nothing where a step is not a program of its own: where it reads a value of
the cycle at another step, or a value along the steps other than by its last
axis, or by an index array, or a sum riding the steps reads none along them.
"""
function step_graph(graph::IndexGraph, stored, g::Group)
    values = graph.values
    cs = recurrence_states(graph, g.outputs)
    fs = [carried_from(graph, c) for c in cs]
    c = first(cs)
    cd = values[c].definition::Carried
    all(c -> values[c].definition.from.index == identity_index(values[c].dims), cs) || return nothing
    cycle = Set(recurrence_cycle(graph, cs, fs))
    places = recurrence_places(graph, g.outputs).places
    # The axis along the steps of each member, from its place.
    along(m) = findfirst(==(var(cd.axis)), places[m])
    # The values the states are carried from first, in the states' order.
    members = [fs; [m for m in g.outputs if !(m in cs) && !(m in fs)]]
    any(m -> along(m) === nothing, members) && return nothing
    # The values a step needs, each with its axis along the steps or 0, and
    # how: :state, :leaf, or :step, computed in the step.
    kinds = Dict{Tuple{Int, Int}, Symbol}()
    order = Tuple{Int, Int}[]
    positions = Set{DataType}()
    ok = Ref(true)
    function visit(u, s)
        haskey(kinds, (u, s)) && return
        kind = if u in cs
            s == cd.axis || (ok[] = false)
            :state
        elseif u in cycle
            (s == 0 || u in members && s != along(u)) && (ok[] = false)
            :step
        elseif stored[u]
            # Sliced along its last axis, a step's slice is contiguous.
            s in (0, length(values[u].dims)) || (ok[] = false)
            :leaf
        else
            :step
        end
        kinds[(u, s)] = kind
        ok[] || return
        if kind === :step
            v = values[u]
            d = v.definition
            d isa Union{Pointwise, Reduction} || (ok[] = false; return)
            si = d isa Reduction && s > 0 ? [k for k in eachindex(d.iteration) if !(k in d.reduced)][s] : s
            foreach(o -> visit_operand(o, si), operands(d))
        end
        push!(order, (u, s))
    end
    function visit_operand(o, si)
        if o isa IndexOperand
            si > 0 && mentions(o.expr, si) || return
            # The step's position, read from an argument a step at a time.
            o.expr == var(si) ? push!(positions, eltype_of(o)) : (ok[] = false)
            return
        end
        o isa Read || return
        all(is_affine, o.index) || (ok[] = false; return)
        at = si == 0 ? Int[] : findall(e -> mentions(e, si), collect(o.index))
        if isempty(at)
            visit(o.value, 0)
        elseif length(at) == 1 && o.index[only(at)] == var(si)
            visit(o.value, only(at))
        else
            ok[] = false
        end
    end
    for m in members
        visit(m, along(m))
        ok[] || return nothing
    end
    ok[] || return nothing
    # Each sum riding the steps, by the axis of its iteration along them: the
    # first it sums whose operand is a step's there.
    sums = g.reductions
    axes = map(sums) do r
        d = values[r].definition
        for k in d.reduced
            d.operand isa Read && any(e -> mentions(e, k), d.operand.index) || continue
            saved = (copy(kinds), copy(order), copy(positions))
            visit_operand(d.operand, k)
            ok[] && return k
            ok[] = true
            merge!(empty!(kinds), saved[1])
            append!(empty!(order), saved[2])
            union!(empty!(positions), saved[3])
        end
        0
    end
    any(==(0), axes) && return nothing
    # The arguments first, the states the first of them, the sums last; then
    # the step's values in the graph's order.
    leaves = Any[[(c, cd.axis) for c in cs]; [key for key in order if kinds[key] === :leaf]; [(:position, T) for T in positions]; [(:sum, r) for r in sums]]
    steps = [key for key in sort(order) if kinds[key] === :step]
    id = Dict{Any, Int}()
    new = Value[]
    for (k, leaf) in enumerate(leaves)
        if first(leaf) === :position
            push!(new, Value(Input(k), (), last(leaf), true))
        elseif first(leaf) === :sum
            v = values[last(leaf)]
            push!(new, Value(Input(k), v.dims, v.eltype, v.scalar))
        else
            u, s = leaf
            v = values[u]
            push!(new, Value(Input(k), drop_axis(v.dims, s), v.eltype, s == 0 && v.scalar))
        end
        id[leaf] = length(new)
    end
    for (u, s) in steps
        v = values[u]
        d = v.definition
        if d isa Reduction
            kept = [k for k in eachindex(d.iteration) if !(k in d.reduced)]
            si = s == 0 ? 0 : kept[s]
            iteration = drop_axis(d.iteration, si)
            o = restricted(d.operand, si, Domain(iteration), id)
            o === nothing && return nothing
            reduced = Tuple(k > si && si > 0 ? k - 1 : k for k in d.reduced)
            push!(new, Value(Reduction(d.fold, o, iteration, reduced), drop_axis(v.dims, s), v.eltype, v.scalar))
        else
            dims = drop_axis(v.dims, s)
            os = map(o -> restricted(o, s, Domain(dims), id), d.operands)
            any(isnothing, os) && return nothing
            push!(new, Value(Pointwise(d.f, Tuple(os)), dims, v.eltype, v.scalar))
        end
        id[(u, s)] = length(new)
    end
    results = Operand[]
    for m in members
        b = id[(m, along(m))]
        push!(results, Read(b, identity_index(new[b].dims)))
    end
    stored_order = members
    # Each sum, the steps before's, plus this step's part: its operand here,
    # summed over its other axes. A part that is a sum of others, each added
    # in turn, lets each product add into the sum as it writes.
    computed = length(leaves)
    for (j, (r, k)) in enumerate(zip(sums, axes))
        d = values[r].definition
        v = values[r]
        kept = drop_axis(d.iteration, k)
        o = restricted(d.operand, k, Domain(kept), id)
        o === nothing && return nothing
        reduced = Tuple(a > k ? a - 1 : a for a in d.reduced if a != k)
        if !isempty(reduced)
            push!(new, Value(Reduction(d.fold, o, kept, reduced), v.dims, v.eltype, v.scalar))
            o = Read(length(new), identity_index(v.dims))
        end
        total = Read(length(leaves) - length(sums) + j, identity_index(v.dims))
        for part in summands(new, o, computed)
            push!(new, Value(Pointwise(+, (total, part)), v.dims, v.eltype, v.scalar))
            total = Read(length(new), identity_index(v.dims))
        end
        push!(results, total)
    end
    dims = [new[r.value].dims for r in results]
    body = IndexGraph(new, collect(1:length(leaves)), results, dims, [isempty(d) for d in dims], [new[r.value].eltype for r in results])
    StepGraph(body, leaves, stored_order, length(cs), sums)
end

# The operands sum `o` adds, a value of the step past `computed` read whole:
# those of each sum of two it is.
function summands(new, o, computed)
    o isa Read && o.value > computed && o.index == identity_index(new[o.value].dims) || return Operand[o]
    d = new[o.value].definition
    d isa Pointwise && d.f === (+) && length(d.operands) == 2 || return Operand[o]
    Operand[summands(new, d.operands[1], computed); summands(new, d.operands[2], computed)]
end

# Operand `o` of a value computed in the step, over the step's domain `d`:
# axis `s` of the value's own, the steps, left out.
function restricted(o::Operand, s, d::Domain, id)
    o isa Literal && return o
    if o isa IndexOperand
        s > 0 && o.expr == var(s) && return Read(id[(:position, eltype_of(o))], ())
        e = renumbered(o.expr, s, d)
        return e === nothing ? nothing : typeof(o)(e)
    end
    o isa Read || return nothing
    at = s == 0 ? 0 : something(findfirst(e -> mentions(e, s), collect(o.index)), 0)
    key = (o.value, at)
    haskey(id, key) || return nothing
    index = Affine[]
    for (k, e) in enumerate(o.index)
        k == at && continue
        r = renumbered(e, s, d)
        r === nothing && return nothing
        push!(index, r)
    end
    Read(id[key], Tuple(index))
end

"""
A recurrence's steps: the step's kernels, launched once per step, each
pointed first at the step's slices of what it reads and stores along the
steps. Before the first, the state's first step is set where the
recurrence starts; after the last, the state at each other step is the
value it is carried from, one step back.
"""
struct SteppedRun
    launches::Vector{Any}
    # Per step, the moves: (launch, value of the step's graph, buffer).
    moves::Vector{Vector{Tuple{Int, Int, Any}}}
    # After each step's kernels, the copies of the sums its plan could not
    # write in place.
    copies::Vector{Any}
    start::Any
    finish::Any
    # What the step's kernels read and write, kept alive.
    keep::Any
end

function (r::SteppedRun)()
    r.start()
    for moves in r.moves
        for (l, v, buffer) in moves
            repoint!(r.launches[l], v, buffer)
        end
        foreach(l -> l(), r.launches)
        foreach(c -> c(), r.copies)
    end
    r.finish()
    nothing
end

"What a recurrence's steps run: its step's kernels, and the sums riding them."
struct SteppedKernels
    steps::Int
    kernels::Vector{Any}
    sums::Vector{Int}
end

function Base.show(io::IO, k::SteppedKernels)
    print(io, k.steps, " steps of ", length(k.kernels), " kernels")
    isempty(k.sums) || print(io, ", adding to ", join(map(r -> "%$r", k.sums), " "))
    for x in k.kernels
        print(io, "; ", x)
    end
end

"""
A recurrence's step planned: its graph, the decider that planned it, the
plan with its kernels in order, their decisions and its storage, and the
number of steps.
"""
struct SteppedPlan
    step::StepGraph
    decider::Decider
    plan::Plan
    decisions::Vector{Any}
    storage::Storage
    steps::Int
end

"""
    stepped_plan(d, p, g) -> SteppedPlan or nothing

Recurrence group `g`'s step planned as a program of its own, once per group
alike in the decider; nothing where a step is not a program of its own. Each
sum riding the steps is written over the storage of the sum the steps before
left, where the step's kernels allow it.
"""
stepped_plan(d::Decider, p::Plan, g::Group) = get!(() -> plan_step(d, p, g), d.stepped, group_identity(p, g))

function plan_step(d::Decider, p::Plan, g::Group)
    s = step_graph(p.graph, p.stored, g)
    s === nothing && return nothing
    values = p.graph.values
    c = first(g.outputs)
    cd = values[c].definition::Carried
    cd.axis == length(values[c].dims) || return nothing
    bd = Decider(d.target, s.graph; measure = d.measure === :dry ? :dry : d.measure)
    bp, _ = cover(bd)
    search!(bd, bp)
    ds = decisions(bd, bp)
    writes_in_place(x) = x !== nothing && all(o -> o.implementation === nothing || Base.invokelatest(in_place, o.implementation, o.claim), x.options)
    first_sum = length(s.leaves) - length(s.sums)
    replaces = [bp.graph.results[length(s.stored) + j].value => first_sum + j for j in eachindex(s.sums)]
    bp, order, st = storage(bp, replaces; in_place = k -> writes_in_place(ds[k]))
    ds = ds[order]
    VERIFYING[] && verify(bp, st; in_place = k -> writes_in_place(ds[k]))
    SteppedPlan(s, bd, bp, ds, st, values[c].dims[cd.axis])
end

"""
    stepped_seconds(d, p, g) -> seconds or nothing

The time of recurrence group `g` run a step at a time: each step's kernels
at their measured times, else their estimates; nothing where it does not run so.
"""
function stepped_seconds(d::Decider, p::Plan, g::Group)
    sp = stepped_plan(d, p, g)
    sp === nothing && return nothing
    sp.steps * sum(x -> something(x.measured, x.choice.estimate), sp.decisions; init = 0.0)
end

# The shape value `v` has, read at `index` by a state of shape `dims`, among
# the state's axes: where each of its axes is one of the state's, in their
# order, its extent there, and one along the state's others, which it is
# spread over. Nothing where it is read otherwise.
function spread_start(v::Value, index, dims)
    (v.scalar || length(index) != length(v.dims)) && return nothing
    along = Int[]
    for (j, e) in enumerate(index)
        k = findfirst(q -> e == var(q), eachindex(dims))
        (k === nothing || !isempty(along) && k <= last(along) || v.dims[j] != dims[k]) && return nothing
        push!(along, k)
    end
    ntuple(q -> q in along ? dims[q] : 1, length(dims))
end

# Of the storage a value of the step is at across the steps, the one at the
# address least aligned: a kernel is built for the alignment of the storage
# it is built on, as cuTile's arrays state it in their type, and may be
# pointed only at storage aligned as much. A step's slice of a value of `n`
# elements along the steps starts at `(t - 1)n` of them, aligned as `n` is.
least_aligned(buffers) = argmin(b -> trailing_zeros(reinterpret(UInt, pointer(b))), collect(buffers))

"""
    build_stepped(d, p, g, buffers) -> (launch, description) or nothing

Recurrence group `g` as a plan of one step, launched once per step on
`buffers`, the outer plan's storage, or nothing where a step is not a
program of its own or its kernels cannot be pointed elsewhere.
"""
function build_stepped(d::Decider, p::Plan, g::Group, buffers)
    sp = stepped_plan(d, p, g)
    sp === nothing && return nothing
    s, bd, bp, ds, st, steps = sp.step, sp.decider, sp.plan, sp.decisions, sp.storage, sp.steps
    values = p.graph.values
    target = d.target
    cs = [first(leaf) for leaf in s.leaves[1:s.states]]
    fs = [carried_from(p.graph, c) for c in cs]
    cd = values[first(cs)].definition::Carried
    # A value a state is carried from, where the plan does not store it, is
    # stored by the steps, each step's where the next reads it.
    own = Dict(f => allocate(target, values[f].eltype, storage_length(values[f])) for f in fs if buffers[f] === nothing)
    stacked(u) = haskey(own, u) ? own[u] : buffers[u]
    # The step's slice of value `u` along its last axis, at step `t`.
    function slice(u, t)
        n = storage_length(values[u]) ÷ values[u].dims[end]
        view(stacked(u), ((t - 1) * n + 1):(t * n))
    end
    # The start: each state's first step where the recurrence starts, and
    # each sum riding the steps at zero.
    first_step = cd.reverse ? steps : 1
    starts = map(cs) do c
        init = values[c].definition.init
        if init isa Literal
            () -> fill!(slice(c, first_step), init.value)
        elseif init isa Read && !any(e -> mentions(e, cd.axis), init.index) && buffers[init.value] !== nothing &&
                reads_storage(values[init.value], init.index[1:end], drop_axis(values[c].dims, cd.axis))
            () -> copyto!(slice(c, first_step), view(buffers[init.value], 1:(storage_length(values[c]) ÷ steps)))
        elseif init isa Read && buffers[init.value] !== nothing &&
                (spread = spread_start(values[init.value], init.index, drop_axis(values[c].dims, cd.axis))) !== nothing
            # A start read along some of the state's axes, in their order, and
            # the same along the others, as a learned first state repeated
            # over a batch, is spread over the others.
            from = reshape(view(buffers[init.value], 1:storage_length(values[init.value])), spread)
            to = reshape(slice(c, first_step), drop_axis(values[c].dims, cd.axis))
            () -> (to .= from; nothing)
        else
            nothing
        end
    end
    any(isnothing, starts) && return nothing
    sums = s.sums
    start = () -> (foreach(x -> x(), starts); foreach(r -> fill!(buffers[r], zero(values[r].eltype)), sums))
    sequence = cd.reverse ? (steps:-1:1) : (1:steps)
    # Where state `k` is at step `t`: where the recurrence starts at the
    # first, else the value it is carried from one step back.
    state_at(k, t) = t == first_step ? slice(cs[k], t) : slice(fs[k], cd.reverse ? t + 1 : t - 1)
    bvalues = bp.graph.values
    slots = Any[allocate(target, UInt8, bytes) for bytes in st.sizes]
    bbuffers = Vector{Any}(nothing, length(bvalues))
    taken = sort!([st.donated; st.overwrites]; by = first)
    # Values along the steps, by the step graph's value: an argument sliced, or a result.
    moving = Dict{Int, Int}()
    # The steps' positions, as each type read, uploaded once.
    positions = Dict{Int, Any}()
    for (k, leaf) in enumerate(s.leaves)
        k <= s.states && continue
        if first(leaf) === :position
            positions[k] = upload(target, [last(leaf)(t - 1) for t in 1:steps])
            bbuffers[k] = least_aligned(view(positions[k], t:t) for t in sequence)
            continue
        end
        first(leaf) === :sum && (bbuffers[k] = buffers[last(leaf)]; continue)
        u, a = leaf
        a == 0 ? (bbuffers[k] = buffers[u]) : (moving[k] = u)
    end
    for (k, u) in enumerate(s.stored)
        moving[bp.graph.results[k].value] = u
    end
    any(t -> haskey(moving, first(t)), taken) && return nothing
    for (id, v) in enumerate(bvalues)
        bbuffers[id] === nothing || continue
        any(t -> first(t) == id, taken) && continue
        def = v.definition
        if haskey(moving, id)
            bbuffers[id] = least_aligned(slice(moving[id], t) for t in sequence)
        elseif id <= s.states
            bbuffers[id] = least_aligned(state_at(id, t) for t in sequence)
        elseif def isa ConstantArray
            bbuffers[id] = upload(target, vec(collect(def.data)))
        elseif st.slots[id] > 0
            bbuffers[id] = slot_view(slots[st.slots[id]], v.eltype, storage_length(v))
        elseif bp.stored[id]
            bbuffers[id] = allocate(target, v.eltype, storage_length(v))
        end
    end
    foreach(((r, a),) -> bbuffers[r] = bbuffers[a], taken)
    # A sum its step's kernels could not write over the steps before's is
    # copied there after them.
    copies = Any[(let to = bbuffers[a], from = bbuffers[r]; () -> copyto!(to, from) end) for (r, a) in st.staged]
    built = [build_group(bd, bp, bg, x, bbuffers) for (bg, x) in zip(bp.groups, ds)]
    launches = Any[b[1] for b in built]
    all(repoints, launches) || return nothing
    # Per step, where each launch reads the states and the values along the
    # steps: each state where the recurrence starts at the first, else the
    # value it is carried from one step back.
    readers(v) = [k for (k, bg) in enumerate(bp.groups) if v in kernel_slots(bp, bg)]
    users = Dict(v => readers(v) for v in [1:s.states; collect(keys(moving)); collect(keys(positions))])
    moves = map(sequence) do t
        ms = Tuple{Int, Int, Any}[]
        for k in eachindex(cs)
            state = state_at(k, t)
            foreach(l -> push!(ms, (l, k, state)), users[k])
        end
        for (v, u) in moving
            buffer = slice(u, t)
            foreach(l -> push!(ms, (l, v, buffer)), users[v])
        end
        for (v, x) in positions
            buffer = view(x, t:t)
            foreach(l -> push!(ms, (l, v, buffer)), users[v])
        end
        ms
    end
    # Each state at every step but the first is the value it is carried
    # from one step back: one copy of all of them.
    n = storage_length(values[first(cs)]) ÷ steps
    total = storage_length(values[first(cs)])
    finish = if steps == 1
        () -> nothing
    elseif cd.reverse
        () -> foreach(((c, f),) -> copyto!(view(buffers[c], 1:(total - n)), view(stacked(f), (n + 1):total)), zip(cs, fs))
    else
        () -> foreach(((c, f),) -> copyto!(view(buffers[c], (n + 1):total), view(stacked(f), 1:(total - n))), zip(cs, fs))
    end
    run = SteppedRun(launches, moves, copies, start, finish, (slots, bbuffers, built, positions, own))
    (run, SteppedKernels(steps, Any[b[2] for b in built], sums))
end
