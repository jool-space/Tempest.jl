# A recurrence stepped: its group's cycle restricted to one step, planned as
# a program of its own, and its kernels launched once per step, pointed at
# that step's place in the recurrence's storage. A one-kernel claim keeps
# the state on chip; this runs a step too large for one kernel, a neural
# ODE's field evaluated four times an RK4 step, as the kernels of an
# unrolled step, from one plan of it.

"""
One step of a recurrence as a graph of its own: its arguments are the state
(argument 1) and the values outside the cycle it reads, each whole or one
step's slice of it; its results are the value the state is carried from and
the cycle's values stored at each step.
"""
struct StepGraph
    graph::IndexGraph
    # Each argument's value outside and the axis it is sliced along, 0 for
    # whole; or `(:position, T)`, the step's position as a `T`.
    leaves::Vector{Any}
    # Each result's value outside, stored a step at a time.
    stored::Vector{Int}
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
outside the cycle with it, or nothing where a step is not a program of its
own: where it reads a value of the cycle at another step, or a value along
the steps other than by its last axis, or by an index array.
"""
function step_graph(graph::IndexGraph, stored, g::Group)
    values = graph.values
    c, f = first(g.outputs), last(g.outputs)
    cd = values[c].definition::Carried
    cd.from isa Read && cd.from.value == f && cd.from.index == identity_index(values[c].dims) || return nothing
    cycle = Set(recurrence_cycle(graph, c, f))
    places = recurrence_places(graph, g.outputs).places
    # The axis along the steps of each member, from its place.
    along(m) = findfirst(==(var(cd.axis)), places[m])
    members = [m for m in g.outputs if m != c]
    any(m -> along(m) === nothing, members) && return nothing
    # The values a step needs, each with its axis along the steps or 0, and
    # how: :state, :leaf, or :step, computed in the step.
    kinds = Dict{Tuple{Int, Int}, Symbol}()
    order = Tuple{Int, Int}[]
    positions = Set{DataType}()
    ok = Ref(true)
    function visit(u, s)
        haskey(kinds, (u, s)) && return
        kind = if u == c
            s == cd.axis || (ok[] = false)
            :state
        elseif u in cycle
            s == 0 && (ok[] = false)
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
    # The arguments first, the state the first of them; then the step's
    # values in the graph's order.
    leaves = Any[(c, cd.axis); [key for key in order if kinds[key] === :leaf]; [(:position, T) for T in positions]]
    steps = [key for key in sort(order) if kinds[key] === :step]
    id = Dict{Any, Int}()
    new = Value[]
    for (k, leaf) in enumerate(leaves)
        if first(leaf) === :position
            push!(new, Value(Input(k), (), last(leaf), true))
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
    # The value the state is carried from first.
    results = [results[end]; results[1:(end - 1)]]
    stored_order = [f; members[1:(end - 1)]]
    dims = [new[r.value].dims for r in results]
    body = IndexGraph(new, collect(1:length(leaves)), results, dims, [isempty(d) for d in dims], [new[r.value].eltype for r in results])
    StepGraph(body, leaves, stored_order)
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
    end
    r.finish()
    nothing
end

"What a recurrence's steps run: its step's kernels."
struct SteppedKernels
    steps::Int
    kernels::Vector{Any}
end

function Base.show(io::IO, k::SteppedKernels)
    print(io, k.steps, " steps of ", length(k.kernels), " kernels")
    for x in k.kernels
        print(io, "; ", x)
    end
end

"""
    build_stepped(d, p, g, buffers) -> (launch, description) or nothing

Recurrence group `g` as a plan of one step, launched once per step on
`buffers`, the outer plan's storage, or nothing where a step is not a
program of its own or its kernels cannot be pointed elsewhere.
"""
function build_stepped(d::Decider, p::Plan, g::Group, buffers)
    s = step_graph(p.graph, p.stored, g)
    s === nothing && return nothing
    values = p.graph.values
    target = d.target
    c = first(g.outputs)
    cd = values[c].definition::Carried
    steps = values[c].dims[cd.axis]
    cd.axis == length(values[c].dims) || return nothing
    # The value the state is carried from, where the plan does not store it,
    # is stored by the steps, each step's where the next reads it.
    f = last(g.outputs)
    own = buffers[f] === nothing ? allocate(target, values[f].eltype, storage_length(values[f])) : nothing
    stacked(u) = u == f && own !== nothing ? own : buffers[u]
    # The step's slice of value `u` along its last axis, at step `t`.
    function slice(u, t)
        n = storage_length(values[u]) ÷ values[u].dims[end]
        view(stacked(u), ((t - 1) * n + 1):(t * n))
    end
    # The start: the state's first step where the recurrence starts.
    first_step = cd.reverse ? steps : 1
    init = cd.init
    start = if init isa Literal
        () -> fill!(slice(c, first_step), init.value)
    elseif init isa Read && !any(e -> mentions(e, cd.axis), init.index) && buffers[init.value] !== nothing &&
            reads_storage(values[init.value], init.index[1:end], drop_axis(values[c].dims, cd.axis))
        () -> copyto!(slice(c, first_step), view(buffers[init.value], 1:(storage_length(values[c]) ÷ steps)))
    else
        return nothing
    end
    bd = Decider(target, s.graph; measure = d.measure === :dry ? :dry : d.measure)
    bp, _ = cover(bd)
    search!(bd, bp)
    ds = decisions(bd, bp)
    writes_in_place(x) = x !== nothing && all(o -> o.implementation === nothing || Base.invokelatest(in_place, o.implementation, o.claim), x.options)
    bp, order, st = storage(bp, Pair{Int, Int}[]; in_place = k -> writes_in_place(ds[k]))
    ds = ds[order]
    bvalues = bp.graph.values
    slots = Any[allocate(target, UInt8, bytes) for bytes in st.sizes]
    bbuffers = Vector{Any}(nothing, length(bvalues))
    taken = sort!([st.donated; st.overwrites]; by = first)
    # Values along the steps, by the step graph's value: an argument sliced, or a result.
    moving = Dict{Int, Int}()
    # The steps' positions, as each type read, uploaded once.
    positions = Dict{Int, Any}()
    for (k, leaf) in enumerate(s.leaves)
        k == 1 && continue
        if first(leaf) === :position
            positions[k] = upload(target, [last(leaf)(t - 1) for t in 1:steps])
            bbuffers[k] = view(positions[k], first_step:first_step)
            continue
        end
        u, a = leaf
        a == 0 ? (bbuffers[k] = buffers[u]) : (moving[k] = u)
    end
    for (k, r) in enumerate(bp.graph.results)
        moving[r.value] = s.stored[k]
    end
    any(t -> haskey(moving, first(t)), taken) && return nothing
    for (id, v) in enumerate(bvalues)
        bbuffers[id] === nothing || continue
        any(t -> first(t) == id, taken) && continue
        def = v.definition
        if haskey(moving, id)
            bbuffers[id] = slice(moving[id], first_step)
        elseif id == 1
            bbuffers[id] = slice(c, first_step)
        elseif def isa ConstantArray
            bbuffers[id] = upload(target, vec(collect(def.data)))
        elseif st.slots[id] > 0
            bbuffers[id] = slot_view(slots[st.slots[id]], v.eltype, storage_length(v))
        elseif bp.stored[id]
            bbuffers[id] = allocate(target, v.eltype, storage_length(v))
        end
    end
    foreach(((r, a),) -> bbuffers[r] = bbuffers[a], taken)
    built = [build_group(bd, bp, bg, x, bbuffers) for (bg, x) in zip(bp.groups, ds)]
    launches = Any[b[1] for b in built]
    all(repoints, launches) || return nothing
    # Per step, where each launch reads the state and the values along the
    # steps: the state where the recurrence starts at the first, else the
    # value it is carried from one step back.
    readers(v) = [k for (k, bg) in enumerate(bp.groups) if v in kernel_slots(bp, bg)]
    users = Dict(v => readers(v) for v in [1; collect(keys(moving)); collect(keys(positions))])
    sequence = cd.reverse ? (steps:-1:1) : (1:steps)
    moves = map(sequence) do t
        ms = Tuple{Int, Int, Any}[]
        state = t == first_step ? slice(c, t) : slice(f, cd.reverse ? t + 1 : t - 1)
        foreach(l -> push!(ms, (l, 1, state)), users[1])
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
    # The state at every step but the first is the value it is carried from
    # one step back: one copy of all of them.
    n = storage_length(values[c]) ÷ steps
    total = storage_length(values[c])
    finish = if steps == 1
        () -> nothing
    elseif cd.reverse
        () -> copyto!(view(buffers[c], 1:(total - n)), view(stacked(f), (n + 1):total))
    else
        () -> copyto!(view(buffers[c], (n + 1):total), view(stacked(f), 1:(total - n)))
    end
    run = SteppedRun(launches, moves, start, finish, (slots, bbuffers, built, positions, own))
    (run, SteppedKernels(steps, Any[b[2] for b in built]))
end
