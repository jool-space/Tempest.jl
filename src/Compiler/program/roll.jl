# Loops rolled from the tape. A `for` loop traces as its iterations, each the
# same nodes up to renaming: a window of the tape that repeats with a period,
# each node reading what the node one period back read (a value the loop
# captures) or the node one period after that (a value of its step). Such a
# window is the loop `accumulate` traces (see `traced_loop`), and is
# rewritten as it: its step once, a `Loop` node for each of the step's
# results the program reads after the window, and a selection of the step
# it reads. The program and the eager code are the same; the tape is shorter.

# A step's selection of what the loop scans differs from the next step's at
# one index, an integer one apart: `xs[:, :, t]` in a loop over `t`. The
# rolled loop scans the array, or the range of it the steps select, as
# `accumulate` over its slices does, and runs reversed where the index falls.

# The fewest periods a window rolls: two are a step and its copy, cheaper
# unrolled than as a loop.
const ROLLED_PERIODS = 3

"The index at which selections `a` and `b` differ by an integer one apart, or `nothing`."
function slid(a::Select, b::Select)
    length(a.indices) == length(b.indices) || return nothing
    differ = findall(k -> !isequal(a.indices[k], b.indices[k]), eachindex(a.indices))
    length(differ) == 1 || return nothing
    i, j = a.indices[only(differ)], b.indices[only(differ)]
    i isa Integer && j isa Integer && abs(i - j) == 1 ? only(differ) : nothing
end
Base.@nospecializeinfer slid(@nospecialize(a), @nospecialize(b)) = nothing

# Operations alike in a loop's steps: one, selections one index apart, or
# constants of one value, as an array the loop captures is copied at each use.
# Compiled once, whatever the operations' functions: an operation is equal to
# another when it is the same, as no operation defines its own equality.
Base.@nospecializeinfer alike_ops(@nospecialize(a), @nospecialize(b)) =
    typeof(a) === typeof(b) && (a === b || slid(a, b) !== nothing || a isa Constant && isequal(a.value, b.value))

"Whether nodes `m` and `n`, `p` apart, are the same up to renaming, as a loop's are."
function alike_nodes(m::Node, n::Node, p)
    m.dims == n.dims && m.eltype === n.eltype && m.scalar == n.scalar && length(m.inputs) == length(n.inputs) &&
        alike_ops(m.op, n.op) || return false
    all(((a, b),) -> a == b || a - b == p, zip(m.inputs, n.inputs))
end

# Whether nodes `m` and `n` hold values of one type: a loop's state keeps its.
same_type(m::Node, n::Node) = m.dims == n.dims && m.eltype === n.eltype && m.scalar == n.scalar

# Whether node `k` of a loop's first step and the node a period on, of the
# second step starting at `first`, are alike but where `k` reads, before
# both steps, a value of the type of what the other reads of the step before it.
function entered(nodes, k, p, first)
    m, n = nodes[k + p], nodes[k]
    same_type(m, n) && length(m.inputs) == length(n.inputs) && alike_ops(m.op, n.op) || return false
    all(((a, b),) -> a == b || a - b == p || (first - p <= a < first && b < first - p && same_type(nodes[a], nodes[b])),
        zip(m.inputs, n.inputs))
end

# A step folds: one of maps alone fuses with the next into a kernel unrolled,
# and rolled would run a step at a time.
folds(n::Node) = n.op isa Union{Reduce, Contract, Accumulate}

# Or node `k` of a step selects from a value of its steps, not one the loop
# captures, as a stencil reads its neighbours shifted: unrolled, a step
# reading the last at other positions than its own does not fuse with it,
# a kernel a step.
shifts(nodes, k, p) = nodes[k].op isa Select && nodes[k].inputs != nodes[k - p].inputs

# What an operation shares with every operation alike it (see `alike_ops`):
# itself, but a constant's value, which may be another array of equal
# elements, and a selection's integers, of which one may slide.
op_class(@nospecialize(op)) = op
op_class(::Constant) = nothing
op_class(op::Select) = map(i -> i isa Integer ? nothing : i, op.indices)

# Nodes alike may be: of one operation's type, shape and element type, their
# operations of one class.
signature(n::Node) = hash((typeof(n.op), op_class(n.op), n.dims, n.eltype, n.scalar, length(n.inputs)))

# Operations a step of a loop may not hold: a loop's own, or a stand-in's.
rollable(n::Node) = !(n.op isa Union{Loop, Placeholder, Argument, Carry})

"""
What `roll!` seeks windows with, among `nodes[lo:hi]`, from each node down
from the last: the nodes of each signature in order, the only ones alike
each other; of each node from `lo`, its signature, how many of its signature
come before it, and the nearest node before it with fewer before it, or
`lo - 1` (see `Periods`); and the runs found so far (see `run_start`), which
each search from a node further down reuses.
"""
struct WindowSearch
    nodes::Vector{Node}
    lo::Int
    alike::Dict{UInt, Vector{Int}}
    signatures::Vector{UInt}
    earlier::Vector{Int}
    rarer::Vector{Int}
    runs::Dict{Int, Tuple{Int, Int}}
end

function WindowSearch(nodes, lo, hi)
    alike = Dict{UInt, Vector{Int}}()
    signatures, earlier, rarer = UInt[], Int[], Int[]
    # The nodes so far each with fewer before it than every later one.
    fewest = Int[]
    for k in lo:hi
        s = signature(nodes[k])
        members = get!(() -> Int[], alike, s)
        push!(signatures, s)
        push!(earlier, length(members))
        push!(members, k)
        while !isempty(fewest) && earlier[fewest[end] - lo + 1] >= last(earlier)
            pop!(fewest)
        end
        push!(rarer, isempty(fewest) ? lo - 1 : fewest[end])
        push!(fewest, k)
    end
    WindowSearch(nodes, lo, alike, signatures, earlier, rarer, Dict{Int, Tuple{Int, Int}}())
end

"""
    Periods(search, last)

The periods a window ending at `last` may have, in increasing order. Its
last period, the `p` nodes up to `last`, holds nodes each alike its node `p`
back, so a period whose last period holds node `a` is the distance from `a`
back to a node of `a`'s signature. The periods are taken from the nodes of
`last`'s signature, up to those whose last period would reach `b`, the
nearest node before `last` with fewer of its signature before it; then from
`b`'s, up to the next such node before `b`; and so on. A period left out has
a node in its last period with no node of its signature a period back, where
the window's run ends (see `run_start`): the search finds what it found
trying every node alike `last`. They are few where `last`'s signature is
common and a rarer node lies just below it: after a loop whose results
`stack` copies one by one, a write of the stack's is alike only the same
write after each other such loop, and the nodes between the writes alike
those between every two.
"""
struct Periods
    search::WindowSearch
    last::Int
end

Base.IteratorSize(::Type{Periods}) = Base.SizeUnknown()
Base.eltype(::Type{Periods}) = Int

# From node `a`, the periods whose last period holds `a` but not `below`, the
# nearest node before `a` with fewer of its signature before it: the nodes of
# `a`'s signature they reach back to, as positions among those, from the
# nearest down to `stop`.
function periods_from(ps::Periods, a)
    (; search, last) = ps
    members = search.alike[search.signatures[a - search.lo + 1]]
    below = search.rarer[a - search.lo + 1]
    (a, members, searchsortedlast(members, 2a - last - 1), searchsortedfirst(members, a - last + below), below)
end

function Base.iterate(ps::Periods, state = periods_from(ps, ps.last))
    a, members, k, stop, below = state
    while k < stop
        below < ps.search.lo && return nothing
        a, members, k, stop, below = periods_from(ps, below)
    end
    (a - members[k], (a, members, k - 1, stop, below))
end

"""
    periodic_window(search, last) -> (; first, period, periods) or nothing

The window of `nodes[lo:last]` ending at `last` that repeats with a period,
over the most nodes, the shortest period among those: each node of it after
the first period alike its node a period back. Sought from each node down
from the last of a search, in turn.
"""
function periodic_window(search::WindowSearch, last)
    (; nodes, lo) = search
    best = nothing
    accepted = Int[]
    n = nodes[last]
    for p in Periods(search, last)
        (last - lo + 1) ÷ p < ROLLED_PERIODS && break
        # Every period is a multiple of one.
        1 in accepted && break
        any(q -> p % q == 0, accepted) && continue
        alike_nodes(n, nodes[last - p], p) || continue
        i = run_start(search, last, p)
        matched = last - i
        periods = matched ÷ p + 1
        periods >= 2 || continue
        push!(accepted, p)
        all(k -> rollable(nodes[k]), (last - periods * p + 1):last) || continue
        any(k -> folds(nodes[k]) || shifts(nodes, k, p), (last - p + 1):last) || continue
        first = last - periods * p + 1
        # The period before, reading where the loop starts what each step
        # after reads of the step before, is the loop's first step.
        if first - p >= lo && all(k -> rollable(nodes[k]) && entered(nodes, k, p, first), (first - p):(first - 1))
            first -= p
            periods += 1
        end
        periods >= ROLLED_PERIODS || continue
        if best === nothing || periods * p > best.periods * best.period
            best = (; first, period = p, periods)
        end
    end
    best
end

"""
The node before the run of nodes up to `last`, each alike its node `p`
back: walked down from `last` once per stretch, and kept in the search's
runs as the stretch's ends by period. A window is sought from every node
down from the last, and walked afresh each time, a run was walked again from
each of its nodes: a heat equation's 800 slices of its source took 0.9 s to
trace.
"""
function run_start(search::WindowSearch, last, p)
    (; nodes, lo, runs) = search
    start, stop = get(runs, p, (0, 0))
    start < last <= stop && return start
    last == start && return last
    i = last
    while i - p >= lo && rollable(nodes[i]) && alike_nodes(nodes[i], nodes[i - p], p)
        i -= 1
    end
    runs[p] = (i, last)
    i
end

"""
    roll_window(nodes, backward, lo, hi, required) -> (nodes, at, first) or nothing

The last window of `nodes[lo:hi]` that repeats, rewritten as a loop: the
nodes before it as they are, the loop and what it computes once, each value
read after the window, or `required`, as its step's slice of the loop, then
the nodes after the window, renumbered. `at` holds each node's new place, or
0 for a value of the window no node reads any more; `first` the window's
first place. Nothing where no window repeats. A window of the `backward`,
where a reverse pass recorded it, is not rolled.
"""
function roll_window(nodes::Vector{Node}, backward, lo, hi, required)
    hi - lo + 1 >= 2 * ROLLED_PERIODS || return nothing
    # The window ends at the last node alike one a period back.
    window = nothing
    search = WindowSearch(nodes, lo, hi)
    for at in hi:-1:lo
        rollable(nodes[at]) || continue
        window = periodic_window(search, at)
        window === nothing || break
    end
    window === nothing && return nothing
    (; period, periods) = window
    w0 = window.first
    p = period
    stop = w0 + periods * p - 1
    any(id -> id in backward, w0:stop) && return nothing
    step = w0:(w0 + p - 1)
    # Each step's selection of an array from before the window, by its
    # offset in the step: the index `j` that slides, `i` at the first step
    # and `d` further at each next, one `d` for every selection.
    scans = Dict{Int, Tuple{Int, Int}}()
    d = 0
    for (o, id) in enumerate(step)
        n = nodes[id]
        j = slid(n.op, nodes[id + p].op)
        j === nothing && continue
        length(n.inputs) == 1 && only(n.inputs) < w0 || return nothing
        selection = (n.op::Select).indices
        i = selection[j]
        d == 0 && (d = (nodes[id + p].op::Select).indices[j] - i)
        all(k -> isequal((nodes[id + k * p].op::Select).indices, Base.setindex(selection, i + k * d, j)), 1:(periods - 1)) || return nothing
        scans[o] = (j, i)
    end
    # What the first step reads from before the window: a state where the
    # second reads the node a period on, else a capture.
    states, captures = Tuple{Int, Int}[], Int[]
    for (o, id) in enumerate(step), (k, x) in enumerate(nodes[id].inputs)
        (x >= w0 || haskey(scans, o)) && continue
        y = nodes[id + p].inputs[k]
        if y == x
            x in captures || push!(captures, x)
        elseif w0 <= y < w0 + p && same_type(nodes[x], nodes[y])
            (x, y) in states || push!(states, (x, y))
        else
            return nothing
        end
    end
    any(((x, _),) -> x in captures, states) && return nothing
    # A step's nodes that read neither a state nor a scanned slice are
    # computed once, before the loop.
    varies = falses(p)
    for (o, id) in enumerate(step)
        varies[o] = haskey(scans, o) || any(x -> any(st -> first(st) == x, states) || (x >= w0 && varies[x - w0 + 1]), nodes[id].inputs)
    end
    # What is read after the window, or required, by step and offset.
    wanted = Set{Int}(r for r in required if w0 <= r <= stop)
    for id in (stop + 1):length(nodes), x in nodes[id].inputs
        w0 <= x <= stop && push!(wanted, x)
    end
    invariant = [w0 + o - 1 for o in 1:p if !varies[o]]
    # The step's results: each state's next, then the values read after.
    carried_from = [y for (_, y) in states]
    all(y -> varies[y - w0 + 1], carried_from) || return nothing
    offsets = Int[y - w0 + 1 for y in carried_from]
    for r in sort!(collect(wanted))
        o = (r - w0) % p + 1
        varies[o] && !(o in offsets) && push!(offsets, o)
    end
    loops = length(offsets)
    # The new nodes at the window's places: the invariant nodes, then the loop's.
    place = Dict{Int, Int}()
    foreach(((k, id),) -> place[id] = w0 + k - 1, enumerate(invariant))
    moved(x) = x >= w0 ? place[x] : x
    hoist = [Node(nodes[id].op, map(moved, nodes[id].inputs), nodes[id].dims, nodes[id].eltype, nodes[id].scalar) for id in invariant]
    # Each scanned array, or the range of it the steps select, computed
    # once too, and the axis the steps slice it along.
    scanned, sliced = Int[], Int[]
    for o in sort!(collect(keys(scans)))
        n = nodes[w0 + o - 1]
        (j, i) = scans[o]
        x = only(n.inputs)
        low = d > 0 ? i : i - periods + 1
        selection = (n.op::Select).indices
        indices = Base.setindex(selection, low:(low + periods - 1), j)
        axis = 1 + count(k -> !(k isa Integer), selection[1:(j - 1)])
        push!(sliced, axis)
        whole = length(indices) == length(nodes[x].dims) && all(k -> indices[k] isa AbstractUnitRange && indices[k] == 1:nodes[x].dims[k], eachindex(indices))
        if whole
            push!(scanned, x)
        else
            push!(hoist, Node(Select(indices), (x,), (n.dims[1:(axis - 1)]..., periods, n.dims[axis:end]...), n.eltype, false))
            push!(scanned, w0 + length(hoist) - 1)
        end
    end
    hoisted = length(hoist)
    # The step's program: the states, the scanned slices, then the captures,
    # as arguments.
    captured = sort!(unique(append!(copy(captures), [place[id] for id in invariant])))
    arguments = Int[]
    body = Node[]
    # The step's nodes by their places before the window, or in the step;
    # those computed once, by their new places; each state by where it
    # starts and what the next step reads, as two may start from one value.
    renumbered, once, held = Dict{Int, Int}(), Dict{Int, Int}(), Dict{Tuple{Int, Int}, Int}()
    for (x, y) in states
        push!(body, Node(Argument(), (), nodes[x].dims, nodes[x].eltype, nodes[x].scalar))
        held[(x, y)] = length(body)
    end
    for o in sort!(collect(keys(scans)))
        n = nodes[w0 + o - 1]
        push!(body, Node(Argument(), (), n.dims, n.eltype, false))
        renumbered[w0 + o - 1] = length(body)
    end
    for x in captured
        n = x >= w0 ? hoist[x - w0 + 1] : nodes[x]
        push!(body, Node(Argument(), (), n.dims, n.eltype, n.scalar))
        x >= w0 ? (once[x] = length(body)) : (renumbered[x] = length(body))
    end
    append!(arguments, eachindex(body))
    for (o, id) in enumerate(step)
        varies[o] && !haskey(scans, o) || continue
        n = nodes[id]
        inputs = map(n.inputs, nodes[id + p].inputs) do x, y
            x >= w0 ? (varies[x - w0 + 1] ? renumbered[x] : once[place[x]]) : get(() -> renumbered[x], held, (x, y))
        end
        push!(body, Node(n.op, inputs, n.dims, n.eltype, n.scalar))
        renumbered[id] = length(body)
    end
    results = [renumbered[w0 + o - 1] for o in offsets]
    program = Program(body, arguments, append!(copy(results), 1:length(states)), Tuple(eachindex(arguments)), Tuple(eachindex(results)))
    loop = LoopBody(program, periods, collect(1:length(states)), sliced, d < 0)
    inputs = (first.(states)..., scanned..., captured...)
    stacked = map(enumerate(offsets)) do (k, o)
        n = nodes[w0 + o - 1]
        Node(Loop(loop, k), inputs, n.scalar ? (periods,) : (n.dims..., periods), n.eltype, false)
    end
    # Each value read after the window: its step's slice of the loop's, or
    # the node computed once.
    selected = Dict{Int, Node}()
    for r in wanted
        o = (r - w0) % p + 1
        t = (r - w0) ÷ p + 1
        d < 0 && (t = periods - t + 1)
        n = nodes[r]
        selected[r] = if varies[o]
            k = findfirst(==(o), offsets)
            Node(Select((map(m -> 1:m, n.dims)..., t)), (w0 + hoisted + k - 1,), n.dims, n.eltype, n.scalar)
        else
            Node(Map(identity), (place[w0 + o - 1],), n.dims, n.eltype, n.scalar)
        end
    end
    # The nodes after the window read each value of it they read at its new
    # place.
    rolled = nodes[1:(w0 - 1)]
    at = zeros(Int, length(nodes))
    at[1:(w0 - 1)] .= 1:(w0 - 1)
    append!(rolled, hoist, stacked)
    for r in sort!(collect(wanted))
        push!(rolled, selected[r])
        at[r] = length(rolled)
    end
    for id in (stop + 1):length(nodes)
        n = nodes[id]
        push!(rolled, Node(n.op, map(x -> at[x], n.inputs), n.dims, n.eltype, n.scalar))
        at[id] = length(rolled)
    end
    rolled, at, w0
end

"""
Whether a trace rolls the loops its function runs: within `with(ROLLING =>
false)`, a program is its trace as the function ran, every iteration's nodes
in place, which is how a roll is checked: the two interpret the same.
"""
const ROLLING = Base.ScopedValues.ScopedValue(true)

"""
    rolled(nodes, backward, lo, required) -> (nodes, backward, at)

The nodes with every window of `nodes[lo:end]` that repeats rolled into a
loop, the last first, as `roll_window` rolls one; the backward among them,
and `at`, each node's new place, or 0 for one no node reads any more. The
nodes given are left as they are, so that a handle to one names what it did.
"""
function rolled(nodes::Vector{Node}, backward::BitSet, lo, required)
    at = collect(eachindex(nodes))
    ROLLING[] || return nodes, backward, at
    required = collect(Int, required)
    hi = length(nodes)
    while hi >= lo
        window = roll_window(nodes, backward, lo, hi, required)
        window === nothing && break
        nodes, moved, w0 = window
        at = map(i -> i == 0 ? 0 : moved[i], at)
        required = map(r -> moved[r], required)
        backward = BitSet(moved[i] for i in backward if moved[i] != 0)
        hi = w0 - 1
    end
    nodes, backward, at
end

"""
    roll!(tape, lo, required) -> place

The loops of `tape.nodes[lo:end]` rolled while the program is traced, as a
gradient rolls its function's before its reverse pass: the nodes `required`
and those they read, rolled, appended to the tape, and `place`, which gives
each node of `required` its place among them. The nodes rolled stay where
they are, so that a handle the function left to one, in a `Ref` say, names
what it did; the program keeps them only where something reads them.
"""
function roll!(tape::Tape, lo, required)
    nodes, backward, at = rolled(tape.nodes, tape.backward, lo, required)
    nodes === tape.nodes && return identity
    keep = live(nodes, [at[r] for r in required])
    place = zeros(Int, length(nodes))
    place[1:(lo - 1)] .= 1:(lo - 1)
    for id in lo:length(nodes)
        keep[id] || continue
        n = nodes[id]
        push!(tape.nodes, Node(n.op, map(x -> place[x], n.inputs), n.dims, n.eltype, n.scalar))
        place[id] = length(tape.nodes)
        id in backward && push!(tape.backward, place[id])
    end
    r -> place[at[r]]
end
