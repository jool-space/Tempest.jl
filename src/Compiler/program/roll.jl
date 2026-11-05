# Loops rolled from the tape. A `for` loop traces as its iterations, each the
# same nodes up to renaming: a window of the tape that repeats with a period,
# each node reading what the node one period back read (a value the loop
# captures) or the node one period after that (a value of its step). Such a
# window is the loop `accumulate` traces (see `traced_loop`), and is
# rewritten as it: its step once, a `Loop` node for each of the step's
# results the program reads after the window, and a selection of the step
# it reads. The program and the eager code are the same; the tape is shorter.

# The fewest periods a window rolls: two are a step and its copy, cheaper
# unrolled than as a loop.
const ROLLED_PERIODS = 3

"Whether nodes `m` and `n`, `p` apart, are the same up to renaming, as a loop's are."
function alike_nodes(m::Node, n::Node, p)
    m.dims == n.dims && m.eltype === n.eltype && m.scalar == n.scalar && length(m.inputs) == length(n.inputs) &&
        typeof(m.op) === typeof(n.op) && isequal(m.op, n.op) || return false
    all(((a, b),) -> a == b || a - b == p, zip(m.inputs, n.inputs))
end

# Whether nodes `m` and `n` hold values of one type: a loop's state keeps its.
same_type(m::Node, n::Node) = m.dims == n.dims && m.eltype === n.eltype && m.scalar == n.scalar

# Whether node `k` of a loop's first step and the node a period on, of the
# second step starting at `first`, are alike but where `k` reads, before
# both steps, a value of the type of what the other reads of the step before it.
function entered(nodes, k, p, first)
    m, n = nodes[k + p], nodes[k]
    same_type(m, n) && length(m.inputs) == length(n.inputs) && typeof(m.op) === typeof(n.op) && isequal(m.op, n.op) || return false
    all(((a, b),) -> a == b || a - b == p || (first - p <= a < first && b < first - p && same_type(nodes[a], nodes[b])),
        zip(m.inputs, n.inputs))
end

# A step folds: one of maps alone fuses with the next into a kernel unrolled,
# and rolled would run a step at a time.
folds(n::Node) = n.op isa Union{Reduce, Contract, Accumulate}

# Nodes alike may be: of one operation's type, shape and element type.
signature(n::Node) = hash((typeof(n.op), n.dims, n.eltype, n.scalar, length(n.inputs)))

# Operations a step of a loop may not hold: a loop's own, or a stand-in's.
rollable(n::Node) = !(n.op isa Union{Loop, Placeholder, Argument, Carry})

"""
    periodic_window(nodes, lo, last) -> (; first, period, periods) or nothing

The window of `nodes[lo:last]` ending at `last` that repeats with a period,
over the most nodes, the shortest period among those: each node of it after
the first period alike its node a period back.
"""
function periodic_window(nodes, lo, last, alike_at = nothing)
    best = nothing
    accepted = Int[]
    n = nodes[last]
    candidates = alike_at === nothing ? ((last - 1):-1:lo) : Iterators.reverse(filter(j -> lo <= j < last, alike_at))
    for j in candidates
        p = last - j
        (last - lo + 1) ÷ p < ROLLED_PERIODS && break
        any(q -> p % q == 0, accepted) && continue
        alike_nodes(n, nodes[j], p) || continue
        i = last
        while i - p >= lo && rollable(nodes[i]) && alike_nodes(nodes[i], nodes[i - p], p)
            i -= 1
        end
        matched = last - i
        periods = matched ÷ p + 1
        periods >= 2 || continue
        push!(accepted, p)
        all(k -> rollable(nodes[k]), (last - periods * p + 1):last) || continue
        any(k -> folds(nodes[k]), (last - p + 1):last) || continue
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
    roll!(tape, range, required) -> Int

Rewrite the last window of `tape.nodes[range]` that repeats as a loop, in
place: the nodes keep their places, those `required` and those read after
the window defined as before, the rest left dead. The window's first place,
or 0 where no window repeats or its rewrite would not keep them. A window
of the backward, where a reverse pass recorded it, is not rolled.
"""
function roll!(tape::Tape, range, required)
    nodes = tape.nodes
    lo, hi = range[begin], range[end]
    hi - lo + 1 >= 2 * ROLLED_PERIODS || return 0
    # The window ends at the last node alike one a period back.
    window = nothing
    # Each node's earlier ones of its signature, the only ones alike it.
    alike = Dict{UInt, Vector{Int}}()
    for k in lo:hi
        push!(get!(() -> Int[], alike, signature(nodes[k])), k)
    end
    for at in hi:-1:lo
        rollable(nodes[at]) || continue
        window = periodic_window(nodes, lo, at, alike[signature(nodes[at])])
        window === nothing || break
    end
    window === nothing && return 0
    (; period, periods) = window
    w0 = window.first
    p = period
    stop = w0 + periods * p - 1
    any(id -> id in tape.backward, w0:stop) && return 0
    step = w0:(w0 + p - 1)
    # What the first step reads from before the window: a state where the
    # second reads the node a period on, else a capture.
    states, captures = Tuple{Int, Int}[], Int[]
    for (o, id) in enumerate(step), (k, x) in enumerate(nodes[id].inputs)
        x >= w0 && continue
        y = nodes[id + p].inputs[k]
        if y == x
            x in captures || push!(captures, x)
        elseif w0 <= y < w0 + p && same_type(nodes[x], nodes[y])
            (x, y) in states || push!(states, (x, y))
        else
            return 0
        end
    end
    any(((x, _),) -> x in captures, states) && return 0
    # A step's nodes that read no state are computed once, before the loop.
    reads_state = falses(p)
    for (o, id) in enumerate(step)
        reads_state[o] = any(x -> any(st -> first(st) == x, states) || (x >= w0 && reads_state[x - w0 + 1]), nodes[id].inputs)
    end
    # What is read after the window, or required, by step and offset.
    wanted = Set{Int}(r for r in required if w0 <= r <= stop)
    for id in (stop + 1):length(nodes), x in nodes[id].inputs
        w0 <= x <= stop && push!(wanted, x)
    end
    invariant = [w0 + o - 1 for o in 1:p if !reads_state[o]]
    # The step's results: each state's next, then the values read after.
    carried_from = [y for (_, y) in states]
    all(y -> reads_state[y - w0 + 1], carried_from) || return 0
    offsets = Int[y - w0 + 1 for y in carried_from]
    for r in sort!(collect(wanted))
        o = (r - w0) % p + 1
        reads_state[o] && !(o in offsets) && push!(offsets, o)
    end
    hoisted = length(invariant)
    loops = length(offsets)
    any(r -> r < w0 + hoisted + loops, wanted) && return 0
    # The new nodes at the window's places: the invariant nodes, then the loop's.
    place = Dict{Int, Int}()
    foreach(((k, id),) -> place[id] = w0 + k - 1, enumerate(invariant))
    moved(x) = x >= w0 ? place[x] : x
    hoist = [Node(nodes[id].op, map(moved, nodes[id].inputs), nodes[id].dims, nodes[id].eltype, nodes[id].scalar) for id in invariant]
    # The step's program: the states, then the captures, as arguments.
    captured = sort!(unique([captures; [place[id] for id in invariant]]))
    arguments = Int[]
    body = Node[]
    # The step's nodes by their places before the window, or in the step;
    # those computed once, by their new places.
    renumbered, once = Dict{Int, Int}(), Dict{Int, Int}()
    for (x, _) in states
        push!(body, Node(Argument(), (), nodes[x].dims, nodes[x].eltype, nodes[x].scalar))
        renumbered[x] = length(body)
    end
    for x in captured
        n = x >= w0 ? hoist[x - w0 + 1] : nodes[x]
        push!(body, Node(Argument(), (), n.dims, n.eltype, n.scalar))
        x >= w0 ? (once[x] = length(body)) : (renumbered[x] = length(body))
    end
    append!(arguments, eachindex(body))
    for (o, id) in enumerate(step)
        reads_state[o] || continue
        n = nodes[id]
        inputs = map(n.inputs) do x
            x >= w0 && !reads_state[x - w0 + 1] ? once[place[x]] : renumbered[x]
        end
        push!(body, Node(n.op, inputs, n.dims, n.eltype, n.scalar))
        renumbered[id] = length(body)
    end
    results = [renumbered[w0 + o - 1] for o in offsets]
    program = Program(body, arguments, [results; 1:length(states)], Tuple(eachindex(arguments)), Tuple(eachindex(results)))
    loop = LoopBody(program, periods, collect(1:length(states)), Int[], false)
    inputs = (first.(states)..., captured...)
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
        n = nodes[r]
        selected[r] = if reads_state[o]
            k = findfirst(==(o), offsets)
            Node(Select((map(d -> 1:d, n.dims)..., t)), (w0 + hoisted + k - 1,), n.dims, n.eltype, n.scalar)
        else
            Node(Map(identity), (place[w0 + o - 1],), n.dims, n.eltype, n.scalar)
        end
    end
    dead = Node(Constant(0), (), (), Int, true)
    for id in w0:stop
        k = id - w0 + 1
        nodes[id] = k <= hoisted ? hoist[k] : k <= hoisted + loops ? stacked[k - hoisted] : get(selected, id, dead)
    end
    w0
end

# Every window of `range` that repeats rolled, the last first.
function roll_all!(tape::Tape, range, required)
    hi = last(range)
    while hi >= first(range)
        w = roll!(tape, first(range):hi, required)
        w == 0 && break
        hi = w - 1
    end
    tape
end
