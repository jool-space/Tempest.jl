# The invariants a plan and its storage keep: what the kernels, their order
# and their buffers rely on, which nothing between `plan` and execution
# otherwise checks. A grouping or a rewrite that broke one went unseen until a
# result came out wrong: a recurrence's value joined into a second kernel, two
# kernels writing it, gave a neural ODE's weights a gradient 39% off in one
# preparation in three. Each is checked against the graph and the plan, and
# what storage recorded deciding is tested again by the rule that allows it:
# its pairs each by its own rule, its slots by running the kernels' order.

public verify, InvalidPlan

"""
    InvalidPlan

A plan, or its storage, that breaks one of the invariants `verify` checks:
the invariant's number there, and what breaks it, naming the value and the
kernel.
"""
struct InvalidPlan <: Exception
    invariant::Int
    message::String
end

Base.showerror(io::IO, e::InvalidPlan) = print(io, "invalid plan, invariant ", e.invariant, ": ", e.message)

invalid(n, parts...) = throw(InvalidPlan(n, string(parts...)))

# Whether a preparation verifies the plans it forms: `prepare`'s default,
# from the environment, and within a preparation what it chose.
verifying() = parse(Bool, get(ENV, "TEMPEST_VERIFY", "false"))
const VERIFYING = Base.ScopedValues.ScopedValue(false)

# Whether a result reached by a shorter way is checked against the whole
# computation it stands for, from the environment's TEMPEST_CROSSCHECK: the
# decisions a preparation replays against those a cover reaches (see
# `reach!`), and a plan formed near another against the plan formed whole
# (see `replanned`). Verifying checks a plan against the invariants, at little cost;
# a crosscheck checks the shorter way itself, at the cost of the whole
# computation each time, and is the tests' to ask for.
crosschecking() = parse(Bool, get(ENV, "TEMPEST_CROSSCHECK", "false"))

kernel_name(p::Plan, k) = string("kernel ", k, " (", p.groups[k].kind, " over ", p.groups[k].domain, ")")
value_name(p::Plan, v) = describe_value(p.graph, v)

# Values that have storage before any kernel runs.
given(graph, v) = graph.values[v].definition isa Union{Input, ConstantArray}

"""
    verify(plan) -> plan
    verify(plan, storage; in_place = k -> true) -> plan

Check the invariants the kernels of `plan` rely on, and throw an
`InvalidPlan` naming the value and the kernel at the first that fails:

 1. Every value a kernel stores is a member of that kernel's group and of no
    other; every value a kernel or a result reads is an input, a constant, or
    stored by a kernel earlier in the order.
 2. With `storage`, simulating the order: no slot is given to a value while
    the value it holds has a reader yet to run, but where a pair the storage
    records allows it, and each donated, staged, overwritten and reused pair
    holds by the rule that allows it, tested again; each lifetime covers its
    value's kernel and every reader.
 3. A recurrence's group has every state of its cycle first among its
    members, each carried from a different value it computes, and stores the
    values of the cycle no other kernel stores; each member outside the
    cycle rides its steps along them or at their last, and each fold of it
    is a sum that rides them, through values nothing stores.
 4. Each member's term loads values stored apart, its own only in a
    recurrence's; each member has a place within the domain; a fold read as
    a `Stat` is a primary fold of the group complete in an earlier phase;
    and a fold a term recomputes is not a member of the group.

`in_place(k)` says whether kernel `k`'s realization reads each element a set
of its outputs needs before it writes any of them, as `storage` was told.
`prepare` checks every plan it forms, and its storage, with `verify = true`,
the default when the environment sets `TEMPEST_VERIFY=true`, as the tests do.
"""
function verify(p::Plan)
    reads = [group_reads(p.graph, g) for g in p.groups]
    verify_owners(p, reads)
    verify_recurrences(p)
    verify_terms(p)
    p
end

function verify(p::Plan, s::Storage; in_place = k -> true)
    verify(p)
    verify_storage(p, s, in_place, [group_reads(p.graph, g) for g in p.groups])
    p
end

# Invariant 1: one owner, and every read written before.
function verify_owners(p::Plan, reads)
    graph = p.graph
    holder = zeros(Int, length(graph.values))
    for (k, g) in enumerate(p.groups), m in members(g)
        holder[m] == 0 || invalid(1, value_name(p, m), " is a member of ", kernel_name(p, holder[m]), " and of ", kernel_name(p, k))
        holder[m] = k
        p.owner[m] == k ||
            invalid(1, value_name(p, m), " is a member of ", kernel_name(p, k), ", but its owner is ", p.owner[m] == 0 ? "no kernel" : kernel_name(p, p.owner[m]))
    end
    for v in eachindex(graph.values)
        p.stored[v] && p.owner[v] > 0 && holder[v] != p.owner[v] &&
            invalid(1, value_name(p, v), " is stored by ", kernel_name(p, p.owner[v]), ", which does not compute it")
    end
    for (k, rs) in enumerate(reads), r in rs
        written_before(p, r, k) || invalid(1, kernel_name(p, k), " reads ", value_name(p, r), ", ", unwritten(p, r, k))
    end
    for o in graph.results, r in read_values(o)
        written_before(p, r, length(p.groups) + 1) || invalid(1, "a result reads ", value_name(p, r), ", ", unwritten(p, r, length(p.groups) + 1))
    end
end

written_before(p::Plan, r, k) = given(p.graph, r) || p.stored[r] && 0 < p.owner[r] < k

unwritten(p::Plan, r, k) = !p.stored[r] ? "which nothing stores" : p.owner[r] == 0 ? "which no kernel writes" :
    p.owner[r] == k ? "which it computes but is not its member" : string("which ", kernel_name(p, p.owner[r]), " writes after it")

# Invariant 3: a recurrence's group owns its cycle and its riders.
function verify_recurrences(p::Plan)
    graph = p.graph
    values = graph.values
    live = nothing
    for (k, g) in enumerate(p.groups)
        g.kind isa RecurrenceKind || continue
        cs = recurrence_states(graph, g.outputs)
        isempty(cs) && invalid(3, kernel_name(p, k), " carries no state")
        g.outputs[1:length(cs)] == cs || invalid(3, "the states of ", kernel_name(p, k), " are not its first members")
        fs = [carried_from(graph, c) for c in cs]
        for (c, f) in zip(cs, fs)
            f in g.outputs || invalid(3, value_name(p, c), " is carried from ", value_name(p, f), ", which ", kernel_name(p, k), " does not compute")
        end
        # Two states carried from one value would be one argument standing for two.
        allunique([cs; fs]) || invalid(3, "two states of ", kernel_name(p, k), " are carried from one value, or a state from a state")
        cycle = recurrence_cycle(graph, cs, fs)
        for w in cycle
            values[w].definition isa Carried && !(w in cs) &&
                invalid(3, "state ", value_name(p, w), " of the cycle of ", kernel_name(p, k), " is not among its states")
            p.owner[w] == k || !p.stored[w] && p.owner[w] == 0 ||
                invalid(3, value_name(p, w), ", of the cycle of ", kernel_name(p, k), ", is ", p.owner[w] == 0 ? "stored by no kernel" : string("owned by ", kernel_name(p, p.owner[w])))
        end
        (; domain) = recurrence_places(graph, g.outputs)
        domain == g.domain || invalid(3, kernel_name(p, k), " steps over ", g.domain, " where its members lie over ", domain)
        riders = [[m for m in g.outputs if !(m in cycle)]; g.reductions]
        isempty(riders) && continue
        live === nothing && (live = planned_live(p))
        for r in riders
            rider = riding(graph, live, cycle, r)
            summed = r in g.reductions
            (rider === nothing || folded(rider.how) != summed) &&
                invalid(3, value_name(p, r), summed ? ", a fold of " : ", a member of ", kernel_name(p, k), " outside its cycle, does not ride its steps")
            for u in rider.between
                p.stored[u] || p.owner[u] > 0 || continue
                invalid(3, value_name(p, u), ", which ", value_name(p, r), " reads at each step of ", kernel_name(p, k), ", is ",
                    p.owner[u] == 0 ? "stored" : string("owned by ", kernel_name(p, p.owner[u])))
            end
        end
    end
end

# The values a plan computes, and those they read: what runs.
function planned_live(p::Plan)
    values = p.graph.values
    live = falses(length(values))
    foreach(g -> foreach(m -> live[m] = true, members(g)), p.groups)
    for id in length(values):-1:1
        live[id] || continue
        foreach(o -> foreach(r -> live[r] = true, read_values(o)), operands(values[id].definition))
    end
    live
end

# Invariant 4: each member's term, and its place.
function verify_terms(p::Plan)
    graph = p.graph
    for (k, g) in enumerate(p.groups)
        own = Set(members(g))
        places = g.kind isa RecurrenceKind ? recurrence_places(graph, g.outputs).places : nothing
        for m in members(g)
            haskey(g.terms, m) || invalid(4, value_name(p, m), " has no term in ", kernel_name(p, k))
            t = g.terms[m]
            for x in subterms((t,); nested = true)
                if x isa Load
                    length(x.index) == length(graph.values[x.value].dims) ||
                        invalid(4, "the term of ", value_name(p, m), " in ", kernel_name(p, k), " loads ", value_name(p, x.value), " at ", length(x.index), " indices")
                    # A recurrence's step reads what it computes where the last step stored it.
                    x.value in own && !(g.kind isa RecurrenceKind) &&
                        invalid(4, "the term of ", value_name(p, m), " in ", kernel_name(p, k), " loads ", value_name(p, x.value), ", which the kernel computes")
                elseif x isa Stat
                    stat_read(p, g, k, m, x.value)
                elseif x isa Recomputed
                    # A recurrence's fold is its own term, recomputed at each step.
                    x.value in own && !(g.kind isa RecurrenceKind && x === t && x.value == m) &&
                        invalid(4, "the term of ", value_name(p, m), " in ", kernel_name(p, k), " recomputes ", value_name(p, x.value), ", which the kernel computes")
                end
            end
            # A sum riding a recurrence's steps lies over its own iteration.
            g.kind isa RecurrenceKind && m in g.reductions && continue
            at = places === nothing ? placement(graph, g, m) : places[m]
            at === nothing && invalid(4, value_name(p, m), " has no place in ", kernel_name(p, k))
            within(graph, g, m, at) || invalid(4, value_name(p, m), " is placed at ", at, ", outside the domain of ", kernel_name(p, k))
        end
    end
end

# Fold `s` read as a `Stat` by member `m` of group `g`, the `k`th: a primary
# fold of the group, complete in an earlier phase than `m`'s, or in its own
# for an output, which is written after the folds of its phase complete.
function stat_read(p::Plan, g::Group, k, m, s)
    graph = p.graph
    reader = string(value_name(p, m), " in ", kernel_name(p, k))
    s in g.reductions || invalid(4, reader, " reads ", value_name(p, s), " as a fold of the kernel, which it is not")
    primary(graph, g, s) && !restricted(graph, g, s) ||
        invalid(4, reader, " reads ", value_name(p, s), ", which is not a fold over the kernel's primary axes")
    earlier = m in g.reductions ? g.phase[s] < g.phase[m] : g.phase[s] <= g.phase[m]
    earlier || invalid(4, reader, ", of phase ", g.phase[m], ", reads ", value_name(p, s), ", complete in phase ", g.phase[s])
end

# Whether place `at` of member `m` lies within group `g`'s domain: an index
# per axis of the value, over the domain's variables alone, and where it is a
# variable, over as many positions as the value has along that axis.
function within(graph, g::Group, m, at)
    dims = graph.values[m].dims
    length(at) == length(dims) || return false
    extents = restricted(graph, g, m) ? graph.values[m].definition.iteration : g.domain
    n = length(g.domain)
    all(eachindex(at)) do j
        e = at[j]
        n < 63 && IR.variables(e) >> n != 0 && return false
        k = findfirst(a -> e == var(a), 1:n)
        k === nothing || extents[k] == dims[j]
    end
end

# Invariant 2: storage, its pairs by their rules and its slots by the order.
function verify_storage(p::Plan, s::Storage, in_place, reads)
    graph = p.graph
    values = graph.values
    results = Set(o.value for o in graph.results if o isa Read)
    final = copy(p.owner)
    for (k, rs) in enumerate(reads), r in rs
        final[r] = max(final[r], k)
    end
    for v in eachindex(values)
        p.stored[v] && p.owner[v] > 0 || continue
        span = s.lifetimes[v]
        p.owner[v] in span && final[v] in span ||
            invalid(2, "the lifetime ", span, " of ", value_name(p, v), " does not cover kernels ", p.owner[v], " to ", final[v])
    end
    for v in eachindex(values)
        slot = s.slots[v]
        slot == 0 && continue
        p.stored[v] && p.owner[v] > 0 && !(v in results) ||
            invalid(2, value_name(p, v), " is in slot ", slot, ", though it is ", v in results ? "a result" : "not stored by a kernel")
        value_bytes(values[v]) <= s.sizes[slot] ||
            invalid(2, value_name(p, v), " is in slot ", slot, " of ", s.sizes[slot], " bytes, too few for it")
    end
    # The kernels other than `k` that read `b` run before it.
    before(b, k) = all(j -> j == k || !(b in reads[j]) || j < k, eachindex(reads))
    replaced = [s.donated; s.staged]
    allunique(first.(replaced)) && allunique(last.(replaced)) ||
        invalid(2, "a result replaces two arguments, or an argument is replaced twice")
    for (r, a) in replaced
        r in results && values[a].definition isa Input ||
            invalid(2, value_name(p, r), " replaces ", value_name(p, a), ", though ", r in results ? "that is not an input" : "it is not a result")
    end
    over = Dict(s.overwrites)
    for (r, a) in s.donated
        k = p.owner[r]
        k > 0 || invalid(2, "result ", value_name(p, r), ", written into the storage of ", value_name(p, a), ", has no kernel")
        # Through folds by key, each over the last: each pair is tested below.
        v = r
        while haskey(over, v) && over[v] != a
            v = over[v]
        end
        haskey(over, v) && continue
        in_place(k) && writes_over(graph, p.groups[k], r, a) && before(a, k) ||
            invalid(2, kernel_name(p, k), " writes result ", value_name(p, r), " into the storage of ", value_name(p, a),
                ", which it ", in_place(k) ? "reads elsewhere than where it writes, or a later kernel reads" : "does not read before it writes")
    end
    for (v, b) in s.overwrites
        k = p.owner[v]
        k > 0 && overwritable(graph, p.groups[k], v) == b ||
            invalid(2, value_name(p, v), " is written over ", value_name(p, b), ", which it does not start from as stored, or its kernel reads otherwise")
        before(b, k) || invalid(2, value_name(p, v), " is written over ", value_name(p, b), ", which a kernel after ", kernel_name(p, k), " reads")
        b in results && !given(graph, b) && invalid(2, value_name(p, v), " is written over result ", value_name(p, b))
        s.slots[v] == s.slots[b] || invalid(2, value_name(p, v), " is written over ", value_name(p, b), " but lies in another slot")
    end
    for (v, b) in s.reused
        k = p.owner[v]
        g = p.groups[k]
        reusable = g.kind isa Union{ReductionKind, PointwiseKind} && in_place(k) && final[b] == k && p.owner[b] != k && !(b in results) &&
            value_bytes(values[v]) == value_bytes(values[b]) && values[v].eltype == values[b].eltype && writes_over(graph, g, v, b)
        reusable && s.slots[v] == s.slots[b] > 0 ||
            invalid(2, value_name(p, v), " takes the slot of ", value_name(p, b), ", which ", kernel_name(p, k), " does not read last where it writes")
    end
    # The order run again: each slot's value, as each kernel writes into it.
    allowed = Set([s.overwrites; s.reused])
    holder = zeros(Int, length(s.sizes))
    for (k, g) in enumerate(p.groups)
        for v in sort!(filter(m -> s.slots[m] > 0, members(g)))
            slot = s.slots[v]
            u = holder[slot]
            u == 0 || final[u] < k || final[u] == k && (v => u) in allowed ||
                invalid(2, kernel_name(p, k), " writes ", value_name(p, v), " into slot ", slot, ", which holds ", value_name(p, u), " until kernel ", final[u], " reads it")
            holder[slot] = v
        end
    end
end
