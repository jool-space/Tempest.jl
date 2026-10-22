# What the backward reads of the forward: saved, or recomputed where the
# backward reads it. Decided as one minimum cut of the forward's values, after
# PyTorch's rematerialization partitioner (torch/_functorch/partitioners.py):
# each value is an edge from its input side to its output side whose capacity
# is what saving it costs in bytes; reads are edges of infinite capacity;
# the source feeds every value that may not be recomputed, with infinite
# capacity, and every value the backward reads drains into the sink. A cut
# then saves the cheapest set of values from which the backward can be
# computed, recomputing the rest, which the objective counts as free: folds,
# scans and folds by key may not be recomputed, maps may. The planner recomputes a
# map the cut does not save in the backward kernels that read it, and stores
# what the cut saves, since two kernels read it.
#
# Bytes stand in for time: a map is cheap to recompute against what moving
# its output costs. A map one forward kernel reads is not written today, so
# saving it costs its bytes twice, written and read; a map several read is
# written anyway, and saving it costs one read.

"""
    partition(graph) -> (saved, recomputed)

The forward values of a program with a backward that the backward reads:
those saved for it, and those recomputed where the backward reads them.
Both empty when the program has no backward.
"""
function partition(graph::IndexGraph)
    any(graph.backward) || return (Int[], Int[])
    values = graph.values
    n = length(values)
    live = live_values(graph)
    users = readers(graph)
    forward = [v for v in 1:n if live[v] && !graph.backward[v]]
    required = [v for v in forward if any(u -> live[u] && graph.backward[u], users[v])]
    isempty(required) && return (Int[], Int[])
    materialized = storing(graph).materialized
    inside, outside = v -> 2v - 1, v -> 2v
    source, sink = 2n + 1, 2n + 2
    flow = Flow(2n + 2)
    for v in forward
        d = values[v].definition
        free = d isa Union{Input, ConstantArray}
        banned = free || d isa Union{Keyed, Scan} || materialized[v] || d isa Reduction && 4 * value_bytes(values[v]) < reduced_bytes(graph, v) ||
            factor_of_backward_product(graph, users, live, v)
        alone = d isa Pointwise && count(u -> live[u] && !graph.backward[u], users[v]) == 1
        edge!(flow, inside(v), outside(v), free ? 0.0 : value_bytes(values[v]) * (alone ? 2.0 : 1.0))
        banned && edge!(flow, source, inside(v), Inf)
        v in required && edge!(flow, outside(v), sink, Inf)
        for o in operands(d), r in read_values(o)
            edge!(flow, outside(r), inside(v), Inf)
        end
    end
    maxflow!(flow, source, sink)
    reached = reachable(flow, source)
    saved = [v for v in forward if reached[inside(v)] && !reached[outside(v)]]
    recomputed = [v for v in required if !reached[inside(v)]]
    (saved, recomputed)
end

# Whether a backward product reads `v` as a factor: then recomputing `v` is
# never free, since a product's sweep absorbs no prologue cheaply, and the
# value is written by the backward anyway (PyTorch's "materialized in
# backward"). A product at the index level is a sum fold of a pointwise
# product of two factors.
function factor_of_backward_product(graph, users, live, v)
    any(users[v]) do u
        live[u] && graph.backward[u] || return false
        d = graph.values[u].definition
        d isa Pointwise && d.f === (*) && length(d.operands) == 2 || return false
        any(w -> live[w] && graph.values[w].definition isa Reduction{SumFold}, users[u])
    end
end

# The bytes a fold reads: its iteration's elements in its operand's type.
function reduced_bytes(graph, v)
    d = graph.values[v].definition
    prod(d.iteration; init = 1) * sizeof(eltype_of(graph, d.operand))
end

# A flow network: edges in pairs, each with its reverse, so that residual
# capacity is the pair's. Dinic's algorithm: level graphs by breadth-first
# search, blocking flows by depth-first search with a current-arc pointer.
struct Flow
    adjacent::Vector{Vector{Int}}
    to::Vector{Int}
    capacity::Vector{Float64}
end

Flow(n) = Flow([Int[] for _ in 1:n], Int[], Float64[])

function edge!(f::Flow, u, v, c)
    push!(f.to, v); push!(f.capacity, c); push!(f.adjacent[u], length(f.to))
    push!(f.to, u); push!(f.capacity, 0.0); push!(f.adjacent[v], length(f.to))
    nothing
end

function levels(f::Flow, s)
    level = fill(-1, length(f.adjacent))
    level[s] = 0
    queue = [s]
    while !isempty(queue)
        u = popfirst!(queue)
        for e in f.adjacent[u]
            f.capacity[e] > 0 && level[f.to[e]] < 0 || continue
            level[f.to[e]] = level[u] + 1
            push!(queue, f.to[e])
        end
    end
    level
end

function augment!(f::Flow, u, t, limit, level, next)
    u == t && return limit
    while next[u] <= length(f.adjacent[u])
        e = f.adjacent[u][next[u]]
        v = f.to[e]
        if f.capacity[e] > 0 && level[v] == level[u] + 1
            pushed = augment!(f, v, t, min(limit, f.capacity[e]), level, next)
            if pushed > 0
                f.capacity[e] -= pushed
                f.capacity[isodd(e) ? e + 1 : e - 1] += pushed
                return pushed
            end
        end
        next[u] += 1
    end
    0.0
end

function maxflow!(f::Flow, s, t)
    total = 0.0
    while true
        level = levels(f, s)
        level[t] < 0 && return total
        next = ones(Int, length(f.adjacent))
        while (pushed = augment!(f, s, t, Inf, level, next)) > 0
            total += pushed
            isinf(total) && throw(ArgumentError("a value the backward reads can neither be saved nor recomputed"))
        end
    end
end

# Which vertices the source still reaches in the residual network: the
# source side of a minimum cut.
function reachable(f::Flow, s)
    seen = falses(length(f.adjacent))
    seen[s] = true
    stack = [s]
    while !isempty(stack)
        u = pop!(stack)
        for e in f.adjacent[u]
            f.capacity[e] > 0 && !seen[f.to[e]] || continue
            seen[f.to[e]] = true
            push!(stack, f.to[e])
        end
    end
    seen
end
