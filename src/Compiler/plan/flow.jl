# Minimum cuts, which the planner decides two things by: what the backward
# saves of the forward (`partition`) and which device each value lives on
# (`assigned_devices`).

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

# The maximum flow from `s` to `t`, the capacity of a minimum cut: infinite,
# found as soon as it is, where every cut crosses an edge of infinite capacity.
function maxflow!(f::Flow, s, t)
    total = 0.0
    while true
        level = levels(f, s)
        level[t] < 0 && return total
        next = ones(Int, length(f.adjacent))
        while (pushed = augment!(f, s, t, Inf, level, next)) > 0
            total += pushed
            isinf(total) && return total
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

# Which vertices still reach the sink in the residual network: the sink side
# of the minimum cut nearest the sink, every other vertex on the source's.
# An edge's pair runs the other way, so a vertex `w` reaches `u` where the
# pair of an edge from `u` to `w` has capacity left.
function reaching(f::Flow, t)
    seen = falses(length(f.adjacent))
    seen[t] = true
    stack = [t]
    while !isempty(stack)
        u = pop!(stack)
        for e in f.adjacent[u]
            w = f.to[e]
            f.capacity[isodd(e) ? e + 1 : e - 1] > 0 && !seen[w] || continue
            seen[w] = true
            push!(stack, w)
        end
    end
    seen
end
