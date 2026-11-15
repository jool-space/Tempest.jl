# A check of an execution's order between streams that shares nothing with
# its derivation (`Tempest.Compiler.stream_schedule`): from the runs and the
# waits alone, which steps run before which, whatever the streams' speeds;
# and orders the streams may run the steps in, drawn at random.
using Random

# Each step's predecessors as the runs launch them: the step before it on its
# stream, and the steps it waits for. Throws where the runs launch the steps
# otherwise than a recording can: a step twice or never, a stream's steps out
# of plan order, or a wait for a step not launched before the run that waits.
function predecessors(steps, waits, runs)
    launched = reduce(vcat, runs; init = Int[])
    sort(launched) == eachindex(steps) || error("the runs launch steps $(launched), not each once")
    run_of = zeros(Int, length(steps))
    foreach(((r, run),) -> foreach(k -> run_of[k] = r, run), enumerate(runs))
    last_on = Dict{Any, Int}()
    preds = [Int[] for _ in steps]
    for k in launched
        σ = C.stream_of(steps[k])
        if haskey(last_on, σ)
            last_on[σ] < k || error("step $k runs after step $(last_on[σ]) of its stream")
            push!(preds[k], last_on[σ])
        end
        last_on[σ] = k
        for j in waits[k]
            run_of[j] < run_of[k] || error("step $k waits for step $j, launched with or after it")
            push!(preds[k], j)
        end
    end
    preds, launched
end

# The steps each step runs after, however the streams run.
function happens_before(steps, waits, runs)
    preds, launched = predecessors(steps, waits, runs)
    before = [falses(length(steps)) for _ in steps]
    for k in launched, j in preds[k]
        before[k][j] = true
        before[k] .|= before[j]
    end
    before
end

# The pairs of steps, in plan order, that touch one storage, one writing it,
# neither running after the other: none in an order that holds.
function unordered(steps, touches, waits, runs)
    before = happens_before(steps, waits, runs)
    pairs = Tuple{Int, Int}[]
    for j in eachindex(steps), i in 1:(j - 1)
        conflict(touches[i], touches[j]) && !before[j][i] && push!(pairs, (i, j))
    end
    pairs
end

conflict(a, b) = !isempty(intersect(a.writes, union(b.reads, b.writes))) || !isempty(intersect(b.writes, a.reads))

unordered(ex::C.Executable) = (s = C.stream_schedule(ex.steps, ex.touches); unordered(ex.steps, ex.touches, s.waits, s.runs))

# An order the streams may run the steps in: each next step drawn from those
# whose predecessors have run.
function interleaving(rng, steps, waits, runs)
    preds, _ = predecessors(steps, waits, runs)
    done = falses(length(steps))
    order = Int[]
    while length(order) < length(steps)
        ready = [k for k in eachindex(steps) if !done[k] && all(j -> done[j], preds[k])]
        k = rand(rng, ready)
        done[k] = true
        push!(order, k)
    end
    order
end

# Whether `order` runs every two steps that touch one storage, one writing
# it, in plan order.
function in_plan_order(order, touches)
    at = invperm(order)
    all(Iterators.product(eachindex(order), eachindex(order))) do (i, j)
        i >= j || !conflict(touches[i], touches[j]) || at[i] < at[j]
    end
end
