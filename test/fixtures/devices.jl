# Arrays "on" devices for the CPU target, which plans across devices by
# label: a host array that answers `device_of` with the label it is given,
# and the placed model of the trial of the devices at toy size, residual
# blocks whose first half's weights are on one device and second half's on
# another.
using Statistics, Random

struct Labelled{T, N} <: AbstractArray{T, N}
    data::Array{T, N}
    device::Int
end

Base.size(x::Labelled) = size(x.data)
Base.IndexStyle(::Type{<:Labelled}) = IndexLinear()
Base.getindex(x::Labelled, i::Int) = x.data[i]
Base.setindex!(x::Labelled, v, i::Int) = (x.data[i] = v; x)
C.device_of(x::Labelled) = x.device

placed_norm(h) = h ./ sqrt.(mean(abs2, h; dims = 1) .+ eltype(h)(1.0e-6))
placed_block(h, w) = h .+ w[2] * max.(w[1] * placed_norm(h), zero(eltype(h)))
placed_half(ws, h) = (for w in ws; h = placed_block(h, w); end; h)
placed_loss(ws, h, t) = mean(abs2, placed_half(ws, h) .- t)
placed_step(ws, x, t) = Tempest.withgradient(placed_loss, ws, x, t)

# The model's weights, input and target, of element type `T`: `blocks`
# blocks of `width` through four times it, over `tokens`.
function placed_model(T; blocks = 4, width = 16, tokens = 32)
    rng = Xoshiro(1)
    ws = Tuple((randn(rng, T, 4width, width) ./ sqrt(T(width)), T(0.5) .* randn(rng, T, width, 4width) ./ sqrt(T(4width))) for _ in 1:blocks)
    (ws, randn(rng, T, width, tokens), randn(rng, T, width, tokens))
end

# The same arrays labelled, the blocks' weights in `devices` runs of blocks
# alike, the first run's on device 0 with the input, the last's with the
# target: for two, the first half's weights on 0 and the second half's on 1.
function labelled_model(ws, x, t; devices = 2)
    run(k) = (k - 1) * devices ÷ length(ws)
    (Tuple(map(a -> Labelled(a, run(k)), w) for (k, w) in enumerate(ws)), Labelled(x, 0), Labelled(t, devices - 1))
end

# The model's arrays again, other numbers, as a later call brings them.
# Two towers of the placed model's blocks, each with its own weights, input
# and target, joined where their losses are added: what two devices
# compute apart.
towers_loss(wa, xa, ta, wb, xb, tb) = placed_loss(wa, xa, ta) + placed_loss(wb, xb, tb)
towers_step(wa, xa, ta, wb, xb, tb) = Tempest.withgradient(towers_loss, wa, xa, ta, wb, xb, tb)

# Data parallelism by hand, the trial's shards.jl at any size: a replica of
# the weights on each of two devices and a shard of the batch beside each,
# the loss the mean of the shards', the replicas' gradients summed and a
# step of SGD written into both replicas.
shards_loss(wa, wb, xa, ta, xb, tb) = (placed_loss(wa, xa, ta) + placed_loss(wb, xb, tb)) / 2
function shards_sgd!(wa, wb, xa, ta, xb, tb)
    l, (ga, gb) = Tempest.withgradient((a, b) -> shards_loss(a, b, xa, ta, xb, tb), wa, wb)
    for (w, v, g, h) in zip(wa, wb, ga, gb), (p, q, r, s) in zip(w, v, g, h)
        d = r .+ s
        p .-= eltype(p)(0.01) .* d
        q .-= eltype(q)(0.01) .* d
    end
    l
end

# The arrays of `shards_sgd!` for the placed model's weights, input and
# target: the replicas and each half of the batch, all copies.
function shards_of(ws, x, t)
    half = size(x, 2) ÷ 2
    replica() = map(w -> map(copy, w), ws)
    (replica(), replica(), x[:, 1:half], t[:, 1:half], x[:, (half + 1):end], t[:, (half + 1):end])
end

# The same labelled, the first replica and shard on device 0, the second on 1.
labelled_shards(wa, wb, xa, ta, xb, tb) = (map(w -> map(a -> Labelled(a, 0), w), wa), map(w -> map(a -> Labelled(a, 1), w), wb),
    Labelled(xa, 0), Labelled(ta, 0), Labelled(xb, 1), Labelled(tb, 1))

# Data parallelism in its own spelling, over any number of replicas: the
# replicas of the weights as arguments, the loss the mean over the shards of
# each replica's loss on its shard, the replicas' gradients reduced by a map
# across their trees, and each replica updated from the reduced gradient. The
# tree map is written here, as `Functors.fmap` would walk the trees when the
# program is traced.
tree_map(f, x::AbstractArray, ys::AbstractArray...) = f(x, ys...)
tree_map(f, x::Tuple, ys::Tuple...) = map((a, bs...) -> tree_map(f, a, bs...), x, ys...)
function replicated_sgd!(replicas, xs, ts)
    l, (gs,) = Tempest.withgradient(rs -> mean(placed_loss(w, x, t) for (w, x, t) in zip(rs, xs, ts)), replicas)
    g = tree_map((ds...) -> +(ds...), gs...)
    foreach(w -> tree_map((p, d) -> (p .-= eltype(p)(0.01) .* d; p), w, g), replicas)
    l
end

# The arrays of `replicated_sgd!` for the placed model: `n` replicas of its
# weights and the batch in `n` shards of its tokens, all copies; and the same
# labelled, the `k`th replica and shard on device `k - 1`.
function replicas_of(ws, x, t, n)
    cuts = round.(Int, range(0, size(x, 2); length = n + 1))
    shard(a, k) = a[:, (cuts[k] + 1):cuts[k + 1]]
    (Tuple(map(w -> map(copy, w), ws) for _ in 1:n), Tuple(shard(x, k) for k in 1:n), Tuple(shard(t, k) for k in 1:n))
end
labelled_replicas(rs, xs, ts) = (Tuple(map(w -> map(a -> Labelled(a, k - 1), w), r) for (k, r) in enumerate(rs)),
    Tuple(Labelled(x, k - 1) for (k, x) in enumerate(xs)), Tuple(Labelled(t, k - 1) for (k, t) in enumerate(ts)))

# The devices a value's terms lie on: what it is computed from through
# elementwise values no larger than what they read, down to values that are
# not, a crossing's the terms of what it copies.
function term_devices(g, v)
    d = g.values[v].definition
    d isa C.Crossed && return term_devices(g, d.operand.value)
    reads = [r for o in C.operands(d) for r in C.read_values(o)]
    joined = d isa C.Pointwise && !isempty(reads) && all(r -> C.value_bytes(g.values[r]) >= C.value_bytes(g.values[v]), reads)
    joined ? reduce(union, (term_devices(g, r) for r in reads)) : Set([g.devices[v]])
end

# Value `v` written out as each device computes it: a crossing as what it
# copies, a commutative function's operands in either order alike, and an
# argument by `argument(k)`.
function computed(g, v, argument)
    d = g.values[v].definition
    d isa C.Crossed && return computed(g, d.operand.value, argument)
    d isa C.Input && return argument(d.argument)
    d isa C.Pointwise || return v
    parts = Any[o isa C.Read ? (computed(g, o.value, argument), o.index) : o for o in d.operands]
    (d.f, C.IR.commutative(d.f) ? sort!(parts; by = repr) : parts)
end

# A tower's arrays all on `device`.
on_label(device, (ws, x, t)) = (map(w -> map(a -> Labelled(a, device), w), ws), Labelled(x, device), Labelled(t, device))

# The devices of the kernels of plan `p`, a crossing's its reader's, each
# run of kernels on one device once.
function device_runs(p)
    devices = [C.group_device(p, g) for g in p.groups]
    [devices[k] for k in eachindex(devices) if k == 1 || devices[k] != devices[k - 1]]
end

# The device each segment of recording `r` runs on: its steps', a
# crossing's its reader's.
segment_devices(r) = [r.schedule.steps[first(s.steps)].step.device for s in r.segments]

again(ws, x, t) = (map(w -> map(a -> a .+ 1, w), ws), 2 .* x, t .- 1)

crossings(p) = count(g -> g.kind isa C.CrossingKind, p.groups)

# The crossings of plan `p`, each as the bytes it copies.
crossed_bytes(p) = [C.value_bytes(p.graph.values[only(g.outputs)]) for g in p.groups if g.kind isa C.CrossingKind]

# Each gradient a program of `withgradient` returns, paired with the argument
# it is the gradient of: its output's `grad` is in the inputs' structure,
# result numbers where the input has an argument's.
function gradient_pairs(program)
    pairs = Pair{Int, Int}[]
    pair!(g::Int, a::Int) = push!(pairs, g => a)
    pair!(g::Tuple, a::Tuple) = foreach(pair!, g, a)
    pair!(g, a) = nothing
    pair!(program.output_structure.grad, program.input_structure)
    pairs
end

# How many gradients plan `p` of the program `program` of `withgradient`
# returns on another device than their argument's.
function gradients_off(p, program)
    graph = p.graph
    count(gradient_pairs(program)) do (r, a)
        result = graph.results[r]
        result isa C.Read && graph.devices[result.value] != graph.devices[graph.arguments[a]]
    end
end
