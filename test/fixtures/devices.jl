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

# A tower's arrays all on `device`.
on_label(device, (ws, x, t)) = (map(w -> map(a -> Labelled(a, device), w), ws), Labelled(x, device), Labelled(t, device))

# The devices of the kernels of plan `p`, a crossing's its reader's, each
# run of kernels on one device once.
function device_runs(p)
    devices = [C.group_device(p, g) for g in p.groups]
    [devices[k] for k in eachindex(devices) if k == 1 || devices[k] != devices[k - 1]]
end

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
