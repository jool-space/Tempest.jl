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

# The same arrays labelled: the first half's weights and the input on device
# 0, the second half's weights and the target on device 1.
function labelled_model(ws, x, t)
    half = length(ws) ÷ 2
    on(d) = w -> map(a -> Labelled(a, d), w)
    ((map(on(0), ws[1:half])..., map(on(1), ws[(half + 1):end])...), Labelled(x, 0), Labelled(t, 1))
end

crossings(p) = count(g -> g.kind isa C.CrossingKind, p.groups)
