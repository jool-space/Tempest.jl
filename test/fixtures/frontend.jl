# Functions shared by frontend/arrays.jl and cuda/frontend.jl.
const TA = Tempest.TracedArray
const TS = Tempest.TracedScalar

function typed_rmsnorm(x::AbstractMatrix{T}, gain::AbstractVector{T}; eps = T(1e-4)) where {T}
    r = inv.(sqrt.(sum(abs2, x; dims = 1) / size(x, 1) .+ eps))
    x .* r .* gain
end

function norm_loss(x::AbstractMatrix, w::AbstractVector, b::AbstractVector)
    sum(abs2, typed_rmsnorm(x, w) .+ b)
end

triple(x, y, z) = tanh(x * y) + sin(z)

function Tempest.scalar_partials(::typeof(triple))
    (
        (x, y, z) -> y * (one(x) - tanh(x * y)) * (one(x) + tanh(x * y)),
        (x, y, z) -> x * (one(x) - tanh(x * y)) * (one(x) + tanh(x * y)),
        (x, y, z) -> cos(z),
    )
end
