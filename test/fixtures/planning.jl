# Losses shared by compiler/planning.jl and cuda/kernels.jl.
# A custom scalar function has no Tempest opcode or dedicated kernel.
ripple(x) = sin(x) * tanh(x)

function Tempest.scalar_partials(::typeof(ripple))
    (x -> cos(x) * tanh(x) + sin(x) * (one(x) - tanh(x)) * (one(x) + tanh(x)),)
end

test_silu(x) = x / (one(x) + exp(-x))

function test_dsilu(x)
    s = inv(one(x) + exp(-x))
    s + (x * s) * (one(x) - s)
end

Tempest.scalar_partials(::typeof(test_silu)) = (test_dsilu,)

function gated_loss(x, gate, up, down, target)
    h = test_silu.(gate' * x) .* (up' * x)
    y = x + down' * h
    sum(abs2, y .- target) / length(y)
end
