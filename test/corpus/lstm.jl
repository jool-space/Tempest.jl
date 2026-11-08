# An LSTM's training step: Flux's cell, the sequence a plain loop over its
# steps' slices `xs[:, :, t]`, a mean squared error on the last hidden state,
# the gradient of every weight and of the first state, and a step of gradient
# descent, the parameters replaced in place. 32 inputs through 256 hidden, 100
# steps, TF32, as the torch challenger (torch_lstm.py) runs it: batch 64 is the
# small shape, 1024 the full one. The challenger takes no optimizer step.
using Random
import Tempest as T

σ(x) = 1.0f0 / (1.0f0 + exp(-x))

function cell(p, h, c, x)
    i = σ.(p.Wi * x .+ p.Ui * h .+ p.bi)
    f = σ.(p.Wf * x .+ p.Uf * h .+ p.bf)
    g = tanh.(p.Wg * x .+ p.Ug * h .+ p.bg)
    o = σ.(p.Wo * x .+ p.Uo * h .+ p.bo)
    c = f .* c .+ i .* g
    (o .* tanh.(c), c)
end

function sequence(p, xs)
    h, c = p.h0, p.h0
    for t in 1:size(xs, 3)
        h, c = cell(p, h, c, xs[:, :, t])
    end
    h
end

loss(p, xs, y) = sum(abs2, sequence(p, xs) .- y) / length(y)

function train_step(p, xs, y; rate = 1.0f-2)
    l, (g,) = T.withgradient(q -> loss(q, xs, y), p)
    (l, map((x, d) -> x .- rate .* d, p, g))
end

function instance(shape)
    D, H, S = 32, 256, 100
    B = shape === :small ? 64 : 1024
    rng = Xoshiro(0)
    w(o, i) = 0.1f0 .* randn(rng, Float32, o, i)
    p = (; Wi = w(H, D), Wf = w(H, D), Wg = w(H, D), Wo = w(H, D), Ui = w(H, H), Uf = w(H, H), Ug = w(H, H), Uo = w(H, H),
        bi = w(H, 1), bf = w(H, 1), bg = w(H, 1), bo = w(H, 1), h0 = zeros(Float32, H, B))
    xs, y = randn(rng, Float32, D, B, S), randn(rng, Float32, H, B)
    (; step = train_step, args = (p, xs, y), replaces = (2 => 1,), target = T.CUDATarget(; precision = :tf32), precision = "tf32")
end
