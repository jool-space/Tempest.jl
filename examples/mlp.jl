# An MLP trained, one compiled step at a time
#
# Four dense layers with tanh between them, a squared-error loss, and a step
# of gradient descent: the loss, its gradient through every layer and the
# update, written in plain Julia on structs. `compile` turns the whole step
# into kernels; each call runs one step on the GPU. The reference is the first
# step in Float64 on the CPU, which Tempest's interpreter computes.
#
#   julia --project=test examples/mlp.jl

using Tempest, CUDACore, cuTile, cuBLAS, Printf, Random

struct Dense{W, B}
    w::W
    b::B
end
(d::Dense)(x) = d.w * x .+ d.b

function forward(layers, x)
    h = x
    for (i, d) in enumerate(layers)
        h = d(h)
        i < length(layers) && (h = tanh.(h))
    end
    h
end

loss(layers, x, y) = sum(abs2, forward(layers, x) .- y) / size(y, 2)

# The loss before the step, and the layers after it. The gradient comes back
# in the layers' structure: a named tuple (w = ..., b = ...) for each.
function step(layers, x, y; rate = 0.05f0)
    l, g = Tempest.gradient(m -> loss(m, x, y), layers)
    l, map((d, dd) -> Dense(d.w .- rate .* dd.w, d.b .- rate .* dd.b), layers, g)
end

# 4 layers of 1024, a batch of 1024, learning y = sin(2x) elementwise.
n, batch = 1024, 1024
rng = Xoshiro(0)
layers = Tuple(Dense(randn(rng, n, n) ./ sqrt(n), zeros(n)) for _ in 1:4)
x = randn(rng, n, batch)
y = sin.(2 .* x)

gpu(a) = CuArray(Float32.(a))
on_gpu(layers) = map(d -> Dense(gpu(d.w), gpu(d.b)), layers)
c = compile(step, on_gpu(layers), gpu(x), gpu(y))

# The first step against Float64.
l, after = c(on_gpu(layers), gpu(x), gpu(y))
l64, after64 = step(layers, x, y)
relative(a, b) = maximum(abs, Array(a) .- b) / maximum(abs, b)
@printf("first step: loss relative difference %.1e, updated weights' largest %.1e\n",
    abs(l - l64) / l64, maximum(relative(a.w, b.w) for (a, b) in zip(after, after64)))

# Training: each call is one step, its result the next step's layers.
println()
explain(c)
println()
m, xs, ys = on_gpu(layers), gpu(x), gpu(y)
first_loss = nothing
t = time_ns()
for s in 1:200
    global l, m = c(m, xs, ys)
    global first_loss = something(first_loss, l)
    s % 50 == 0 && @printf("step %3d: loss %.4f\n", s, l)
end
CUDACore.synchronize()
ms = (time_ns() - t) / 1.0e6 / 200
@printf("\nmlp step: %d kernels, loss %.4f → %.4f in 200 steps, %.2f ms a step\n",
    length(c.executable.plan.groups), first_loss, l, ms)
