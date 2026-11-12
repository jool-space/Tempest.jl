# A neural ODE's training step, the NeuralODE trial's program (TRIALS.md): the
# vector field du/dt = dec(tanh(enc(u))) integrated over [0, 1] by fixed-step
# RK4 written as a plain loop, a mean squared error on the final state, and
# RAdam as torch.optim.RAdam computes it. The challengers' scripts are
# PathMNISTReactant.jl's benchmarks/{julia,pytorch}/bench_node.*, at 7684532.
#
# The step is one traced program, prepared with the parameters and RAdam's
# state replaced by its results in place, measuring close calls, and replayed
# as one captured graph.
#
#   julia --project=test test/bench/node.jl
#
# TEMPEST_NODE_{DIMS,HIDDEN,STEPS,BATCH} give the shape (32, 256, 100, 128 as
# the challengers'; STEPS is RK4's), TEMPEST_NODE_PRECISION is full or tf32,
# TEMPEST_NODE_TIMED the training steps timed (default 200; 0 times nothing),
# and TEMPEST_NODE_CHECK=0 skips the interpreter.
module NODE

using Tempest, CUDACore, Random, Printf
using Tempest.Compiler: trace, interpret, prepare, bind!, execute!, capture!, replay!, outputs
import Tempest as T
import Tempest.Compiler as C

## The model, as the challengers' common.jl and common.py.

dense(layer, u) = layer.weight * u .+ layer.bias

field(p, u) = dense(p.dec, tanh.(dense(p.enc, u)))

# Fixed-step RK4 for du/dt = f(u) over [0, 1]; the field is autonomous.
function rk4(f, u, steps)
    dt = 1.0f0 / Float32(steps)
    half = dt / 2.0f0
    sixth = dt / 6.0f0
    for _ in 1:steps
        k1 = f(u)
        k2 = f(u .+ half .* k1)
        k3 = f(u .+ half .* k2)
        k4 = f(u .+ dt .* k3)
        u = u .+ sixth .* (k1 .+ 2.0f0 .* k2 .+ 2.0f0 .* k3 .+ k4)
    end
    u
end

function loss(p, batch, steps)
    d = rk4(u -> field(p, u), batch.u0, steps) .- batch.y
    sum(abs2.(d)) / length(d)
end

# Uniform in ±1/sqrt(fan_in), as torch.nn.Linear initializes.
function layer(rng, inputs, outputs)
    bound = 1.0f0 / sqrt(Float32(inputs))
    (weight = bound .* (2.0f0 .* rand(rng, Float32, outputs, inputs) .- 1.0f0),
     bias = bound .* (2.0f0 .* rand(rng, Float32, outputs) .- 1.0f0))
end

parameters(rng, cfg) = (enc = layer(rng, cfg.dims, cfg.hidden), dec = layer(rng, cfg.hidden, cfg.dims))

batch(rng, cfg) = (u0 = randn(rng, Float32, cfg.dims, cfg.batch), y = randn(rng, Float32, cfg.dims, cfg.batch))

## RAdam, as torch.optim.RAdam with its defaults and no weight decay.

map_leaves(f, xs::Union{Tuple, NamedTuple}...) = map((ys...) -> map_leaves(f, ys...), xs...)
map_leaves(f, xs...) = f(xs...)

state_of(params) = (first = map_leaves(zero, params), second = map_leaves(zero, params), step = zeros(Float32, 1))

function radam(p, state, gradients; rate = 1.0f-3, beta1 = 0.9f0, beta2 = 0.999f0, epsilon = 1.0f-8)
    step = state.step .+ 1.0f0
    first = map_leaves((m, g) -> beta1 .* m .+ (1 - beta1) .* g, state.first, gradients)
    second = map_leaves((v, g) -> beta2 .* v .+ (1 - beta2) .* abs2.(g), state.second, gradients)
    correction1 = 1 .- beta1 .^ step
    correction2 = 1 .- beta2 .^ step
    # The length of the approximated simple moving average, and the
    # rectification that applies once it passes 5; before, the step is momentum's.
    limit = 2 / (1 - beta2) - 1
    averaged = limit .- 2 .* step .* beta2 .^ step ./ correction2
    rectified = averaged .> 5.0f0
    rectification = sqrt.(max.((averaged .- 4) .* (averaged .- 2) .* limit ./ ((limit - 4) * (limit - 2) .* averaged), 0.0f0))
    updated = map_leaves(p, first, second) do x, m, v
        adaptive = ifelse.(rectified, rectification .* sqrt.(correction2) ./ (sqrt.(v) .+ epsilon), 1.0f0)
        x .- rate .* (m ./ correction1) .* adaptive
    end
    updated, (; first, second, step)
end

function train_step(p, state, batch, steps)
    l, (g,) = Tempest.withgradient(q -> loss(q, batch, steps), p)
    p, state = radam(p, state, g)
    (l, p, state)
end

step_program(params, state, batch, steps) = trace((p, s, b) -> train_step(p, s, b, steps), params, state, batch)

function configuration()
    int(name, default) = parse(Int, get(ENV, "TEMPEST_NODE_$name", string(default)))
    (; dims = int("DIMS", 32), hidden = int("HIDDEN", 256), steps = int("STEPS", 100), batch = int("BATCH", 128))
end

## Measuring, as test/bench/gpt.jl.

launch!(ex) = ex.captured === nothing ? execute!(ex) : replay!(ex)

loss_value(ex) = Array(first(outputs(ex)))[]

"Device time of one step, the minimum over samples of a captured graph of several."
function device_milliseconds(ex; samples = 10, milliseconds = 200.0)
    launch!(ex)
    capture!(ex)
    launch!(ex)
    CUDACore.synchronize()
    once = minimum(_ -> CUDACore.@elapsed(launch!(ex)), 1:3)
    repetitions = clamp(round(Int, 1.0e-3 * milliseconds / once), 1, 100)
    capture!(ex; repetitions)
    launch!(ex)
    t = minimum(_ -> 1.0e3 * CUDACore.@elapsed(launch!(ex)) / repetitions, 1:samples)
    capture!(ex)
    t
end

"""
The steady time per step as the challengers measure it: a batch from a pool of
eight bound before each replay, the loss read at the end, the time over the
second half of the run.
"""
function steady_milliseconds(ex, pool; steps)
    ex.captured === nothing && capture!(ex)
    halfway = 0.0
    for i in 1:steps
        bind!(ex, 3 => pool[mod1(i, length(pool))])
        launch!(ex)
        i == steps ÷ 2 && (CUDACore.synchronize(); halfway = time())
    end
    loss_value(ex)
    1.0e3 * (time() - halfway) / (steps - steps ÷ 2)
end

function main()
    cfg = configuration()
    rng = Xoshiro(0)
    params = parameters(rng, cfg)
    pool = [batch(rng, cfg) for _ in 1:8]
    precision = Symbol(get(ENV, "TEMPEST_NODE_PRECISION", "tf32"))
    target = T.CUDATarget(; precision)
    println("dims ", cfg.dims, ", hidden ", cfg.hidden, ", ", cfg.steps, " RK4 steps, batch ", cfg.batch, "; ",
        sum(length, C.leaves(params)), " parameters; ", target)
    state = state_of(params)
    traced = @elapsed program = step_program(params, state, pool[1], cfg.steps)
    device(x) = map_leaves(CuArray, x)
    prepared = @elapsed ex = prepare(program, device(params), device(state), device(pool[1]); target,
        replaces = (2 => 1, 3 => 2), measure = :close)
    @printf("trace %.1f s, prepare %.1f s; %d kernels\n", traced, prepared, length(ex.kernels))
    for (k, g) in zip(ex.kernels, ex.plan.groups)
        k isa C.FloorKernels && println("  on the floor: a $(g.kind) group over ", g.domain, " (", k, ")")
    end
    first_step = @elapsed (launch!(ex); CUDACore.synchronize())
    @printf("first execution %.2f s; trace to first result %.1f s\n", first_step, traced + prepared + first_step)
    if get(ENV, "TEMPEST_NODE_CHECK", "1") != "0"
        bind!(ex, device(params), device(state), device(pool[1]))
        launch!(ex)
        reference = interpret(program, params, state, pool[1])
        moments = map(Array, C.leaves(outputs(ex)[3].first))
        errors = map((a, b) -> maximum(abs.(a .- b)) / maximum(abs.(b)), moments, C.leaves(reference[3].first))
        @printf("loss %.7f, the interpreter's %.7f; gradients' largest relative difference %.1e\n",
            loss_value(ex), reference[1], maximum(errors))
    end
    timed = parse(Int, get(ENV, "TEMPEST_NODE_TIMED", "200"))
    timed > 0 || return ex
    bind!(ex, device(params), device(state), device(pool[1]))
    @printf("device time per step: %.3f ms\n", device_milliseconds(ex))
    @printf("steady: %.3f ms/step over the last %d of %d steps\n", steady_milliseconds(ex, map(device, pool); steps = timed),
        timed - timed ÷ 2, timed)
    ex
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    import cuBLAS, cuTile
    # A cold preparation: nothing an earlier run kept is read.
    include(joinpath(@__DIR__, "..", "fixtures", "stores.jl"))
    run_stores!()
    NODE.main()
end
