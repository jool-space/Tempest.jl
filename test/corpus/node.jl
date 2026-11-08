# The neural ODE's training step, test/bench/node.jl's (module NODE): the
# vector field dec(tanh(enc(u))) integrated by fixed-step RK4 as a plain loop,
# a mean squared error on the final state, and RAdam, the parameters and
# RAdam's state replaced in place. 32 inputs through 256 hidden, 100 RK4
# steps, TF32, as the challengers' scripts (PathMNISTReactant.jl's
# benchmarks/*/bench_node.*) run it: batch 128 is the small shape, 1024 the
# full one.
include(joinpath(@__DIR__, "..", "bench", "node.jl"))
using Random
import Tempest as T

function instance(shape)
    cfg = (; dims = 32, hidden = 256, steps = 100, batch = shape === :small ? 128 : 1024)
    rng = Xoshiro(0)
    params = NODE.parameters(rng, cfg)
    batch = NODE.batch(rng, cfg)
    step = (p, s, b) -> NODE.train_step(p, s, b, cfg.steps)
    (; step, args = (params, NODE.state_of(params), batch), replaces = (2 => 1, 3 => 2),
        target = T.CUDATarget(; precision = :tf32), precision = "tf32")
end
