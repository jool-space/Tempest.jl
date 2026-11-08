# A 2-D heat equation's gradient: u on a 256×256 grid, its interior diffusing
# by the five-point stencil, its border held, a source added at each step;
# the loss the sum of the final state squared, and its gradient with respect
# to the initial state. 100 steps is the small shape, 1000 the full one;
# TEMPEST_CORPUS_HEAT_STEPS sets them.
#
# The steps are `accumulate` over the source's slices, the form Tempest rolls
# into one loop (TRIALS: recurrences): at 20 steps 2 kernels forward and 4 for
# the gradient. A plain `for` loop over `q[:, :, t]` rolls as well, its steps
# selecting their neighbours from the step before.
using Random
import Tempest as T

const α = 0.2f0

function interior(u)
    i = reshape(Int32(1):Int32(size(u, 1)), :, 1)
    j = reshape(Int32(1):Int32(size(u, 2)), 1, :)
    (i .> 1) .& (i .< size(u, 1)) .& (j .> 1) .& (j .< size(u, 2))
end

# One step: each neighbour read from the grid shifted by one, the border
# repeated where a shift runs off it, which the interior's mask never reads.
function diffuse(u, inside)
    up, down = vcat(u[1:1, :], u[1:(end - 1), :]), vcat(u[2:end, :], u[end:end, :])
    left, right = hcat(u[:, 1:1], u[:, 1:(end - 1)]), hcat(u[:, 2:end], u[:, end:end])
    ifelse.(inside, u .+ α .* (up .+ down .+ left .+ right .- 4 .* u), u)
end

function evolve(u, q)
    inside = interior(u)
    last(accumulate((v, s) -> diffuse(v, inside) .+ s, eachslice(q; dims = 3); init = u))
end

loss(u, q) = sum(abs2, evolve(u, q))

step(u, q) = T.withgradient(v -> loss(v, q), u)

function instance(shape)
    steps = parse(Int, get(ENV, "TEMPEST_CORPUS_HEAT_STEPS", shape === :small ? "100" : "1000"))
    rng = Xoshiro(0)
    u = randn(rng, Float32, 256, 256)
    q = 0.01f0 .* randn(rng, Float32, 256, 256, steps)
    (; step, args = (u, q), replaces = (), target = T.CUDATarget(), precision = "fp32", note = "$steps steps")
end
