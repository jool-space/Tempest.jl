# Run: julia --project=. examples/research/native_region_baseline.jl
# Count the existing CPU planner's regions on the scope probe's formulas.
# These are storage/region observations, not GPU launch or timing measurements.
include("reduction_scopes.jl")

using Tempest
using Random

function native_region_baseline()
    rng = MersenneTwister(23)
    x = randn(rng, Float32, 259, 7)
    residual = randn(rng, Float32, size(x))
    weight = randn(rng, Float32, size(x, 1))
    cases = (
        (:rmsnorm, ReductionScopes.rmsnorm, (x, residual, weight)),
        (:layernorm, ReductionScopes.layernorm, (x, weight)),
        (:rms_input_backward, ReductionScopes.rms_input_backward, (x, weight, residual)),
        (:composed, ReductionScopes.composed, (x,)),
    )
    for (name, f, args) in cases
        expected = Tempest.tree_leaves(f(args...))
        for enabled in (false, true)
            executable = prepare_forward(f, args...; fuse_regions = enabled)
            actual = Tempest.tree_leaves(execute!(executable))
            @assert length(actual) == length(expected)
            @assert all(
                isapprox(a, b; rtol = 2f-5, atol = 2f-6)
                for (a, b) in zip(actual, expected)
            )
            steps = executable.storage.steps
            println((;
                name,
                fuse_regions = enabled,
                regions = length(steps),
                graph_regions = count(s -> s.region isa Tempest.GraphRegion, steps),
                value_bytes = sum(sizeof, executable.buffers; init = 0),
            ))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    native_region_baseline()
end
