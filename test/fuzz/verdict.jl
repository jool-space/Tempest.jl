# What a stage's result says, without loading Tempest: summarize.jl reads
# results with it, and the shrinker asks it whether a failure persists.

# The stages that measure an error, in the order they run, and the element
# type each runs in. The reference is eager Julia in Float64, never eager
# half precision: Tempest sums half-precision values in Float32, closer to
# Float64 than eager's sums are.
const MEASURED = [
    :fwd_interp => Float64, :fwd_cpu => Float64, :grad_fd => Float64, :grad_interp => Float64, :grad_cpu => Float64,
    :grad_ride => Float64, :grad_recompute => Float64, :cuda_fwd => Float32, :cuda_grad => Float32, :cuda_ride => Float32,
    :cuda_repeat => Float32,
]

# Relative to the reference's norm, by element type; finite differences are
# the derivative to about the square root of Float64's precision; two runs on
# a device agree bitwise.
const TOLERANCE = Dict(Float64 => 1.0e-8, Float32 => 2.0e-3, :grad_fd => 1.0e-4, :cuda_repeat => 0.0)

tolerance(stage) = get(TOLERANCE, stage, TOLERANCE[Dict(MEASURED)[stage]])

# The stages in the order they run, those that trace among them.
const STAGES = [:trace, :fwd_interp, :fwd_cpu, :grad_eager, :grad_fd, :grad_trace, :grad_interp, :grad_cpu, :grad_ride,
    :grad_recompute, :cuda_trace, :cuda_fwd, :cuda_grad_trace, :cuda_grad, :cuda_ride, :cuda_repeat]

# Messages differ in numbers from case to case; the same fault's do not
# otherwise.
normalized(message) = replace(message, r"%\d+" => "%N", r"\d+" => "N")

"""
    verdict(stage, value) -> nothing or String

How stage `stage` failed with `value`, or `nothing` where it passed: `"error:
…"` for a stage that threw, `"nonfinite"` for a result not finite where the
reference is, `"mismatch"` for one past the stage's tolerance.
"""
function verdict(stage, value)
    value isa String && startswith(value, "error: ") && return "error: " * first(normalized(value[8:end]), 120)
    value == "nonfinite" && return "nonfinite"
    any(p -> first(p) === stage, MEASURED) && value isa Real && !(value <= tolerance(stage)) && return "mismatch"
    nothing
end

"The failed stages of a case's results, each with its verdict."
function failures(out)
    found = Dict{Symbol, String}()
    for stage in STAGES
        haskey(out, stage) || continue
        v = verdict(stage, out[stage])
        v === nothing || (found[stage] = v)
    end
    found
end
