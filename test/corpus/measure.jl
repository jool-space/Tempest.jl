# One preparation of a corpus program and its measurements, in the worker
# process run.jl starts; included into the program's module beside its
# `instance`. Returns the result line's fields.
using Tempest, CUDACore
using Tempest.Compiler: trace, prepare, bind!, execute!, capture!, replay!, outputs, leaves

device(x::AbstractArray) = CuArray(x)
device(x::Union{Tuple, NamedTuple}) = map(device, x)
device(x) = x

host(x::CuArray) = Array(x)
host(x) = x

# The inputs as the Float64 reference takes them: each Float32 array widened,
# so both sides start from the same rounded numbers.
wide(x::AbstractArray{Float32}) = Float64.(x)
wide(x::Union{Tuple, NamedTuple}) = map(wide, x)
wide(x) = x

# An output's error relative to its largest magnitude; an output that is zero
# throughout, as a gradient through nothing is, by its largest error alone.
relative(a, b) = maximum(abs.(a .- b)) / max(maximum(abs.(b)), floatmin(Float64))

"""
Milliseconds of one step: the minimum over samples of replays of a graph
capturing a few milliseconds of steps; three samples of a step past a second.
"""
function step_milliseconds(ex; milliseconds = 20.0)
    replay!(capture!(ex))
    CUDACore.synchronize()
    once = minimum(_ -> CUDACore.@elapsed(replay!(ex)), 1:3)
    samples = once > 1.0 ? 3 : 10
    repetitions = clamp(round(Int, 1.0e-3 * milliseconds / once), 1, 200)
    replay!(capture!(ex; repetitions))
    minimum(_ -> 1.0e3 * CUDACore.@elapsed(replay!(ex)) / repetitions, 1:samples)
end

# Where each output is in the step's results, in the order of `leaves`.
paths(x::NamedTuple, at) = reduce(vcat, [paths(v, "$at.$k") for (k, v) in pairs(x)]; init = String[])
paths(x::Tuple, at) = reduce(vcat, [paths(v, "$at[$i]") for (i, v) in enumerate(x)]; init = String[])
paths(x, at) = [at]

"""
The step's outputs from the inputs as given, against the step run eagerly in
Float64: the largest relative error, which output it is, and the first
output's, the loss's where the step returns it first.
"""
function agreement(ex, case)
    bind!(ex, case.args...)
    execute!(ex)
    got = map(host, leaves(outputs(ex)))
    expected = case.step(map(wide, case.args)...)
    errors = map(relative, got, leaves(expected))
    worst = argmax(errors)
    (errors[worst], paths(expected, "")[worst], first(errors))
end

function measure(case; check)
    check &= get(case, :check, true)
    started = time()
    program = trace(case.step, case.args...)
    ex = prepare(program, map(device, case.args)...; target = case.target, replaces = case.replaces)
    execute!(ex)
    CUDACore.synchronize()
    prep_s = time() - started
    println("prepared in ", round(prep_s; digits = 1), " s, ", length(ex.kernels), " kernels")
    agrees, worst, first_output = check ? agreement(ex, case) : (NaN, "", NaN)
    check && println("agreement with Float64 eager ", agrees, " at ", worst, "; the first output's ", first_output)
    step_ms = step_milliseconds(ex)
    println("step ", step_ms, " ms")
    note = join(filter(!isempty, [get(case, :note, ""), check ? "worst at $worst, the first output to $(round(first_output; sigdigits = 2))" : ""]), "; ")
    (; prep_s, kernels = length(ex.kernels), step_ms, agreement = agrees, precision = case.precision, note)
end
