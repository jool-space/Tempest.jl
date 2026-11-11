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

# A case is checked as it is, or, where its reference is past the host, by a
# smaller case standing for it, prepared and checked in its place.
checked(case) = (c = get(case, :check, true); c isa NamedTuple ? c : c ? case : nothing)

# Every plan a preparation forms is verified (see `Tempest.Compiler.verify`),
# the cover's trials among them: under 1% of a cold preparation, the plan
# built's check of a warm one, so that no time the corpus records is of a
# plan the verifier would reject.
prepared(case) = prepare(trace(case.step, case.args...), map(device, case.args)...; target = case.target, replaces = case.replaces, verify = true)

"""
    measure(case; check, report) -> result

Trace `case`'s step, prepare it and execute it once, the seconds to that first
result; then, with `check`, its agreement with the step run in Float64; then
the step's milliseconds. With `report`, the profile and the explanation the
environment names are written last.
"""
function measure(case; check, report)
    stand_in = checked(case)
    check &= stand_in !== nothing
    started = time()
    ex = prepared(case)
    execute!(ex)
    CUDACore.synchronize()
    first_s = time() - started
    println("prepared in ", round(first_s; digits = 1), " s, ", length(ex.kernels), " kernels")
    agrees, worst, first_output = check ? agreement(stand_in === case ? ex : prepared(stand_in), stand_in) : (NaN, "", NaN)
    check && println("agreement with Float64 eager ", agrees, " at ", worst, "; the first output's ", first_output)
    step_ms = step_milliseconds(ex)
    println("step ", step_ms, " ms")
    if report
        path = get(ENV, "TEMPEST_CORPUS_PROFILE", "")
        isempty(path) || profile(path, ex)
        path = get(ENV, "TEMPEST_CORPUS_EXPLAIN", "")
        isempty(path) || open(io -> Tempest.Compiler.explain(io, ex), path, "w")
    end
    # A device Tempest ships rates for is not calibrated, cold or warm.
    shipped = isempty(Tempest.Compiler.shipped_rates(case.target)) ? "" : "rates shipped for the device"
    note = join(filter(!isempty, [get(case, :note, ""), "plans verified", shipped, check ? "worst at $worst, the first output to $(round(first_output; sigdigits = 2))" : ""]), "; ")
    (; first_s, kernels = length(ex.kernels), step_ms, agreement = agrees, precision = case.precision, note)
end

"""
Every kernel of a prepared step timed alone, as rows of a tab-separated file
at `path`: its place, microseconds, what runs it, the group's kind and
domain, the bytes it moves at least (what it reads once and writes), the
flops of its products as its implementation estimates them, the cover's
estimate in microseconds, and the kernel as it describes itself.
"""
function profile(path, ex)
    C = Tempest.Compiler
    open(path, "w") do io
        println(io, join(("kernel", "us", "runner", "kind", "domain", "bytes", "flops", "estimate_us", "description"), '\t'))
        for k in eachindex(ex.kernels)
            g = ex.plan.groups[k]
            d = ex.decisions[k]
            seconds = C.time_launch(ex.target, ex.launches[k])
            implementation = d === nothing ? nothing : d.choice.implementation
            work = implementation === nothing ? Any[] :
                Base.invokelatest(C.Implement.estimate, implementation, d.choice.claim, ex.target, C.group_view(ex.plan, g))
            flops = sum(w -> w.flops + w.recomputed, work; init = 0.0)
            estimate = d === nothing ? NaN : 1.0e6 * d.choice.estimate
            description = first(replace(string(ex.kernels[k]), r"[\t\n]" => " "), 300)
            println(io, join((k, round(1.0e6 * seconds; digits = 2), C.runner(ex, k), g.kind, g.domain, C.group_bytes(ex.plan, g),
                flops, round(estimate; digits = 2), description), '\t'))
        end
    end
end
