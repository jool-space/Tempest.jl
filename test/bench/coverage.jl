# Device time of the coverage suite's plans (test/fixtures/compositions.jl), each
# against its time recorded in a TSV, which the first run writes: a plan more
# than 1.5 times slower than its record sounds an alarm, and the script exits
# with an error. Time is an alarm, not a target; the suite itself checks
# structure (test/cuda/coverage.jl). Hold the GPU alone while this runs.
#   julia --project=test test/bench/coverage.jl [record.tsv]
# The record is test/bench/coverage-<device>.tsv by default; delete a line of
# it to record that case again.
using Tempest, CUDACore, cuBLAS, cuTile, Printf
using Tempest.Compiler: trace, differentiate, interpret, prepare, bind!, execute!, capture!, replay!, outputs
import Tempest as T

# Plans by estimate, as the tests make them.
ENV["TEMPEST_CACHE"] = mktempdir()
include(joinpath(@__DIR__, "..", "fixtures", "compositions.jl"))

const ALARM = 1.5

# Microseconds of one execution: the minimum over samples of replays of a
# graph that captures a few milliseconds of executions.
function device_microseconds(ex; samples = 15, milliseconds = 4.0)
    execute!(ex)
    replay!(capture!(ex))
    once = minimum(_ -> CUDACore.@elapsed(replay!(ex)), 1:3)
    repetitions = clamp(round(Int, 1.0e-3 * milliseconds / once), 1, 200)
    replay!(capture!(ex; repetitions))
    minimum(_ -> 1.0e6 * CUDACore.@elapsed(replay!(ex)) / repetitions, 1:samples)
end

device = CUDACore.name(CUDACore.device())
path = isempty(ARGS) ? joinpath(@__DIR__, "coverage-" * replace(device, r"[^A-Za-z0-9._-]" => "_") * ".tsv") : ARGS[1]
recorded = Dict{String, Float64}()
isfile(path) && foreach(l -> (s = split(l, '\t'); recorded[s[1]] = parse(Float64, s[2])), readlines(path))

println("coverage on ", device, ", against ", path)
alarms, new = String[], Pair{String, Float64}[]
for c in COMPOSITIONS
    ex = prepare(c.program, map(CuArray, c.args)...; target = T.CUDATarget(), measure = false)
    t = device_microseconds(ex)
    r = get(recorded, c.name, nothing)
    @printf("%-44s %2d kernels %9.1f μs", c.name, length(ex.kernels), t)
    if r === nothing
        push!(new, c.name => t)
        println("  recorded now")
    else
        t > ALARM * r && push!(alarms, c.name)
        @printf("  record %9.1f μs  %5.2f×%s\n", r, t / r, t > ALARM * r ? "  SLOWER" : "")
    end
end
isempty(new) || open(io -> foreach(((name, t),) -> println(io, name, '\t', t), new), path, "a")
if isempty(alarms)
    println("no alarms")
else
    println("alarm: ", length(alarms), " slower than ", ALARM, "× their record: ", join(alarms, ", "))
    exit(1)
end
