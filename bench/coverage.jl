# Device time of the coverage suite's plans (test/compositions.jl), each
# against its time recorded in a TSV, which the first run writes: a plan more
# than 1.5 times slower than its record sounds an alarm, and the script exits
# with an error. Time is an alarm, not a target; the suite itself checks
# structure (test/coverage.jl). Hold the GPU alone while this runs.
#   julia --project=dev bench/coverage.jl [record.tsv]
# The record is bench/coverage-<device>.tsv by default; delete a line of it to
# record that case again.
using Tempest, CUDACore, cuBLAS, cuTile, Printf
import Tempest as T

# Plans by estimate, as the tests make them.
ENV["TEMPEST_CACHE"] = mktempdir()
include(joinpath(@__DIR__, "..", "test", "compositions.jl"))

const ALARM = 1.5

function replay_graph(ex; repetitions)
    execute!(ex)
    CUDACore.synchronize()
    graph = CUDACore.capture() do
        for _ = 1:repetitions
            foreach(launch -> launch(), ex.launches)
        end
    end
    CUDACore.instantiate(graph)
end

# Each captured graph is destroyed once timed: the garbage collector may
# otherwise unload the kernels' modules first, which crashes the driver.
function device_microseconds(ex; samples = 15, milliseconds = 4.0)
    once = let g = replay_graph(ex; repetitions = 1)
        CUDACore.launch(g)
        t = minimum(_ -> CUDACore.@elapsed(CUDACore.launch(g)), 1:3)
        finalize(g)
        t
    end
    repetitions = clamp(round(Int, 1.0e-3 * milliseconds / once), 1, 200)
    g = replay_graph(ex; repetitions)
    CUDACore.launch(g)
    t = minimum(_ -> 1.0e6 * CUDACore.@elapsed(CUDACore.launch(g)) / repetitions, 1:samples)
    finalize(g)
    t
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
