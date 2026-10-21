# The tuning record: measured seconds by claim, and the rates file beside it,
# one of each per device.

using Scratch: @get_scratch!

# Tempest's `tuning` scratch space in the depot. Its path is found when Tempest
# loads, so it is not part of the precompiled image; while Tempest precompiles
# there is none, and its workload records nothing unless an environment
# variable names a directory.
const TUNING = Ref{Union{Nothing, String}}(nothing)

__init__() = (TUNING[] = @get_scratch!("tuning"); nothing)

"""
    Tempest.Compiler.tuning_directory() -> directory or nothing

Where Tempest keeps what it measures: `tuning` in `ENV["TEMPEST_CACHE"]` when
set, otherwise Tempest's `tuning` scratch space.
"""
function tuning_directory()
    dir = get(ENV, "TEMPEST_CACHE", nothing)
    dir === nothing ? TUNING[] : joinpath(dir, "tuning")
end

device_file(target, suffix) = (dir = tuning_directory(); dir === nothing ? nothing :
    joinpath(dir, replace(device_name(target), r"[^A-Za-z0-9._-]" => "_") * suffix))
record_path(target) = device_file(target, ".tsv")
rates_path(target) = device_file(target, ".rates.tsv")

function load_record(path)
    record = Dict{String, Float64}()
    path !== nothing && isfile(path) || return record
    for line in eachline(path)
        key, value = rsplit(line, '\t'; limit = 2)
        record[key] = parse(Float64, value)
    end
    record
end

function save_record(path, key, seconds)
    path === nothing && return
    mkpath(dirname(path))
    open(io -> println(io, key, '\t', seconds), path, "a")
end

keep!(calibration, path, key, value) = (calibration[key] = value; save_record(path, key, value); value)
