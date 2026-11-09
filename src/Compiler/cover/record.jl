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
# Times taken with each launch's data in memory, the cache cleared before it
# (see `time_launch`), in files named so: the files before them hold times
# taken from the cache, which are not to be compared with these.
record_path(target) = device_file(target, ".cold.tsv")
rates_path(target) = device_file(target, ".cold.rates.tsv")

function load_record(path)
    record = Dict{String, Float64}()
    path !== nothing && isfile(path) || return record
    for line in eachline(path)
        key, value = rsplit(line, '\t'; limit = 2)
        record[key] = parse(Float64, value)
    end
    record
end

function save_record(path, key, value::Float64)
    path === nothing && return
    mkpath(dirname(path))
    open(io -> println(io, key, '\t', value), path, "a")
end

# `value` kept under `key` in `table`, the record or the rates as a decider
# reads them, and appended to its file at `path`: the one writer of both
# files, and of each value they hold. A table in memory holds what its file
# reads back, but for the claims that failed to build in this preparation,
# losing in it alone (see `build_option!`), with which decisions are not
# settled (see `settled`). Both are given the same Float64, which prints as
# the shortest decimal that parses back to it; a Float32 a timer returned,
# kept as a Float64 and printed as itself, reads back as another value, and
# decisions whose evidence held the one would not hold on the other (see
# `holds`).
function keep!(table, path, key, value)
    value = Float64(value)
    table[key] = value
    save_record(path, key, value)
    value
end

"""
The tuning record or the rates file as a decider reads it: a table that notes
each key it is asked for, and whether anything was written to it. The keys
it noted, with the values they hold, are what decisions reached on it rest
on (see `Reached`), and a table nothing was written to while they were
reached is one they would be reached on again. Read and written under a
lock: kernels built side by side keep the claims that fail to build, and the
floor's options are priced as their groups build.
"""
struct Noted <: AbstractDict{String, Float64}
    table::Dict{String, Float64}
    read::Set{String}
    written::Base.RefValue{Bool}
    lock::ReentrantLock
end

Noted(table::Dict{String, Float64} = Dict{String, Float64}()) = Noted(table, Set{String}(), Ref(false), ReentrantLock())

# The value under `key`, or nothing, the key noted.
noted(t::Noted, key) = @lock t.lock (push!(t.read, key); get(t.table, key, nothing))

Base.haskey(t::Noted, key) = noted(t, key) !== nothing
Base.getindex(t::Noted, key) = (value = noted(t, key); value === nothing ? throw(KeyError(key)) : value)
Base.get(t::Noted, key, default) = (value = noted(t, key); value === nothing ? default : value)
Base.get!(f::Base.Callable, t::Noted, key) = (value = noted(t, key); value === nothing ? (t[key] = f()) : value)
Base.setindex!(t::Noted, value, key) = @lock t.lock (t.written[] = true; t.table[key] = value; t)
Base.length(t::Noted) = length(t.table)
Base.iterate(t::Noted, state...) = iterate(t.table, state...)

# Each key the table noted with the value it holds, or nothing: the evidence
# decisions reached on it rest on.
evidence(t::Noted) = @lock t.lock sort!([key => get(t.table, key, nothing) for key in t.read]; by = first)

# Whether the table holds what `evidence` found, each key's value or its absence.
holds(t::Noted, evidence) = all(((key, value),) -> isequal(get(t.table, key, nothing), value), evidence)

# Whether anything was written to the table since it was made or `unwritten!`.
written(t::Noted) = t.written[]
unwritten!(t::Noted) = (t.written[] = false; t)
