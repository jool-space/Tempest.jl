# The tuning record: measured seconds by claim, and the rates file beside it,
# one of each per device and toolchain.

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

# A file of the device `target` runs on and of the toolchain that builds its
# kernels (see `toolchain`): the times it holds are of kernels built so on
# that device, and those another toolchain builds are timed into a file of
# their own, a kernel's time taken on one never read for the other's.
device_file(target, suffix) = (dir = tuning_directory(); dir === nothing ? nothing : joinpath(dir, basis_file_name(target, suffix)))
basis_file_name(target, suffix) = file_name(join([device_name(target); [string(k, " ", v) for (k, v) in toolchain(target)]], " ")) * suffix
device_file_name(target, suffix) = file_name(device_name(target)) * suffix
file_name(name) = replace(name, r"[^A-Za-z0-9._-]" => "_")
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

# Rates fitted on devices Tempest knows, shipped with it, a file per device
# named for the device alone (`device_file_name`), so that a first
# preparation there prices its plans without calibrating. Rates only price
# options, and the close calls among them are measured whichever rates priced
# them, so a file from another version of a package, whose kernels are faster
# or slower than it says, can make a plan slower, never wrong. test/bench/rates.jl writes
# them, into this file's directory `rates`, found where the package is rather
# than where it was precompiled.
shipped_directory() = joinpath(pkgdir(ROOT), "src", "Compiler", "cover", "rates")

"""
    shipped_rates(target, directory = shipped_directory()) -> Dict

The rates shipped for the device `target` runs on: the file in `directory`
named for it whose identity, the lines `device_identity` gives, is the
device's too. None for a device no file names, nor for one a file names whose
identity differs in any line, as a product of the name with other clocks or
fewer multiprocessors: its decider then fits the rates on first need, as with
nothing shipped, or prices on nominal rates where it measures nothing.
"""
function shipped_rates(target, directory = shipped_directory())
    identity = device_identity(target)
    identity === nothing && return Dict{String, Float64}()
    path = joinpath(directory, device_file_name(target, ".cold.rates.tsv"))
    rates = load_record(path)
    get(rates, "timing basis", nothing) == TIMING_BASIS || return Dict{String, Float64}()
    all(((key, value),) -> get(rates, key, nothing) == value, identity) || return Dict{String, Float64}()
    rates
end

# The rates a decider of `target` measuring at `measure` prices on where it is
# given none: those fitted on this machine, in its rates file, over those
# shipped for its device; none where it prices on nominal rates alone.
known_rates(target, measure) =
    Noted(level(measure) === :dry ? Dict{String, Float64}() : merge(shipped_rates(target), load_record(rates_path(target))))

# The basis the times of a shipped file were taken on, which the file states:
# each launch timed among the launches of a captured graph, its data in memory
# rather than in the cache (see `time_launch`), its buffers holding numbers
# as a program's are (see `buffers!`; on basis 1 their floats were zeros). A
# change to how launches are timed changes it, so that the files are written
# again rather than read on another basis.
const TIMING_BASIS = 2.0

# What tells the device `target` runs on from another of its name, as pairs of
# a name and a number, which a shipped rates file states: the rates the device
# reports (see `device_rates`), so that a product of the name with fewer
# multiprocessors, a smaller cache or slower memory is another device to
# price; nothing for a target that reports none. They are the configuration's
# rated values, which the driver reports alike for every unit of it, not
# measurements: two GB10s report the same, and a unit's own clocks and power
# cap show as the spread of the rates fitted on each.
function device_identity(target)
    measurable(target) || return nothing
    r = device_rates(target)
    ["device bandwidth" => Float64(r.bandwidth), "device cache" => Float64(r.cache), "device blocks" => Float64(r.blocks)]
end
