# The corpus: whole programs, each a training or inference step written as
# plain Julia the way a user writes it, run against its challengers by one
# command, and recorded.
#
#     julia --project=test test/corpus/run.jl PROGRAM [--shape small|full] [--scenarios cold,warm] [--runs 2] [--challengers]
#
# PROGRAM names a file here defining `instance(shape)`: the step as a
# function, its inputs in Float32, which results replace which inputs, and
# the target; its challengers are in challengers/challengers.jl.
#
# Tempest's side is a set of scenarios, each a way a program comes to its
# first result, all of them by default or those `--scenarios` names, and the
# set is run `--runs` times (2), each round in processes and stores of its
# own, so that each scenario's numbers come from as many processes:
#
#   cold    A new process whose stores are empty: TEMPEST_CACHE and
#           JULIA_OBJCACHE_PATH new directories, so no kernel compiled, no
#           time recorded and no decision kept. It prepares as the
#           environment's settings say, by default measuring close calls and
#           settling what it measured.
#   warm    A new process on the stores the round's cold process left.
#
# In each, the process traces the step, prepares it, verifying every plan it
# forms, and executes it once: the seconds from the trace to that first
# result. Then it times the step as replays of a captured graph of a few
# milliseconds of steps, the minimum over samples (measure.jl). The warm
# process also takes the step's outputs from the inputs as given and compares
# them with the step run eagerly in Float64 on the same inputs, widened: the
# agreement is the largest error of any output relative to that output's
# largest magnitude, the note saying which output and the first output's.
# With `--challengers`, each challenger then runs by its own script, through
# its launcher in challengers/.
#
# One row per scenario of each round, and per challenger's run, is appended
# to corpus.tsv in Tempest's scratch space `corpus`
# (~/.julia/scratchspaces/<Tempest's uuid>/corpus), or in the directory
# TEMPEST_CORPUS names, beside each job's log in logs/; the rows of one
# invocation share its `pass`, and each names the machine it ran on. A record
# begins with its columns' names, and holds them again wherever the columns
# change, so that rows of an earlier layout stay readable. Read it with
#
#     julia --project=test test/corpus/summarize.jl
#
# Every job runs under `flock` of TEMPEST_GPU_LOCK when it is set, as every
# job on a shared GPU must, and is stopped after TEMPEST_CORPUS_TIMEOUT
# seconds (1800). The SM clock recorded is the median of the clocks seen
# while the GPU was busy (challengers/clocked.sh), in the row's own process.
# TEMPEST_CORPUS_CHECK=0 skips the Float64 reference: its gradient is the
# interpreter's on the CPU, which holds every value it computes, and a
# program whose reference is past the host's memory is checked by a smaller
# instance its `check` names. TEMPEST_CORPUS_PROFILE names a file for every
# kernel of the step timed alone, with what runs it, its bytes and flops and
# the cover's estimate, and TEMPEST_CORPUS_EXPLAIN one for the plan; the warm
# process writes them last. A job killed by a signal stops the run: on this
# machine that is the memory the GPU shares running out. runtests.jl skips
# this directory.

const HERE = @__DIR__
const PROJECT = dirname(HERE)
const COLUMNS = ("when", "pass", "host", "program", "shape", "scenario", "side", "commit", "device", "sm_mhz", "precision",
    "first_s", "step_ms", "kernels", "agreement", "note")
const SCENARIOS = ("cold", "warm")

include(joinpath(HERE, "challengers", "challengers.jl"))

option(args, name, default) = (i = findfirst(==(name), args); i === nothing ? default : args[i + 1])

# Tempest's scratch space `corpus`, as test/fuzz's `fuzz`: Scratch is
# Tempest's dependency rather than the tests', so it is loaded by its identity.
function scratch_corpus()
    scratch = Base.require(Base.PkgId(Base.UUID("6c6a2e73-6563-6170-7368-637461726353"), "Scratch"))
    Base.invokelatest(scratch.get_scratch!, Base.UUID("aba8ef0c-1387-4c24-84ad-a51c8dc32f50"), "corpus")
end

record_directory() = haskey(ENV, "TEMPEST_CORPUS") ? abspath(ENV["TEMPEST_CORPUS"]) : scratch_corpus()

commit() = readchomp(`git -C $PROJECT describe --always --dirty --abbrev=7`)

device_name() = first(readlines(`nvidia-smi --query-gpu=name --format=csv,noheader`))

"""
Runs `cmd` as a GPU job in `env`: under the lock, the clock sampled, stopped
after the timeout, its output in `log`. Returns the exit code and the median
busy clock.
"""
function gpu_job(cmd, log; env = ())
    seconds = get(ENV, "TEMPEST_CORPUS_TIMEOUT", "1800")
    job = `bash $(joinpath(HERE, "challengers", "clocked.sh")) $seconds $cmd`
    lock = get(ENV, "TEMPEST_GPU_LOCK", nothing)
    lock === nothing || (job = `flock $lock $job`)
    process = open(io -> run(pipeline(ignorestatus(addenv(job, env...)); stdout = io, stderr = io)), log, "w")
    # A job the kernel killed for memory leaves the GPU's memory to the
    # machine again; another job now would meet the same end.
    process.termsignal != 0 && error("$cmd was killed by signal $(process.termsignal); see $log")
    status = process.exitcode
    status in (137, 139) && error("$cmd was killed (exit $status); see $log")
    line = last_line(l -> startswith(l, "corpus clock "), log)
    clock = line === nothing ? nothing : tryparse(Int, last(split(line)))
    (status, clock)
end

# The last line of a log that `f` accepts.
last_line(f, log) = (lines = readlines(log); i = findlast(f, lines); i === nothing ? nothing : lines[i])

# Why a job failed, in a line: the first error it printed, or how it ended.
function failure(log, status)
    status == 124 && return "timed out after $(get(ENV, "TEMPEST_CORPUS_TIMEOUT", "1800")) s"
    lines = readlines(log)
    i = findfirst(l -> occursin(r"ERROR|Error|error:", l), lines)
    i === nothing ? "exit $status" : first(strip(lines[i]), 200)
end

function row!(io, values)
    fields = map(c -> replace(string(something(get(values, Symbol(c), ""), "")), r"[\t\n]" => " "), COLUMNS)
    println(io, join(fields, '\t'))
    flush(io)
end

# A record's columns are named by its first line and again wherever they
# change: a line beginning with the first column's name.
is_header(line) = startswith(line, first(COLUMNS) * '\t')

function open_record(directory)
    path = joinpath(directory, "corpus.tsv")
    last_header = isfile(path) ? last_line(is_header, path) : nothing
    io = open(path, "a")
    last_header == join(COLUMNS, '\t') || println(io, join(COLUMNS, '\t'))
    io
end

stamp() = string(round(Int, time()))

## Tempest's side: the scenarios, each prepared in a process of a round.

"""
One of a round's processes: the scenarios of the preparations it makes in
turn, each a row, and the directory of the stores it starts on.
"""
struct Session
    scenarios::Vector{String}
    stores::String
end

# What a process keeps on disk, in one directory: Tempest's tuning directory
# and the object cache of compiled kernels inside it (test/fixtures/stores.jl).
stores_env(dir) = ["TEMPEST_CACHE" => dir, "JULIA_OBJCACHE_PATH" => joinpath(dir, "objcache")]

# The processes of a round for the scenarios wanted, in the order they run:
# the warm process starts on the stores the cold one left.
function sessions(wanted, round)
    "warm" in wanted && !("cold" in wanted) && error("the warm scenario starts on what the cold one leaves: run them together")
    cold = joinpath(round, "cold")
    all = [Session(["cold"], cold), Session(["warm"], cold)]
    filter(s -> any(in(wanted), s.scenarios), all)
end

# One process of a round: a row for each of its scenarios, with its result
# or why it failed.
function session(program, shape, s::Session, run, directory)
    log = joinpath(directory, "logs", "$program-$shape-tempest-$(join(s.scenarios, '-'))-$run-$(stamp()).log")
    check = "warm" in s.scenarios && get(ENV, "TEMPEST_CORPUS_CHECK", "1") != "0"
    flags = check ? ["--check"] : String[]
    preparations = reduce(vcat, [[program, shape, scenario] for scenario in s.scenarios])
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$PROJECT $(joinpath(HERE, "run.jl")) --worker $flags $preparations`
    println("tempest $program $shape, $(join(s.scenarios, " and ")), run $run: ", log)
    status, clock = gpu_job(cmd, log; env = stores_env(s.stores))
    results = Dict{String, Any}()
    for line in filter(l -> startswith(l, "corpus result "), readlines(log))
        r = eval(Meta.parse(line[(length("corpus result ") + 1):end]))
        results[r.scenario] = r
    end
    map(s.scenarios) do scenario
        r = get(results, scenario, nothing)
        r === nothing && return (; scenario, side = "tempest", sm_mhz = clock, note = failure(log, status))
        merge(r, (; side = "tempest", sm_mhz = clock))
    end
end

# Tempest's rows: each round's processes in turn, in stores of the round's own.
function tempest(program, shape, wanted, runs, directory)
    rows = []
    for run in 1:runs
        round = mktempdir()
        for s in sessions(wanted, round)
            append!(rows, session(program, shape, s, run, directory))
        end
        rm(round; recursive = true)
    end
    rows
end

function challengers(program, shape, directory)
    rows = []
    for c in CHALLENGERS[program]
        log = joinpath(directory, "logs", "$program-$shape-$(c.name)-$(stamp()).log")
        println("challenger ", c.name, " for $program $shape: ", log)
        command = c.command(shape)
        if command isa String
            push!(rows, (; side = c.name, note = "unavailable: " * command))
            continue
        end
        status, clock = gpu_job(command, log)
        found = status == 0 ? c.parse(readlines(log)) : []
        isempty(found) && (push!(rows, (; side = c.name, sm_mhz = clock, note = "unavailable: " * failure(log, status))); continue)
        append!(rows, map(r -> merge(r, (; sm_mhz = clock)), found))
    end
    rows
end

## The worker: preparations and their measurements, in a process of its own.

# The worker's arguments: `--check`, then each preparation as its program,
# shape and scenario, in the order it makes them.
function worker(args)
    check = "--check" in args
    rest = filter(!=("--check"), args)
    @eval Main begin
        import cuBLAS, cuTile
        using Tempest, CUDACore
    end
    for (program, shape, scenario) in Iterators.partition(rest, 3)
        m = Core.eval(Main, :(module Program end))
        Base.include(m, joinpath(HERE, program * ".jl"))
        Base.include(m, joinpath(HERE, "measure.jl"))
        result = Base.invokelatest(() -> m.measure(m.instance(Symbol(shape)); check, report = scenario == "warm"))
        println("corpus result ", repr(merge((; scenario), result)))
    end
end

function main(args)
    if "--worker" in args
        return worker(args[(findfirst(==("--worker"), args) + 1):end])
    end
    program = first(args)
    isfile(joinpath(HERE, program * ".jl")) && haskey(CHALLENGERS, program) || error("$program is no program of the corpus")
    shape = option(args, "--shape", "small")
    wanted = split(option(args, "--scenarios", join(SCENARIOS, ',')), ',')
    foreach(s -> s in SCENARIOS || error("$s is no scenario; the scenarios are $(join(SCENARIOS, ", "))"), wanted)
    runs = parse(Int, option(args, "--runs", "2"))
    directory = record_directory()
    mkpath(joinpath(directory, "logs"))
    println("record: ", joinpath(directory, "corpus.tsv"))
    common = (; pass = stamp(), host = gethostname(), program, shape, commit = commit(), device = device_name())
    io = open_record(directory)
    for r in tempest(program, shape, wanted, runs, directory)
        row!(io, merge(common, (; when = stamp()), r))
        println(r)
    end
    if "--challengers" in args
        for r in challengers(program, shape, directory)
            row!(io, merge(common, (; when = stamp()), r))
            println(r)
        end
    end
    close(io)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
