# The corpus: whole programs, each a training or inference step written as
# plain Julia the way a user writes it, run against its challengers by one
# command, and recorded.
#
#     julia --project=test test/corpus/run.jl PROGRAM [--shape small|full] [--scenarios cold,warm,...] [--runs 2]
#         [--after PROGRAM[:SHAPE]] [--challengers]
#
# PROGRAM names a file here defining `instance(shape)`: the step as a
# function, its inputs in Float32, which results replace which inputs, and
# the target; its challengers are in challengers/challengers.jl.
#
# Tempest's side is a set of scenarios, each a way a program comes to its
# first result, all of them by default or those `--scenarios` names (none
# where it names none, for the challengers alone), and the
# set is run `--runs` times (2), each round in processes and stores of its
# own, so that each scenario's numbers come from as many processes. A
# process's stores are a directory of its own: TEMPEST_CACHE, whose `tuning`
# holds the record of times measured, the rates fitted and the decisions
# kept, and JULIA_OBJCACHE_PATH in it, the object cache of compiled kernels.
#
#   cold              A new process whose stores are empty: no kernel
#                     compiled, no time recorded and no decision kept. It
#                     prepares as the environment's settings say, by default
#                     measuring close calls and settling what it measured, on
#                     the rates Tempest ships for the device where it ships
#                     them.
#   cold-unmeasured   The same with nothing timed (TEMPEST_MEASURE=none).
#   cold-known        A new process holding the tuning directory the round's
#                     cold process left, its record and its decisions, with an
#                     empty object cache: it replays the decisions and builds
#                     only its plan's kernels.
#   warm              A new process on everything the round's cold process
#                     left.
#   again             The program prepared a second time in the warm process,
#                     the same instance.
#   after             A new process with empty stores that prepares another
#                     program (`--after`: heat at its small shape, or node
#                     before heat), then this one with TEMPEST_CACHE moved to a
#                     new directory, so that no time is recorded and no
#                     decision kept for it. What it shares with the other is
#                     what a session shares and a store's path does not name:
#                     Julia's compiled code, the session's tables
#                     (`forget_session!`'s: kernels resolved by their content,
#                     packages' identities) and the object cache, which a
#                     process opens once before Julia 1.14, so that a kernel
#                     of the other's with the same content loads.
#   after-unmeasured  The same with nothing timed in either.
#
# Each preparation traces the step, prepares it, verifying every plan it
# forms, and executes it once: the seconds from the trace to that first
# result, the packages loaded and the instance made before. Of those seconds,
# Julia's compilation as its cumulative counter reads it, which sums the
# threads that compile and so may pass the seconds themselves. With them, the
# kernels compiled and those loaded from the object cache in the preparation,
# beside the plan's; the bytes the plan stores; the deciders, a recurrence's
# step's among them, that replayed their decisions and those the cover
# reached; and a hash of the plan, of what `explain` says of it and what runs
# each kernel, alike where two preparations reached one plan. Then it takes
# the step's outputs from the inputs as given and compares them with the step
# run eagerly in Float64 on the same inputs, widened, computed once a run of
# this command and kept for its processes: the agreement is the largest error
# of any output relative to that output's largest magnitude, the note saying
# which output and the first output's. Last it times the step as replays of a
# captured graph of a few milliseconds of steps, the minimum over samples
# (measure.jl). With `--challengers`, each challenger then runs by its own
# script, through its launcher in challengers/.
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
    "first_s", "julia_s", "step_ms", "kernels", "built", "loaded", "stored_bytes", "replayed", "covered", "plan", "agreement", "note")
const SCENARIOS = ("cold", "cold-unmeasured", "cold-known", "warm", "again", "after", "after-unmeasured")

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
A preparation a process makes: of a program at a shape, recorded under its
scenario unless that is empty, with TEMPEST_CACHE moved first to `tuning`
unless that is empty.
"""
struct Preparation
    program::String
    shape::String
    scenario::String
    tuning::String
end

"""
One of a round's processes: the preparations it makes in turn, the directory
of the stores it starts on, the stores whose tuning directory it is given
before it starts unless that is empty, and the environment's TEMPEST_MEASURE
unless that is empty.
"""
struct Session
    preparations::Vector{Preparation}
    stores::String
    tuning_of::String
    measure::String
end

scenarios(s::Session) = filter(!isempty, map(p -> p.scenario, s.preparations))

# The settings that point a process at its stores (test/fixtures/stores.jl),
# and what else the session sets.
function session_env(s::Session)
    env = ["TEMPEST_CACHE" => s.stores, "JULIA_OBJCACHE_PATH" => joinpath(s.stores, "objcache")]
    isempty(s.measure) ? env : [env; "TEMPEST_MEASURE" => s.measure]
end

# The processes of a round for the scenarios wanted, in the order they run:
# cold-known starts on the tuning directory the cold process left, before the
# warm one adds to it, and the warm one on all of it.
function sessions(program, shape, wanted, other, round)
    any(in(wanted), ("cold-known", "warm", "again")) && !("cold" in wanted) &&
        error("cold-known, warm and again start on what cold leaves: run them with it")
    at(scenario; tuning = "") = Preparation(program, shape, scenario, tuning)
    before = Preparation(other..., "", "")
    stores(name) = joinpath(round, name)
    warm = "again" in wanted ? [at("warm"), at("again")] : [at("warm")]
    all = [
        Session([at("cold")], stores("cold"), "", ""),
        Session([at("cold-known")], stores("cold-known"), stores("cold"), ""),
        Session(warm, stores("cold"), "", ""),
        Session([at("cold-unmeasured")], stores("cold-unmeasured"), "", "none"),
        Session([before, at("after"; tuning = stores("after-own"))], stores("after"), "", ""),
        Session([before, at("after-unmeasured"; tuning = stores("after-unmeasured-own"))], stores("after-unmeasured"), "", "none")]
    filter(s -> any(in(wanted), scenarios(s)), all)
end

# One process of a round: a row for each of its scenarios, with its result
# or why it failed.
function session(s::Session, run, reference, directory)
    program, shape = s.preparations[end].program, s.preparations[end].shape
    names = join(scenarios(s), " and ")
    log = joinpath(directory, "logs", "$program-$shape-tempest-$(replace(names, " and " => "+"))-$run-$(stamp()).log")
    mkpath(s.stores)
    if !isempty(s.tuning_of)
        given = joinpath(s.tuning_of, "tuning")
        isdir(given) || return map(scenario -> (; scenario, side = "tempest", note = "no tuning directory left to start on"), scenarios(s))
        cp(given, joinpath(s.stores, "tuning"))
    end
    check = get(ENV, "TEMPEST_CORPUS_CHECK", "1") != "0" ? ["--check", reference] : String[]
    preparations = reduce(vcat, [[p.program, p.shape, dash(p.scenario), dash(p.tuning)] for p in s.preparations])
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$PROJECT $(joinpath(HERE, "run.jl")) --worker $check $preparations`
    println("tempest $program $shape, $names, run $run: ", log)
    status, clock = gpu_job(cmd, log; env = session_env(s))
    results = Dict{String, Any}()
    for line in filter(l -> startswith(l, "corpus result "), readlines(log))
        r = eval(Meta.parse(line[(length("corpus result ") + 1):end]))
        results[r.scenario] = r
    end
    map(scenarios(s)) do scenario
        r = get(results, scenario, nothing)
        r === nothing && return (; scenario, side = "tempest", sm_mhz = clock, note = failure(log, status))
        merge(r, (; side = "tempest", sm_mhz = clock))
    end
end

# An empty argument to the worker, and back.
dash(x) = isempty(x) ? "-" : x
undash(x) = x == "-" ? "" : x

# Tempest's rows, each given to `record` as its process ends: each round's
# processes in turn, in stores of the round's own, and the Float64 reference
# kept for all of them in one directory.
function tempest(record, program, shape, wanted, other, runs, directory)
    reference = mktempdir()
    for run in 1:runs
        round = mktempdir()
        for s in sessions(program, shape, wanted, other, round), r in session(s, run, reference, directory)
            r.scenario in wanted && record(r)
        end
        rm(round; recursive = true)
    end
    rm(reference; recursive = true)
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

# The worker's arguments: `--check` and the directory its references are kept
# in, then each preparation as its program, shape, scenario and tuning
# directory, `-` for one empty, in the order it makes them. A program's file
# is included once a process, and its instance at a shape made once.
function worker(args)
    i = findfirst(==("--check"), args)
    reference = i === nothing ? nothing : args[i + 1]
    rest = i === nothing ? args : args[[1:(i - 1); (i + 2):end]]
    @eval Main begin
        import cuBLAS, cuTile
        using Tempest, CUDACore
    end
    modules, cases = Dict{String, Module}(), Dict{Tuple{String, String}, Any}()
    for (program, shape, scenario, tuning) in Iterators.partition(rest, 4)
        scenario, tuning = undash(scenario), undash(tuning)
        m = get!(modules, program) do
            m = Core.eval(Main, :(module $(Symbol(program)) end))
            Base.include(m, joinpath(HERE, program * ".jl"))
            Base.include(m, joinpath(HERE, "measure.jl"))
            m
        end
        case = get!(() -> Base.invokelatest(m.instance, Symbol(shape)), cases, (program, shape))
        isempty(tuning) || (ENV["TEMPEST_CACHE"] = mkpath(tuning))
        kept = reference === nothing || isempty(scenario) ? nothing : joinpath(reference, "$program-$shape.jls")
        result = Base.invokelatest(() -> m.measure(case; reference = kept, report = scenario == "warm"))
        println("corpus result ", repr(merge((; scenario), result)))
    end
end

# The program prepared before this one in the after scenarios.
function before(program, args)
    given = option(args, "--after", program == "heat" ? "node" : "heat")
    name, shape = occursin(':', given) ? split(given, ':'; limit = 2) : (given, "small")
    name == program && error("--after names the program itself")
    (String(name), String(shape))
end

function main(args)
    if "--worker" in args
        return worker(args[(findfirst(==("--worker"), args) + 1):end])
    end
    program = first(args)
    isfile(joinpath(HERE, program * ".jl")) && haskey(CHALLENGERS, program) || error("$program is no program of the corpus")
    shape = option(args, "--shape", "small")
    wanted = filter(!isempty, split(option(args, "--scenarios", join(SCENARIOS, ',')), ','))
    foreach(s -> s in SCENARIOS || error("$s is no scenario; the scenarios are $(join(SCENARIOS, ", "))"), wanted)
    runs = parse(Int, option(args, "--runs", "2"))
    directory = record_directory()
    mkpath(joinpath(directory, "logs"))
    println("record: ", joinpath(directory, "corpus.tsv"))
    common = (; pass = stamp(), host = gethostname(), program, shape, commit = commit(), device = device_name())
    io = open_record(directory)
    tempest(program, shape, wanted, before(program, args), runs, directory) do r
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
