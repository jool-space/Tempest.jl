# The corpus: whole programs, each a training or inference step written as
# plain Julia the way a user writes it, run against its challengers by one
# command, and recorded.
#
#     julia --project=test test/corpus/run.jl PROGRAM [--shape small|full] [--challengers]
#
# PROGRAM names a file here defining `instance(shape)`: the step as a
# function, its inputs in Float32, which results replace which inputs, and
# the target; its challengers are in challengers/challengers.jl. Tempest's
# side runs twice, each in a new process: cold, with an empty kernel cache
# and no tuning record, then warm, with what the cold run left. Each traces
# the step, prepares it and executes it once, its preparation the seconds to
# that first result, and times it as replays of a captured graph of a few
# milliseconds of steps, the minimum over samples (measure.jl). The warm run
# also takes the step's outputs from the inputs as given and compares them
# with the step run eagerly in Float64 on the same inputs, widened: the
# agreement is the largest error of any output relative to that output's
# largest magnitude, the note saying which output and the first output's.
# With `--challengers`, each challenger then runs by its own script, through
# its launcher in challengers/.
#
# One row per run, Tempest's and each challenger's, is appended to
# corpus.tsv in Tempest's scratch space `corpus`
# (~/.julia/scratchspaces/<Tempest's uuid>/corpus), or in the directory
# TEMPEST_CORPUS names, beside each job's log in logs/. Read them with
#
#     julia --project=test test/corpus/summarize.jl
#
# Every job runs under `flock` of TEMPEST_GPU_LOCK when it is set, as every
# job on a shared GPU must, and is stopped after TEMPEST_CORPUS_TIMEOUT
# seconds (1800). The SM clock recorded is the median of the clocks seen
# while the GPU was busy (challengers/clocked.sh). TEMPEST_CORPUS_CHECK=0
# skips the Float64 reference: its gradient is the interpreter's on the CPU,
# which holds every value it computes, and a program whose reference is past
# the host's memory is checked by a smaller instance its `check` names.
# TEMPEST_CORPUS_PROFILE names a file for every kernel of the step timed
# alone, with what runs it, its bytes and flops and the cover's estimate, and
# TEMPEST_CORPUS_EXPLAIN one for the plan; the warm run writes them last.
# A job killed by a signal stops the
# run: on this machine that is the memory the GPU shares running out.
# runtests.jl skips this directory.

const HERE = @__DIR__
const PROJECT = dirname(HERE)
const COLUMNS = ("when", "program", "shape", "side", "commit", "device", "sm_mhz", "precision", "cold_prep_s", "warm_prep_s",
    "step_ms", "kernels", "agreement", "note")

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

function open_record(directory)
    path = joinpath(directory, "corpus.tsv")
    fresh = !isfile(path)
    io = open(path, "a")
    fresh && println(io, join(COLUMNS, '\t'))
    io
end

stamp() = string(round(Int, time()))

# Tempest's side: a cold and a warm process, each printing one result line.
function tempest(program, shape, directory)
    cache = mktempdir()
    env = ["TEMPEST_CACHE" => cache, "JULIA_OBJCACHE_PATH" => joinpath(cache, "objcache")]
    results = map(("cold", "warm")) do phase
        log = joinpath(directory, "logs", "$program-$shape-tempest-$phase-$(stamp()).log")
        check = phase == "warm" && get(ENV, "TEMPEST_CORPUS_CHECK", "1") != "0"
        flags = check ? ["--check"] : String[]
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$PROJECT $(joinpath(HERE, "run.jl")) --worker $program $shape $flags`
        println("tempest $program $shape, $phase: ", log)
        status, clock = gpu_job(cmd, log; env)
        line = last_line(l -> startswith(l, "corpus result "), log)
        status == 0 && line !== nothing || return (; failed = failure(log, status), clock)
        merge(eval(Meta.parse(line[(length("corpus result ") + 1):end])), (; clock))
    end
    rm(cache; recursive = true)
    cold, warm = results
    haskey(cold, :failed) && return (; side = "tempest", sm_mhz = cold.clock, note = "cold: " * cold.failed)
    haskey(warm, :failed) && return (; side = "tempest", precision = cold.precision, cold_prep_s = cold.prep_s,
        step_ms = cold.step_ms, kernels = cold.kernels, sm_mhz = cold.clock, note = "warm: " * warm.failed)
    note = cold.kernels == warm.kernels ? "" : "cold run $(cold.kernels) kernels"
    (; side = "tempest", precision = warm.precision, cold_prep_s = cold.prep_s, warm_prep_s = warm.prep_s,
        step_ms = min(cold.step_ms, warm.step_ms), kernels = warm.kernels, agreement = warm.agreement,
        sm_mhz = warm.clock, note = join(filter(!isempty, [note, warm.note]), "; "))
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

## The worker: one preparation and its measurements, in a process of its own.

function worker(program, shape, check)
    @eval Main begin
        import cuBLAS, cuTile
        using Tempest, CUDACore
    end
    m = Core.eval(Main, :(module Program end))
    Base.include(m, joinpath(HERE, program * ".jl"))
    Base.include(m, joinpath(HERE, "measure.jl"))
    result = Base.invokelatest(() -> m.measure(m.instance(Symbol(shape)); check))
    println("corpus result ", repr(result))
end

function main(args)
    if "--worker" in args
        i = findfirst(==("--worker"), args)
        return worker(args[i + 1], args[i + 2], "--check" in args)
    end
    program = first(args)
    isfile(joinpath(HERE, program * ".jl")) && haskey(CHALLENGERS, program) || error("$program is no program of the corpus")
    shape = option(args, "--shape", "small")
    directory = record_directory()
    mkpath(joinpath(directory, "logs"))
    println("record: ", joinpath(directory, "corpus.tsv"))
    common = (; program, shape, commit = commit(), device = device_name())
    io = open_record(directory)
    r = tempest(program, shape, directory)
    row!(io, merge(common, (; when = stamp()), r))
    println(r)
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
