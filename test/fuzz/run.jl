# The randomized program checker.
#
#     julia --project=test test/fuzz/run.jl --seeds LO:HI [--cuda] [--big] [--out DIR]
#         [--tag NAME] [--watch SECONDS] [--chunk N] [--lock FILE]
#
# draws a program for each seed from LO to HI (gen.jl: maps, folds, scans,
# layouts, gathers, writes, products and loops, as source) and checks it
# stage by stage (check.jl): eager Julia in Float64 is the reference; the
# program traced, interpreted and prepared on the CPU; its gradient against
# central differences, interpreted, prepared, and prepared again with random
# sums riding a recurrence and random folds recomputed; with `--cuda` on the
# device in Float32, forward and gradient, and two runs compared bitwise.
# `--big` (and `--cuda`) draws extents around the tiles' multiples (31, 32,
# 33, 127, 128, 129) besides small ones and ones.
#
# Workers check chunks of `--chunk` seeds (25) in turn; a case that runs past
# `--watch` seconds (600) is killed and logged as a timeout, one that crashes
# its worker as a crash, and the next worker starts after it. With `--cuda`
# each worker holds `--lock` (or ENV["TEMPEST_GPU_LOCK"]) by `flock` while it
# runs. Each case is one line of TAG.log in `--out`, by default Tempest's
# scratch space `fuzz` (~/.julia/scratchspaces/<Tempest's uuid>/fuzz), never
# the repository; TAG defaults to cpu-LO-HI or cuda-LO-HI. A round loads the
# Tempest of the project it runs in, so run long rounds from a worktree of
# their own that nothing edits meanwhile.
#
# Reading results:
#
#     julia --project=test test/fuzz/summarize.jl [DIR or FILES]
#
# prints the cases, the generator's rejections, and the failures by stage:
# mismatches past the stage's tolerance (verdict.jl), results not finite
# where the reference is, errors by message, timeouts and crashes, each with
# its seeds. One case again, its source and every stage:
#
#     julia --project=test test/fuzz/repro.jl SEED [--cuda] [--big] [--show] [--lock FILE]
#         [--shrink] [--stage NAME]
#
# and with `--shrink` its statements removed and extents made smaller while
# the same stage fails the same way, the smallest case written as a script to
# shrunk-SEED.jl in `--out`. Faults found and not yet closed are in found.jl.
# runtests.jl skips this directory.

const HERE = @__DIR__
const PROJECT = dirname(HERE)

function option(args, name, default)
    i = findfirst(==(name), args)
    i === nothing ? default : args[i + 1]
end

function seeds(text)
    lo, hi = split(text, ':')
    parse(Int, lo):parse(Int, hi)
end

# Tempest's scratch space `fuzz`. Scratch is Tempest's dependency rather than
# the tests', so it is loaded by its identity.
function scratch_fuzz()
    scratch = Base.require(Base.PkgId(Base.UUID("6c6a2e73-6563-6170-7368-637461726353"), "Scratch"))
    Base.invokelatest(scratch.get_scratch!, Base.UUID("aba8ef0c-1387-4c24-84ad-a51c8dc32f50"), "fuzz")
end

# Runs `script` again under `flock` when `--lock FILE` is given, as every job
# on the GPU must; true when it did.
function relocked(script, args)
    lock = option(args, "--lock", nothing)
    (lock === nothing || haskey(ENV, "TEMPEST_FUZZ_LOCKED")) && return false
    i = findfirst(==("--lock"), args)
    rest = args[[1:(i - 1); (i + 2):end]]
    run(addenv(`flock $lock $(Base.julia_cmd()) --project=$PROJECT $script $rest`, "TEMPEST_FUZZ_LOCKED" => "1"))
    true
end

# The last case a worker started since line `base` of its stderr: its seed
# and when.
function last_start(err, base)
    isfile(err) || return nothing
    lines = filter(l -> startswith(l, "start "), collect(Iterators.drop(eachline(err), base)))
    isempty(lines) && return nothing
    _, seed, t = split(last(lines))
    (parse(Int, seed), parse(Int, t))
end

logged(log, seed) = isfile(log) && any(l -> occursin("seed = $seed,", l) || occursin("seed = $seed)", l), eachline(log))

# The worker and what it started: with a lock, `flock` runs julia as its child.
function stop(process)
    pid = getpid(process)
    children = readlines(ignorestatus(`pgrep -P $pid`))
    foreach(c -> run(ignorestatus(`kill $c`)), children)
    kill(process)
    sleep(2)
end

function drive(range, cuda, big, out, tag, watch, chunk, lock)
    mkpath(out)
    log, err = joinpath(out, tag * ".log"), joinpath(out, tag * ".err")
    flags = [cuda ? ["--cuda"] : String[]; big ? ["--big"] : String[]]
    println("results in ", log)
    s = first(range)
    while s <= last(range)
        e = min(s + chunk - 1, last(range))
        base = isfile(err) ? countlines(err) : 0
        cmd = `$(Base.julia_cmd()) --project=$PROJECT $(joinpath(HERE, "run.jl")) --worker $s:$e $flags`
        lock === nothing || (cmd = `flock $lock $cmd`)
        process = run(pipeline(cmd; stdout = log, stderr = err, append = true); wait = false)
        killed = false
        while process_running(process)
            sleep(5)
            started = last_start(err, base)
            if started !== nothing && time() - started[2] > watch
                stop(process)
                killed = true
            end
        end
        started = last_start(err, base)
        # A worker that started nothing failed as it loaded: its stderr says why.
        started === nothing && error("worker for $s:$e started no case; see $err")
        seed = started[1]
        logged(log, seed) || open(io -> println(io, "(seed = $seed, ", killed ? "timeout" : "crash", " = true)"), log, "a")
        println("checked to ", seed)
        s = seed + 1
    end
    open(io -> println(io, "done ", tag), err, "a")
end

function main(args)
    range = seeds(option(args, "--worker", option(args, "--seeds", nothing)))
    cuda, big = "--cuda" in args, "--big" in args
    if "--worker" in args
        Base.include(Main, joinpath(HERE, "check.jl"))
        return Base.invokelatest(() -> Main.work(range, cuda, big))
    end
    out = option(args, "--out", nothing)
    out = out === nothing ? scratch_fuzz() : abspath(out)
    tag = option(args, "--tag", (cuda ? "cuda-" : "cpu-") * "$(first(range))-$(last(range))")
    watch = parse(Int, option(args, "--watch", "600"))
    chunk = parse(Int, option(args, "--chunk", "25"))
    lock = cuda ? option(args, "--lock", get(ENV, "TEMPEST_GPU_LOCK", nothing)) : nothing
    drive(range, cuda, big, out, tag, watch, chunk, lock)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
