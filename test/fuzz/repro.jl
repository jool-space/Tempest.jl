# One case again by its seed: its source, then every stage's result.
#
#     julia --project=test test/fuzz/repro.jl SEED [--cuda] [--big] [--show]
#         [--shrink] [--stage NAME] [--budget N] [--out DIR] [--lock FILE]
#
# `--show` prints the source only. `--shrink` makes the case smaller while
# `--stage` (by default the first failing stage) fails the same way, and
# writes the smallest as a script to shrunk-SEED.jl in `--out` (by default
# Tempest's scratch space `fuzz`). `--lock` runs it under `flock FILE`, as a
# case on CUDA must run.
include(joinpath(@__DIR__, "run.jl"))
relocked(@__FILE__, ARGS) && exit()
include(joinpath(@__DIR__, "gen.jl"))

seed = parse(Int, ARGS[1])
cuda, big = "--cuda" in ARGS, "--big" in ARGS
case = Gen.generate(seed; big = big || cuda)
println("# seed ", seed, ", family ", case.family, ", extents ", sort!(collect(case.extents)))
println(Gen.standalone(case))
if !("--show" in ARGS)
    include(joinpath(@__DIR__, "check.jl"))
    include(joinpath(@__DIR__, "shrink.jl"))
    out = check(case; cuda)
    for (k, v) in sort!(collect(out); by = first)
        println(rpad(k, 18), v)
    end
    println("failures: ", failures(out))
    if "--shrink" in ARGS
        stage = option(ARGS, "--stage", nothing)
        stage = stage === nothing ? nothing : Symbol(stage)
        smallest, stage, kind = shrink(case; cuda, stage, budget = parse(Int, option(ARGS, "--budget", "300")))
        dir = option(ARGS, "--out", nothing)
        dir = dir === nothing ? scratch_fuzz() : abspath(dir)
        mkpath(dir)
        path = joinpath(dir, "shrunk-$seed.jl")
        text = script(smallest, stage, kind, filter(in(("--cuda", "--big")), ARGS))
        write(path, text)
        println("\n", path, ":\n", text)
    end
end
