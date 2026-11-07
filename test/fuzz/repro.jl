# One case again by its seed: its source, then every stage's result.
#
#     julia --project=test test/fuzz/repro.jl SEED [--cuda] [--big] [--show] [--lock FILE]
#
# `--show` prints the source only. `--lock` runs it under `flock FILE`, as a
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
    out = check(case; cuda)
    for (k, v) in sort!(collect(out); by = first)
        println(rpad(k, 18), v)
    end
    println("failures: ", failures(out))
end
