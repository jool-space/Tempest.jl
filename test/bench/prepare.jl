# Wall-clock time of preparing each guard-rail program, measuring close calls,
# and of its first execution, in one process, each preparation covering
# (TEMPEST_REPLAY=false) rather than replaying what an earlier run kept. `--stores DIR` keeps what a
# run leaves on disk in DIR, the kernels compiled and the record measured: run
# it twice with the same new DIR for a cold run and then a warm one. Without
# it, a directory of the run's own, removed after. `--save path` writes the
# results, `--compare path` checks them bit for bit against a saved run.
#   julia --project=test test/bench/prepare.jl [--stores DIR] [--save|--compare path]
include("guard_rails.jl")
include(joinpath(@__DIR__, "..", "fixtures", "stores.jl"))
let i = findfirst(==("--stores"), ARGS)
    println("stores in ", i === nothing ? run_stores!() : run_stores!(ARGS[i + 1]))
    i === nothing || deleteat!(ARGS, i:(i + 1))
end
ENV["TEMPEST_REPLAY"] = "false"
using Serialization
module Ops
using Tempest
include(joinpath(@__DIR__, "..", "fixtures", "operations.jl"))
end

D, H, S, B = 64, 12, 256, 16
q, k, v = guard_data(D, H, S, B; scale = 0.001f0), guard_data(D, H, S, B; scale = 0.002f0), guard_data(D, H, S, B; scale = 0.003f0)
r = guard_data(D, H, S, B; scale = 0.5f0)
attention_loss(q, k, v, r) = sum(abs2, Ops.attention(q, k, v; causal = true) .+ r)

const PROGRAMS = [
    [(name, false, f, args, TARGET) for (name, f, args) in CASES];
    [(name, true, f, args, TARGET) for (name, f, args) in TRAINING];
    [("attention training step (TF32)", true, attention_loss, (q, k, v, r), T.CUDATarget(precision = :tf32))]]

function prepare_times()
    results = Any[]
    totals = zeros(4)
    @printf("%-44s %8s %8s %8s %8s %7s %8s\n", "case", "trace", "plan", "build", "execute", "loaded", "compiled")
    for (name, training, f, args, target) in PROGRAMS
        counts = C.cache_counts()
        t0 = time()
        p = training ? differentiate(trace(f, args...)) : trace(f, args...)
        t1 = time()
        C.plan(C.lower(p))
        t2 = time()
        ex = prepare(p, map(CuArray, args)...; target, measure = :close)
        t3 = time()
        got = map(host, C.leaves(outputs(execute!(ex))))
        CUDACore.synchronize()
        t4 = time()
        # Preparing plans again; the build is the rest.
        times = (t1 - t0, t2 - t1, (t3 - t2) - (t2 - t1), t4 - t3)
        totals .+= times
        @printf("%-44s %8.2f %8.2f %8.2f %8.2f %7d %8d\n", name, times...,
            C.cache_counts().loaded - counts.loaded, C.cache_counts().compiled - counts.compiled)
        push!(results, name => got)
    end
    @printf("%-44s %8.2f %8.2f %8.2f %8.2f\n", "total", totals...)
    results
end

started = time()
results = prepare_times()
@printf("all cases: %.1f s\n", time() - started)
if length(ARGS) >= 2 && ARGS[1] == "--save"
    serialize(ARGS[2], results)
elseif length(ARGS) >= 2 && ARGS[1] == "--compare"
    saved = deserialize(ARGS[2])
    same = [isequal(a, b) for ((_, a), (_, b)) in zip(saved, results)]
    println(all(same) ? "results identical to the saved run" :
        "results differ: " * join(first.(results[.!same]), ", "))
end
