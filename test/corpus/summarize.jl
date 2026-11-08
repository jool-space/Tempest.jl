# The corpus's record as one table: each program and shape, each side's
# latest run, its preparation cold and warm, its step, its kernels and its
# agreement with the Float64 reference.
#
#     julia --project=test test/corpus/summarize.jl [corpus.tsv]
#
# reads corpus.tsv in Tempest's scratch space `corpus`, or in TEMPEST_CORPUS,
# by default. Tempest's row is marked where its step is more than 1.1 times
# its best challenger's, or its cold preparation past 60 s: the corpus's
# gates, each a trial to open (TRIALS.md), not a failure.
using Printf
include(joinpath(@__DIR__, "run.jl"))

const SLOWER, COLD = 1.1, 60.0

path = isempty(ARGS) ? joinpath(record_directory(), "corpus.tsv") : ARGS[1]
lines = readlines(path)
header = split(first(lines), '\t')
rows = [Dict(zip(header, split(l, '\t'))) for l in lines[2:end]]

number(r, key) = something(tryparse(Float64, get(r, key, "")), NaN)

# The latest row of each program, shape and side, in the order first run.
latest = Dict{Tuple{String, String, String}, Dict}()
order = Tuple{String, String, String}[]
for r in rows
    key = (r["program"], r["shape"], r["side"])
    key in order || push!(order, key)
    latest[key] = r
end

cell(x, digits) = isnan(x) ? "" : @sprintf("%.*f", digits, x)
exponent(x) = isnan(x) ? "" : @sprintf("%.1e", x)

@printf("%-7s %-6s %-22s %9s %9s %10s %8s %10s %6s  %s\n", "program", "shape", "side", "cold s", "warm s", "step ms", "kernels",
    "agreement", "MHz", "marks and notes")
for (program, shape) in unique(map(k -> k[1:2], order))
    sides = filter(k -> k[1:2] == (program, shape), order)
    challengers = filter(k -> k[3] != "tempest", sides)
    best = minimum(filter(!isnan, map(k -> number(latest[k], "step_ms"), challengers)); init = Inf)
    for key in sides
        r = latest[key]
        step, cold = number(r, "step_ms"), number(r, "cold_prep_s")
        marks = String[]
        if key[3] == "tempest"
            step > SLOWER * best && push!(marks, @sprintf("SLOWER %.2f× best challenger", step / best))
            cold > COLD && push!(marks, "COLD past 60 s")
        end
        isempty(r["note"]) || push!(marks, r["note"])
        @printf("%-7s %-6s %-22s %9s %9s %10s %8s %10s %6s  %s\n", program, shape, key[3], cell(cold, 1),
            cell(number(r, "warm_prep_s"), 1), cell(step, 3), r["kernels"], exponent(number(r, "agreement")), r["sm_mhz"],
            join(marks, "; "))
    end
end
