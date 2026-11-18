# The session's numbers (README.md): each program's seconds, the sum over its
# three runs of each run's seconds less those reading a checkpoint, and with a
# second session the factors, each program's the first's seconds over the
# second's, a set's their geometric mean; then the seconds by phase of each.
#
#     julia factors.jl BASELINE [CANDIDATE]
#
# BASELINE and CANDIDATE are sessions' OUT directories.
const SESSION = ("first", "kept", "doubled")
const SETS = ("development" => ["gpt", "lstm", "qwen-gradient"], "held out" => ["heat", "node", "qwen-decode"])

function rows(directory)
    header, out = String[], Dict{String, String}[]
    for line in readlines(joinpath(directory, "corpus.tsv"))
        fields = split(line, '\t')
        first(fields) == "when" && (header = String.(fields); continue)
        r = Dict(zip(header, String.(fields)))
        get(r, "scenario", "") in SESSION && push!(out, r)
    end
    out
end

number(x) = (v = tryparse(Float64, x); v === nothing ? NaN : v)

"Each program's runs, by scenario, and its seconds."
function programs(directory)
    byname = Dict{String, Dict{String, Dict{String, String}}}()
    for r in rows(directory)
        get!(byname, r["program"], Dict{String, Dict{String, String}}())[r["scenario"]] = r
    end
    byname
end

seconds(runs) = all(s -> haskey(runs, s) && !isnan(number(runs[s]["seconds"])), SESSION) ?
    sum(s -> number(runs[s]["seconds"]) - number(runs[s]["checkpoint_s"]), SESSION) : NaN

geomean(xs) = isempty(xs) ? NaN : exp(sum(log, xs) / length(xs))

set_of(name) = startswith(name, "unseen-") ? "unseen" : something(findfirst(s -> name in last(s), collect(SETS)) |> i -> i === nothing ? nothing : first(SETS[i]), "other")

function session_line(directory)
    path = joinpath(directory, "session.tsv")
    isfile(path) ? last(readlines(path)) : ""
end

function report(base, cand = nothing)
    a = programs(base)
    b = cand === nothing ? nothing : programs(cand)
    println("baseline ", base, ": ", session_line(base))
    cand === nothing || println("candidate ", cand, ": ", session_line(cand))
    names = sort!(collect(keys(a)); by = n -> (set_of(n), n))
    factors = Dict{String, Vector{Float64}}()
    println(rpad("program", 24), rpad("set", 13), join(rpad.(["first", "kept", "doubled", "seconds"], 9)), cand === nothing ? "" : join(rpad.(["now", "factor", "agree"], 10)))
    for n in names
        runs = a[n]
        line = rpad(n, 24) * rpad(set_of(n), 13) * join(rpad.([haskey(runs, s) ? string(round(number(runs[s]["seconds"]) - number(runs[s]["checkpoint_s"]); digits = 1)) : "-" for s in SESSION], 9))
        line *= rpad(string(round(seconds(runs); digits = 1)), 9)
        if b !== nothing && haskey(b, n)
            now = seconds(b[n])
            f = seconds(runs) / now
            agree = join((get(b[n], s, Dict("agreement" => "-"))["agreement"] for s in SESSION), "/")
            isnan(f) || push!(get!(factors, set_of(n), Float64[]), f)
            line *= rpad(string(round(now; digits = 1)), 10) * rpad(string(round(f; digits = 2)), 10) * agree
        end
        println(line)
    end
    for (set, fs) in sort!(collect(factors))
        println("factor of ", set, ": ", round(geomean(fs); digits = 2), " over ", length(fs), " programs; slowest ", round(minimum(fs); digits = 2))
    end
    println()
    println("seconds by phase, summed over a program's three runs (", cand === nothing ? base : cand, "):")
    c = b === nothing ? a : b
    for n in sort!(collect(keys(c)); by = n -> (set_of(n), n))
        totals = Dict{String, Float64}()
        order = String[]
        for s in SESSION
            haskey(c[n], s) || continue
            r = c[n][s]
            for p in split(get(r, "phases", ""), ';'; keepempty = false)
                k, v = split(p, '=')
                k in order || push!(order, k)
                totals[k] = get(totals, k, 0.0) + number(v)
            end
            for k in ("compile_s", "calls_s", "julia_s", "checkpoint_s")
                key = "($k)"
                key in order || push!(order, key)
                totals[key] = get(totals, key, 0.0) + number(get(r, k, "NaN"))
            end
        end
        println(rpad(n, 24), join(("$k $(round(totals[k]; digits = 1))" for k in order), ", "))
    end
end

report(ARGS...)
