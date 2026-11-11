# The corpus's record as tables, one for each program and shape: a line for
# each scenario of Tempest's latest pass (run.jl), each cell the range of its
# runs, then each challenger's latest row, with when and where it ran.
#
# Tempest's columns: the seconds to the first result, and of them Julia's
# compilation summed over threads; the step's milliseconds; the kernels
# compiled and loaded in the preparation, and the plan's; the bytes the plan
# stores; the deciders that replayed their decisions and those the cover
# reached, as replayed/covered; the plan, a letter for each plan the table's
# rows reached, alike where two reached one; the agreement with Float64; and
# the SM clock.
#
#     julia --project=test test/corpus/summarize.jl [RECORD...]
#
# reads corpus.tsv in Tempest's scratch space `corpus`, or in TEMPEST_CORPUS,
# by default; records given are read together, Tempest's from one machine and
# challengers' from another say. A range is the least and the greatest of the
# runs, one value where they print alike. A scenario is marked where its step
# is more than 1.1 times the best challenger's on the same machine and
# device, or where its first result took past 60 s cold: the corpus's gates,
# each a trial to open (TRIALS.md), not a failure.
using Printf, Dates
include(joinpath(@__DIR__, "run.jl"))

const SLOWER, COLD = 1.1, 60.0

## Reading: rows as dictionaries of their columns' names.

# A row of the layout before scenarios: Tempest's held a cold and a warm
# preparation and the faster one's step, a challenger's its first step.
function scenario_rows(r)
    haskey(r, "scenario") && return [r]
    common = merge(r, Dict("pass" => r["when"], "host" => ""))
    r["side"] == "tempest" || return [merge(common, Dict("scenario" => "cold", "first_s" => r["cold_prep_s"]))]
    [merge(common, Dict("scenario" => "cold", "first_s" => r["cold_prep_s"], "step_ms" => "", "agreement" => "")),
        merge(common, Dict("scenario" => "warm", "first_s" => r["warm_prep_s"], "note" => "the faster of cold and warm's step; " * r["note"]))]
end

function read_record(path)
    rows = Dict{String, String}[]
    header = nothing
    for line in eachline(path)
        isempty(line) && continue
        fields = split(line, '\t')
        if is_header(line)
            header = fields
            continue
        end
        append!(rows, scenario_rows(Dict{String, String}(zip(header, fields))))
    end
    rows
end

value(r, key) = get(r, key, "")
number(r, key) = something(tryparse(Float64, value(r, key)), NaN)
when(r) = Dates.format(unix2datetime(parse(Int, r["when"])), "yyyy-mm-dd HH:MM")
host(r) = isempty(value(r, "host")) ? "a host not recorded" : r["host"]

## Printing: a cell is the range of a column over a scenario's rows.

function span(xs, format)
    xs = filter(!isnan, xs)
    isempty(xs) && return ""
    low, high = format(minimum(xs)), format(maximum(xs))
    low == high ? low : low * "–" * high
end

fixed(digits) = x -> @sprintf("%.*f", digits, x)
exponent(x) = @sprintf("%.1e", x)
bytes(x) = Base.format_bytes(round(Int, x))
unit(x, format, name) = isnan(x) ? "" : format(x) * " " * name
column(key, format) = rs -> span(map(r -> number(r, key), rs), format)
each(f) = rs -> join(unique(filter(!isempty, map(f, rs))), ",")

# The columns of a scenario's line: a title, a width (negative to the left)
# and the cell of its rows.
const TABLE = [
    ("scenario", -18, rs -> value(first(rs), "scenario")),
    ("runs", 4, rs -> string(length(rs))),
    ("first s", 11, column("first_s", fixed(1))),
    ("julia s", 11, column("julia_s", fixed(1))),
    ("step ms", 13, column("step_ms", fixed(3))),
    ("built", 9, column("built", fixed(0))),
    ("loaded", 7, column("loaded", fixed(0))),
    ("kernels", 9, column("kernels", fixed(0))),
    ("stored", 21, column("stored_bytes", bytes)),
    ("decided", 9, each(r -> isempty(value(r, "replayed")) ? "" : value(r, "replayed") * "/" * value(r, "covered"))),
    ("plan", 4, each(r -> value(r, "letter"))),
    ("agreement", 15, column("agreement", exponent)),
    ("MHz", 9, column("sm_mhz", fixed(0)))]

pad(s, width) = width < 0 ? rpad(s, -width) : lpad(s, width)
line(cells) = join(map(((_, width, _), c) -> pad(c, width), TABLE, cells), " ")

# A row's note in parts; the parts every run of a table shares are said once
# under it.
parts(r) = filter(!isempty, strip.(split(value(r, "note"), ';')))

# The best step among challengers that ran where Tempest's pass did.
function best_challenger(tempest, others)
    isempty(tempest) && return Inf
    place(r) = (value(r, "host"), value(r, "device"))
    here = filter(r -> !isempty(value(r, "host")) && place(r) == place(first(tempest)), others)
    minimum(filter(!isnan, map(r -> number(r, "step_ms"), here)); init = Inf)
end

function table(program, shape, tempest, others)
    plans = unique(filter(!isempty, map(r -> value(r, "plan"), tempest)))
    tempest = map(r -> merge(r, Dict("letter" => isempty(value(r, "plan")) ? "" : string('A' + findfirst(==(r["plan"]), plans) - 1))), tempest)
    if isempty(tempest)
        println("\n$program $shape: no pass of Tempest's")
    else
        r = first(tempest)
        println("\n$program $shape: Tempest's pass of $(when(r)) UTC at $(value(r, "commit")) on $(host(r)), $(value(r, "device"))")
        println(line(map(first, TABLE)), "  marks and notes")
    end
    best = best_challenger(tempest, others)
    shared = isempty(tempest) ? String[] : intersect(map(parts, tempest)...)
    for scenario in unique(map(r -> r["scenario"], tempest))
        rs = filter(r -> r["scenario"] == scenario, tempest)
        marks = String[]
        step = minimum(filter(!isnan, map(r -> number(r, "step_ms"), rs)); init = Inf)
        step > SLOWER * best && push!(marks, @sprintf("SLOWER %.2f× best challenger", step / best))
        scenario == "cold" && any(r -> number(r, "first_s") > COLD, rs) && push!(marks, "COLD past 60 s")
        notes = setdiff(unique(reduce(vcat, map(parts, rs))), shared)
        println(line(map(c -> c[3](rs), TABLE)), "  ", join([marks; notes], "; "))
    end
    isempty(shared) || println("  every run: ", join(shared, "; "))
    isempty(others) && return
    println("  challengers, each side's latest run:")
    for r in others
        @printf("  %-22s %-8s %11s %13s  %s UTC on %s, %s MHz  %s\n", r["side"], value(r, "scenario"), unit(number(r, "first_s"), fixed(1), "s"),
            unit(number(r, "step_ms"), fixed(3), "ms"), when(r), host(r), value(r, "sm_mhz"), value(r, "note"))
    end
end

rows = reduce(vcat, map(read_record, isempty(ARGS) ? [joinpath(record_directory(), "corpus.tsv")] : ARGS))

# Each program and shape in the order first recorded: Tempest's rows of its
# latest pass, and each challenger's latest row of each scenario.
for (program, shape) in unique(map(r -> (r["program"], r["shape"]), rows))
    mine = filter(r -> (r["program"], r["shape"]) == (program, shape), rows)
    ours = filter(r -> r["side"] == "tempest", mine)
    latest = maximum(r -> parse(Int, r["pass"]), ours; init = 0)
    tempest = filter(r -> parse(Int, r["pass"]) == latest, ours)
    theirs = filter(r -> r["side"] != "tempest", mine)
    sides = unique(map(r -> (r["side"], value(r, "scenario")), theirs))
    table(program, shape, tempest, [last(filter(r -> (r["side"], value(r, "scenario")) == s, theirs)) for s in sides])
end
