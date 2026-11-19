# What Julia compiled in the session's processes (README.md, "What
# compiles"), read from the trace each process writes as it compiles
# (`--trace-compile`, each signature with its milliseconds). run.jl counts
# each run's from it; run as a script, this file reports a session's counts,
# and with a second session each count's baseline over its candidate.
#
#     julia compiled.jl BASELINE [CANDIDATE]
#     julia compiled.jl --census TRACE [FROM:TO]
#
# BASELINE and CANDIDATE are sessions' OUT directories. `--census` lists what
# a trace's window compiled that names none of the program's code: by
# function, with its copies, their milliseconds and the kinds of type its
# copies differ by.

"""
A signature Julia compiled, as the trace writes it: the milliseconds it took,
the types it was compiled for, and whether it is code a package's image held
that loading other packages threw away (`# recompile`).
"""
struct Compiled
    ms::Float64
    types::String
    again::Bool
end

const LINE = r"^#=\s*([0-9.]+) ms =# precompile\((.*)\)( # recompile)?$"

"""
How far this process's trace of what Julia compiles has been written, in
bytes, each signature written and flushed as it is compiled; nothing without
a trace.
"""
function traced()
    trace = Base.JLOptions().trace_compile
    trace == C_NULL ? 0 : filesize(unsafe_string(trace))
end

"The signatures a trace holds whose lines begin in the bytes `from` to `to`."
function compiled(trace, from = 0, to = filesize(trace))
    out = Compiled[]
    open(trace) do io
        seek(io, from)
        while !eof(io) && position(io) < to
            m = match(LINE, readline(io))
            m === nothing || push!(out, Compiled(parse(Float64, m[1]), m[2], m[3] !== nothing))
        end
    end
    out
end

# A module's name in a type's text: an identifier, or `var"…"`, followed by a
# dot and not itself following one, so the first of a qualified name's.
const ROOT = r"(?<![\p{L}\p{N}_!.#\"])(var\"[^\"]*\"|[\p{L}_][\p{L}\p{N}_!]*)\."

unquoted(name) = startswith(name, "var\"") ? name[5:(end - 1)] : name

"The modules a signature's types are named under, each by its outermost."
roots(types) = unique!([unquoted(m[1]) for m in eachmatch(ROOT, types)])

"""
Whether a signature names the program's own code: any module its types are
named under that is not one of `foreign`, the packages the process loaded
but the program did not (run.jl's worker writes them beside the trace).
Main, which holds the program's file and the session's own code, is never
foreign, nor a module made at run time with no package of its own.
"""
names_program(c::Compiled, foreign) = any(r -> !(r in foreign), roots(c.types))

"What a process's trace says beside it: where it stood once the packages were loaded, and the packages foreign to the programs."
function beside(trace)
    lines = split.(readlines(trace * ".modules"), '\t')
    (; loaded = parse(Int, only(l[2] for l in lines if l[1] == "loaded")), foreign = Set(String(l[2]) for l in lines if l[1] == "foreign"))
end

"The signatures of a trace's window `from:to` (the whole trace for none) that name none of the program's code."
function counted(trace, window = nothing)
    from, to = window === nothing ? (0, filesize(trace)) : parse.(Int, split(window, ':'))
    foreign = beside(trace).foreign
    filter(c -> !names_program(c, foreign), compiled(trace, from, to))
end

# The types between a tuple type's outer braces, split where they are joined.
function parameters(types)
    inner = types[(findfirst('{', types) + 1):prevind(types, lastindex(types))]
    out, depth, quoted, start = String[], 0, false, 1
    for (i, ch) in pairs(inner)
        if ch == '"'
            quoted = !quoted
        elseif !quoted && ch in "{("
            depth += 1
        elseif !quoted && ch in "})"
            depth -= 1
        elseif !quoted && depth == 0 && ch == ','
            push!(out, strip(inner[start:prevind(inner, i)]))
            start = nextind(inner, i)
        end
    end
    push!(out, strip(inner[start:end]))
    out
end

# A type's name without its parameters.
function bare(t)
    quoted = false
    for (i, ch) in pairs(t)
        ch == '"' && (quoted = !quoted)
        !quoted && ch == '{' && return t[1:prevind(t, i)]
    end
    t
end

"""
The function a signature compiles a body of: the function its first type is
the type of, the keyword function's where that is `Core.kwcall`, the type a
constructor makes, or a callable type's name, each closure a function of its
own; parameters left out.
"""
function callee(types)
    params = parameters(types)
    f = first(params)
    f == "typeof(Core.kwcall)" && length(params) >= 3 && (f = params[3])
    m = match(r"^typeof\((.*)\)$", f)
    m === nothing || return m[1]
    m = match(r"^Type\{(.*)\}$", f)
    m === nothing || return "Type{" * bare(m[1]) * "}"
    bare(f)
end

"The signatures over the distinct functions among them."
copies(cs) = isempty(cs) ? NaN : length(cs) / length(unique(callee(c.types) for c in cs))

## Census: what still compiles, by function and by what splits it.

# The kinds of type a parameter's text holds.
const KINDS = ("tuple" => r"(?<![\p{L}])(Tuple|NTuple)\{", "named tuple" => r"NamedTuple\{", "closure" => r"var\"#",
    "tracer" => r"Traced(Array)?\{", "value" => r"Val\{", "function" => r"typeof\(")

kinds_of(text) = [k for (k, r) in KINDS if occursin(r, text)]

"""
What splits a function's copies: for each argument position where the copies
differ, the kinds of type found there, and `arity` where the copies differ in
how many arguments they take.
"""
function splits(types)
    params = map(t -> parameters(t)[2:end], types)
    out = String[]
    length(unique(length.(params))) > 1 && push!(out, "arity")
    for i in 1:minimum(length.(params))
        column = unique(p[i] for p in params)
        length(column) > 1 || continue
        found = unique(reduce(vcat, map(kinds_of, column)))
        append!(out, isempty(found) ? ["other"] : found)
    end
    unique!(out)
end

function census(cs; io = stdout, top = 60)
    by = Dict{String, Vector{Compiled}}()
    foreach(c -> push!(get!(by, callee(c.types), Compiled[]), c), cs)
    total = sum(c -> c.ms, cs; init = 0.0)
    println(io, length(cs), " signatures, ", length(by), " functions, ", round(copies(cs); digits = 2), " copies each, ",
        round(total / 1000; digits = 1), " s compiling")
    kinds = Dict{String, Vector{Int}}()
    for (f, group) in by
        length(group) > 1 || continue
        for k in splits(map(c -> c.types, group))
            v = get!(kinds, k, [0, 0])
            v[1] += 1
            v[2] += length(group)
        end
    end
    println(io, "the kinds of type copies differ by (functions, their copies):")
    for (k, (n, m)) in sort!(collect(kinds); by = kv -> -kv[2][2])
        println(io, "    ", rpad(k, 14), n, " functions, ", m, " copies")
    end
    println(io, "by function, the most copies first:")
    for (f, group) in first(sort!(collect(by); by = kv -> (-length(kv[2]), kv[1])), top)
        ms = sum(c -> c.ms, group)
        println(io, lpad(length(group), 6), lpad(round(Int, ms), 8), " ms  ", f, "  ", join(length(group) > 1 ? splits(map(c -> c.types, group)) : String[], ", "))
    end
end

## The session's counts.

const DEVELOPMENT = ["gpt", "lstm", "qwen-gradient"]
const HELD_OUT = ["heat", "node", "qwen-decode"]

function record_rows(directory)
    header, out = String[], Dict{String, String}[]
    for line in readlines(joinpath(directory, "corpus.tsv"))
        fields = split(line, '\t')
        first(fields) == "when" && (header = String.(fields); continue)
        push!(out, Dict(zip(header, String.(fields))))
    end
    out
end

count_of(r) = something(tryparse(Int, get(r, "compiled", "")), missing)

median_of(xs) = isempty(xs) ? missing : (s = sort(xs); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)

unseen(name) = startswith(name, "unseen-")

"""
A session's counts: each program's of its `kept` run (a new process), and of
its `second` (after the corpus's other programs, in one process), by name;
the sets' counts, a named set's the sum over its programs and the unseen
set's the median; and the copies per function of the 4B gradient's `first`
run.
"""
function counts(directory)
    rows = filter(r -> get(r, "side", "") == "tempest", record_rows(directory))
    of(scenario) = Dict(r["program"] => count_of(r) for r in rows if r["scenario"] == scenario)
    kept, second = of("kept"), of("second")
    named(set) = all(p -> haskey(kept, p), set) ? sum(p -> kept[p], set) : missing
    unseen_kept = [v for (p, v) in kept if unseen(p) && v !== missing]
    unseen_second = [v for (p, v) in second if unseen(p) && v !== missing]
    gradient = [r for r in rows if r["program"] == "qwen-gradient" && r["scenario"] == "first"]
    per_function = isempty(gradient) ? missing : copies(counted(joinpath(directory, only(gradient)["trace"]), only(gradient)["window"]))
    (; kept, second, second_program = median_of(unseen_second), new_process = (development = named(DEVELOPMENT),
        held_out = named(HELD_OUT), unseen = median_of(unseen_kept)), per_function, unseen_programs = length(unseen_second))
end

shown(x) = x === missing ? "-" : x isa AbstractFloat ? string(round(x; digits = 2)) : string(x)
ratio(a, b) = a === missing || b === missing ? "-" : string(round(a / b; digits = 2))

function report(base, cand = nothing)
    a = counts(base)
    b = cand === nothing ? nothing : counts(cand)
    line(name, f) = println(rpad(name, 34), rpad(shown(f(a)), 10), b === nothing ? "" : rpad(shown(f(b)), 10) * ratio(f(a), f(b)))
    println(rpad("", 34), rpad("baseline", 10), b === nothing ? "" : rpad("now", 10) * "baseline/now")
    line("second program (median of unseen)", c -> c.second_program)
    line("new process, development", c -> c.new_process.development)
    line("new process, held out", c -> c.new_process.held_out)
    line("new process, unseen (median)", c -> c.new_process.unseen)
    line("copies per function (4B, first)", c -> c.per_function)
    println()
    println("each program's count, a new process (kept) and after the others (second):")
    for p in sort!(unique([collect(keys(a.kept)); collect(keys(a.second))]); by = p -> (unseen(p), p))
        k(c) = get(c.kept, p, missing)
        s(c) = get(c.second, p, missing)
        println("    ", rpad(p, 30), rpad(shown(k(a)), 8), rpad(shown(s(a)), 8),
            b === nothing ? "" : "now " * rpad(shown(k(b)), 8) * shown(s(b)))
    end
    if b !== nothing
        for p in sort!(unique([collect(keys(b.kept)); collect(keys(b.second))]))
            unseen(p) && !haskey(a.kept, p) && println("    ", rpad(p, 30), rpad("", 16), "now ", rpad(shown(get(b.kept, p, missing)), 8), shown(get(b.second, p, missing)))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    if first(ARGS) == "--census"
        census(counted(ARGS[2], length(ARGS) > 2 ? ARGS[3] : nothing))
    else
        report(ARGS...)
    end
end
