# Each corpus program's challengers: a launcher here that runs the
# challenger's own script, and what its printed lines mean as rows of the
# record. A challenger's command for a shape is a `Cmd`, or a string saying
# why it does not run at that shape. Included by run.jl, which loads nothing
# else, so these read JSON lines by their fields.

const CHALLENGERS = Dict{String, Vector{Any}}()

launcher(name) = joinpath(@__DIR__, name)

# A field of a flat JSON line: a number, or a string's contents.
function field(line, key)
    m = match(Regex("\"$key\": *(\"([^\"]*)\"|[-0-9.eE+]+)"), line)
    m === nothing && return nothing
    m.captures[2] === nothing ? parse(Float64, m.captures[1]) : m.captures[2]
end

json_lines(lines) = filter(l -> startswith(l, "{"), lines)

## node

# bench_node.py: a line per compile mode, its first step's seconds and its
# steady milliseconds a step, or the launcher's line for a mode that failed.
node_torch(lines) = map(json_lines(lines)) do l
    side = "torch " * field(l, "compile")
    failed = field(l, "failed")
    failed === nothing || return (; side, note = "unavailable: " * failed)
    (; side, precision = field(l, "precision"), cold_prep_s = field(l, "first_step_s"), step_ms = field(l, "step_ms"),
        note = "torch " * field(l, "torch") * ", steady over 100 steps")
end

# bench_node_port.jl: its first run, which compiles, and a second.
function node_reactant(lines)
    runs = json_lines(lines)
    length(runs) == 2 || return []
    first_run, second = runs
    [(; side = "reactant", precision = "xla " * field(first_run, "precision"), cold_prep_s = field(first_run, "first_step_s"),
        warm_prep_s = field(second, "first_step_s"), step_ms = min(field(first_run, "step_ms"), field(second, "step_ms")),
        note = "reactant " * field(first_run, "reactant") * ", steady over 100 steps, loss read each step")]
end

CHALLENGERS["node"] = [
    (; name = "torch", command = shape -> `bash $(launcher("node_torch.sh")) $shape`, parse = node_torch),
    (; name = "reactant", command = shape -> shape == "small" ? `bash $(launcher("node_reactant.sh"))` :
        "its script fixes batch 128", parse = node_reactant)]

## lstm

# torch_lstm.py: `  NAME   T ms` for each way it runs the step, best of 20
# events; the ones captured as graphs are recorded.
function lstm_torch(lines)
    header = something(findfirst(l -> startswith(l, "B="), lines), 0)
    header == 0 && return []
    compile = match(r"compile ([0-9.]+) s", lines[header])
    version = match(r"torch (\S+)", lines[header]).captures[1]
    rows = []
    for (label, side) in (("cudnn graph", "torch cudnn"), ("loop graph", "torch loop"), ("loop compiled graph", "torch compiled loop"))
        i = findfirst(l -> occursin(Regex("^  $label +[0-9]"), l), lines)
        i === nothing && continue
        ms = parse(Float64, match(r"([0-9.]+) ms", lines[i]).captures[1])
        prep = side == "torch compiled loop" ? parse(Float64, compile.captures[1]) : nothing
        push!(rows, (; side, precision = "tf32", cold_prep_s = something(prep, ""), step_ms = ms,
            note = "torch $version, a CUDA graph's replay, best of 20, no optimizer step"))
    end
    rows
end

CHALLENGERS["lstm"] = [(; name = "torch", command = shape -> `bash $(launcher("lstm_torch.sh")) $shape`, parse = lstm_torch)]

## gpt

# gpt_torch.py, and heat_cuda.jl, print one JSON line per way they run.
function json_rows(lines)
    map(json_lines(lines)) do l
        kernels = field(l, "kernels")
        (; side = field(l, "side"), precision = field(l, "precision"), cold_prep_s = field(l, "first_step_s"),
            step_ms = field(l, "step_ms"), kernels = kernels === nothing ? "" : round(Int, kernels),
            agreement = something(field(l, "agreement"), ""), note = something(field(l, "note"), ""))
    end
end

CHALLENGERS["gpt"] = [(; name = "torch", command = shape -> `bash $(launcher("gpt_torch.sh")) $shape`, parse = json_rows)]

## heat

CHALLENGERS["heat"] = [(; name = "cuda", command = shape -> `bash $(launcher("heat_cuda.sh")) $shape`, parse = json_rows)]
