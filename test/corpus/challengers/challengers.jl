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
