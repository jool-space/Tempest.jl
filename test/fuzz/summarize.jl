# The results of rounds: cases by family, the generator's rejections, and the
# failures by stage and verdict, each with its seeds.
#
#     julia --project=test test/fuzz/summarize.jl [DIR or FILE.log ...]
#
# reads every .log in Tempest's scratch space `fuzz` by default.
include(joinpath(@__DIR__, "run.jl"))
include(joinpath(@__DIR__, "verdict.jl"))

logs(path) = isdir(path) ? filter(f -> endswith(f, ".log"), readdir(path; join = true)) : [path]
files = isempty(ARGS) ? logs(scratch_fuzz()) : reduce(vcat, map(logs, ARGS))

cases = []
for f in files, line in eachline(f)
    startswith(line, "(") || continue
    c = eval(Meta.parse(line))
    c isa NamedTuple && haskey(c, :seed) && push!(cases, (basename(f), c))
end

where_(found) = join(("$(f):$(s)" for (f, s) in found[1:min(end, 10)]), " ")

families = Dict{Symbol, Int}()
rejected = Dict{String, Vector{Any}}()
stopped = Dict{Symbol, Vector{Any}}()
failed = Dict{Tuple{Symbol, String}, Vector{Any}}()
unchecked = Dict{Tuple{Symbol, String}, Int}()
slow = []
for (file, c) in cases
    for k in (:timeout, :crash)
        haskey(c, k) && push!(get!(stopped, k, []), (file, c.seed))
    end
    (haskey(c, :timeout) || haskey(c, :crash)) && continue
    families[c.family] = get(families, c.family, 0) + 1
    haskey(c, :generator) && (push!(get!(rejected, normalized(c.generator), []), (file, c.seed)); continue)
    for (stage, kind) in failures(c)
        push!(get!(failed, (stage, kind), []), (file, c.seed))
    end
    for (k, v) in pairs(c)
        v isa String && startswith(v, "unchecked") && (unchecked[(k, v)] = get(unchecked, (k, v), 0) + 1)
        endswith(string(k), "_s") && v isa Real && v > 60 && push!(slow, (file, c.seed, k, v))
    end
end

println(length(cases), " cases in ", length(files), " files; by family ", sort!(collect(families)))
println("\n== rejected by the generator: ", sum(length, values(rejected); init = 0))
for (k, v) in sort!(collect(rejected); by = x -> -length(x[2]))
    println(lpad(length(v), 5), "  ", first(k, 140), "\n       ", where_(v))
end
println("\n== failures by stage")
order(stage) = something(findfirst(==(stage), STAGES), 0)
for ((stage, kind), v) in sort!(collect(failed); by = x -> (order(x[1][1]), -length(x[2])))
    println(lpad(length(v), 5), "  ", rpad(stage, 16), first(kind, 140), "\n       ", where_(v))
end
println("\n== unchecked")
for ((stage, why), n) in sort!(collect(unchecked))
    println(lpad(n, 5), "  ", rpad(stage, 16), why)
end
for (k, v) in sort!(collect(stopped))
    println("\n== ", k, " ", length(v), "  ", where_(v))
end
println("\n== stages over 60 s: ", length(slow), "  ", join(("$(s[1]):$(s[2]) $(s[3])=$(s[4])" for s in slow[1:min(end, 10)]), " "))
