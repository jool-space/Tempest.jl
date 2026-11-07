# A failing case made smaller while the same stage fails the same way:
# statements removed (a loop with its body), terms of the loss removed,
# extents made smaller, and the arguments and globals nothing reads left out.

indentation(line) = length(line) - length(lstrip(line))

# What may be removed, the last lines first: each line, and each loop whole.
function removals(lines)
    out = UnitRange{Int}[]
    for (i, line) in enumerate(lines)
        push!(out, i:i)
        startswith(lstrip(line), "for ") || continue
        j = findfirst(k -> strip(lines[k]) == "end" && indentation(lines[k]) == indentation(line), (i + 1):length(lines))
        j === nothing || push!(out, i:(i + j))
    end
    reverse!(out)
end

mentions(c, name) = occursin(Regex("\\b" * name * "\\b"), join([c.lines; c.terms], "\n"))

"""
    shrink(case; cuda, stage = nothing, budget = 300) -> (case, stage, verdict)

The smallest case found, in at most `budget` checks, on which `stage` (by
default the first that fails) fails as it fails on `case`.
"""
function shrink(case; cuda, stage = nothing, budget = 300)
    fails = failures(check(case; cuda))
    isempty(fails) && error("seed $(case.seed) fails no stage")
    stage = something(stage, first(s for s in STAGES if haskey(fails, s)))
    kind = fails[stage]
    UPTO[] = stage
    tries = 0
    function persists(c)
        tries += 1
        out = check(c; cuda)
        !haskey(out, :generator) && get(failures(out), stage, nothing) == kind
    end
    best = case
    changed = true
    while changed && tries < budget
        changed = false
        for r in removals(best.lines)
            c = copy(best; lines = deleteat!(copy(best.lines), r))
            persists(c) || continue
            best, changed = c, true
            break
        end
        changed && continue
        for i in eachindex(best.terms)
            length(best.terms) > 1 || break
            c = copy(best; terms = deleteat!(copy(best.terms), i))
            persists(c) || continue
            best, changed = c, true
            break
        end
        changed && continue
        for (name, v) in sort!(collect(best.extents))
            for w in sort!(unique(filter(w -> 1 <= w < v, [1, 2, 3, v ÷ 2, v - 1])))
                c = copy(best; extents = merge(best.extents, Dict(name => w)))
                persists(c) || continue
                best, changed = c, true
                break
            end
            changed && break
        end
    end
    UPTO[] = :none
    best = copy(best; args = filter(a -> mentions(best, a.name), best.args), globals = filter(g -> mentions(best, g.name), best.globals))
    best, stage, kind
end

# How each stage runs, for the script a shrunk case is written as.
const HOW = Dict(
    :trace => "C.trace(f, args...)",
    :fwd_interp => "C.interpret(C.trace(f, args...), args...)  # against f(args...)",
    :fwd_cpu => "C.outputs(C.execute!(C.prepare(C.trace(f, args...), args...)))  # against f(args...)",
    :grad_eager => "T.withgradient(f, args...)",
    :grad_fd => "T.withgradient(f, args...)  # against central differences of f",
    :grad_trace => "C.trace((xs...) -> T.withgradient(f, xs...), args...)",
    :grad_interp => "C.interpret(C.trace((xs...) -> T.withgradient(f, xs...), args...), args...)  # against T.withgradient(f, args...)",
    :grad_cpu => "C.outputs(C.execute!(C.prepare(C.trace((xs...) -> T.withgradient(f, xs...), args...), args...)))  # against T.withgradient(f, args...)",
    :cuda_fwd => "C.outputs(C.execute!(C.prepare(C.trace(f, single...), map(CuArray, single)...)))  # single: the arguments in Float32",
    :grad_ride => "C.prepare(C.trace((xs...) -> T.withgradient(f, xs...), args...), args...; ride)  # ride: some of C.riding_sums(plan)",
    :grad_recompute => "C.prepare(C.trace((xs...) -> T.withgradient(f, xs...), args...), args...; recompute)  # some of C.recomputable_folds(plan.graph)",
    :cuda_grad => "C.outputs(C.execute!(C.prepare(C.trace((xs...) -> T.withgradient(f, xs...), single...), map(CuArray, single)...)))",
)

# The case as a script; `flags` those it was drawn with, as repro.jl takes them.
function script(case, stage, kind, flags)
    E = startswith(string(stage), "cuda") ? Float32 : Float64
    io = IOBuffer()
    println(io, "# Shrunk from seed ", case.seed, " (", case.family, "): stage ", stage, " fails, ", kind, ".")
    println(io, "# As drawn: julia --project=test test/fuzz/repro.jl ", join([case.seed; flags], " "))
    print(io, Gen.standalone(case, E))
    println(io, "# ", get(HOW, stage, "check.jl's stage $stage"))
    String(take!(io))
end
