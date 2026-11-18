# An unseen program of the session: a case of the fuzzer's generator
# (test/fuzz/gen.jl) at its largest (`big`), on the device in Float32, taken
# as it is or with its gradient. The file that includes this one names it:
# `SEED`, and `GRADIENT`, whether the program is the case's gradient
# (`Tempest.withgradient` of its function with respect to every argument)
# rather than the case's function. A loop's steps are LONG at the small
# shape; the doubled shape has twice a loop's steps, or twice the extent
# most of the arguments' dimensions name. select.jl draws them.
include(joinpath(@__DIR__, "..", "..", "fuzz", "gen.jl"))
using Random, LinearAlgebra
import Tempest
using CUDACore: CuArray

const LONG = 64

# The extent most of the arguments' dimensions name, the first by name of
# those that tie.
function busiest(c)
    names = String[]
    for a in [c.args; c.globals], d in a.dims
        collect_names!(names, Meta.parse(d))
    end
    counts = Dict(n => count(==(n), names) for n in unique(names))
    first(sort!(collect(keys(counts)); by = n -> (-counts[n], n)))
end
collect_names!(names, x::Symbol) = push!(names, string(x))
collect_names!(names, x::Expr) = foreach(a -> collect_names!(names, a), x.args[2:end])
collect_names!(names, x) = names

function case_of(seed, shape)
    c = Gen.generate(seed; big = true)
    extents = copy(c.extents)
    if c.family === :loop
        extents["T"] = shape === :doubled ? 2LONG : LONG
    elseif shape === :doubled
        extents[busiest(c)] *= 2
    end
    copy(c; extents)
end

# The case's function in element type `E`, in a module of its own, the
# arrays the generator makes globals bound to locals it captures, as
# `compile` asks of a function that holds arrays, each placed by `place` (the
# device's array type, or nothing for the host's).
function instantiate(c, E, place = nothing)
    m = Module(:Unseen)
    Core.eval(m, Meta.parse("begin\n" * Gen.PRELUDE * "end"))
    Core.eval(m, :(const place = $(something(place, identity))))
    bound = join((g.name * " = place(" * string(E) * ".(" * Gen.construction(g, c.seed, c.extents) * "))" for g in c.globals), ", ")
    Core.eval(m, Meta.parse("f = let " * bound * "\n" * Gen.source(c, E) * "\nf\nend"))
    Core.eval(m, :f)
end

narrow(x::AbstractArray{Float64}) = Float32.(x)
narrow(x) = x

program(f, gradient) = gradient ? (xs...) -> Tempest.withgradient(f, xs...) : f

function instance(shape)
    c = case_of(SEED, shape)
    f = instantiate(c, Float32, CuArray)
    (; step = program(f, GRADIENT), args = Tuple(map(narrow, Gen.materialize(c))), target = Tempest.CUDATarget(), precision = "fp32",
        note = "seed $SEED, $(c.family)$(GRADIENT ? ", gradient" : ""), extents $(join(("$k $v" for (k, v) in sort!(collect(c.extents))), ", "))")
end

# The same function untraced, in Float64 on the host.
function untraced(shape)
    c = case_of(SEED, shape)
    Base.invokelatest(program(instantiate(c, Float64), GRADIENT), Gen.materialize(c)...)
end
