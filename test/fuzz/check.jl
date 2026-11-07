# Each case checked stage by stage against eager Julia in Float64, and its
# gradient against central differences: traced and interpreted, prepared on
# the CPU, its gradient prepared with random sums riding a recurrence and
# random folds recomputed, and on CUDA in Float32, where two runs must agree
# bitwise. A stage's result is an error measure (see verdict.jl), "nonfinite"
# where the reference is finite and the result is not, "unchecked: …" where
# the reference is not finite, or "error: …".
import cuBLAS, cuTile
using Tempest, CUDACore, Random, LinearAlgebra
import Tempest as T, Tempest.Compiler as C, Tempest.Compiler.IR as IR

isdefined(@__MODULE__, :Gen) || include(joinpath(@__DIR__, "gen.jl"))
isdefined(@__MODULE__, :verdict) || include(joinpath(@__DIR__, "verdict.jl"))

const CUDA_OK = CUDACore.functional()

host(x) = x isa CuArray ? Array(x) : x
num(x) = x isa AbstractArray ? (ndims(x) == 0 ? host(x)[] : host(x)) : x

# Relative to the reference, or to `floor` where its norm is smaller: Float32
# leaves 1e-8 where a gradient cancels to zero in Float64, and Float64 1e-16.
function relerr(a, b; floor = 1.0e-6)
    a === nothing && b === nothing && return 0.0
    (a === nothing || b === nothing) && return Inf
    a, b = num(a), num(b)
    size(a) == size(b) || return Inf
    all(isfinite, b) || return "unchecked: reference not finite"
    all(isfinite, a) || return "nonfinite"
    norm(a .- b) / max(norm(b), floor)
end

function leaves_err(a, b; floor = 1.0e-6)
    length(a) == length(b) || return Inf
    errors = map((x, y) -> relerr(x, y; floor), a, b)
    any(==("nonfinite"), errors) && return "nonfinite"
    measured = filter(e -> e isa Real, errors)
    isempty(measured) ? 0.0 : maximum(measured)
end

function instantiate(case, E)
    m = Module(:FuzzCase)
    Core.eval(m, Meta.parse("begin\n" * Gen.PRELUDE * "end"))
    for g in case.globals
        Core.eval(m, Meta.parse(g.name * " = " * string(E) * ".(" * Gen.construction(g, case.seed, case.extents) * ")"))
    end
    Core.eval(m, Meta.parse(Gen.source(case, E)))
    Core.eval(m, :f)
end

oneline(e) = first(split(sprint(showerror, e), '\n'))

# The stage after which `check` returns, as the shrinker asks; none by default.
const UPTO = Ref(:none)
struct Reached <: Exception end

# Runs a stage: its error, else what it returns.
function stage!(out, name, f)
    t = time()
    out[name] = try
        Base.invokelatest(f)
    catch e
        e isa InterruptException && rethrow()
        "error: " * first(oneline(e), 300)
    end
    out[Symbol(name, :_s)] = round(time() - t; digits = 2)
    name === UPTO[] && throw(Reached())
    out[name]
end

some(rng, xs) = xs[randperm(rng, length(xs))[1:rand(rng, 1:length(xs))]]
floats(args) = map(a -> a isa AbstractArray{<:AbstractFloat}, args)
snapshot(x) = x === nothing ? nothing : copy(num(x))

function check(case; cuda = CUDA_OK)
    out = Dict{Symbol, Any}(:seed => case.seed)
    merge!(out, case.notes)
    out[:extents] = Tuple(sort!(collect(case.extents)))
    try
        stages!(out, case, cuda)
    catch e
        e isa Reached || rethrow()
    end
    out
end

function stages!(out, case, cuda)
    args = Gen.materialize(case)
    # A case the shrinker cut short may not parse, or not run.
    f = nothing
    ref = try
        f = instantiate(case, Float64)
        Base.invokelatest(f, args...)
    catch e
        out[:generator] = first(oneline(e), 200)
        return
    end
    # Values past 1e8 are a recurrence blowing up, where differences mislead.
    if !(ref isa Real && isfinite(ref) && abs(ref) < 1.0e8)
        out[:generator] = "not a finite number below 1e8"
        return
    end
    rng = Xoshiro(case.seed + 1)
    p = nothing
    stage!(out, :trace, () -> (p = C.trace(f, args...); count(n -> n.op isa C.Loop, p.nodes)))
    p === nothing && return
    out[:nodes] = length(p.nodes)
    stage!(out, :fwd_interp, () -> relerr(C.interpret(p, args...), ref))
    stage!(out, :fwd_cpu, () -> relerr(only(C.leaves(C.outputs(C.execute!(C.prepare(p, args...))))), ref))
    g = (xs...) -> T.withgradient(f, xs...)
    eager = nothing
    # Eager `withgradient` is Tempest's own differentiation, interpreted: only
    # central differences check it.
    stage!(out, :grad_eager, () -> (eager = Base.invokelatest(g, args...); 0.0))
    if eager !== nothing
        stage!(out, :grad_fd, function ()
            vs = map((a, float) -> float ? randn(rng, size(a)...) : nothing, args, floats(args))
            ε = 1.0e-6
            step(s) = map((a, v) -> v === nothing ? a : a .+ s .* v, args, vs)
            dd = (Base.invokelatest(f, step(ε)...) - Base.invokelatest(f, step(-ε)...)) / 2ε
            gv = sum(map((gr, v) -> gr === nothing || v === nothing ? 0.0 : dot(gr, v), eager.grad, vs))
            abs(dd - gv) / max(abs(dd), abs(gv), 1.0e-3)
        end)
        q = nothing
        stage!(out, :grad_trace, () -> (q = C.trace(g, args...); count(n -> n.op isa C.Loop, q.nodes)))
        if q !== nothing
            reference = C.leaves(eager)
            stage!(out, :grad_interp, () -> leaves_err(C.leaves(C.interpret(q, args...)), reference))
            planned = nothing
            stage!(out, :grad_cpu, function ()
                ex = C.prepare(q, args...)
                planned = ex.plan
                leaves_err(C.leaves(C.outputs(C.execute!(ex))), reference)
            end)
            if planned !== nothing
                sums = C.riding_sums(planned)
                out[:sums] = length(sums)
                if !isempty(sums)
                    ride = some(rng, sums)
                    stage!(out, :grad_ride, () -> leaves_err(C.leaves(C.outputs(C.execute!(C.prepare(q, args...; ride)))), reference))
                end
                folds = collect(C.recomputable_folds(planned.graph))
                out[:folds] = length(folds)
                if !isempty(folds)
                    recompute = some(rng, folds)
                    stage!(out, :grad_recompute, () -> leaves_err(C.leaves(C.outputs(C.execute!(C.prepare(q, args...; recompute)))), reference))
                end
            end
        end
    end
    cuda || return
    f32 = instantiate(case, Float32)
    a32 = map((a, float) -> float ? Float32.(a) : a, args, floats(args))
    dev = map(a -> a isa AbstractArray ? CuArray(a) : a, a32)
    p32 = nothing
    stage!(out, :cuda_trace, () -> (p32 = C.trace(f32, a32...); 0.0))
    p32 === nothing && return
    stage!(out, :cuda_fwd, () -> relerr(only(C.leaves(C.outputs(C.execute!(C.prepare(p32, dev...; measure = :none))))), ref))
    eager === nothing && return
    g32 = (xs...) -> T.withgradient(f32, xs...)
    q32 = nothing
    stage!(out, :cuda_grad_trace, () -> (q32 = C.trace(g32, a32...); 0.0))
    q32 === nothing && return
    reference = C.leaves(eager)
    measure = rand(rng) < 0.25 ? :close : :none
    out[:measure] = measure
    planned = nothing
    stage!(out, :cuda_grad, function ()
        ex = C.prepare(q32, dev...; measure)
        planned = ex.plan
        leaves_err(C.leaves(C.outputs(C.execute!(ex))), reference; floor = 1.0e-4)
    end)
    if planned !== nothing
        sums = C.riding_sums(planned)
        if !isempty(sums)
            ride = some(rng, sums)
            stage!(out, :cuda_ride, () -> leaves_err(C.leaves(C.outputs(C.execute!(C.prepare(q32, dev...; ride, measure = :none)))), reference; floor = 1.0e-4))
        end
    end
    # Two runs agree bitwise on a target that keeps sums' rounding from the
    # order blocks run in, unless the plan sums by key, which accumulates
    # atomically on any target (see `CUDATarget`).
    stage!(out, :cuda_repeat, function ()
        ex = C.prepare(q32, dev...; target = T.CUDATarget(; deterministic = true), measure = :none)
        keyed = any(v -> v.definition isa C.Keyed && !(v.definition.fold isa IR.Overwrite), ex.plan.graph.values)
        out[:keyed] = keyed
        first_run = map(snapshot, C.leaves(C.outputs(C.execute!(ex))))
        second_run = map(snapshot, C.leaves(C.outputs(C.execute!(ex))))
        keyed && return "unchecked: a sum by key"
        all(map((x, y) -> x === y || isequal(x, y), first_run, second_run)) ? 0.0 : 1.0
    end)
    nothing
end

# A worker of run.jl: the cases of its seeds, each started on stderr with the
# time, so that the driver's watchdog knows which runs and since when, and
# each one's results a line on stdout.
function work(range, cuda, big)
    cuda && !CUDA_OK && error("--cuda, but CUDA is not functional")
    for seed in range
        println(stderr, "start ", seed, " ", round(Int, time()))
        flush(stderr)
        out = check(Gen.generate(seed; big = big || cuda); cuda)
        println(repr((; sort!(collect(out); by = first)...)))
        flush(stdout)
    end
end
