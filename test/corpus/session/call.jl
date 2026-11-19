# One run of the session (session/README.md), in the worker process run.jl
# starts: a program called as its user calls it, `compile` with its defaults
# and two calls, its result checked against the same function untraced.
# Included into the program's module beside its `instance`; `called` returns
# the result line's fields.
using Tempest, CUDACore, Serialization
using Tempest.Compiler: leaves

device(x::AbstractArray{<:Number}) = CuArray(x)
device(x::CuArray) = x
device(x::Union{Tuple, NamedTuple}) = map(device, x)
device(x) = x

# The inputs as the reference takes them where the program has no `untraced`
# of its own: each Float32 array widened, as measure.jl's.
wide(x::AbstractArray{Float32}) = Float64.(x)
wide(x::Union{Tuple, NamedTuple}) = map(wide, x)
wide(x) = x

# What a run compares: the result, or what the case says, as the decode's
# states it writes beside the tokens it returns.
compared_of(case, result, args) = haskey(case, :compared) ? case.compared(result, args) : result

# An output too large to hold twice on the host is compared by a digest: its
# sum, the sum of its magnitudes, of its squares, and its sum weighted by a
# hash of each position.
const WHOLE = 2^22

weight(i) = Float64((UInt32(i) * 0x9e3779b1) >> 8) / 2.0^24 - 0.5

function digest(x::AbstractArray)
    s, a, q, w = 0.0, 0.0, 0.0, 0.0
    for (i, v) in enumerate(x)
        v = Float64(v)
        s += v
        a += abs(v)
        q += v * v
        w += weight(i) * v
    end
    (s, a, q, w)
end

"An output as a reference holds it: whole in Float64, or its digest."
held(x::Number) = Float64(x)
held(x::AbstractArray{<:Number}) = length(x) <= WHOLE ? Float64.(Array(x)) : digest(Array(x))

relative(a::Float64, b::Float64) = abs(a - b) / max(abs(b), floatmin(Float64))
relative(a::AbstractArray, b::AbstractArray) = size(a) == size(b) ? maximum(abs.(a .- b); init = 0.0) / max(maximum(abs, b; init = 0.0), floatmin(Float64)) : Inf
function relative(a::NTuple{4, Float64}, b::NTuple{4, Float64})
    scale = max(b[2], floatmin(Float64))
    max(abs(a[1] - b[1]) / scale, abs(a[3] - b[3]) / max(b[3], floatmin(Float64)), abs(a[4] - b[4]) / scale)
end
relative(a, b) = Inf

"The untraced outputs, as `held`, kept at `path` for every run to compare with."
reference!(path, outputs) = serialize(path, map(held, collect(Any, leaves(outputs))))

"The largest relative error of an output against the reference, and which."
function agreement(outputs, reference)
    got = map(held, collect(Any, leaves(outputs)))
    length(got) == length(reference) || return (Inf, 0)
    errors = map(relative, got, reference)
    isempty(errors) && return (0.0, 0)
    worst = argmax(errors)
    (errors[worst], worst)
end

# A phase's seconds, as `name=seconds` joined, in the order each first ended.
phase_text(phases) = join(("$(first(p))=$(round(first(last(p)); digits = 2))" for p in phases), ";")

plan_word(ex) = first(string(hash((sprint(Tempest.Compiler.explain, ex.plan), map(string, ex.kernels))); base = 16, pad = 16), 8)

"""
    called(case; reference, started, using_s, instance_s) -> result

`compile(case.step, args...)` with its defaults but the case's target, its
arguments on the device, then two calls, each waited for, and the second's
outputs checked against the reference kept at `reference`, unless that is
`nothing`. `seconds` runs from `started`, the moment the process was
started, to the check's end; `window` is the bytes of the trace written from
the moment `compile` was called to the second call's end, what Julia
compiled to prepare the program and call it (`traced`, of compiled.jl, which
run.jl's worker loads).
"""
function called(case; reference, started, using_s, instance_s)
    C = Tempest.Compiler
    expected = reference === nothing ? nothing : deserialize(reference)
    args = map(device, case.args)
    Base.cumulative_compile_timing(true)
    compiling, kernels, decided = first(Base.cumulative_compile_time_ns()), C.cache_counts(), C.reached_counts()
    from = Main.traced()
    t = time()
    c, _, phases = C.tallied(() -> compile(case.step, args...; target = case.target))
    compile_s = time() - t
    t = time()
    c(args...)
    CUDACore.synchronize()
    result = c(args...)
    CUDACore.synchronize()
    calls_s = time() - t
    to = Main.traced()
    julia_s = 1.0e-9 * (first(Base.cumulative_compile_time_ns()) - compiling)
    t = time()
    agrees, worst = expected === nothing ? (missing, 0) : agreement(compared_of(case, result, args), expected)
    check_s = time() - t
    seconds = time() - started
    Base.cumulative_compile_timing(false)
    ex = c.executable
    built, loaded = C.cache_counts().compiled - kernels.compiled, C.cache_counts().loaded - kernels.loaded
    replayed, covered = C.reached_counts().replayed - decided.replayed, C.reached_counts().covered - decided.covered
    note = join(filter(!isempty, [get(case, :note, ""), "output $worst worst", "using $(round(using_s; digits = 1)) s",
        "instance $(round(instance_s; digits = 1)) s", "check $(round(check_s; digits = 1)) s"]), "; ")
    (; seconds, checkpoint_s = get(case, :checkpoint_s, 0.0), compile_s, calls_s, first_s = compile_s + calls_s, julia_s,
        kernels = length(ex.kernels), built, loaded, stored_bytes = C.structure(ex.plan).bytes, replayed, covered, plan = plan_word(ex),
        agreement = agrees, precision = case.precision, phases = phase_text(phases), note, window = "$from:$to")
end
