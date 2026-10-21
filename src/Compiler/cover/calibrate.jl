# Measuring on the device: its rates, each package's line for products of a
# class and its memory, and the seconds per position of each scalar function
# in each regime, fitted on first need to reference programs timed as claims
# are, and kept in the rates file.

# Whether the rates file holds every rate of a package's line for a class: a
# file from before a line had its split and recompute rates is fitted again.
fitted_line(d, package, class) =
    all(what -> haskey(d.calibration, string(package, " ", class, " ", what)), ("latency", "rate", "split", "recompute"))

"""
The line of `package` for products of `T` factors on the decider's target,
fitted on first need and kept in the rates file. A package that claims no
reference product borrows the class's fastest line.
"""
function line!(d::Decider, package, T)
    class = product_class(d.target, T)
    get!(d.lines, (package, class)) do
        measures(d) || return Line(0.0, 1.0e10, 1.0e10, 1.0e10, false)
        key(p, what) = string(p, " ", class, " ", what)
        own(p) = Line((d.calibration[key(p, what)] for what in ("latency", "rate", "split", "recompute"))..., false)
        fitted_line(d, package, class) || fit!(d, T)
        fitted_line(d, package, class) && return own(package)
        fastest = argmax(l -> l.rate, [own(nameof(m)) for m in used_packages(d.target) if fitted_line(d, nameof(m), class)])
        Line(fastest.latency, fastest.rate, fastest.split, fastest.recompute, true)
    end
end

# The scalar functions whose seconds per position are measured: the
# primitives programs apply elementwise, by name in the rates file.
const APPLIED = (
    ("exp", exp), ("log", log), ("tanh", tanh), ("sqrt", sqrt), ("sin", sin), ("cos", cos), ("abs", abs), ("abs2", abs2),
    ("inv", inv), ("*", *), ("+", +), ("-", -), ("/", /), ("^", ^), ("max", max), ("min", min),
    ("identity", identity), ("Float32", Cast{Float32}()), ("BFloat16", Cast{BFloat16}()),
)

# The regimes a function is applied in, each priced by its own measurement:
# streamed, in a kernel that reads and writes its tiles once; swept, in a
# product's sweep over its factors' tiles; inner, in the sweep of a product
# that recomputes a fold, over the fold's tiles.
const REGIMES = (:streamed, :swept, :inner)

"""
The seconds per position of a scalar function `f` applied in `regime` in a
kernel whose products are of `class`, measured on first need and kept in the
rates file. Streamed: the time of a fold over eight applications of the
function beyond that of the fold alone, per application, the same for every
class. Swept: the time a product's factor computed by the function adds to
the product. Inner: the same for a product that recomputes the fold the
function is applied to. A function priced where other work hides it measures
as nothing, which is what it costs there. A function the table does not name
costs what addition does.
"""
function price!(d::Decider, regime, class, f)
    regime === :streamed && (class = :any)
    get!(d.applied, (regime, class, f)) do
        i = findfirst(((_, g),) -> g == f, APPLIED)
        i === nothing && return price!(d, regime, class, +)
        measures(d) || return 0.0
        get!(d.calibration, join(("applied", regime, class, APPLIED[i][1]), " ")) do
            fit_applied!(d, regime, class, f)
        end
    end
end

# Every function priced in every regime, for the classes `types` make.
function applied!(d::Decider, types = (Float32,))
    for (_, f) in APPLIED, regime in REGIMES, T in types
        price!(d, regime, product_class(d.target, T), f)
    end
    d.applied
end

const APPLIED_SHAPE = (4096, 512)
const APPLIED_TIMES = 8
const SWEEP_SHAPES = ((256, 512), (512, 2048))

function fit_applied!(d::Decider, regime, class, f)
    packages = used_packages(d.target)
    key = join(("applied", regime, class, APPLIED[findfirst(((_, g),) -> g == f, APPLIED)][1]), " ")
    if regime === :streamed
        x = zeros(Float32, APPLIED_SHAPE)
        chain(x) = foldl((y, _) -> apply_chain(f, y), 1:APPLIED_TIMES; init = x)
        fastest(r) = minimum(first.(values(r)); init = Inf)
        base = fastest(reference(d, packages, x -> sum(x; dims = 1), x))
        timed = fastest(reference(d, packages, x -> sum(chain(x); dims = 1), x))
        return keep!(d.calibration, d.calibration_path, key, max(0.0, timed - base) / (APPLIED_TIMES * length(x)))
    end
    # What the function adds to the product of the package that computes it,
    # against that package's product without it; the package it costs least.
    T = class in (:Float16, :BFloat16) ? getfield(@__MODULE__, class) : Float32
    if regime === :swept
        a, b = zeros(T, SWEEP_SHAPES[1]), zeros(T, SWEEP_SHAPES[2])
        base = reference(d, packages, (a, b) -> a * b, a, b)
        r = reference(d, packages, (a, b) -> a * apply_chain(f, b), a, b)
    else
        a, b, c = map(dims -> zeros(T, dims), RECOMPUTED_SHAPES)
        base = reference(d, packages, (a, b, c) -> (a * b) * c, a, b, c; recomputing = true)
        r = reference(d, packages, (a, b, c) -> apply_chain(f, a * b) * c, a, b, c; recomputing = true)
    end
    both = [p for p in keys(r) if haskey(base, p)]
    isempty(both) && return keep!(d.calibration, d.calibration_path, key, 0.0)
    package = argmin(p -> first(r[p]) - first(base[p]), both)
    n = sum(n for w in last(r[package]) for (g, n) in getfield(w, regime) if g == f; init = 0.0)
    keep!(d.calibration, d.calibration_path, key, n > 0 ? max(0.0, first(r[package]) - first(base[package])) / n : 0.0)
end

# One application of `f` to an array, with a literal second operand where it
# takes one, so that the chain stays one term of one kernel.
apply_chain(f, y) = f.(y)
apply_chain(f::Union{typeof(*), typeof(+), typeof(-), typeof(/), typeof(max), typeof(min)}, y) = f.(y, 1.5f0)
apply_chain(::typeof(^), y) = y .^ 3
apply_chain(c::Cast{T}, y) where {T} = Float32.(c.(y)) .+ 1.5f0

"""
The memory of `package` on the decider's target, fitted on first need and
kept in the rates file; the device's rates for a package that claims no
reference fold.
"""
function memory!(d::Decider, package)
    get!(d.memories, package) do
        measures(d) || return Memory(d.rates.launch, d.rates.cached)
        key(what) = string(package, " memory ", what)
        haskey(d.calibration, key("bandwidth")) || fit_memory!(d)
        Memory(d.calibration[key("latency")], d.calibration[key("bandwidth")])
    end
end

"""
Each package's fastest claim among `packages` on the group the planner forms
for `f` over arrays like `args`: its seconds, timed as claims are, and its
launches as the package estimates them.
"""
function reference(d::Decider, packages, f, args...; recomputing = false)
    graph = lower(trace(f, args...))
    timing = Decider(d.target, graph; measure = :none)
    p = plan(graph; recompute = recomputing ? recomputable_folds(graph) : ())
    g = only(p.groups)
    view = group_view(p, g)
    fastest = Dict{Symbol, Tuple{Float64, Vector{Work}}}()
    # Each implementation's first claim: what it stands by unmeasured, which
    # a reference is measured under, not the fastest of its alternatives.
    for (m, is) in REGISTERED, i in is, c in (m in packages ? Iterators.take(claims(i, d.target, view), 1) : ())
        t = time_launch(d.target, first(build_claim(i, c, d.target, view, buffers!(timing, p, g))))
        t < first(get(fastest, nameof(m), (Inf,))) && (fastest[nameof(m)] = (t, estimate(i, c, d.target, view)))
    end
    fastest
end

# Products of factors of these shapes fit a package's line: the small one, too
# small to split, its latency; the large one its rate; and the middle one,
# whose few output tiles leave the device idle unless the package splits the
# sums among blocks, its rate when it does. Products of 128×64 and 64×16384
# factors, summed with a 16384×128 one, are recomputed inside that product, as
# attention's scores are inside its backward pass, which splits its sum too.
const REFERENCE_SHAPES = (((64, 16), (16, 64)), ((256, 256), (256, 256)), ((2048, 2048), (2048, 2048)))
const RECOMPUTED_SHAPES = ((128, 64), (64, 16384), (16384, 128))

"""
Fit the line of every package on the decider's target that has none for
products of `T` factors: the fastest of the package's claims on `a * b` for
each reference shape, timed as claims are, the line through the small and the
large one's times, the rate at which the middle one computes when it splits
the sums, and the rate at which the package's claim on `(a * b) * c`, with
`a * b` recomputed, recomputes it when that claim splits; the line's rate for
either that does not split.
"""
function fit!(d::Decider, T)
    class = product_class(d.target, T)
    key(p, what) = string(p, " ", class, " ", what)
    unfitted = [m for m in used_packages(d.target) if !fitted_line(d, nameof(m), class)]
    times = map(shapes -> reference(d, unfitted, *, map(dims -> zeros(T, dims), shapes)...), REFERENCE_SHAPES)
    recomputing = reference(d, unfitted, (a, b, c) -> (a * b) * c, map(dims -> zeros(T, dims), RECOMPUTED_SHAPES)...; recomputing = true)
    # A reference's first launch: its time less those of the launches after
    # it, which merge split sums, and its flops over its share of the device.
    first_launch(t, work, package) = (t - sum(w -> seconds(w, d.rates, memory!(d, package), nothing), work[2:end]; init = 0.0),
        first(work).flops / min(1.0, first(work).blocks / d.rates.blocks), first(work))
    for (package, _) in times[1]
        (t1, f1, _), (t2, f2, middle), (t3, f3, _) = map(fastest -> first_launch(fastest[package]..., package), times)
        rate = (f3 - f1) / (t3 - t1)
        latency = t1 - f1 / rate
        split = middle.split ? f2 / (t2 - latency) : rate
        t, f, w = haskey(recomputing, package) ? first_launch(recomputing[package]..., package) : (0.0, 0.0, Work(0.0))
        recompute = w.split ? w.recomputed / (t - latency - f / split) : rate
        foreach(((what, value),) -> keep!(d.calibration, d.calibration_path, key(package, what), value),
            (("latency", latency), ("rate", rate), ("split", split), ("recompute", recompute)))
    end
end

# Folds over the columns of 1024-row matrices with these many columns fit a
# package's memory: the bytes the claim moves, against its time.
const MEMORY_COLUMNS = (256, 1024)

"""
Fit the memory of every package on the decider's target that has none: the
line through the times of its fastest claims on a fold over each column at
the two reference sizes, against the bytes the claims move. A package that
claims no such fold moves bytes at the device's rates.
"""
function fit_memory!(d::Decider)
    key(p, what) = string(p, " memory ", what)
    unfitted = [m for m in used_packages(d.target) if !haskey(d.calibration, key(nameof(m), "bandwidth"))]
    small, large = map(n -> reference(d, unfitted, x -> sum(x; dims = 1), zeros(Float32, 1024, n)), MEMORY_COLUMNS)
    for m in unfitted
        package = nameof(m)
        latency, bandwidth = if haskey(small, package)
            (t, a), (u, b) = small[package], large[package]
            bytes(work) = sum(w -> w.bytes, work)
            rate = (bytes(b) - bytes(a)) / (u - t)
            t - bytes(a) / rate, rate
        else
            d.rates.launch, d.rates.cached
        end
        keep!(d.calibration, d.calibration_path, key(package, "latency"), latency)
        keep!(d.calibration, d.calibration_path, key(package, "bandwidth"), bandwidth)
    end
end

"Fit the lines of every package on `target` for Float32, Float16 and BFloat16 factors, their memories, and the seconds of each scalar function in products of Float32 factors, now rather than on first need; the half classes' are priced on first need."
function calibrate!(target)
    d = Decider(target, lower(trace(identity, zeros(Float32, 1))); measure = :none)
    foreach(m -> memory!(d, nameof(m)), used_packages(target))
    foreach(T -> foreach(m -> line!(d, nameof(m), T), used_packages(target)), (Float32, Float16, BFloat16))
    applied!(d)
end

# The device's rates from its rates file, measured and kept there first when
# the file has none. A target that cannot measure has fixed rates.
function device_rates!(target, calibration, path)
    measurable(target) || return rates(target)
    haskey(calibration, "launch") || foreach(((k, v),) -> keep!(calibration, path, string(k), v), pairs(calibrate(target)))
    c = calibration
    Rates(c["launch"], c["cached"], c["bandwidth"], c["cache"], Int(c["blocks"]), NOISE)
end
