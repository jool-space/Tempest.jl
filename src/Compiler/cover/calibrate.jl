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
reference product, as a library of attention does not, borrows the class's
fastest line of the target's packages; or, when none of them has one, of
Tempest's own whose packages are loaded, fitted for it; or, when none does
products, one that bounds no time, for the decision to measure.
"""
function line!(d::Decider, package, T)
    class = product_class(d.target, T)
    get!(d.lines, (package, class)) do
        measures(d) || return Line(0.0, 1.0e10, 1.0e10, 1.0e10, false)
        key(p, what) = string(p, " ", class, " ", what)
        own(p) = Line((d.calibration[key(p, what)] for what in ("latency", "rate", "split", "recompute"))..., false)
        fitted_line(d, package, class) || fit!(d, T)
        fitted_line(d, package, class) && return own(package)
        fitted(is) = [own(p) for p in unique(map(package_name, is)) if fitted_line(d, p, class)]
        lines = fitted(implementations(d.target))
        if isempty(lines)
            loaded = loaded_implementations()
            fit!(d, T, loaded)
            lines = fitted(loaded)
        end
        isempty(lines) && return Line(0.0, Inf, Inf, Inf, true)
        fastest = argmax(l -> l.rate, lines)
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
Base.@nospecializeinfer price!(d::Decider, regime, class, @nospecialize(f)) = price_applied!(d, regime, class, applied_function(f))

# The table's function `f` is, or addition where the table does not name it:
# what `f` is priced as. Compiled once, as `price!` is, whatever function a
# program applies; what prices it is compiled for the table's functions alone.
Base.@nospecializeinfer function applied_function(@nospecialize(f))
    for (_, g) in APPLIED
        g === f && return g
    end
    +
end

function price_applied!(d::Decider, regime, class, f)
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

# Every function priced in every regime, for the classes `types` make: those
# the rates file lacks are fitted together.
function applied!(d::Decider, types = (Float32,))
    needs = unique((regime, regime === :streamed ? :any : product_class(d.target, T), f) for (_, f) in APPLIED, regime in REGIMES, T in types)
    key((regime, class, f)) = join(("applied", regime, class, APPLIED[findfirst(((_, g),) -> g == f, APPLIED)][1]), " ")
    unfitted = [n for n in needs if measures(d) && !haskey(d.calibration, key(n))]
    isempty(unfitted) || fit_applied_all!(d, unfitted)
    for (regime, class, f) in needs
        price!(d, regime, class, f)
    end
    d.applied
end

const APPLIED_SHAPE = (4096, 512)
const APPLIED_TIMES = 8
const SWEEP_SHAPES = ((256, 512), (512, 2048))

# The seconds per position of each of `needs`, triples `(regime, class, f)`,
# fitted together: the reference programs of all of them are built side by
# side, their kernels compiling at once, then timed one at a time, as claims
# are. Each regime's plain reference (the fold, or the product, without the
# function) is one program for every function of a class.
function fit_applied_all!(d::Decider, needs)
    phase(:calibration) do
        specs = Dict{Any, Tuple}()
        for (regime, class, f) in needs
            plain, applied = applied_references(regime, class, f)
            get!(specs, (regime, class), plain)
            specs[(regime, class, f)] = applied
        end
        keys_ = collect(keys(specs))
        built = concurrently(k -> build_reference(d, implementations(d.target), specs[k][1], specs[k][2]...; recomputing = specs[k][3]), keys_, d.target)
        timed = Dict(k => time_reference(d, r::Reference) for (k, r) in zip(keys_, built))
        for (regime, class, f) in needs
            name = APPLIED[findfirst(((_, g),) -> g == f, APPLIED)][1]
            keep!(d.calibration, d.calibration_path, join(("applied", regime, class, name), " "),
                applied_seconds(regime, f, timed[(regime, class)], timed[(regime, class, f)]))
        end
        nothing
    end
end

function fit_applied!(d::Decider, regime, class, f)
    fit_applied_all!(d, [(regime, class, f)])
    d.calibration[join(("applied", regime, class, APPLIED[findfirst(((_, g),) -> g == f, APPLIED)][1]), " ")]
end

# The two reference programs that price `f` in a regime for a class: the plain
# one, and the one applying `f`, each `(function, arguments, recomputing)`.
function applied_references(regime, class, f)
    if regime === :streamed
        x = zeros(Float32, APPLIED_SHAPE)
        chain(x) = foldl((y, _) -> apply_chain(f, y), 1:APPLIED_TIMES; init = x)
        return ((x -> sum(x; dims = 1), (x,), false), (x -> sum(chain(x); dims = 1), (x,), false))
    end
    T = class in (:Float16, :BFloat16) ? getfield(@__MODULE__, class) : Float32
    if regime === :swept
        a, b = zeros(T, SWEEP_SHAPES[1]), zeros(T, SWEEP_SHAPES[2])
        return (((a, b) -> a * b, (a, b), false), ((a, b) -> a * apply_chain(f, b), (a, b), false))
    end
    a, b, c = map(dims -> zeros(T, dims), RECOMPUTED_SHAPES)
    (((a, b, c) -> (a * b) * c, (a, b, c), true), ((a, b, c) -> apply_chain(f, a * b) * c, (a, b, c), true))
end

# Seconds per position from the two references' times: streamed, the time
# the applications add to the fold, per application; swept or inner, what
# the function adds to the product of the package that computes it, against
# that package's product without it, for the package it costs least.
function applied_seconds(regime, f, plain, applied)
    fastest(r) = minimum(first, values(r); init = Inf)
    regime === :streamed && return max(0.0, fastest(applied) - fastest(plain)) / (APPLIED_TIMES * prod(APPLIED_SHAPE))
    both = [p for p in keys(applied) if haskey(plain, p)]
    isempty(both) && return 0.0
    package = argmin(p -> first(applied[p]) - first(plain[p]), both)
    n = sum(n for w in last(applied[package]) for (g, n) in getfield(w, regime) if g == f; init = 0.0)
    n > 0 ? max(0.0, first(applied[package]) - first(plain[package])) / n : 0.0
end

# One application of `f` to an array, with a literal second operand where it
# takes one, so that the chain stays one term of one kernel.
apply_chain(f, y) = f.(y)
apply_chain(f::Union{typeof(*), typeof(+), typeof(-), typeof(/), typeof(max), typeof(min)}, y) = f.(y, 1.5f0)
apply_chain(::typeof(^), y) = y .^ 3
apply_chain(c::Cast{T}, y) where {T} = Float32.(c.(y)) .+ 1.5f0

"""
The memory of `package` on the decider's target, or of the floor's kernels
for `:floor`, fitted on first need, with the target's packages, and kept in
the rates file; the device's rates for a package that claims no reference
fold.
"""
function memory!(d::Decider, package)
    get!(d.memories, package) do
        measures(d) || return Memory(d.rates.launch, d.rates.cached)
        key(what) = string(package, " memory ", what)
        own = implementations(d.target)
        haskey(d.calibration, key("bandwidth")) ||
            fit_memory!(d, Implementation[own..., (i for i in loaded_implementations() if package_name(i) === package && !(i in own))...])
        Memory(d.calibration[key("latency")], d.calibration[key("bandwidth")])
    end
end

"""
A reference program, `f` over arrays like `args`, built: planned as one group,
with each implementation's first claim on it built on buffers of its own, for
`time_reference` to time as claims are, and the floor's kernels too when
`floor`, as package `:floor`. The first claim is what a package stands by
unmeasured, which a reference is measured under, not the fastest of its
alternatives.
"""
struct Reference
    launches::Vector{Tuple{Symbol, Any, Vector{Work}}} # package, launch, estimate
end

function build_reference(d::Decider, among, f, args...; recomputing = false, floor = false)
    graph = lower(trace(f, args...))
    timing = Decider(d.target, graph; measure = :none)
    p = plan(graph; recompute = recomputing ? recomputable_folds(graph) : ())
    g = only(p.groups)
    view = group_view(p, g)
    launches = Tuple{Symbol, Any, Vector{Work}}[]
    for i in among, c in Iterators.take(Base.invokelatest(claims, i, d.target, view), 1)
        push!(launches, (package_name(i), first(Base.invokelatest(build_claim, i, c, d.target, view, buffers!(timing, p, g))), Base.invokelatest(estimate, i, c, d.target, view)))
    end
    floor && push!(launches, (:floor, first(build_floor(d.target, p, g, buffers!(timing, p, g))), floor_estimate(p, g, floor_splits(timing))))
    Reference(launches)
end

"Each package's fastest launch of a built reference: its seconds, and its launches as the package estimates them."
function time_reference(d::Decider, r::Reference)
    fastest = Dict{Symbol, Tuple{Float64, Vector{Work}}}()
    for (package, launch, work) in r.launches
        t = time_launch(d.target, launch)
        t < first(get(fastest, package, (Inf,))) && (fastest[package] = (t, work))
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
Fit the line of every package of the implementations `among`, by default the
decider's target's, that has none for products of `T` factors: the fastest of the package's claims on `a * b` for
each reference shape, timed as claims are, the line through the small and the
large one's times, the rate at which the middle one computes when it splits
the sums, and the rate at which the package's claim on `(a * b) * c`, with
`a * b` recomputed, recomputes it when that claim splits; the line's rate for
either that does not split.
"""
function fit!(d::Decider, T, among = implementations(d.target))
    phase(:calibration) do
        class = product_class(d.target, T)
        key(p, what) = string(p, " ", class, " ", what)
        unfitted = [i for i in among if !fitted_line(d, package_name(i), class)]
        isempty(unfitted) && return nothing
        # The references built side by side, their kernels compiling at once,
        # then timed one at a time, as `fit_applied_all!` builds its own.
        products = [map(dims -> zeros(T, dims), shapes) for shapes in REFERENCE_SHAPES]
        recomputed = map(dims -> zeros(T, dims), RECOMPUTED_SHAPES)
        built = concurrently(1:(length(products) + 1), d.target) do k
            k <= length(products) ? build_reference(d, unfitted, *, products[k]...) :
                build_reference(d, unfitted, (a, b, c) -> (a * b) * c, recomputed...; recomputing = true)
        end
        timed = [time_reference(d, r::Reference) for r in built]
        times, recomputing = timed[1:(end - 1)], timed[end]
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
end

# Folds over the columns of 1024-row matrices with these many columns fit a
# package's memory: the bytes the claim moves, against its time. The smallest
# moves 64 KB, its time nearly all latency, which a line through larger folds
# alone extrapolates to: through the 1 and 4 MB ones, a jitter of a tenth in
# the smaller's time moved the latency between 0.4 and 2.1 μs, and with it
# whether the cover measured a split it would otherwise not.
const MEMORY_COLUMNS = (16, 256, 1024, 4096)

"""
Fit the memory of every package of the implementations `among`, by default
the decider's target's, and of the floor, `:floor`, that has none, from the
times of its fastest claims on a fold over each column at the reference
sizes and the bytes the claims move: its bandwidth is the slope of their
least-squares line, which the largest folds set, and its latency, a launch's,
the smallest fold's time less its bytes at that bandwidth, over its
launches. A package that claims no such fold moves bytes at the device's
rates.
"""
function fit_memory!(d::Decider, among = implementations(d.target))
    phase(:calibration) do
        key(p, what) = string(p, " memory ", what)
        unfitted = [i for i in among if !haskey(d.calibration, key(package_name(i), "bandwidth"))]
        floor = !haskey(d.calibration, key(:floor, "bandwidth"))
        built = concurrently(n -> build_reference(d, unfitted, x -> sum(x; dims = 1), zeros(Float32, 1024, n); floor), collect(MEMORY_COLUMNS), d.target)
        folds = [time_reference(d, r::Reference) for r in built]
        for package in [unique(map(package_name, unfitted)); floor ? [:floor] : Symbol[]]
            latency, bandwidth = if all(fold -> haskey(fold, package), folds)
                bytes = [sum(w -> w.bytes, last(fold[package])) for fold in folds]
                seconds = [first(fold[package]) for fold in folds]
                b, s = sum(bytes) / length(bytes), sum(seconds) / length(seconds)
                per_byte = sum((bytes .- b) .* (seconds .- s)) / sum((bytes .- b) .^ 2)
                (first(seconds) - first(bytes) * per_byte) / length(last(first(folds)[package])), 1 / per_byte
            else
                d.rates.launch, d.rates.cached
            end
            keep!(d.calibration, d.calibration_path, key(package, "latency"), latency)
            keep!(d.calibration, d.calibration_path, key(package, "bandwidth"), bandwidth)
        end
    end
end

"Fit the lines of every package on `target` for Float32, Float16 and BFloat16 factors, their memories, and the seconds of each scalar function in products of each, now rather than on first need, so that a rates file it writes prices every plan alike wherever it is read."
function calibrate!(target)
    d = Decider(target, lower(trace(identity, zeros(Float32, 1))); measure = :none)
    packages = unique(map(package_name, implementations(target)))
    foreach(p -> memory!(d, p), packages)
    types = (Float32, Float16, BFloat16)
    foreach(T -> foreach(p -> line!(d, p, T), packages), types)
    applied!(d, types)
end

# The device's rates as it reports them. A target that cannot measure has fixed rates.
function target_rates(target)
    measurable(target) || return rates(target)
    r = device_rates(target)
    Rates(r.launch, r.cached, r.bandwidth, r.cache, Int(r.blocks), NOISE)
end
