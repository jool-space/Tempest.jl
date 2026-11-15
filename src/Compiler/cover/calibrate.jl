# Measuring on the device: its rates, each package's line for products of a
# class and its memory, and the seconds per position of each scalar function
# in each regime, fitted on first need to reference programs timed as claims
# are, and kept in the rates file.

# What a fitted rate can be, ends included: seconds, a latency, a step, or
# what a function adds, no fewer than none; a rate of flops, more than none;
# a memory's bandwidth, more than none and no more than the memory's peak
# (`Rates`), to within the noise, as the timer clears the cache before each
# reference it times (`time_launch`) and the reference reads from memory.
const SECONDS = (0.0, Inf)
const RATE = (floatmin(Float64), Inf)
bandwidths(d) = (floatmin(Float64), (1 + NOISE) * d.rates.bandwidth)

"""
Keep the rates `values` a fit yields under `keys` in the rates file where
each lies in its `domains`, what the rate can be; where any does not, the
fit has failed: it is reported, and `instead`, what the rates' docstrings
say stands in their absence, is kept in their place, or nothing where
`instead` is `nothing`. The rates of one fit stand or fall together, as a
memory's latency is fitted through its bandwidth.
"""
function keep_fit!(d::Decider, keys, values, domains, instead)
    within(value, (low, high)) = isfinite(value) && low <= value <= high
    if !all(splat(within), zip(values, domains))
        @warn "The fit of $(join(keys, ", ")) failed: $(join(values, ", ")) outside $(join(domains, ", ")); " *
            (instead === nothing ? "nothing is kept" : "$(join(instead, ", ")) stands in its place")
        values = instead
    end
    values === nothing || foreach((key, value) -> keep!(d.calibration, d.calibration_path, key, value), keys, values)
    values
end

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
# that recomputes a fold, over the fold's tiles; stepped, in each step of a
# sweep, along the chain a step computes on what it loaded before the next.
const REGIMES = (:streamed, :swept, :inner, :stepped)

# The class a regime prices a function for: streamed and stepped, the same
# for every class, the function's work neither in a product's sweep nor in
# its tiles; swept and inner, that of the product's factors.
regime_class(regime, class) = regime in (:streamed, :stepped) ? :any : class

"""
The seconds per position of a scalar function `f` applied in `regime` in a
kernel whose products are of `class`, measured on first need and kept in the
rates file. Streamed: the time of a fold over eight applications of the
function beyond that of the fold alone, per application, the same for every
class. Swept: the time a product's factor computed by the function adds to
the product. Inner: the same for a product that recomputes the fold the
function is applied to. Stepped: the time a matrix-vector product's vector
computed by the function in each step adds to that product's sweep, per
step and application, the same for every class. A function priced where other work hides it, or that
adds less than a timing can tell, measures as nothing, which is what it
costs there. A function the table does not name costs what addition does.
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
    class = regime_class(regime, class)
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
    needs = unique((regime, regime_class(regime, product_class(d.target, T)), f) for (_, f) in APPLIED, regime in REGIMES, T in types)
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

# A matrix-vector product's sweep prices the functions a step computes, and
# a step of a fold a block recomputes apart (see `fit_refold!`): a
# half-precision matrix stored along the sum, as a checkpoint stores its
# weights, of enough outputs that the generator keeps its sum whole rather
# than splitting it among blocks, and a long sum, which its blocks sweep in
# many narrow steps, side by side. On GB10 the generator's first claim
# takes 64 blocks of 32 outputs and steps 32 along the sum, and computed a
# vector's silu in 0.3 μs a step more than its product alone, on numbers as
# a program's are (see `buffers!`): on zeros, whose division runs its slow
# path, the silu took 1.2 to 2.5.
const STEPPED_OUTPUTS = 2048
const STEPPED_SUMS = (8192, 16384)

stepped_reference(sum) = (zeros(BFloat16, sum, STEPPED_OUTPUTS), zeros(BFloat16, sum))

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
            keep_fit!(d, (join(("applied", regime, class, name), " "),),
                (applied_seconds(regime, f, timed[(regime, class)], timed[(regime, class, f)]),), (SECONDS,), (0.0,))
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
    if regime === :stepped
        w, x = stepped_reference(first(STEPPED_SUMS))
        return (((w, x) -> transpose(w) * x, (w, x), false), ((w, x) -> transpose(w) * BFloat16.(apply_chain(f, Float32.(x))), (w, x), false))
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
# that package's product without it, for the package it costs least;
# stepped, the same per application in a step along the sweep's chain.
function applied_seconds(regime, f, plain, applied)
    fastest(r) = minimum(first, values(r); init = Inf)
    regime === :streamed && return added_seconds(fastest(applied), fastest(plain)) / (APPLIED_TIMES * prod(APPLIED_SHAPE))
    both = [p for p in keys(applied) if haskey(plain, p)]
    isempty(both) && return 0.0
    package = argmin(p -> first(applied[p]) - first(plain[p]), both)
    n = sum(n for w in last(applied[package]) for (g, n) in getfield(w, regime) if g == f; init = 0.0)
    n > 0 ? added_seconds(first(applied[package]), first(plain[package])) / n : 0.0
end

# The seconds a function adds to a reference that takes `plain` without it:
# none where the two tie, within the margin two measurements tie by
# (`NOISE`), so that a function no timing tells from nothing is priced at
# nothing, and a fit again writes the same. Two timings of one plain
# reference, eighteen timed ten times each on the GB10, differed by at most
# 5.1% of the faster (1.4% at the 90th percentile); what the functions that
# cost something add is 13% (exp) to 4.3 times (inv) of the reference at
# the median.
added_seconds(applied, plain) = applied - plain <= NOISE * plain ? 0.0 : applied - plain

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
        measures(d) || return Memory(d.rates.launch, d.rates.cached, 0.0, d.rates.cached, ESTIMATE_ERROR, 0.0)
        own = implementations(d.target)
        fitted_memory(d, package) ||
            fit_memory!(d, Implementation[own..., (i for i in loaded_implementations() if package_name(i) === package && !(i in own))...])
        Memory((d.calibration[memory_key(package, what)] for what in MEMORY_RATES)...)
    end
end

"""
A reference program, `f` over arrays like `args`, built: planned as one group,
with each implementation's first claim on it built on buffers of its own and
the resources of the decider that times it, for `time_reference` to time as
claims are, and the floor's kernels too when `floor`, as package `:floor`;
with `offered`, every claim each implementation makes and every variant it
offers of them, and of those, as far as they reach (see
`Tempest.Compiler.Implement.variants`). The first claim is what a package
stands by unmeasured but where its estimate ranks another of its offer
first (see `standing`), and a reference is measured under it, not the
fastest of its alternatives.
"""
struct Reference
    launches::Vector{Tuple{Symbol, Any, Vector{Work}}} # package, launch, estimate
end

function build_reference(d::Decider, among, f, args...; recomputing = false, floor = false, offered = false)
    graph = lower(trace(f, args...))
    timing = Decider(d.target, graph; measure = :none)
    p = plan(graph; recompute = recomputing ? recomputable_folds(graph) : ())
    g = only(p.groups)
    view = group_view(p, g)
    launches = Tuple{Symbol, Any, Vector{Work}}[]
    for i in among, c in (offered ? offer(i, d.target, view) : Iterators.take(Base.invokelatest(claims, i, d.target, view), 1))
        built = Base.invokelatest(build_claim, i, c, d.target, view, buffers!(timing, p, g), resources!(d.resources, i))
        push!(launches, (package_name(i), first(built), Base.invokelatest(estimate, i, c, d.target, view)))
    end
    floor && push!(launches, (:floor, first(build_floor(d.target, p, g, buffers!(timing, p, g))), floor_estimate(p, g, floor_splits(timing))))
    Reference(launches)
end

# Every claim implementation `i` makes on a group and every variant it offers
# of them, and of those, each once (by `record_name`).
function offer(i, target, view)
    found, names = Any[], Set{String}()
    next = collect(Any, Base.invokelatest(claims, i, target, view))
    while !isempty(next)
        c = popfirst!(next)
        name = Base.invokelatest(record_name, i, c)
        name in names && continue
        push!(names, name)
        push!(found, c)
        append!(next, Base.invokelatest(variants, i, target, view, c))
    end
    found
end

"Each package's launches of a built reference: their seconds, and their launches as the package estimates them."
function time_launches(d::Decider, r::Reference)
    timed = Dict{Symbol, Vector{Tuple{Float64, Vector{Work}}}}()
    for (package, launch, work) in r.launches
        push!(get!(() -> Tuple{Float64, Vector{Work}}[], timed, package), (d.timer(d.target, launch), work))
    end
    timed
end

"Each package's fastest launch of a built reference: its seconds, and its launches as the package estimates them."
time_reference(d::Decider, r::Reference) = Dict(package => argmin(first, timed) for (package, timed) in time_launches(d, r))

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
either that does not split, or whose fit fails (see `keep_fit!`). A line
whose rate fails is not kept, and the package borrows one; a latency that
fails, the small product taking less than its flops at the rate over its
share of the device, is none.
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
            fitted = keep_fit!(d, (key(package, "rate"),), ((f3 - f1) / (t3 - t1),), (RATE,), nothing)
            fitted === nothing && continue
            rate = only(fitted)
            latency = only(keep_fit!(d, (key(package, "latency"),), (t1 - f1 / rate,), (SECONDS,), (0.0,)))
            split = only(keep_fit!(d, (key(package, "split"),), (middle.split ? f2 / (t2 - latency) : rate,), (RATE,), (rate,)))
            t, f, w = haskey(recomputing, package) ? first_launch(recomputing[package]..., package) : (0.0, 0.0, Work(0.0))
            recompute = w.split ? w.recomputed / (t - latency - f / split) : rate
            keep_fit!(d, (key(package, "recompute"),), (recompute,), (RATE,), (rate,))
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

# What a package's memory is in the rates file, each a line of its own (see `Memory`).
const MEMORY_RATES = ("latency", "bandwidth", "step", "block", "error", "refold")
memory_key(package, what) = string(package, " memory ", what)
memory_fitted(d, package, whats) = all(what -> haskey(d.calibration, memory_key(package, what)), whats)
fitted_memory(d, package) = memory_fitted(d, package, MEMORY_RATES)

"""
Fit the memory of every package of the implementations `among`, by default
the decider's target's, and of the floor, `:floor`, that has none: its
latency and bandwidth, where the rates file lacks them, from the times of
its fastest claims on a fold over each column at the reference sizes and
the bytes the claims move: its bandwidth is the slope of their
least-squares line, which the largest folds set, and its latency, a launch's,
the smallest fold's time less its bytes at that bandwidth, over its
launches. A package that claims no such fold, or whose fit fails (see
`keep_fit!`), moves bytes at the device's rates. Then its step and its
refold, where the file lacks them (see `fit_step!` and `fit_refold!`).
"""
function fit_memory!(d::Decider, among = implementations(d.target))
    phase(:calibration) do
        key(p, what) = memory_key(p, what)
        unfitted = [i for i in among if !memory_fitted(d, package_name(i), ("latency", "bandwidth"))]
        floor = !memory_fitted(d, :floor, ("latency", "bandwidth"))
        built = concurrently(n -> build_reference(d, unfitted, x -> sum(x; dims = 1), zeros(Float32, 1024, n); floor), collect(MEMORY_COLUMNS), d.target)
        folds = [time_reference(d, r::Reference) for r in built]
        device = (d.rates.launch, d.rates.cached)
        for package in [unique(map(package_name, unfitted)); floor ? [:floor] : Symbol[]]
            keys = key.(package, ("latency", "bandwidth"))
            if all(fold -> haskey(fold, package), folds)
                bytes = [sum(w -> w.bytes, last(fold[package])) for fold in folds]
                seconds = [first(fold[package]) for fold in folds]
                b, s = sum(bytes) / length(bytes), sum(seconds) / length(seconds)
                per_byte = sum((bytes .- b) .* (seconds .- s)) / sum((bytes .- b) .^ 2)
                latency = (first(seconds) - first(bytes) * per_byte) / length(last(first(folds)[package]))
                keep_fit!(d, keys, (latency, 1 / per_byte), (SECONDS, bandwidths(d)), device)
            else
                foreach((key, value) -> keep!(d.calibration, d.calibration_path, key, value), keys, device)
            end
        end
        fit_step!(d, among)
        fit_refold!(d, among)
    end
end

# Matrix-vector products of a half-precision matrix stored along the sum fit
# a package's step (see `fit_step!`): 9216 outputs, a projection's of a
# decoding step, so that the blocks of every tile offered fill the device a
# wave or more, summing 2560 and 5120, each moving tens of megabytes, many
# ticks of the coarsest event clock (the RTX 6000 Ada's, 1.024 μs).
const STEP_OUTPUTS = 9216
const STEP_SUMS = (2560, 5120)

"""
Fit the step of every package of the implementations `among`, and of the
floor, that has none: its latency and the rate one block reaches (see
`Memory`), by least squares over every tile the package offers on
matrix-vector products (see `STEP_SUMS`), timed as claims are, of the
logarithm of each time against its seconds (see `seconds`). Where the
tiles cannot bound the block's rate from above, the fit with the block at
the memory's bandwidth no worse than the best by more than two standard
errors, the block's rate is none of its own, the memory's bandwidth. The
form's error is what the fit leaves unexplained: twice the standard error
of its residuals, as a factor, within which two launches' estimates are
not told apart (see `standing`). A package whose claims state no steps, as
the floor's do not, or whose fit fails (see `keep_fit!`), steps in no time,
a block at the memory's bandwidth, its estimates told apart beyond
`ESTIMATE_ERROR`.
"""
function fit_step!(d::Decider, among = implementations(d.target))
    rates = ("step", "block", "error")
    unfitted = [i for i in among if !memory_fitted(d, package_name(i), rates)]
    floor = !memory_fitted(d, :floor, rates)
    reference(n) = (zeros(BFloat16, n, STEP_OUTPUTS), zeros(BFloat16, n))
    built = concurrently(n -> build_reference(d, unfitted, (w, x) -> transpose(w) * x, reference(n)...; floor, offered = true),
        collect(STEP_SUMS), d.target)
    sweeps = [time_launches(d, r::Reference) for r in built]
    for package in [unique(map(package_name, unfitted)); floor ? [:floor] : Symbol[]]
        latency, bandwidth = (d.calibration[memory_key(package, what)] for what in ("latency", "bandwidth"))
        timed = [(t, only(work)) for sweep in sweeps for (t, work) in get(sweep, package, ()) if length(work) == 1 && stepped(only(work))]
        squares(step, block) = sum(((t, w),) -> log(t / seconds(w, d.rates, Memory(latency, bandwidth, step, block, ESTIMATE_ERROR, 0.0), nothing))^2, timed; init = 0.0)
        values = step_fit(squares, bandwidth, length(timed))
        keep_fit!(d, memory_key.(package, rates), values === nothing ? (0.0, bandwidth, ESTIMATE_ERROR) : values,
            (SECONDS, (RATE[1], bandwidth), (1.0, Inf)), (0.0, bandwidth, ESTIMATE_ERROR))
    end
end

"""
Fit the refold of every package of the implementations `among`, and of the
floor, that has none: the slope of its first claims' times on a
matrix-vector product (see `STEPPED_SUMS`) over the steps their sweeps
take, at the two sums, what a sweep's step took, its loads and all, as
sweeps were priced before their loads were (see `fit_step!`). It prices a
step of a fold a block recomputes apart (see `seconds`), as it did then,
though on GB10 it is 0.79 μs where such a step takes 60 to 70 ns, until
work a launch recomputes is priced by the steps it takes. None where its
claims state no steps, as the floor's do not, it claims no such product,
or the fit fails (see `keep_fit!`).
"""
function fit_refold!(d::Decider, among = implementations(d.target))
    unfitted = [i for i in among if !memory_fitted(d, package_name(i), ("refold",))]
    floor = !memory_fitted(d, :floor, ("refold",))
    built = concurrently(n -> build_reference(d, unfitted, (w, x) -> transpose(w) * x, stepped_reference(n)...; floor), collect(STEPPED_SUMS), d.target)
    sweeps = [time_reference(d, r::Reference) for r in built]
    for package in [unique(map(package_name, unfitted)); floor ? [:floor] : Symbol[]]
        steps = all(sweep -> haskey(sweep, package), sweeps) ? [first(last(sweep[package])).steps for sweep in sweeps] : [0.0, 0.0]
        if steps[2] > steps[1] > 1
            refold = (first(sweeps[2][package]) - first(sweeps[1][package])) / (steps[2] - steps[1])
            keep_fit!(d, (memory_key(package, "refold"),), (refold,), (SECONDS,), (0.0,))
        else
            keep!(d.calibration, d.calibration_path, memory_key(package, "refold"), 0.0)
        end
    end
end

# The steps' latencies and blocks' rates a step's fit searches (see
# `step_fit`): to 4 μs, past what a step of the tables took, 10 ns apart;
# from a thousandth of the bandwidth to the whole, evenly in its logarithm.
const STEP_LATENCIES = range(0.0, 4.0e-6; length = 401)
const BLOCK_SHARES = exp10.(range(-3.0, 0.0; length = 121))

# A step's latency, a block's rate and the fit's error, `(latency, rate,
# error)`, minimizing `squares(latency, rate)`, a sum of `n` squares of
# logarithms, over `STEP_LATENCIES` and `BLOCK_SHARES` of `bandwidth`; the
# rate the bandwidth where the fit at it is within two standard errors of
# the best, and the error twice the standard error of the fit kept, as a
# factor (see `fit_step!`); nothing from fewer than three tiles.
function step_fit(squares, bandwidth, n)
    n >= 3 || return nothing
    best(rate) = argmin(last, ((l, squares(l, rate)) for l in STEP_LATENCIES))
    fits = [(rate, best(rate)...) for rate in bandwidth .* BLOCK_SHARES]
    rate, latency, least = argmin(last, fits)
    _, at_bandwidth, whole = last(fits)
    unbounded = whole - least <= 4 * least / (n - 2)
    unbounded ? (at_bandwidth, bandwidth, exp(2 * sqrt(whole / (n - 2)))) : (latency, rate, exp(2 * sqrt(least / (n - 2))))
end

"Fit the lines of every package on `target` for Float32, Float16 and BFloat16 factors, their memories and the floor's, and the seconds of each scalar function in products of each, now rather than on first need, so that a rates file it writes prices every plan alike wherever it is read. What the rates file lacks is fitted, whatever is shipped for the device (see `shipped_rates`)."
function calibrate!(target)
    d = Decider(target, lower(trace(identity, zeros(Float32, 1))); measure = :none, calibration = Noted(load_record(rates_path(target))))
    packages = unique(map(package_name, implementations(target)))
    foreach(p -> memory!(d, p), [packages; :floor])
    types = (Float32, Float16, BFloat16)
    foreach(T -> foreach(p -> line!(d, p, T), packages), types)
    applied!(d, types)
end

# The device's rates as it reports them. A target that cannot measure has fixed rates.
function target_rates(target)
    measurable(target) || return rates(target)
    r = device_rates(target)
    Rates(r.launch, r.cached, r.bandwidth, r.cache, Int(r.blocks), NOISE, LINK)
end
