# A target: where a program runs. The compiler asks a target for buffers, for
# the floor's launches, for captured graphs, for the device's name, rates and
# times, and for the options it was made with (the kernel packages it uses, its
# numerics, whether it permits rewrites that change rounding, and how it
# classes products), through the protocol below: the public names of this
# file, `packages`, `product_class`, and `ScalarKernel`, `Split`,
# `accumulate!` and `merge_lanes` of the floor's code. The CPU's definitions are here; a package that adds a
# target defines them for its type (see CUDAExt).

public Target, CPUTarget, CUDATarget, target_for, allocate, upload, bindable, task_state, device_name,
    measurable, time_launch, calibrate, capture, replay, release, floor_launch, floor_split, ScalarKernel, Split,
    accumulate!, merge_lanes, packages, numerics, permits_rewrites, product_class

"Where a program runs: `CPUTarget`, `CUDATarget`, or a type a package adds, defining the protocol of this file for it."
abstract type Target end

"Run on the CPU, through the same generated floor code CUDA runs."
struct CPUTarget <: Target end

Base.show(io::IO, ::CPUTarget) = print(io, "CPUTarget()")

"""
    CUDATarget(packages...; precision = :full, online = true)

Run on an NVIDIA GPU; needs CUDACore loaded. `packages` names the packages whose
implementations the target may use, as in `CUDATarget(cuTile)`; by default it
uses every implementation available when a program is prepared. The floor is
always used. `precision` is the numerics policy for products: `:full` keeps
Float32 products in Float32, `:tf32` allows TF32 tensor cores for them, and
`:tf32x3` splits each Float32 factor into two TF32 parts and sums three
tensor-core products (close to Float32 accuracy). Half-precision products
always accumulate in Float32. `online` permits the online softmax (see
`rewrite`): a sum weighted by a softmax accumulates in one pass with the
softmax's maximum, rescaled as it grows, which changes rounding.
"""
struct CUDATarget <: Target
    precision::Symbol
    online::Bool
    packages::Union{Nothing, Tuple{Vararg{Module}}}
end

function CUDATarget(packages::Module...; precision::Symbol = :full, online::Bool = true)
    precision in (:full, :tf32, :tf32x3) || throw(ArgumentError("precision must be :full, :tf32 or :tf32x3, got $precision"))
    CUDATarget(precision, online, isempty(packages) ? nothing : packages)
end

Base.show(io::IO, t::CUDATarget) = print(io, "CUDATarget(", t.packages === nothing ? "" : join(map(string ∘ nameof, t.packages), ", ") * "; ",
    "precision = :", t.precision, t.online ? "" : ", online = false", ")")

"""
    target_for(x) -> target or nothing

The target an argument selects: `CPUTarget()` for an `Array`, a package's
target for its arrays, nothing for anything else. A program is prepared on
the first target among its inputs.
"""
target_for(x::AbstractArray) = CPUTarget()
target_for(x) = nothing

function default_target(inputs)
    for leaf in leaves(inputs)
        t = target_for(leaf)
        t === nothing || return t
    end
    CPUTarget()
end

"The target's numerics, as part of the tuning record's key."
numerics(::Target) = :default
numerics(target::CUDATarget) = target.online ? target.precision : Symbol(target.precision, "-two-pass")

"Whether a target permits the rewrites of sums that change rounding (see `rewrite`)."
permits_rewrites(::Target) = true
permits_rewrites(target::CUDATarget) = target.online

## Storage.

"""
    allocate(target, T, n) -> buffer

A buffer of `n` elements of `T` on `target`, dense and one-dimensional: the
storage of a program's values.
"""
allocate(::CPUTarget, T, n) = Vector{T}(undef, n)

"""
    upload(target, x) -> buffer

The elements of the array `x` as a buffer on `target`: how a program's
constants get there.
"""
upload(::CPUTarget, x) = x

"Whether array `x` can be storage on `target`, bound by reference."
bindable(::Target, x) = false
bindable(::CPUTarget, x) = x isa Array

"""
    task_state(target) -> function

A function that, called first in a new task, gives the task the state on
`target` of the task that asked for it (for CUDA, its context and stream).
"""
task_state(target) = () -> nothing

## The floor's launches.

"""
    floor_launch(target, kernel::ScalarKernel, args) -> launch

A launch of a floor kernel on `target`: `launch()` computes every item of
`kernel` over `args`, the buffers of its slots and workspaces as a tuple. The
CPU runs the items in order, in a function `generated` from the kernel's body;
a device runs them at once, one thread per item, so a scatter's items add with
`accumulate!`. A kernel with a `split` runs each lane of each part of each
item, in the names its body reads: `lin` the item, `part` of `parts`, and
`lane` of `lanes` in the part; the CPU runs a part as one lane.
"""
function floor_launch(::CPUTarget, k::ScalarKernel, args)
    if k.split === nothing
        f = generated(:(function (buffers, n)
            for lin in 0:(n - 1)
                $(k.body)
            end
            nothing
        end))
        return HostLaunch(f, (args, k.items))
    end
    k.split.lanes == 1 || throw(ArgumentError("the CPU runs each part of a split item as one lane, not $(k.split.lanes)"))
    f = generated(:(function (buffers, n, parts)
        lane, lanes = 0, 1
        for lin in 0:(n - 1), part in 0:(parts - 1)
            $(k.body)
        end
        nothing
    end))
    HostLaunch(f, (args, k.items, k.split.parts))
end

"""
A floor kernel's launch on the CPU. Each kernel is a new function, so its
launch holds the function and arguments untyped: a closure over them would be
a new type too, and every method it passes through would compile again for it.
"""
struct HostLaunch
    f::Any
    arguments::Tuple
end

(l::HostLaunch)() = Base.invokelatest(l.f, l.arguments...)

"""
    floor_split(target, items, positions, adjacent) -> Split or nothing

How the floor folds `positions` positions for each of `items` items on
`target`, neighbouring items neighbours in memory when `adjacent`: nothing for
one lane per item, which folds them in order, as on the CPU; or a `Split`,
which deals them among lanes. A target that splits runs a part's lanes
together and merges their states with `merge_lanes`.
"""
floor_split(::Target, items, positions, adjacent) = nothing

## Graphs.

"""
    capture(target, run, keep) -> graph

`run`, which launches work on `target`, captured as a graph for `replay(target,
graph)` to launch again. The graph keeps `keep`, what it launches, alive until
`release(target, graph)` or the collector destroys it. A target without graphs
replays `run` itself.
"""
capture(::Target, run, keep) = run

"Launch `graph`, what `capture` recorded on `target`."
replay(::Target, run) = run()

"Destroy `graph`, what `capture` recorded on `target`, once nothing replays it."
release(::Target, graph) = nothing

## Measuring.

"The name of the device `target` runs on, which names its tuning record and its rates file."
device_name(::CPUTarget) = "cpu"

"Whether claims on `target` can be timed at preparation; `time_launch` times one."
measurable(::Target) = false

"""
    time_launch(target, launch) -> seconds

The seconds one call of `launch()` takes on the device among the launches of
a captured graph, its data in memory rather than in the cache, as the cover
times claims: a step reads its weights once, and finds them in memory. Only
a `measurable` target defines it.
"""
function time_launch end

"""
    calibrate(target) -> rates

The device's rates, measured once and kept in its rates file (see `Rates`):
a named tuple of `launch`, the seconds between launches in a captured graph;
`cached` and `bandwidth`, bytes per second for a launch that moves what the
cache holds and for one that moves more, each from memory, as `time_launch`
times; `cache`, its bytes; and `blocks`, the blocks that fill the device.
Only a `measurable` target defines it.
"""
function calibrate end
