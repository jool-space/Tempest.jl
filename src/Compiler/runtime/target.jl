# A target: where a program runs. The compiler asks a target for buffers, for
# the floor's launches, for captured graphs, for the device's name, rates and
# times, and for the options it was made with (the implementations it uses,
# its numerics, whether it permits rewrites that change rounding, and how it
# classes products), through the protocol below: the public names of this
# file, `implementations`, `product_class`, and `ScalarKernel`, `Split`,
# `accumulate!` and `merge_lanes` of the floor's code. The CPU's definitions are here; a package that adds a
# target defines them for its type (see CUDAExt).

public Target, CPUTarget, CUDATarget, target_for, allocate, upload, bindable, task_state, device_name,
    measurable, time_launch, calibrate, capture, replay, release, floor_launch, floor_split, ScalarKernel, Split,
    accumulate!, merge_lanes, implementations, numerics, permits_rewrites, rounded_backward, deterministic, product_class

"Where a program runs: `CPUTarget`, `CUDATarget`, or a type a package adds, defining the protocol of this file for it."
abstract type Target end

"""
    CPUTarget(implementations...)

Run on the CPU, through the same generated floor code CUDA runs, and the
`implementations` given, none by default.
"""
struct CPUTarget <: Target
    implementations::Tuple{Vararg{Implementation}}
    CPUTarget(implementations::Implementation...) = new(implementations)
end

Base.show(io::IO, t::CPUTarget) = print(io, "CPUTarget(", join(map(repr, t.implementations), ", "), ")")

"""
    CUDATarget(implementations...; precision = :full, online = true, rounded_backward = false, deterministic = true)

Run on an NVIDIA GPU; needs CUDACore loaded. The target uses the floor and
the `implementations` given, as in `CUDATarget(cuTileGenerator())`, ties
between their estimates going to the first; by default Tempest's own whose
packages are loaded when it is made, cuBLAS's, cuDNN's and cuTile's (see
`cuBLASProducts`, `cuDNNAttention` and `cuTileGenerator`). Another package's
implementation takes part only when it is given, and with none at all,
`CUDATarget(precision, online, ())`, the floor runs every kernel.
`precision` is the numerics policy for products: `:full` keeps Float32
products in Float32, `:tf32` allows TF32 tensor cores for them, and
`:tf32x3` splits each Float32 factor into two TF32 parts and sums three
tensor-core products (close to Float32 accuracy). Half-precision products
always accumulate in Float32. `online` permits the online softmax (see
`rewrite`): a sum weighted by a softmax accumulates in one pass with the
softmax's maximum, rescaled as it grows, which changes rounding.
`rounded_backward` lets a backward read a forward's sum as the program rounds
it to half precision where the sums of the derivative are exchanged (see
`exchanged_sums`): the softmax backward's Σ dO·o then reads the output o as a
half-precision forward stores it, as FlashAttention and cuDNN read theirs,
rather than in Float32, and the forward stores the rounded output only. It
changes rounding: in BFloat16 at 64 features, 8 heads, 1024 positions, the
gradients' largest error grows from 6.8e-3 to 8.6e-3 of their largest element
when the output's cotangent comes from elsewhere, and from 6.2e-3 to 3.5e-2
for a loss of the output's squares, where Σ dO·o nearly cancels the sums it
is taken from.
`deterministic` keeps a sum's rounding from depending on the order its
blocks run in. With it false, a sweep may compute a sum in the sweep's
orientation and add it into its value atomically, as FlashAttention-2's
backward adds the queries' gradient: attention's training step in Float32
at 1024 positions, 64 features, 8 heads, causal and TF32 then takes 449 μs
against 741, and the gradient differs from run to run in its last bits. A
sum by key, as an embedding's gradient is, accumulates atomically either
way.
"""
struct CUDATarget <: Target
    precision::Symbol
    online::Bool
    implementations::Tuple{Vararg{Implementation}}
    rounded_backward::Bool
    deterministic::Bool
end

CUDATarget(precision::Symbol, online::Bool, implementations::Tuple) = CUDATarget(precision, online, implementations, false, true)

function CUDATarget(implementations::Implementation...; precision::Symbol = :full, online::Bool = true,
        rounded_backward::Bool = false, deterministic::Bool = true)
    precision in (:full, :tf32, :tf32x3) || throw(ArgumentError("precision must be :full, :tf32 or :tf32x3, got $precision"))
    CUDATarget(precision, online, isempty(implementations) ? loaded_implementations() : implementations, rounded_backward, deterministic)
end

Base.show(io::IO, t::CUDATarget) = print(io, "CUDATarget(", join(map(repr, t.implementations), ", "), "; ",
    "precision = :", t.precision, t.online ? "" : ", online = false",
    t.rounded_backward ? ", rounded_backward = true" : "", t.deterministic ? "" : ", deterministic = false", ")")

"The implementations a target uses beside the floor, in order."
implementations(target::Union{CPUTarget, CUDATarget}) = target.implementations
implementations(::Target) = ()

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
numerics(target::CUDATarget) = Symbol(target.precision, target.online ? "" : "-two-pass",
    target.rounded_backward ? "-rounded-backward" : "", target.deterministic ? "" : "-nondeterministic")

"Whether a target permits the rewrites of sums that change rounding (see `rewrite`)."
permits_rewrites(::Target) = true
permits_rewrites(target::CUDATarget) = target.online

"Whether a backward reads a forward's sum as the program rounds it (see `CUDATarget`)."
rounded_backward(::Target) = false
rounded_backward(target::CUDATarget) = permits_rewrites(target) && target.rounded_backward

"Whether a target keeps a sum's rounding from depending on the order its blocks run in (see `CUDATarget`)."
deterministic(::Target) = true
deterministic(target::CUDATarget) = target.deterministic

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
