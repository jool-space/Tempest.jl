# A target: where a program runs. The compiler asks a target for buffers, for
# the floor's launches, for captured graphs, for the device's name, rates and
# times, and for the options it was made with (the implementations it uses,
# its numerics, whether it permits rewrites that change rounding, and how it
# classes products), through the protocol below: the public names of this
# file, `implementations`, `product_class`, and `ScalarKernel`, `Split`,
# `accumulate!` and `merge_lanes` of the floor's code. The CPU's definitions are here; a package that adds a
# target defines them for its type (see CUDAExt).

public Target, CPUTarget, CUDATarget, target_for, device_of, on_device, Step, crosses, walk, follow, written, crossing, allocate, upload, upload!, download!, bindable, pointable, retarget!, task_state, device_name,
    toolchain, measurable, time_launch, quiesce, device_rates, capture, recapture, replay, release, floor_launch, floor_split, ScalarKernel, Split,
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
    CUDATarget(implementations...; device = 0, precision = :full, online = true, rounded_backward = true, deterministic = false)

Run on an NVIDIA GPU, the one CUDA numbers `device`; needs CUDACore loaded.
A program whose arrays are on a device is prepared for that device's target,
whatever device the calling task is on (see `target_for`). The target uses the floor and
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
always accumulate in Float32, and a product of a half-precision value, read
through its conversion to Float32, and a Float32 one is a half-precision
product under any `precision`, the Float32 factor rounded to the half type:
a BFloat16 attention's backward rounds its probabilities and their gradient
so, as FlashAttention and cuDNN do, where the CPU multiplies in Float32 as
the program does. The target trades that precision for speed as `cu` does,
which makes arrays Float32. `online` permits the online softmax (see
`rewrite`): a sum weighted by a softmax accumulates in one pass with the
softmax's maximum, rescaled as it grows, which changes rounding.
`rounded_backward` lets a backward read a forward's sum as the program rounds
it to half precision where the sums of the derivative are exchanged (see
`exchanged_sums`): the softmax backward's Σ dO·o reads the output o as a
half-precision forward stores it, as FlashAttention and cuDNN read theirs,
rather than in Float32, and the forward stores the rounded output only. With
it false the backward reads the sum as the program computes it: in BFloat16
at 64 features, 8 heads, 1024 positions, the gradients' largest error is
6.2e-3 of their largest element rather than 6.8e-3 for a loss of the
output's squares, where Σ dO·o nearly cancels the sums it is taken from,
and 5.5e-3 either way when the output's cotangent comes from elsewhere.
`deterministic` keeps a sum's rounding from depending on the order its
blocks run in. Without it a sweep may compute a sum in the sweep's
orientation and add it into its value atomically, as FlashAttention-2's
backward adds the queries' gradient, which then differs from run to run in
its last bits: attention's training step in Float32 at 1024 positions, 64
features, 8 heads, causal and TF32 takes 437 μs, and 702 deterministic. A
sum by key, as an embedding's gradient is, accumulates atomically either
way.
"""
struct CUDATarget <: Target
    precision::Symbol
    online::Bool
    implementations::Tuple{Vararg{Implementation}}
    rounded_backward::Bool
    deterministic::Bool
    # The device's number, as CUDA counts them from 0.
    device::Int
end

CUDATarget(precision::Symbol, online::Bool, implementations::Tuple) = CUDATarget(precision, online, implementations, true, false, 0)

function CUDATarget(implementations::Implementation...; device::Integer = 0, precision::Symbol = :full, online::Bool = true,
        rounded_backward::Bool = true, deterministic::Bool = false)
    precision in (:full, :tf32, :tf32x3) || throw(ArgumentError("precision must be :full, :tf32 or :tf32x3, got $precision"))
    CUDATarget(precision, online, isempty(implementations) ? loaded_implementations() : implementations, rounded_backward, deterministic, device)
end

Base.show(io::IO, t::CUDATarget) = print(io, "CUDATarget(", join(map(repr, t.implementations), ", "), "; ",
    t.device == 0 ? "" : "device = $(t.device), ", "precision = :", t.precision, t.online ? "" : ", online = false",
    t.rounded_backward ? "" : ", rounded_backward = false", t.deterministic ? ", deterministic = true" : "", ")")

"The implementations a target uses beside the floor, in order."
implementations(target::Union{CPUTarget, CUDATarget}) = target.implementations
implementations(::Target) = ()

"""
    target_for(x) -> target or nothing

The target an argument selects: `CPUTarget()` for an `Array`, a package's
target for its arrays, on the device that holds them, nothing for anything
else. A program is prepared on the first target among its inputs.
"""
target_for(x::AbstractArray) = CPUTarget()
target_for(x) = nothing

"""
    device_of(x) -> Int or nothing

The device whose memory holds `x`, an array or a target's storage, by the
number its package gives it; nothing for the host's memory, the CPU's. A
target reads arrays on the host, copying them where they cannot be its
storage, and the CPU reads a device's arrays so. A program whose arrays
are on several devices is planned with a device for each value and a copy
where a value is read on another (see `plans_across_devices`); on a target
that does not plan so, its arrays and the target are on one device at
most, beside the host, and a call that brings arrays of another device
than its target's throws (see `point!`).
"""
device_of(x) = nothing
device_of(target::CUDATarget) = target.device

"""
Whether a target plans and runs a program whose arrays are on several
devices: the CPU, whose devices are labels on the host's memory, a crossing
between two a copy there, and CUDA's, a crossing a copy staged through the
host's memory. A target that does not refuses such arrays.
"""
plans_across_devices(::Target) = false
plans_across_devices(::Union{CPUTarget, CUDATarget}) = true

"""
Where a preparation's arguments are: the label of each argument's device
(see `with_devices`), a fact of the plan, and the device each label stands
for, as `device_of` numbers it, nothing for the host's memory, a fact of
the arrays. Label 1 is home: the target's device where it has one, else
the first device an argument is on, else the host's memory. Arguments on
the host and values no placed input reaches are there, and the decider
builds and times there. The devices the other arguments are on follow in
the order the arguments first meet them, so that a plan and its decisions
do not depend on which of a machine's devices hold the arrays.
"""
struct Placement
    labels::Vector{Int}
    devices::Vector{Union{Nothing, Int}}
end

# Every argument of `program` at home on `target`.
Placement(target::Target, program) = Placement(ones(Int, length(program.arguments)), Union{Nothing, Int}[device_of(target)])

# Where the arguments of `program` among `inputs` are, on `target`.
function argument_devices(target::Target, program, @nospecialize(inputs))
    found = map(device_of, argument_leaves(program, inputs))
    devices = Union{Nothing, Int}[device_of(target)]
    append!(devices, filter(!isnothing, found))
    unique!(devices)
    # Without a device of its own the target is at home where the first
    # argument on a device is, and on the host when none is.
    length(devices) > 1 && devices[1] === nothing && popfirst!(devices)
    refuse_kinds(target, devices)
    Placement([d === nothing ? 1 : findfirst(isequal(d), devices) for d in found], devices)
end

# Throw where `devices` of `target` are of more than one kind: a decider
# times its claims at home, and a time is the kernel's on every device of
# the kind alone.
function refuse_kinds(target::Target, devices)
    names = [device_name(target, d) for d in devices]
    allequal(names) && return
    held = join([string(device_named(d), " is ", repr(n)) for (d, n) in zip(devices, names)], ", ")
    throw(ArgumentError(string("the arrays are on devices of more than one kind: ", held,
        "; a program is planned over devices of one kind, whose kernels are timed on one of them")))
end

"""
    on_device(f, target, device = device_of(target))

`f()` with the calling task on `device` of `target`'s kind, its state on the
device it was on (for CUDA, its context and stream) as it was when `f`
returns or throws: what touches a device, building, binding and running a
program, runs on its own, whatever device the caller is on. An executable's
steps are put on theirs by `walk` alone. A target of no devices, as the CPU,
runs `f()` as it is.
"""
on_device(f, target::Target) = on_device(f, target, device_of(target))
on_device(f, ::Target, device) = f()

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
    target.rounded_backward ? "" : "-exact-backward", target.deterministic ? "-deterministic" : "")

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

"""
    upload!(target, buffer, x) -> buffer

Copy the array `x`, which is not storage on `target`, into `buffer`, which
is: how a caller's array that cannot be an argument's storage gets into the
buffer that is (see `bind!` and `point!`).
"""
upload!(::Target, buffer, x) = copyto!(buffer, x)

"""
    download!(target, x, buffer) -> x

Copy `buffer`, storage on `target`, into the array `x`, which is not: how
what a program wrote into an argument gets back into a caller's array that
could not be its storage (see `point!`).
"""
download!(::Target, x, buffer) = copyto!(x, buffer)

"Whether array `x` can be storage on `target`, bound by reference."
bindable(::Target, x) = false
bindable(::CPUTarget, x) = x isa Array

"""
    pointable(target, x, at) -> Bool

Whether launches built on storage `target` allocated can read array `x`,
whose first element is at address `at`, where it is instead (see
`repoint!`): `x` is storage the target could have allocated, as aligned as
its kernels assume of what it allocates.
"""
pointable(::Target, x, at) = false
pointable(::CPUTarget, x, at) = x isa Array

"""
    task_state(target) -> function

A function that, called first in a new task, gives the task the state on
`target` of the task that asked for it (for CUDA, its context and stream),
and none where that task has none: a preparation that builds nothing asks
implementations side by side what they offer (see `side_by_side`), and
touches no device, as while a package precompiles without one.
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
        return HostLaunch(f, (args, k.items), k.slots)
    end
    k.split.lanes == 1 || throw(ArgumentError("the CPU runs each part of a split item as one lane, not $(k.split.lanes)"))
    f = generated(:(function (buffers, n, parts)
        lane, lanes = 0, 1
        for lin in 0:(n - 1), part in 0:(parts - 1)
            $(k.body)
        end
        nothing
    end))
    HostLaunch(f, (args, k.items, k.split.parts), k.slots)
end

"""
A floor kernel's launch on the CPU. Each kernel is a new function, so its
launch holds the function and arguments untyped: a closure over them would be
a new type too, and every method it passes through would compile again for it.
Its first argument is the kernel's buffers, the first of them those of the
values it reads and stores, in `values`.
"""
mutable struct HostLaunch
    const f::Any
    arguments::Tuple
    const values::Vector{Int}
end

(l::HostLaunch)() = Base.invokelatest(l.f, l.arguments...)

repoints(::HostLaunch) = true

function repoint!(l::HostLaunch, value, buffer)
    buffers = foldl((b, j) -> Base.setindex(b, buffer, j), findall(==(value), l.values); init = first(l.arguments)::Tuple)
    l.arguments = (buffers, Base.tail(l.arguments)...)
    l
end

"""
    floor_split(target, items, positions, adjacent) -> Split or nothing

How the floor folds `positions` positions for each of `items` items on
`target`, neighbouring items neighbours in memory when `adjacent`: nothing for
one lane per item, which folds them in order, as on the CPU; or a `Split`,
which deals them among lanes. A target that splits runs a part's lanes
together and merges their states with `merge_lanes`.
"""
floor_split(::Target, items, positions, adjacent) = nothing

## Steps.

"""
    Step(run, device, from = device)

A step of an execution: `run()`, a kernel's launch or a copy, and the device
it runs on (see `device_of`), where `walk` puts it; and the device it reads
from, another only for a crossing, a copy from `from` to `device` (see
`crossing`).
"""
struct Step
    run::Any
    device::Union{Nothing, Int}
    from::Union{Nothing, Int}
end

Step(run, device) = Step(run, device, device)

"""
    crosses(step) -> Bool

Whether `step` copies between two devices, a crossing's (see `crossing`):
a target with graphs records a segment of other steps as one device's
graph, and says how a segment of crossings runs (see `capture`).
"""
crosses(s::Step) = s.from != s.device

# The last of the steps from `start` on that run where it does: on one
# device, or crossing between the same two.
function run_end(steps, start)
    stop = start
    while stop < length(steps) && (steps[stop + 1].device, steps[stop + 1].from) == (steps[start].device, steps[start].from)
        stop += 1
    end
    stop
end

"""
    walk(f, target, steps)

`f(step)` for each of `steps` in order, the calling task on the step's
device: the one place an execution's steps are put on their devices, as
they run (`execute!`). Consecutive steps on one device, or consecutive
crossings between the same two, are a run, walked in one entry of its
device (see `on_device`) after the runs before it on another are ordered
before it (see `follow`), so an execution on one device, as a program
planned on one is, enters it once. A recording takes the same runs as its
segments (see `Captured`).
"""
function walk(f, target::Target, steps)
    start = 1
    while start <= length(steps)
        stop = run_end(steps, start)
        device = steps[start].device
        start > 1 && steps[start - 1].device != device && follow(target, steps[start - 1].device, device)
        on_device(() -> foreach(k -> f(steps[k]), start:stop), target, device)
        start = stop + 1
    end
    nothing
end

# Run step `s` where the task is.
run_step(s::Step) = (s.run(); nothing)

"""
    follow(target, before, after)

Order the work launched next on device `after` after what was launched on
device `before`: called between consecutive runs of an execution's steps
on two devices (see `walk`), and between segments of a recording on two
(see `Captured`), with the task where the caller has it. A target whose
devices run apart orders them here, by an event recorded on `before`'s
stream and waited for on `after`'s, or by waiting for `before`. A
crossing's own copy is ordered after what wrote its source either way. The
CPU runs its steps in order, and does nothing.
"""
follow(::Target, before, after) = nothing

"""
    written(target, x)

Make `x`, a result of the call that has just run, storage on the device the
task is on, read after the call by whatever reads it next, on any stream
or task: called as each execution and replay ends. A target whose launches
its arrays' package does not see says here that the launches just made on
the task's stream wrote `x`. The CPU's run in order, and it does nothing.
"""
written(::Target, x) = nothing

"""
    crossing(target, to, from, values) -> run

A step's run that copies buffer `from`, storage on one of `target`'s
devices, into buffer `to`, storage on another, as large: how a value read
on another device than its own gets there (see `Crossed`). `values` names
the two buffers, `to_value => from_value`, for the run to be pointed
elsewhere as a `Copy` is (see `repoint!`). It is built with the task on
`to`'s device; each buffer knows its own. A target with graphs says, as it
records a segment of crossings (see `capture`), whether the crossing is
recorded or runs as the segment is replayed. On the CPU, whose devices are
labels on the host's memory, it is a copy on the host.
"""
crossing(::CPUTarget, to, from, values::Pair{Int, Int}) = Copy(to, from, values)

## Graphs.

"""
    capture(target, steps, keep) -> graph

`steps`, each a `Step` that launches work on `target`, all on one device or
all crossings between the same two (a segment of a `Captured`), captured
in order as one graph for `replay(target, graph)` to launch again;
`retarget!` changes the addresses one step's work reads. Tempest asks with
the task on the steps' device, where the graph is made, launched and
changed. The graph keeps `keep`, what it launches, alive until
`release(target, graph)` or the collector destroys it. A target without
graphs replays the steps themselves.
"""
capture(::Target, steps, keep) = steps

"""
    recapture(target, graph, steps, keep) -> graph

`graph`, which `capture` recorded from `steps`, recorded again after their
launches were pointed at other storage (see `repoint!`). A target with graphs
may update the graph it has rather than make another.
"""
recapture(target::Target, graph, steps, keep) = (release(target, graph); capture(target, steps, keep))

"""
    retarget!(target, graph, steps, moves, repoint) -> Bool

Make what `graph` recorded of a launch, `steps` among the steps `capture`
took, read where `moves` say its values moved, each a named tuple of the
value, its old and new addresses and its bytes, `(value, from, to, bytes)`.
`repoint()` points the launch at the new storage (see `repoint!`), and is
called once, by the target or after it: a target may record the launch
alone before and after, to learn where its recording holds an address.
False when the graph cannot be changed so, and must be recorded again
(`recapture`). The steps themselves, a target's graph without graphs, read
where their launches read: true.
"""
retarget!(::Target, graph, steps, moves, repoint) = true

"Launch `graph`, what `capture` recorded on `target`."
replay(target::Target, steps) = walk(run_step, target, steps)

"Destroy `graph`, what `capture` recorded on `target`, once nothing replays it."
release(::Target, graph) = nothing

"""
A run of a recording's steps on one device, or of crossings between the
same two, as `walk` forms them: its device, where its graph is made,
launched and changed, the positions of its steps among the recording's,
and the graph the target captured of them (see `capture`), recorded again
in place where its launches move (see `recapture`).
"""
mutable struct Segment
    const device::Union{Nothing, Int}
    const steps::UnitRange{Int}
    graph::Any
end

"""
What `capture!` records of an executable: the steps of the executions a
replay runs, and a segment per run of them, each captured as its device's
own graph, a target's graph being one device's. A replay launches the
segments in order, each on its device after the one before it is ordered
before it (see `follow`). An execution on one device is one segment, its
graph the target's capture of every step.
"""
struct Captured
    steps::Vector{Step}
    segments::Vector{Segment}
end

# `steps` recorded on `target`, a segment per run, each graph keeping `keep` alive.
function record(target::Target, steps::Vector{Step}, keep)
    segments = Segment[]
    start = 1
    while start <= length(steps)
        stop = run_end(steps, start)
        device = steps[start].device
        push!(segments, Segment(device, start:stop, on_device(() -> capture(target, steps[start:stop], keep), target, device)))
        start = stop + 1
    end
    Captured(steps, segments)
end

# Launch what `r` recorded, segment by segment.
function replay_recording(target::Target, r::Captured)
    for (k, s) in enumerate(r.segments)
        k > 1 && r.segments[k - 1].device != s.device && follow(target, r.segments[k - 1].device, s.device)
        on_device(() -> replay(target, s.graph), target, s.device)
    end
    nothing
end

# Segment `s` of what `r` recorded, recorded again after its launches moved.
recapture_segment!(target::Target, r::Captured, s::Segment, keep) =
    s.graph = on_device(() -> recapture(target, s.graph, r.steps[s.steps], keep), target, s.device)

# Destroy every segment's graph of what `r` recorded.
release_recording(target::Target, r::Captured) = foreach(s -> on_device(() -> release(target, s.graph), target, s.device), r.segments)

# The segment that holds position `j` among what `r` recorded.
segment_at(r::Captured, j) = findlast(s -> first(s.steps) <= j, r.segments)

## Measuring.

"""
    device_name(target[, device])

The name of the device `target` runs on, which names its tuning record and
its rates file; or of another of its devices, as `device_of` numbers them,
which a program placed over several runs on (see `Placement`). A target's
devices are of one kind unless it says otherwise.
"""
device_name(::CPUTarget) = "cpu"
device_name(target::Target, device) = device_name(target)

"""
    toolchain(target) -> Vector{Pair{String, String}}

What builds the kernels `target` runs, each part by name and version, which
names its tuning record and its rates file with its device: what they hold
are times of kernels built so, and a kernel another toolchain builds is
another kernel to time. Nothing for a target that builds none, as the CPU.
"""
toolchain(::Target) = Pair{String, String}[]

"Whether claims on `target` can be timed at preparation; `time_launch` times one."
measurable(::Target) = false

"""
    time_launch(target, launch) -> seconds::Float64

The seconds one call of `launch()` takes on the device among the launches of
a captured graph, its data in memory rather than in the cache, as the cover
times claims: a step reads its weights once, and finds them in memory. Only
a `measurable` target defines it. Tempest calls it one timing at a time,
once the device's work is done, so what it keeps between timings needs no
lock.
"""
function time_launch end

"""
    quiesce(target)

Wait until the device has run everything launched on it, by any task.
Every timing waits so before it times a claim (see `timed`).
"""
quiesce(::Target) = nothing

# Timings run one at a time in the process: two at once, two preparations'
# on two tasks, would time each other. Nothing else is held off while a
# claim is timed: a compiled program's search has its timings taken between
# the program's own calls (see `Search`), and other work in the process runs
# beside a timing as it does beside any preparation.
const TIMINGS = ReentrantLock()

# `time_launch`, one timing at a time, once the device's work is done: a
# decider's timer unless it is given another (see `decide!`).
timed(target, launch) = @lock TIMINGS (quiesce(target); time_launch(target, launch))

"""
    device_rates(target) -> rates

The device's rates as it reports them (see `Rates`): a named tuple of
`launch`, the seconds between launches in a captured graph; `cached` and
`bandwidth`, bytes per second for a launch that moves what the cache holds
and for one that moves more, each from memory; `cache`, its bytes; and
`blocks`, the blocks that fill the device. They are not measured: off by
twice or half, they change no plan of the corpus's, and the rates that do,
each package's products and each scalar function's seconds, are fitted on
first need. Only a `measurable` target defines it.
"""
function device_rates end
