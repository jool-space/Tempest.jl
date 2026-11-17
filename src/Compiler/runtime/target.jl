# A target: where a program runs. The compiler asks a target for buffers, for
# the floor's launches, for captured graphs, for the device's name, rates and
# times, and for the options it was made with (the implementations it uses,
# its numerics, whether it permits rewrites that change rounding, and how it
# classes products), through the protocol below: the public names of this
# file, `implementations`, `product_class`, and `ScalarKernel`, `Split`,
# `accumulate!` and `merge_lanes` of the floor's code. The CPU's definitions are here; a package that adds a
# target defines them for its type (see CUDAExt).

public Target, CPUTarget, CUDATarget, target_for, device_of, default_device, home_device, at_home, on_device, Step, crosses, taken, crossing, stream, lane, link, on_stream, task_stream, event, record!, wait!, Scheduled, allocate, upload, upload!, download!, bindable, pointable, retarget!, task_state, device_name,
    toolchain, measurable, time_launch, time_crossing, quiesce, device_rates, capture, recapture, replay, release, floor_launch, floor_split, ScalarKernel, Split,
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
    CUDATarget(implementations...; device = nothing, precision = :full, online = true, rounded_backward = true, deterministic = false)

Run on NVIDIA GPUs; needs CUDACore loaded. A program runs where its arrays
are, whatever device the calling task is on, and a target is at home (see
`Placement`) on the first device an argument is on, or, with every array on
the host, on the calling task's device. `device`, as CUDA numbers them from
0, names the home instead: where arrays on the host go, and where claims
are timed. The target uses the floor and
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
    # The home device's number, as CUDA counts them from 0; nothing where
    # the arrays say (see `at_home`).
    device::Union{Nothing, Int}
end

CUDATarget(precision::Symbol, online::Bool, implementations::Tuple) = CUDATarget(precision, online, implementations, true, false, nothing)

function CUDATarget(implementations::Implementation...; device::Union{Nothing, Integer} = nothing, precision::Symbol = :full, online::Bool = true,
        rounded_backward::Bool = true, deterministic::Bool = false)
    precision in (:full, :tf32, :tf32x3) || throw(ArgumentError("precision must be :full, :tf32 or :tf32x3, got $precision"))
    CUDATarget(precision, online, isempty(implementations) ? loaded_implementations() : implementations, rounded_backward, deterministic, device)
end

Base.show(io::IO, t::CUDATarget) = print(io, "CUDATarget(", join(map(repr, t.implementations), ", "), "; ",
    t.device === nothing ? "" : "device = $(t.device), ", "precision = :", t.precision, t.online ? "" : ", online = false",
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
between two a copy there, and CUDA's, a crossing a copy over the link two
devices share, or staged through the host's memory where they share none. A
target that does not refuses such arrays.
"""
plans_across_devices(::Target) = false
plans_across_devices(::Union{CPUTarget, CUDATarget}) = true

"""
Where a preparation's arguments are: the label of each argument's device
(see `with_devices`), a fact of the plan, and the device each label stands
for, as `device_of` numbers it, nothing for the host's memory, a fact of
the arrays. Label 1 is home: the target's device where it names one, else
the first device an argument is on, else the target's default (see
`default_device`). Arguments on
the host and values no placed input reaches are there, and the decider
builds and times there. The devices the other arguments are on follow in
the order the arguments first meet them, so that a plan does not depend
on which of a machine's devices hold the arrays; its decisions do through
the price of each crossing, which is the link's between the two devices
its labels stand for (see `link!`).
"""
struct Placement
    labels::Vector{Int}
    devices::Vector{Union{Nothing, Int}}
end

# Every argument of `program` at home on `target`.
Placement(target::Target, program) = Placement(ones(Int, length(program.arguments)), Union{Nothing, Int}[home_device(target)])

"""
    default_device(target) -> Int or nothing

Where a target that names no device is at home when no argument is on one:
the host's memory for the CPU, the calling task's device for CUDA.
"""
default_device(::Target) = nothing

# The device a target names, else its default.
function home_device(target::Target)
    d = device_of(target)
    d === nothing ? default_device(target) : d
end

"""
    at_home(target, device) -> target

`target` at home on `device`: what a preparation builds and times with,
once its arguments have said where home is.
"""
at_home(target::Target, device) = target
at_home(t::CUDATarget, device::Int) = t.device == device ? t :
    CUDATarget(t.precision, t.online, t.implementations, t.rounded_backward, t.deterministic, device)

# Where the arguments of `program` among `inputs` are, on `target`.
function argument_devices(target::Target, program, @nospecialize(inputs))
    found = map(device_of, argument_leaves(program, inputs))
    devices = Union{Nothing, Int}[device_of(target)]
    append!(devices, filter(!isnothing, found))
    unique!(devices)
    # Naming no device, the target is at home where the first argument on a
    # device is, and at its default when none is.
    length(devices) > 1 && devices[1] === nothing && popfirst!(devices)
    devices[1] === nothing && (devices[1] = default_device(target))
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
    on_device(f, target, device = home_device(target))

`f()` with the calling task on `device` of `target`'s kind, its state on the
device it was on (for CUDA, its context and stream) as it was when `f`
returns or throws: what touches a device, building, binding and running a
program, runs on its own, whatever device the caller is on. An executable's
steps are put on theirs by `walk` alone. A target of no devices, as the CPU,
runs `f()` as it is.
"""
on_device(f, target::Target) = on_device(f, target, home_device(target))
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
it runs on (see `device_of`); and the device it reads from, another only for
a crossing, a copy from `from` to `device` (see `crossing`). The two name
the stream it runs on: a device's own (see `stream`), or the lane of the
ordered pair of devices a crossing copies between (see `lane`).
"""
struct Step
    run::Any
    device::Union{Nothing, Int}
    from::Union{Nothing, Int}
end

Step(run, device) = Step(run, device, device)

"""
    crosses(step) -> Bool

Whether `step` copies between two devices, a crossing's (see `crossing`),
on the lane of the two rather than on a device's own stream: a target with
graphs says how a run of crossings is recorded (see `capture`).
"""
crosses(s::Step) = s.from != s.device

"""
    taken(target, x)

Take `x`, an array a call shares with its caller, a result or an argument
bound by reference, for the launches the call is about to make, with the
task on the array's device: the launches read and write it by its address,
unseen by the package whose array it is. What other streams still have to
run on `x` is waited for, and whatever reads it after the call, on any
stream or task, waits for the launches. A target whose arrays' package
orders an array's uses by the stream each was submitted on takes the
array's address here, which that package counts as the calling task's use
of it from then on, until another stream's. The CPU's run in order, and it
does nothing.
"""
taken(::Target, x) = nothing

"""
    crossing(target, to, from, values, offset) -> run

A step's run that copies into buffer `to`, storage on another of `target`'s
devices, as many elements of buffer `from`, storage on one, as `to` holds,
those after the first `offset`: how a value read on another device than its
own gets there (see `Crossed`), all of it at offset 0, or a span of it,
which is all its readers there read (see `storage_span`). `values` names
the two buffers, `to_value => from_value`, for the run to be pointed
elsewhere as a `Copy` is (see `repoint!`); the offset is the run's own,
where in the source value's storage the span starts, kept as either buffer
is replaced. It is built with the task on
`to`'s device, and runs where its lane puts the task (see `on_stream`); each
buffer knows its own device. Its schedule orders it after what wrote its
source and before what reads its copy (see `Schedule`). A target with graphs
says, as it records a run of crossings (see `capture`), whether the crossing
is recorded or runs as the run is replayed. On the CPU, whose devices are
labels on the host's memory, it is a copy on the host.
"""
crossing(::CPUTarget, to, from, values::Pair{Int, Int}, offset::Int) = Copy(to, from, values, offset)

## Streams.

"""
    stream(target, device) -> stream

What a device's own steps run on (see `stream_of`), its kernels and copies
in order. Made once for each schedule, with the task on the device. By
default the pair `(device, device)`, whose steps run with the task on the
device (see `on_stream`).
"""
stream(::Target, device) = (device, device)

"""
    lane(target, from, to, out) -> stream

What the crossings from device `from` to device `to` run on (see
`stream_of`), in order: the lane of the ordered pair, which leaves `from`
by `out`, its link (see `link`). Made once for each schedule, with the task
on `to`'s device. By default the pair `(from, to)`, whose steps run with
the task on `to` (see `on_stream`).
"""
lane(::Target, from, to, out) = (from, to)

"""
    link(target, device) -> link

What the crossings leaving `device` go out by, every lane from it the same
(see `lane`): it takes them one at a time, in the order they are
launched, which is the order their sources are written (see `walk` and
`run_order`), so that none waits behind one whose source is not. Made once
for each schedule, for each device a crossing leaves, with the task on it.
By default nothing: the CPU's crossings are copies on the host, which run
as they are launched.
"""
link(::Target, device) = nothing

"""
    on_stream(f, target, stream)

`f()` with the calling task where the steps of `stream` run, as `on_device`
puts it on a device. By default on the stream's last device: a device's
own, or the reader's of a lane, where a crossing's run is built and expects
to run (see `crossing`). A target whose lanes are streams of their own
enters them here.
"""
on_stream(f, target::Target, stream) = on_device(f, target, last(stream))

"""
    task_stream(target) -> stream

The stream the calling task's launches run on, on the device the task is on
(see `on_device`): what an executable keeps of its last use on each device,
and compares with the next (see `taken!`). On a target whose steps have run
as they return, the CPU's, `nothing`.
"""
task_stream(::Target) = nothing

"""
    quiesce(target, stream)

Wait until `stream`, a task's as `task_stream` gave it, has run everything
that task launched on it, with the calling task on the stream's device: a
task that uses an executable another stream used last waits so (see
`taken!`). The CPU's steps have run as they return, and it does nothing.
"""
quiesce(::Target, stream) = nothing

"""
    event(target, stream) -> event

What a step on `stream` records once it has run, for steps on other streams
to wait for (see `record!` and `wait!`), made with the task where the
stream's steps run. A target whose steps run in the order they are
launched, the CPU's, has none: `nothing`.
"""
event(::Target, stream) = nothing

"""
    record!(target, stream, event)

Record `event` on `stream`, after what was launched on it so far, with the
task where the stream's steps run (see `on_stream`): after a step that a
step on another stream waits for, and as a call begins and ends (see
`walk`). Recorded into a graph (see `capture`), each launch of the graph
records it again.
"""
record!(::Target, stream, event) = nothing

"""
    wait!(target, stream, event)

Make what is launched on `stream` from now on wait for what was launched
before the last `record!` of `event`, with the task where the stream's
steps run (see `on_stream`): before a step that waits for a step on another
stream, and as a call begins and ends (see `walk`). Recorded into a graph,
the wait is for the last record launched before the graph.
"""
wait!(::Target, stream, event) = nothing

## Graphs.

"""
    capture(target, steps, keep) -> graph

`steps`, each a `Scheduled` step that launches work on `target`, all on one
stream (a run of a `Schedule`), captured in order, their waits and records
with them, as one graph for `replay(target, graph)` to launch again;
`retarget!` changes the addresses one step's work reads. Tempest asks with
the task where the stream's steps run (see `on_stream`), where the graph is
made, launched and changed. The graph keeps `keep`, what it launches, alive
until `release(target, graph)` or the collector destroys it. A target
without graphs replays the steps themselves.
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
replay(target::Target, steps) = foreach(s -> run!(target, s), steps)

"Destroy `graph`, what `capture` recorded on `target`, once nothing replays it."
release(::Target, graph) = nothing

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
    time_crossing(target, from, to, bytes) -> seconds::Float64

The seconds a crossing of `bytes` from device `from` to device `to` of
`target` takes (see `crossing`), run alone on the lane of the two (see
`lane`), from when the lane may start to when what reads the copy may:
what a decider fits the link between them on (see `link!`). Only a
`measurable` target that plans across devices defines it. Tempest calls it
one timing at a time, once both devices' work is done.
"""
function time_crossing end

# `time_crossing`, one timing at a time, once the work of both devices is done.
timed_crossing(target, from, to, bytes) =
    @lock TIMINGS (quiesce(at_home(target, from)); quiesce(at_home(target, to)); time_crossing(target, from, to, bytes))

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
