# The CUDA target: buffers are device arrays, and the generated floor runs one
# thread per item, or a block per part of an item it splits.
module CUDAExt

using Tempest
using CUDACore
import Tempest: CUDATarget

Tempest.Compiler.target_for(::CuArray) = CUDATarget()
Tempest.Compiler.allocate(::CUDATarget, T, n) = CuArray{T}(undef, n)
# A copy of `x` on the device, filled rather than constructed from `x`:
# CuArray's constructor from an array dispatches among every package's
# constructor methods, so code that reached it, the cover entire, was compiled
# again in each session where cuBLAS added its own (a Diagonal's).
Tempest.Compiler.upload(::CUDATarget, x) = copyto!(CuArray{eltype(x)}(undef, size(x)), x)
Tempest.Compiler.bindable(::CUDATarget, x) = x isa CuArray

function Tempest.Compiler.task_state(::CUDATarget)
    context, stream = CUDACore.context(), CUDACore.stream()
    () -> (CUDACore.context!(context); CUDACore.stream!(stream))
end

const THREADS = 256

Tempest.Compiler.device_name(::CUDATarget) = CUDACore.name(CUDACore.device())

Tempest.Compiler.measurable(::CUDATarget) = true

# A captured graph. The driver crashes destroying a graph whose kernels'
# modules were unloaded first, and objects that die in one collection are
# finalized in no order we choose: so what a graph launches stays alive in
# `GRAPHS` until the graph's own finalizer has destroyed it, and graphs alive
# at exit are destroyed before finalizers run.
mutable struct Graph
    exec::CUDACore.CuGraphExec
end

const GRAPHS = WeakKeyDict{Graph, Any}()

function Tempest.Compiler.capture(::CUDATarget, run, keep)
    graph = CUDACore.capture(run)
    exec = CUDACore.instantiate(graph)
    # The executable graph does not need the graph it was made from.
    finalize(graph)
    g = finalizer(g -> finalize(g.exec), Graph(exec))
    GRAPHS[g] = keep
    g
end

Tempest.Compiler.replay(::CUDATarget, g::Graph) = CUDACore.launch(g.exec)
Tempest.Compiler.release(::CUDATarget, g::Graph) = finalize(g)

# Iterated rather than collected: `collect` sizes its result first, and a
# graph the collector freed meanwhile, which iteration skips, left a slot
# of it undefined.
__init__() = atexit(() -> foreach(finalize, keys(GRAPHS)))

# Device seconds of one launch among a captured graph's, its data in memory
# rather than in the cache, as a step finds the weights it reads once: the
# minimum over samples of its graph replayed alone after the cache is
# cleared, less what timing a replay alone adds (`replay_overhead`).
# Replayed back to back instead, as it was, a launch whose data fit the
# cache was timed from it: on GB10, a decoding step's output projection, 21
# MB of weights, at 25 μs where reading them from memory takes 93, and its
# tiles were ranked at the cache's rate.
function Tempest.Compiler.time_launch(target::CUDATarget, launch)
    launch()
    CUDACore.synchronize()
    once = CUDACore.@elapsed launch()
    g = Tempest.Compiler.capture(target, () -> launch(), launch)
    t = cleared(target, g; samples = clamp(round(Int, 1.0e-2 / max(once, 1.0e-6)), 3, 20))
    Tempest.Compiler.release(target, g)
    t - replay_overhead(target)
end

# The minimum over samples of graph `g` replayed after the cache is cleared,
# each between events, all enqueued before waiting once, as Triton's
# `do_bench` times them.
function cleared(target, g; samples = 20)
    events = [(CUDACore.CuEvent(), CUDACore.CuEvent()) for _ in 1:samples]
    for (start, stop) in events
        clear_cache!()
        CUDACore.record(start)
        Tempest.Compiler.replay(target, g)
        CUDACore.record(stop)
    end
    CUDACore.synchronize(last(last(events)))
    minimum(((start, stop),) -> CUDACore.elapsed(start, stop), events)
end

# Read the `n` words of `a`, storing their fold only where zeros could not
# give it, so that the reads are kept. It has a floor kernel's arguments, so
# that it is compiled once and kept in the kernel cache (`floor_function`),
# not compiled in each session, which cost the first timing 6 s.
function read_through((a, sink), n)
    i = (CUDACore.blockIdx().x - 1) * CUDACore.blockDim().x + CUDACore.threadIdx().x
    stride = CUDACore.gridDim().x * CUDACore.blockDim().x
    s = UInt32(0)
    while i <= n
        s ⊻= @inbounds a[i]
        i += stride
    end
    s == typemax(UInt32) && (@inbounds sink[1] = s)
    return
end

# The launch that clears the cache and its buffers, made on first need.
const CLEARING = Ref{Any}(nothing)

# Clear the cache, reading a buffer twice its size through it. Nvbench and
# Triton's `do_bench` clear it by writing one, which leaves it dirty: a
# launch is then timed writing it back, the output projection at 160 μs.
function clear_cache!()
    if CLEARING[] === nothing
        bytes = CUDACore.attribute(CUDACore.device(), CUDACore.DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
        a, sink = CUDACore.zeros(UInt32, 2bytes ÷ 4), CUDACore.zeros(UInt32, 1)
        args = CUDACore.cudaconvert((a, sink))
        CLEARING[] = (FloorLaunch(floor_function(read_through, (args, length(a)))..., (args, length(a)), 1024, 256), a, sink)
    end
    first(CLEARING[])()
end

# What timing one replay alone adds to the seconds of its launch among a
# graph's: an empty kernel's replay timed so, 2.5 μs on GB10, less its
# seconds in a graph of a millisecond of them, 0.43. Measured once a
# session.
const OVERHEAD = Ref(NaN)

function replay_overhead(target)
    isnan(OVERHEAD[]) || return OVERHEAD[]
    empty = blank_launch()
    g = Tempest.Compiler.capture(target, () -> empty(), empty)
    alone = cleared(target, g)
    Tempest.Compiler.release(target, g)
    OVERHEAD[] = alone - batched(target, empty)
end

# Device seconds of one launch in a captured graph of about a millisecond of
# them, replayed: the minimum over samples.
function batched(target, launch; samples = 5)
    launch()
    CUDACore.synchronize()
    once = CUDACore.@elapsed launch()
    repetitions = clamp(round(Int, 1.0e-3 / max(once, 1.0e-6)), 1, 100)
    g = Tempest.Compiler.capture(target, () -> foreach(_ -> launch(), 1:repetitions), launch)
    Tempest.Compiler.replay(target, g)
    t = minimum(_ -> CUDACore.@elapsed(Tempest.Compiler.replay(target, g)), 1:samples) / repetitions
    Tempest.Compiler.release(target, g)
    t
end

# Nothing, launched to time the gap between launches; a floor kernel, as
# `read_through` is.
blank(_, _) = nothing

blank_launch() = FloorLaunch(floor_function(blank, ((), 1))..., ((), 1), 1, 1)

# The gap between launches of an empty kernel in a captured graph, copies of
# 4 MiB (what the cache holds) and 256 MiB (more), from memory, the cache's
# size and the multiprocessors.
function Tempest.Compiler.calibrate(target::CUDATarget)
    device = CUDACore.device()
    copying(n) = (a = CUDACore.zeros(UInt8, n); b = similar(a); 2n / Tempest.Compiler.time_launch(target, () -> copyto!(b, a)))
    (launch = batched(target, blank_launch()),
     cached = copying(4 << 20), bandwidth = copying(256 << 20),
     cache = Float64(CUDACore.attribute(device, CUDACore.DEVICE_ATTRIBUTE_L2_CACHE_SIZE)),
     blocks = Float64(CUDACore.attribute(device, CUDACore.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)))
end

# A fold by key's items run at once: a sum's add atomically.
Tempest.Compiler.accumulate!(::Tempest.Compiler.IR.SumFold, buffer::CUDACore.CuDeviceArray, i, v) = (CUDACore.@atomic buffer[i] += v; nothing)

function Tempest.Compiler.floor_launch(::CUDATarget, k::Tempest.Compiler.ScalarKernel, args)
    k.split === nothing || return split_launch(k, args)
    f = Tempest.Compiler.generated(:(function (buffers, n)
        lin = ($(CUDACore.blockIdx)().x - 1) * $(CUDACore.blockDim)().x + $(CUDACore.threadIdx)().x - 1
        lin < n || return
        $(k.body)
        return
    end))
    args = CUDACore.cudaconvert(args)
    threads = min(THREADS, k.items)
    FloorLaunch(floor_function(f, (args, k.items))..., (args, k.items), cld(k.items, threads), threads)
end

# A split kernel's launch: a block per part of `side` neighbouring items, a
# thread per lane of each, the items along the block's first axis, so that a
# warp's threads read neighbouring elements whether the items or the positions
# are neighbours. No thread returns early, since a part's lanes merge their
# states together.
function split_launch(k, args)
    f = Tempest.Compiler.generated(:(function (buffers, n, parts)
        block, part = divrem(Int($(CUDACore.blockIdx)().x) - 1, parts)
        lin = block * Int($(CUDACore.blockDim)().x) + Int($(CUDACore.threadIdx)().x) - 1
        lane, lanes = Int($(CUDACore.threadIdx)().y) - 1, Int($(CUDACore.blockDim)().y)
        $(k.body)
        return
    end))
    args = CUDACore.cudaconvert(args)
    arguments = (args, k.items, k.split.parts)
    FloorLaunch(floor_function(f, arguments)..., arguments, (k.items ÷ k.split.side) * k.split.parts, (k.split.side, k.split.lanes))
end

# Lanes that keep a multiprocessor busy, by GB10's measure: a fold split over
# fewer moves its bytes more slowly, and over more merges more parts.
const RESIDENT_LANES = 1024

# Positions an item's thread folds alone however few the items: a split's
# merges and its launch per phase cost more than so short a sweep, the 32
# positions of each of 8192 columns 18 μs where a thread per column takes 4.
const SHORT_FOLD = 64

# Parts an item splits into at most, so that the launch after merges their
# states in a few loads per lane.
const MAXIMUM_PARTS = 1024

# Items a block holds side by side at most: a warp's, whose threads then read
# neighbouring items' elements at once.
const SIDE = 32

# The multiprocessors of each device, asked once.
const MULTIPROCESSORS = Dict{CUDACore.CuDevice, Int}()

multiprocessors() = get!(() -> CUDACore.attribute(CUDACore.device(), CUDACore.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT), MULTIPROCESSORS, CUDACore.device())

# Items too few to keep the device busy, one thread each, have their positions
# dealt among the lanes that would: as many as they have positions to fold
# four or more each, a part's lanes a power of two, so that they merge by
# halves, and a block of at most `THREADS`. Neighbouring items in memory are
# side by side in a block, as many of a warp's as divide them, so that a warp
# reads neighbouring elements; otherwise a block is one item's part, a warp of
# lanes at least, which read neighbouring positions.
function Tempest.Compiler.floor_split(::CUDATarget, items, positions, adjacent)
    resident = RESIDENT_LANES * multiprocessors()
    (items == 0 || items >= resident || positions <= SHORT_FOLD) && return nothing
    wanted = min(cld(resident, items), cld(positions, 4))
    side = adjacent ? gcd(items, SIDE) : 1
    lanes = clamp(prevpow(2, wanted), cld(SIDE, side), THREADS ÷ side)
    Tempest.Compiler.Split(clamp(wanted ÷ lanes, 1, MAXIMUM_PARTS), lanes, side)
end

# A part's lanes merge their states through shared memory, the first half of
# those holding one merging the second's at each step, until the first lane
# holds the part's; then every lane reads it. Items side by side merge apart.
function Tempest.Compiler.merge_lanes(fold, state::S, ::CUDACore.CuDeviceArray) where {S}
    shared = CUDACore.CuStaticSharedArray(S, THREADS)
    side, item, lane = CUDACore.blockDim().x, CUDACore.threadIdx().x, CUDACore.threadIdx().y
    at(lane) = (lane - 1) * side + item
    holding = CUDACore.blockDim().y
    @inbounds shared[at(lane)] = state
    while holding > 1
        CUDACore.sync_threads()
        holding >>= 1
        lane <= holding && (@inbounds shared[at(lane)] = Tempest.Compiler.IR.fold_combine(fold, shared[at(lane)], shared[at(lane + holding)]))
    end
    CUDACore.sync_threads()
    merged = @inbounds shared[item]
    # The next merge writes where this one is read.
    CUDACore.sync_threads()
    merged
end

# A floor kernel and its exception slot, from the kernel cache or compiled and
# kept. A kernel that refers to Julia values in the session (relocations) is
# compiled once a session and kept out of the cache.
function floor_function(@nospecialize(f), @nospecialize(arguments::Tuple))
    Base.issingletontype(typeof(f)) || error("a floor kernel captures values, which its launch does not pass")
    tt = Tuple{map(typeof, arguments)...}
    device = CUDACore.device()
    key = Tempest.Compiler.content_key("CUDA kernel", f, tt, CUDACore.capability(device), CUDACore.GPUCompiler,
        CUDACore.CUDA_Compiler_jll, Base.libllvm_version)
    entry, binary = Tempest.Compiler.cached_kernel(key) do
        compiled = Base.invokelatest(f, tt) do f, tt
            job = CUDACore.CompilerJob(CUDACore.methodinstance(typeof(f), tt), CUDACore.compiler_config(device))
            Base.@lock CUDACore.cufunction_lock CUDACore.compile_or_lookup(job)
        end
        isempty(compiled.relocations.records) && return Tempest.Compiler.store_kernel(key, compiled.entry, compiled.image)
        Tempest.Compiler.uncached_kernel(compiled.entry, compiled)
    end
    # A kernel with relocations is the compiled kernel itself, and not an image.
    image, relocations = binary isa Vector{UInt8} ? (binary, CUDACore.GPUCompiler.Relocations()) : (binary.image, binary.relocations)
    fun = CUDACore.link_kernel(image, entry, relocations)
    fun, CUDACore.create_exceptions!(fun.mod)
end

"""
A launch of a floor kernel on fixed arguments. Its parameters are those CUDA.jl
gives a kernel: the kernel state, then each argument by value.
"""
struct FloorLaunch
    fun::CuFunction
    exceptions::Any
    state::Base.RefValue{CUDACore.KernelState}
    boxes::Vector{Any}
    pointers::Vector{Ptr{Cvoid}}
    blocks::Int
    threads::Tuple{Int, Int}
end

function FloorLaunch(fun, exceptions, @nospecialize(args::Tuple), blocks, threads)
    state = Base.RefValue(CUDACore.KernelState(exceptions, UInt32(0)))
    boxes = Any[state; [Base.RefValue{typeof(a)}(a) for a in args]]
    FloorLaunch(fun, exceptions, state, boxes, Ptr{Cvoid}[pointer_from_objref(b) for b in boxes], blocks, threads isa Integer ? (threads, 1) : threads)
end

function (l::FloorLaunch)()
    l.state[] = CUDACore.KernelState(l.exceptions, rand(CUDACore.launch_rng(), UInt32))
    GC.@preserve l begin
        CUDACore.cuLaunchKernel(l.fun, l.blocks, 1, 1, l.threads..., 1, 0, CUDACore.stream(), l.pointers, C_NULL)
    end
    nothing
end

end
