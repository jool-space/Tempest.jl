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

# cuTile compiles a kernel for the alignment of each address it was built on,
# up to 128 bytes, which every allocation has.
const ALIGNMENT = 128

Tempest.Compiler.pointable(::CUDATarget, x) = x isa CuArray{<:Any, <:Any, CUDACore.DeviceMemory} && reinterpret(UInt, pointer(x)) % ALIGNMENT == 0

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
#
# A graph keeps, per step it recorded, its nodes as the executable graph has
# them now (`Node`), so that a step whose launches were pointed elsewhere is
# changed in place (`retarget!`): the words of its kernels' arguments that
# hold a moved value's address are rewritten, and the executable graph takes
# the kernels' new arguments, 0.3 μs a kernel on GB10 where recording the
# graph again took 3.7. Which words hold an address is learned, not assumed,
# as a library's kernels say nothing of their arguments: the step is
# recorded alone with its launches pointed elsewhere, and the words that
# changed, each by exactly what its value moved, are the ones to rewrite.
# A library may choose its kernels by an address's alignment, so a move to
# an address aligned otherwise than those seen is learned again. A step whose
# kernels' arguments changed otherwise, as cuBLAS's hold bytes of the host
# stack it built them on, or an address packed with other fields (a tensor
# map's), is recorded alone at each move, and the executable graph takes its
# kernels' arguments from that record; a step whose kernels or other nodes
# changed, as a library choosing another kernel, is recorded again with the
# rest.
mutable struct Graph
    exec::CUDACore.CuGraphExec
    # The graph the executable was made from, whose nodes name its own.
    graph::CUDACore.CuGraph
    steps::Vector{Any}
    nodes::Vector{Vector{Any}}
    # Per step and value, where its address is in the step's kernels, and
    # the alignments it was learned at.
    slots::Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int, Int, Int}}}
    aligned::Dict{Tuple{Int, Int}, Set{Int}}
    # The steps recorded alone at each move.
    recorded::Set{Int}
end

const GRAPHS = WeakKeyDict{Graph, Any}()

# A kernel node as the executable graph launches it: its launch, and its
# arguments' bytes, each argument's own (`cuFuncGetParamInfo`).
struct Kernel
    node::CUDACore.CUgraphNode
    params::CUDACore.CUDA_KERNEL_NODE_PARAMS
    arguments::Vector{Vector{UInt8}}
    pointers::Vector{Ptr{Cvoid}}
end

# Another node: its kind and parameters, which a step must keep to be
# changed in place; a kind whose parameters are not read keeps `nothing`,
# and a step with one is recorded again.
struct Node
    kind::CUDACore.CUgraphNodeType
    params::Any
end

function node_record(node)
    kind = Ref{CUDACore.CUgraphNodeType}()
    CUDACore.cuGraphNodeGetType(node, kind)
    if kind[] == CUDACore.CU_GRAPH_NODE_TYPE_KERNEL
        p = Ref{CUDACore.CUDA_KERNEL_NODE_PARAMS}()
        CUDACore.cuGraphKernelNodeGetParams_v2(node, p)
        p[].kernelParams == C_NULL && return Node(kind[], nothing)
        func = p[].func
        if func == C_NULL
            f = Ref{CUDACore.CUfunction}()
            CUDACore.cuKernelGetFunction(f, p[].kern)
            func = f[]
        end
        arguments = Vector{UInt8}[]
        offset, size = Ref{Csize_t}(), Ref{Csize_t}()
        while CUDACore.unchecked_cuFuncGetParamInfo(func, length(arguments), offset, size) == CUDACore.CUDA_SUCCESS
            argument = unsafe_load(p[].kernelParams, length(arguments) + 1)
            push!(arguments, copy(unsafe_wrap(Array, Ptr{UInt8}(argument), Int(size[]))))
        end
        return Kernel(node, p[], arguments, Ptr{Cvoid}[pointer(a) for a in arguments])
    elseif kind[] == CUDACore.CU_GRAPH_NODE_TYPE_MEMSET
        p = Ref{CUDACore.CUDA_MEMSET_NODE_PARAMS}()
        CUDACore.cuGraphMemsetNodeGetParams(node, p)
        return Node(kind[], p[])
    elseif kind[] == CUDACore.CU_GRAPH_NODE_TYPE_MEMCPY
        p = Ref{CUDACore.CUDA_MEMCPY3D}()
        CUDACore.cuGraphMemcpyNodeGetParams(node, p)
        return Node(kind[], p[])
    end
    Node(kind[], nothing)
end

# A node's dependencies, read with their edges' data: the driver refuses to
# leave it out where an edge has any, as a library's programmatic launch's.
function dependencies(node)
    n = Ref{Csize_t}(0)
    CUDACore.cuGraphNodeGetDependencies_v2(node, C_NULL, C_NULL, n)
    out = Vector{CUDACore.CUgraphNode}(undef, n[])
    edges = Vector{CUDACore.CUgraphEdgeData}(undef, n[])
    n[] > 0 && CUDACore.cuGraphNodeGetDependencies_v2(node, out, edges, n)
    out
end

# The steps captured as one graph, and per step its nodes: those the frontier
# after the step depends on, back to the step before's. Each step is captured
# by itself, after the step before: a library records a call within a longer
# capture otherwise than alone (cuBLAS, a word of its kernels' arguments), and
# `retarget!` learns from a step captured alone.
function steps_recorded(steps)
    graph = CUDACore.CuGraph()
    after = CUDACore.CuGraphNode[]
    marks = map(steps) do step
        after = captured!(() -> step(), graph; after)
        CUDACore.CUgraphNode[n.handle for n in after]
    end
    seen = Set{CUDACore.CUgraphNode}()
    nodes = map(marks) do mark
        mine, todo = CUDACore.CUgraphNode[], copy(mark)
        while !isempty(todo)
            n = pop!(todo)
            n in seen && continue
            push!(seen, n)
            push!(mine, n)
            append!(todo, dependencies(n))
        end
        Any[node_record(n) for n in reverse!(mine)]
    end
    graph, nodes
end

# `f` captured into `graph`, after `after`. A capture that starts while CUDA.jl
# waits for the GPU to release memory, as it does when an allocation of another
# task (a group built at the same time, a kernel's measurement) reclaims, fails
# before it captures anything, and is tried again once the release is done.
function captured!(f, graph; after)
    while true
        try
            return CUDACore.capture!(f, graph; after)
        catch err
            err isa ErrorException && startswith(err.msg, "Cannot start a graph capture while CUDA.jl is waiting") || rethrow()
            sleep(0.001)
        end
    end
end

function Tempest.Compiler.capture(::CUDATarget, steps, keep)
    graph, nodes = steps_recorded(steps)
    exec = CUDACore.instantiate(graph)
    g = finalizer(g -> (finalize(g.exec); finalize(g.graph)), Graph(exec, graph, steps, nodes, Dict(), Dict(), Set{Int}()))
    GRAPHS[g] = keep
    g
end

# Updated in place where the graph recorded again launches the same kernels,
# as it does when only addresses moved; the records take its launches.
function Tempest.Compiler.recapture(target::CUDATarget, g::Graph, steps, keep)
    graph, nodes = steps_recorded(steps)
    if alike(g.nodes, nodes) && CUDACore.update(g.exec, graph; throw_error = false)
        foreach((old, new) -> foreach(adopt!, old, new), g.nodes, nodes)
        finalize(graph)
        g.steps = steps
        return g
    end
    finalize(graph)
    Tempest.Compiler.release(target, g)
    Tempest.Compiler.capture(target, steps, keep)
end

alike(a::Vector{Vector{Any}}, b::Vector{Vector{Any}}) = length(a) == length(b) && all(((x, y),) -> length(x) == length(y) && all(splat(alike), zip(x, y)), zip(a, b))
alike(a::Kernel, b::Kernel) = a.params.func == b.params.func && a.params.kern == b.params.kern &&
    (a.params.gridDimX, a.params.gridDimY, a.params.gridDimZ) == (b.params.gridDimX, b.params.gridDimY, b.params.gridDimZ) &&
    (a.params.blockDimX, a.params.blockDimY, a.params.blockDimZ, a.params.sharedMemBytes) ==
        (b.params.blockDimX, b.params.blockDimY, b.params.blockDimZ, b.params.sharedMemBytes) &&
    length.(a.arguments) == length.(b.arguments)
alike(a::Node, b::Node) = a.kind == b.kind
alike(a, b) = false

# The record of a node takes the arguments of its counterpart in a graph
# recorded again, whose launch it now is.
adopt!(a::Kernel, b::Kernel) = foreach(copyto!, a.arguments, b.arguments)
adopt!(a::Node, b::Node) = nothing

# Launch the kernel node as its record has it now.
function relaunch!(g::Graph, k::Kernel)
    p = k.params
    params = Ref(CUDACore.CUDA_KERNEL_NODE_PARAMS(p.func, p.gridDimX, p.gridDimY, p.gridDimZ, p.blockDimX, p.blockDimY, p.blockDimZ,
        p.sharedMemBytes, pointer(k.pointers), C_NULL, p.kern, p.ctx))
    GC.@preserve k CUDACore.cuGraphExecKernelNodeSetParams_v2(g.exec, k.node, params)
end

# The alignment of an address, in powers of two up to 4096 bytes.
alignment(address) = min(trailing_zeros(address), 12)

function Tempest.Compiler.retarget!(::CUDATarget, g::Graph, steps, moves, repoint)
    key = first(steps)
    key in g.recorded && (repoint(); return record_alone!(g, steps))
    known = all(m -> haskey(g.slots, (key, m.value)) && alignment(m.to) in g.aligned[(key, m.value)], moves)
    known || return learn!(g, steps, moves, repoint)
    repoint()
    foreach(step -> patch!(g, step, key, moves), steps)
    true
end

# Write where each value moved into the words of the step's kernels that
# hold its address, and launch the kernels so.
function patch!(g::Graph, step, key, moves)
    touched = Int[]
    for m in moves, (n, a, byte, offset) in g.slots[(key, m.value)]
        bytes = g.nodes[step][n].arguments[a]
        GC.@preserve bytes unsafe_store!(Ptr{UInt64}(pointer(bytes, byte)), UInt64(Int(m.to) + offset))
        n in touched || push!(touched, n)
    end
    foreach(n -> relaunch!(g, g.nodes[step][n]), touched)
end

# Record the launch alone before and after it is pointed where `moves` say,
# and find where each value's address is: every 8-byte word of its kernels'
# arguments that changed must have changed by what one value moved, from near
# where the value was (an address into it, its end, or rounded down from its
# start), and hold in the graph what it held alone. Recorded one after the
# other, the launch's arguments differ in nothing else: what a library leaves
# unset in them, as the bytes of the host stack cuBLAS builds them on, which
# differ between the graph and a recording alone, stays as the graph has it.
# A launch with other words changed is recorded alone at each move from then
# on. Its kernels and other nodes must be as they were.
function learn!(g::Graph, steps, moves, repoint)
    key = first(steps)
    old, (before,) = steps_recorded(Any[g.steps[key]])
    repoint()
    new, (after,) = steps_recorded(Any[g.steps[key]])
    try
        same(before, after) && all(step -> same(g.nodes[step], after), steps) || return false
        found = Dict(m.value => Tuple{Int, Int, Int, Int}[] for m in moves)
        exact = true
        for (n, (a, b)) in enumerate(zip(before, after))
            a isa Kernel || continue
            for (j, (x, y)) in enumerate(zip(a.arguments, b.arguments))
                x == y && continue
                for byte in 1:8:length(x)
                    last = min(byte + 7, length(x))
                    x[byte:last] == y[byte:last] && continue
                    word = last - byte == 7 ? reinterpret(UInt64, x[byte:last])[1] : nothing
                    i = word === nothing ? nothing : moved_by(moves, word, reinterpret(UInt64, y[byte:last])[1])
                    held = i !== nothing && all(step -> g.nodes[step][n].arguments[j][byte:last] == x[byte:last], steps)
                    held ? push!(found[moves[i].value], (n, j, byte, Int(word - moves[i].from))) : (exact = false)
                end
            end
        end
        if !exact
            push!(g.recorded, key)
            foreach(step -> take_record!(g, g.nodes[step], after), steps)
            return true
        end
        for m in moves
            g.slots[(key, m.value)] = found[m.value]
            push!(get!(() -> Set([alignment(m.from)]), g.aligned, (key, m.value)), alignment(m.to))
        end
        foreach(step -> patch!(g, step, key, moves), steps)
        true
    finally
        finalize(old)
        finalize(new)
    end
end

# The launch recorded alone, its kernels' arguments in each of its steps
# taken from the record.
function record_alone!(g::Graph, steps)
    graph, (fresh,) = steps_recorded(Any[g.steps[first(steps)]])
    try
        all(step -> same(g.nodes[step], fresh), steps) || return false
        foreach(step -> take_record!(g, g.nodes[step], fresh), steps)
        true
    finally
        finalize(graph)
    end
end

# Whether a step recorded again launches the same kernels, and has the same
# other nodes, as the executable graph's.
same(old, fresh) = alike(Vector{Any}[old], Vector{Any}[fresh]) &&
    all(((a, b),) -> a isa Kernel || (a.params !== nothing && a.params == b.params), zip(old, fresh))

# The executable graph's kernels of a step take their arguments from a record of it.
take_record!(g, old, fresh) = foreach(((a, b),) -> a isa Kernel && a.arguments != b.arguments && (adopt!(a, b); relaunch!(g, a)), zip(old, fresh))

# The move a word changed by, from near where its value was: within a page
# of it, so that an address rounded down from its start or its end counts.
function moved_by(moves, before, after)
    near(m) = m.from - min(m.from, 4096) <= before <= m.from + m.bytes + 4096
    by(m) = after - before == m.to - m.from
    something(findfirst(m -> by(m) && m.from <= before < m.from + m.bytes, moves), findfirst(m -> by(m) && near(m), moves), Some(nothing))
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
    g = Tempest.Compiler.capture(target, Any[launch], launch)
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
        CLEARING[] = (FloorLaunch(floor_function(read_through, (args, length(a)))..., (args, length(a)), 1024, 256, Int[]), a, sink)
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
    g = Tempest.Compiler.capture(target, Any[empty], empty)
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
    g = Tempest.Compiler.capture(target, Any[launch for _ in 1:repetitions], launch)
    Tempest.Compiler.replay(target, g)
    t = minimum(_ -> CUDACore.@elapsed(Tempest.Compiler.replay(target, g)), 1:samples) / repetitions
    Tempest.Compiler.release(target, g)
    t
end

# Nothing, launched to time the gap between launches; a floor kernel, as
# `read_through` is.
blank(_, _) = nothing

blank_launch() = FloorLaunch(floor_function(blank, ((), 1))..., ((), 1), 1, 1, Int[])

# Half a microsecond between launches in a captured graph: GB10's 0.47. Ten
# times as long or a tenth changes one plan of the corpus's.
const LAUNCH = 5.0e-7

# The memory's peak: its transfers a second per pin, a bit each, over the
# bus's pins; and what the cache holds twice as fast, as GB10 copies 4 MiB at
# 476 GB/s and 256 MiB at 245. A discrete device reports its memory's clock,
# which transfers twice a cycle; an integrated one, the system's memory, as
# GB10 its LPDDR5X's transfer rate, 8533 MT/s over 256 pins, 273 GB/s.
function Tempest.Compiler.device_rates(::CUDATarget)
    device = CUDACore.device()
    attribute(a) = Float64(CUDACore.attribute(device, a))
    transfers = (attribute(CUDACore.DEVICE_ATTRIBUTE_INTEGRATED) == 1 ? 1 : 2) * 1.0e3 * attribute(CUDACore.DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
    bandwidth = transfers * attribute(CUDACore.DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH) / 8
    (launch = LAUNCH, cached = 2bandwidth, bandwidth,
     cache = attribute(CUDACore.DEVICE_ATTRIBUTE_L2_CACHE_SIZE),
     blocks = attribute(CUDACore.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT))
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
    FloorLaunch(floor_function(f, (args, k.items))..., (args, k.items), cld(k.items, threads), threads, k.slots)
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
    FloorLaunch(floor_function(f, arguments)..., arguments, (k.items ÷ k.split.side) * k.split.parts, (k.split.side, k.split.lanes), k.slots)
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
function Tempest.Compiler.floor_split(::CUDATarget, items::Int, positions::Int, adjacent::Bool)
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
gives a kernel: the kernel state, then each argument by value, the first the
kernel's buffers, the first of them those of the values it reads and stores,
in `values`.
"""
struct FloorLaunch
    fun::CuFunction
    exceptions::Any
    state::Base.RefValue{CUDACore.KernelState}
    boxes::Vector{Any}
    pointers::Vector{Ptr{Cvoid}}
    blocks::Int
    threads::Tuple{Int, Int}
    values::Vector{Int}
end

function FloorLaunch(fun, exceptions, @nospecialize(args::Tuple), blocks, threads, values)
    state = Base.RefValue(CUDACore.KernelState(exceptions, UInt32(0)))
    boxes = Any[state; [Base.RefValue{typeof(a)}(a) for a in args]]
    FloorLaunch(fun, exceptions, state, boxes, Ptr{Cvoid}[pointer_from_objref(b) for b in boxes], blocks,
        threads isa Integer ? (threads, 1) : threads, values)
end

Tempest.Compiler.Implement.repoints(::FloorLaunch) = true

# The buffers' box holds a tuple of device arrays; one of the same type
# replaces it.
function Tempest.Compiler.Implement.repoint!(l::FloorLaunch, value, buffer)
    box = l.boxes[2]
    box[] = foldl((b, j) -> Base.setindex(b, CUDACore.cudaconvert(buffer), j), findall(==(value), l.values); init = box[])
    l
end

function (l::FloorLaunch)()
    l.state[] = CUDACore.KernelState(l.exceptions, rand(CUDACore.launch_rng(), UInt32))
    GC.@preserve l begin
        CUDACore.cuLaunchKernel(l.fun, l.blocks, 1, 1, l.threads..., 1, 0, CUDACore.stream(), l.pointers, C_NULL)
    end
    nothing
end

end
