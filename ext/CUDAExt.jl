# The CUDA target: buffers are device arrays, and the generated floor runs one
# thread per item.
module CUDAExt

using Tempest
using CUDACore
import Tempest: CUDATarget
import Tempest.Compiler: Plan, Group

Tempest.Compiler.target_for(::CuArray) = CUDATarget()
Tempest.Compiler.allocate(::CUDATarget, T, n) = CuArray{T}(undef, n)
Tempest.Compiler.upload(::CUDATarget, x) = CuArray(x)
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

__init__() = atexit(() -> foreach(finalize, collect(keys(GRAPHS))))

# Device seconds of one launch: the minimum over samples of a captured graph
# of about a millisecond of launches.
function Tempest.Compiler.time_launch(target::CUDATarget, launch; samples = 5, milliseconds = 1.0)
    launch()
    CUDACore.synchronize()
    once = CUDACore.@elapsed launch()
    repetitions = clamp(round(Int, 1.0e-3 * milliseconds / max(once, 1.0e-6)), 1, 100)
    g = Tempest.Compiler.capture(target, () -> foreach(_ -> launch(), 1:repetitions), launch)
    Tempest.Compiler.replay(target, g)
    t = minimum(_ -> CUDACore.@elapsed(Tempest.Compiler.replay(target, g)), 1:samples) / repetitions
    Tempest.Compiler.release(target, g)
    t
end

# Nothing, launched to time the gap between launches.
blank() = nothing

# The gap between launches of an empty kernel, copies of 4 MiB (found in the
# cache when replayed) and 256 MiB (not), the cache's size and the
# multiprocessors.
function Tempest.Compiler.calibrate(target::CUDATarget)
    device = CUDACore.device()
    kernel = CUDACore.@cuda launch = false blank()
    copying(n) = (a = CUDACore.zeros(UInt8, n); b = similar(a); 2n / Tempest.Compiler.time_launch(target, () -> copyto!(b, a)))
    (launch = Tempest.Compiler.time_launch(target, () -> kernel(; threads = 1, blocks = 1)),
     cached = copying(4 << 20), bandwidth = copying(256 << 20),
     cache = Float64(CUDACore.attribute(device, CUDACore.DEVICE_ATTRIBUTE_L2_CACHE_SIZE)),
     blocks = Float64(CUDACore.attribute(device, CUDACore.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)))
end

Tempest.Compiler.build_floor(::CUDATarget, p::Plan, g::Group, buffers) = floor(p, g, buffers)

# A scatter's items run at once: they add atomically.
Tempest.Compiler.accumulate!(buffer::CUDACore.CuDeviceArray, i, v) = (CUDACore.@atomic buffer[i] += v; nothing)

function floor(p::Plan, g::Group, buffers)
    ks = Tempest.Compiler.scalar_kernels(p.graph, p.stored, g, Tempest.Compiler.kernel_slots(p, g))
    launches = map(ks) do k
        f = Tempest.Compiler.generated(:(function (buffers, n)
            lin = ($(CUDACore.blockIdx)().x - 1) * $(CUDACore.blockDim)().x + $(CUDACore.threadIdx)().x - 1
            lin < n || return
            $(k.body)
            return
        end))
        args = CUDACore.cudaconvert(Tuple(buffers[v] for v in k.slots))
        threads = min(THREADS, k.items)
        FloorLaunch(floor_function(f, args, k.items)..., (args, k.items), cld(k.items, threads), threads)
    end
    (() -> foreach(l -> l(), launches), Tempest.Compiler.FloorKernels(ks))
end

# A floor kernel and its exception slot, from the kernel cache or compiled and
# kept. A kernel that refers to Julia values in the session (relocations) is
# compiled once a session and kept out of the cache.
function floor_function(@nospecialize(f), @nospecialize(args), n)
    Base.issingletontype(typeof(f)) || error("a floor kernel captures values, which its launch does not pass")
    tt = Tuple{typeof(args), typeof(n)}
    device = CUDACore.device()
    key = Tempest.Compiler.content_key("CUDA kernel", f, tt, CUDACore.capability(device), CUDACore.GPUCompiler,
        CUDACore.CUDA_Compiler_jll, Base.libllvm_version)
    entry, binary = Tempest.Compiler.cached_kernel(key) do
        compiled = Base.invokelatest(f, tt) do f, tt
            job = CUDACore.CompilerJob(CUDACore.methodinstance(typeof(f), tt), CUDACore.compiler_config(device))
            Base.@lock CUDACore.cufunction_lock CUDACore.compile_or_lookup(job)
        end
        isempty(compiled.relocations.records) && return Tempest.Compiler.store_kernel(key, compiled.entry, compiled.image)
        Threads.atomic_add!(Tempest.Compiler.CACHE_COUNTS.compiled, 1)
        compiled.entry, compiled
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
    threads::Int
end

function FloorLaunch(fun, exceptions, @nospecialize(args::Tuple), blocks, threads)
    state = Base.RefValue(CUDACore.KernelState(exceptions, UInt32(0)))
    boxes = Any[state; [Base.RefValue{typeof(a)}(a) for a in args]]
    FloorLaunch(fun, exceptions, state, boxes, Ptr{Cvoid}[pointer_from_objref(b) for b in boxes], blocks, threads)
end

function (l::FloorLaunch)()
    l.state[] = CUDACore.KernelState(l.exceptions, rand(CUDACore.launch_rng(), UInt32))
    GC.@preserve l begin
        CUDACore.cuLaunchKernel(l.fun, l.blocks, 1, 1, l.threads, 1, 1, 0, CUDACore.stream(), l.pointers, C_NULL)
    end
    nothing
end

end
