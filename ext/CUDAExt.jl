# The CUDA target: buffers are device arrays, and the generated floor runs one
# thread per item.
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

# A fold by key's items run at once: a sum's add atomically.
Tempest.Compiler.accumulate!(::Tempest.IR.SumFold, buffer::CUDACore.CuDeviceArray, i, v) = (CUDACore.@atomic buffer[i] += v; nothing)

function Tempest.Compiler.floor_launch(::CUDATarget, k::Tempest.Compiler.ScalarKernel, args)
    f = Tempest.Compiler.generated(:(function (buffers, n)
        lin = ($(CUDACore.blockIdx)().x - 1) * $(CUDACore.blockDim)().x + $(CUDACore.threadIdx)().x - 1
        lin < n || return
        $(k.body)
        return
    end))
    args = CUDACore.cudaconvert(args)
    threads = min(THREADS, k.items)
    FloorLaunch(floor_function(f, args, k.items)..., (args, k.items), cld(k.items, threads), threads)
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
