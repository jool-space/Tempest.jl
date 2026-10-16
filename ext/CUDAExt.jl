# The CUDA target: buffers are device arrays, and the generated floor runs one
# thread per item.
module CUDAExt

using Tempest
using CUDACore
import Tempest: CUDATarget, Plan, Group

Tempest.target_for(::CuArray) = CUDATarget()
Tempest.allocate(::CUDATarget, T, n) = CuArray{T}(undef, n)
Tempest.upload(::CUDATarget, x) = CuArray(x)

function Tempest.task_state(::CUDATarget)
    context, stream = CUDACore.context(), CUDACore.stream()
    () -> (CUDACore.context!(context); CUDACore.stream!(stream))
end

const THREADS = 256

function Tempest.build(target::CUDATarget, p::Plan, g::Group, buffers)
    g.kind === :opaque && return Tempest.build_opaque(target, p, g, buffers)
    for implementation in Tempest.EXTRA_IMPLEMENTATIONS
        for claim in Tempest.claims(implementation, target, p, g)
            built = try
                Tempest.build_claim(implementation, claim, target, p, g, buffers)
            catch err
                @debug "claim failed to build; trying the next" implementation exception = err
                nothing
            end
            built === nothing || return built
        end
    end
    floor(p, g, buffers)
end

function floor(p::Plan, g::Group, buffers)
    ks = Tempest.scalar_kernels(p.graph, p.stored, g, Tempest.kernel_slots(p, g))
    launches = map(ks) do k
        f = Tempest.generated(:(function (buffers, n)
            lin = ($(CUDACore.blockIdx)().x - 1) * $(CUDACore.blockDim)().x + $(CUDACore.threadIdx)().x - 1
            lin < n || return
            $(k.body)
            return
        end))
        args = CUDACore.cudaconvert(Tuple(buffers[v] for v in k.slots))
        threads = min(THREADS, k.items)
        FloorLaunch(floor_function(f, args, k.items)..., (args, k.items), cld(k.items, threads), threads)
    end
    (() -> foreach(l -> l(), launches), Tempest.FloorKernels(ks))
end

# A floor kernel and its exception slot, from the kernel cache or compiled and
# kept. Kernels that refer to Julia values in the session (relocations) are
# compiled every time.
function floor_function(@nospecialize(f), @nospecialize(args), n)
    Base.issingletontype(typeof(f)) || error("a floor kernel captures values, which its launch does not pass")
    tt = Tuple{typeof(args), typeof(n)}
    device = CUDACore.device()
    key = Tempest.content_key("CUDA kernel", f, tt, CUDACore.capability(device), CUDACore.GPUCompiler,
        CUDACore.CUDA_Compiler_jll, Base.libllvm_version)
    found = Tempest.load_kernel(key)
    if found !== nothing
        entry, image = found
        fun = CUDACore.link_kernel(image, entry, CUDACore.GPUCompiler.Relocations())
        return fun, CUDACore.create_exceptions!(fun.mod)
    end
    Base.invokelatest(f, args, n) do f, args, n
        kernel = CUDACore.@cuda(launch = false, f(args, n))
        job = CUDACore.CompilerJob(CUDACore.methodinstance(typeof(f), tt), CUDACore.compiler_config(device))
        compiled = CUDACore.compile_or_lookup(job)
        if isempty(compiled.relocations.records)
            Tempest.store_kernel(key, compiled.entry, compiled.image)
        else
            Threads.atomic_add!(Tempest.CACHE_COUNTS.compiled, 1)
        end
        kernel.fun, kernel.state.exception_info
    end
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
