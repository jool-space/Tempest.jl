# The CUDA target: buffers are device arrays, and the generated floor runs one
# thread per item.
module CUDAExt

using Tempest
using CUDACore
import Tempest: CUDATarget, Plan, Group

Tempest.target_for(::CuArray) = CUDATarget()
Tempest.allocate(::CUDATarget, T, n) = CuArray{T}(undef, n)
Tempest.upload(::CUDATarget, x) = CuArray(x)

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
        kernel = floor_kernel(f, args, k.items)
        threads = min(THREADS, k.items)
        blocks = cld(k.items, threads)
        items = k.items
        () -> kernel(args, items; threads, blocks)
    end
    (() -> foreach(l -> l(), launches), Tempest.FloorKernels(ks))
end

# A floor kernel from the kernel cache, or compiled and kept. Kernels that refer
# to Julia values in the session (relocations) are compiled every time.
function floor_kernel(f, args, n)
    tt = Tuple{typeof(args), typeof(n)}
    device = CUDACore.device()
    key = Tempest.content_key("CUDA kernel", f, tt, CUDACore.capability(device), CUDACore.GPUCompiler,
        CUDACore.CUDA_Compiler_jll, Base.libllvm_version)
    found = Tempest.load_kernel(key)
    if found !== nothing
        entry, image = found
        fun = CUDACore.link_kernel(image, entry, CUDACore.GPUCompiler.Relocations())
        state = CUDACore.KernelState(CUDACore.create_exceptions!(fun.mod), UInt32(0))
        return CUDACore.HostKernel{typeof(f), tt}(f, fun, state)
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
        kernel
    end
end

end
