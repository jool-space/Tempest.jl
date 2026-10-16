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
        f = Core.eval(Tempest.Generated, :(function (buffers, n)
            lin = ($(CUDACore.blockIdx)().x - 1) * $(CUDACore.blockDim)().x + $(CUDACore.threadIdx)().x - 1
            lin < n || return
            $(k.body)
            return
        end))
        args = CUDACore.cudaconvert(Tuple(buffers[v] for v in k.slots))
        kernel = Base.invokelatest((f, args, n) -> CUDACore.@cuda(launch = false, f(args, n)), f, args, k.items)
        threads = min(THREADS, k.items)
        blocks = cld(k.items, threads)
        items = k.items
        () -> kernel(args, items; threads, blocks)
    end
    (() -> foreach(l -> l(), launches), Tempest.FloorKernels(ks))
end

end
