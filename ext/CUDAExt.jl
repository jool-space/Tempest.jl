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

# Rates for estimates, measured on known devices: copy bandwidth, and the
# launch gap in a captured graph. Other devices derive bandwidth from their
# attributes.
const DEVICE_BANDWIDTH = Dict("NVIDIA GB10" => 220.0e9)
const LAUNCH_SECONDS = 1.0e-6
const NOISE = 0.05

Tempest.device_name(::CUDATarget) = CUDACore.name(CUDACore.device())

function Tempest.rates(target::CUDATarget)
    device = CUDACore.device()
    attribute(a) = CUDACore.attribute(device, a)
    sms = attribute(CUDACore.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    bandwidth = get(DEVICE_BANDWIDTH, CUDACore.name(device)) do
        0.85 * 2.0e3 * attribute(CUDACore.DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE) *
            attribute(CUDACore.DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH) / 8
    end
    Tempest.Rates(bandwidth, LAUNCH_SECONDS, sms, NOISE)
end

Tempest.measurable(::CUDATarget) = true

# Device seconds of one launch: the minimum over samples of a captured graph
# of about a millisecond of launches.
function Tempest.time_launch(::CUDATarget, launch; samples = 5, milliseconds = 1.0)
    launch()
    CUDACore.synchronize()
    once = CUDACore.@elapsed launch()
    repetitions = clamp(round(Int, 1.0e-3 * milliseconds / max(once, 1.0e-6)), 1, 100)
    graph = CUDACore.capture() do
        for _ in 1:repetitions
            launch()
        end
    end
    exec = CUDACore.instantiate(graph)
    CUDACore.launch(exec)
    minimum(_ -> CUDACore.@elapsed(CUDACore.launch(exec)), 1:samples) / repetitions
end

Tempest.build_floor(::CUDATarget, p::Plan, g::Group, buffers) = floor(p, g, buffers)

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
