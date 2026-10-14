# Memory-traffic accounting for GPU benchmarks. A kernel that must read and
# write `bytes` cannot finish faster than `bytes / bandwidth`; reporting the
# effective bandwidth `bytes / time` against a copy measured in the same run
# says how far a schedule is from that bound, independent of other schedules.
#
# `minimum_bytes` counts each distinct input and output array once. It is a
# lower bound on DRAM traffic for a single-pass schedule, not a model of what a
# given kernel moves. Warm timings reuse one input set, which may stay in L2;
# cold timings rotate through enough input sets to exceed L2 several times.
module Roofline

using CUDACore, Statistics

export DeviceLimits, device_limits, minimum_bytes, rotation_count, gbps

struct DeviceLimits
    name::String
    multiprocessors::Int
    l2_bytes::Int
    theoretical_bandwidth::Float64
    copy_bandwidth::Float64
end

function Base.show(io::IO, d::DeviceLimits)
    print(
        io, d.name, ": ", d.multiprocessors, " SMs, ",
        round(d.l2_bytes / 2^20; digits = 1), " MiB L2, theoretical ",
        round(d.theoretical_bandwidth / 1e9; digits = 1), " GB/s, measured copy ",
        round(d.copy_bandwidth / 1e9; digits = 1), " GB/s",
    )
end

attribute(d, name) = CUDACore.attribute(d, getfield(CUDACore, name))

# The memory clock attribute is the transfer rate in kHz on GB10 (8533000 for
# LPDDR5X-8533), which with its 256-bit bus gives the published 273 GB/s.
function device_limits(; copy_bytes = 1 << 30, samples = 9)
    d = CUDACore.device()
    bus = attribute(d, :CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
    clock = attribute(d, :CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
    DeviceLimits(
        CUDACore.name(d),
        attribute(d, :CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
        attribute(d, :CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE),
        1e3 * clock * bus / 8,
        copy_bandwidth(copy_bytes, samples),
    )
end

# A device-to-device copy far larger than L2 reads and writes every byte once.
function copy_bandwidth(bytes, samples)
    source = CUDACore.CuArray{UInt8}(undef, bytes)
    destination = similar(source)
    fill!(source, 0x01)
    copyto!(destination, source)
    CUDACore.synchronize()
    times = map(1:samples) do _
        CUDACore.@elapsed copyto!(destination, source)
    end
    2 * bytes / median(times)
end

minimum_bytes(arrays...) = sum(a -> a isa AbstractArray ? sizeof(a) : 0, arrays)

# Input sets needed for one rotation to cover `factor` times the L2 capacity.
function rotation_count(bytes, limits::DeviceLimits; factor = 4, limit = 32)
    min(limit, max(2, cld(factor * limits.l2_bytes, bytes)))
end

gbps(bytes, microseconds) = bytes / (1e3 * microseconds)

end
