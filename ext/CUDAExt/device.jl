# A device's rates for cost models: the half-precision product rate cuBLAS
# reaches, and memory bandwidth. Derived from device attributes where the
# architecture is in the table below, measured once per device otherwise.

# Dense Float16 tensor-core FLOPs per clock per SM with Float32 accumulation,
# and memory transfers per clock of DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE. GeForce
# boards of compute capability 7.5 to 8.9 accumulate in Float32 at half the rate.
const ARCHITECTURE_RATES = Dict(
    (7, 0) => (flops_per_clock = 1024, transfers_per_clock = 2), # V100
    (7, 5) => (flops_per_clock = 1024, transfers_per_clock = 2), # T4
    (8, 0) => (flops_per_clock = 2048, transfers_per_clock = 2), # A100
    (8, 6) => (flops_per_clock = 1024, transfers_per_clock = 2), # A10, A40
    (8, 9) => (flops_per_clock = 1024, transfers_per_clock = 2), # L4, L40
    (9, 0) => (flops_per_clock = 4096, transfers_per_clock = 2), # H100, H200
    (10, 0) => (flops_per_clock = 8192, transfers_per_clock = 2), # B200
    (10, 3) => (flops_per_clock = 8192, transfers_per_clock = 2), # B300
    (12, 0) => (flops_per_clock = 1024, transfers_per_clock = 2), # RTX PRO 6000, RTX 50
    # GB10 reports its LPDDR5X data rate, 8533 MT/s, as the memory clock.
    (12, 1) => (flops_per_clock = 1024, transfers_per_clock = 1),
)

# Shares of the peak rates that cuBLAS and a copy reach. On GB10, cuBLAS's
# Float16 products ran at 108 TFLOP/s of a 119 TFLOP/s peak, and copies at
# 234 GB/s of 273 GB/s.
const PRODUCT_EFFICIENCY = 0.9
const BANDWIDTH_EFFICIENCY = 0.85

const DEVICE_RATES = Dict{CUDACore.CuDevice, Any}()
const DEVICE_RATES_LOCK = ReentrantLock()

"""
    Tempest.device_rates(device::CuDevice) -> (; product_flops, memory_bandwidth, source, detail)

The rates used by cost models on `device`, cached for the process. `source` is
`:derived` from device attributes or `:measured` by one large copy and one
cuBLAS Float16 product, for architectures without a table entry.
"""
function Tempest.device_rates(device::CUDACore.CuDevice)
    lock(DEVICE_RATES_LOCK) do
        get!(() -> derived_rates(device), DEVICE_RATES, device)
    end
end

function derived_rates(device)
    attribute(a) = CUDACore.attribute(device, a)
    capability = CUDACore.capability(device)
    rates = get(ARCHITECTURE_RATES, (capability.major, capability.minor), nothing)
    rates === nothing && return measured_rates(device)
    sms = attribute(CUDACore.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
    clock = 1e3 * attribute(CUDACore.DEVICE_ATTRIBUTE_CLOCK_RATE)
    memory_clock = 1e3 * attribute(CUDACore.DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
    bus_bytes = attribute(CUDACore.DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH) ÷ 8
    product_flops = PRODUCT_EFFICIENCY * sms * clock * rates.flops_per_clock
    memory_bandwidth = BANDWIDTH_EFFICIENCY * memory_clock * rates.transfers_per_clock * bus_bytes
    detail = (
        product = "$sms SMs × $(round(clock / 1e9; digits = 2)) GHz × $(rates.flops_per_clock) FLOP/clock × $PRODUCT_EFFICIENCY",
        memory = "$(round(memory_clock * rates.transfers_per_clock / 1e9; digits = 2)) GT/s × $(8bus_bytes) bits × $BANDWIDTH_EFFICIENCY",
    )
    (; product_flops, memory_bandwidth, source = :derived, detail)
end

# Median device seconds of `f`, after one warm-up call.
function device_seconds(f; samples = 5)
    f()
    times = map(1:samples) do _
        CUDACore.@elapsed f()
    end
    sort!(times)[cld(samples, 2)]
end

function measured_rates(device)
    CUDACore.device!(device) do
        bytes = 256 << 20
        source, destination = CUDACore.CuArray{UInt8}(undef, bytes), CUDACore.CuArray{UInt8}(undef, bytes)
        fill!(source, 0x01)
        memory_bandwidth = 2 * bytes / device_seconds(() -> copyto!(destination, source))
        n = 4096
        a, b = CUDACore.CuArray(rand(Float16, n, n)), CUDACore.CuArray(rand(Float16, n, n))
        c = similar(a)
        resources = BlasResources(4 << 20)
        workspace = CUDACore.CuArray{UInt8}(undef, 4 << 20)
        product = gemm_call(resources, c, a, b, workspace)
        product_flops = 2n^3 / device_seconds(product)
        detail = (product = "cuBLAS Float16 $(n)³ product", memory = "$(bytes >> 20) MiB copy")
        (; product_flops, memory_bandwidth, source = :measured, detail)
    end
end
