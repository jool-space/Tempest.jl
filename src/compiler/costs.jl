# Whether a graph region pays: a bytes-and-launches estimate of the region
# against the default partition's kernels for the same members.
#
# Implementations describe their launches, not their times: for each kernel,
# the bytes it moves through memory and the bytes its threads or tiles request
# at once. The target prices them with three device numbers, so one formula
# compares plans from different providers. By Little's law a kernel with
# `parallel` bytes in flight moves at most `parallel / latency` bytes per
# second, and never more than the bandwidth:
#
#     seconds = Σ launch_seconds + bytes / min(bandwidth, parallel / latency)
#
# The second term is what makes a CUDA kernel with one warp per output slow on
# 128 long columns, a completion that loops over partial sums in one block
# slow when there are many of them, and a schedule with too few blocks slow
# anywhere. Caches and coalescing are not modelled: every byte is priced at
# memory bandwidth, and a strided read like a contiguous one.

# Estimates are on unless a scope turns them off: the guard-rail benchmarks
# compare each choice with the plan it declined.
const REGION_COSTS = ScopedValue(true)

"""
    Tempest.KernelCost(bytes, parallel)

One launch in a cost estimate: `bytes` of memory traffic, inputs read and
outputs written, and `parallel`, the bytes its threads or tiles request at once:
threads times element size, or blocks times tile bytes.
"""
struct KernelCost
    bytes::Float64
    parallel::Float64
end

"""
    Tempest.CostRates(memory_bandwidth, latency, launch_seconds, detail)

The prices of `KernelCost`s on a target: memory bandwidth in bytes per second,
the latency in seconds that bytes in flight wait (a kernel saturates memory
with `memory_bandwidth * latency` bytes in flight), and the seconds between
consecutive launches. `detail` says where they came from, for `explain`.
"""
struct CostRates
    memory_bandwidth::Float64
    latency::Float64
    launch_seconds::Float64
    detail::String
end

"""
    Tempest.cost_rates(target) -> CostRates or nothing

The rates region estimates are priced with. Targets without rates (the CPU)
keep every supported region.
"""
cost_rates(::ExecutionTarget) = nothing

"""
    Tempest.region_cost(implementation, request) -> Vector{KernelCost} or nothing

The launches `implementation` would make for `request`: a `GraphRequest`, or a
`RegionRequest` of the default partition. `nothing`, the default, means no
estimate; a region whose implementation gives none, or whose default kernels
have none, is kept, as before estimates existed.
"""
region_cost(::Implementation, ::ImplementationRequest) = nothing

estimated_seconds(kernels, rates) =
    sum(kernels; init = 0.0) do kernel
        rate = min(rates.memory_bandwidth, kernel.parallel / rates.latency)
        rates.launch_seconds + (iszero(kernel.bytes) ? 0.0 : kernel.bytes / rate)
    end

function cost_summary(name, kernels, rates)
    bytes = sum(kernel -> kernel.bytes, kernels; init = 0.0)
    launches = length(kernels) == 1 ? "1 launch" : "$(length(kernels)) launches"
    "$(round(1.0e6 * estimated_seconds(kernels, rates); digits = 1)) μs on $name " *
    "($launches, $(round(bytes / 2^20; digits = 2)) MiB)"
end

# The default partition, planned without regions, and for each of its steps
# the launches of the first candidate that supports it. A step is keyed by its
# output; steps without an estimate (semantic calls, or an implementation that
# gives none) map to nothing.
function default_kernels(program, target, implementations, live)
    (; layouts, materialized) = plan_layouts(program, live)
    steps = schedule_steps(program, live, layouts, materialized)
    nodes = program.nodes
    estimates = Dict{Int, Any}()
    for step in steps
        estimates[step.output] = nothing
        step.region isa OpaqueCall && continue
        node = nodes[step.output]
        # Scalar regions read flat storage roots; products their logical operands.
        reads, shape = step.region isa ScalarRegion ?
            (step.dependencies, id -> (prod(nodes[id].dims; init = 1),)) : (node.inputs, id -> nodes[id].dims)
        request = RegionRequest(
            target,
            step,
            node.dtype,
            node.dims,
            Tuple(nodes[id].dtype for id in reads),
            Tuple(shape(id) for id in reads),
        )
        for implementation in implementations
            checked_support(implementation, request) isa Supported || continue
            kernels = region_cost(implementation, request)
            kernels === nothing || (estimates[step.output] = (implementation_name(implementation), kernels))
            break
        end
    end
    estimates
end

# A region estimated to be slower by less than this factor is kept. The model
# ignores caches and coalescing, so estimates this close do not say which plan
# is faster; the provider that built the region for these members gets the
# benefit of the doubt. On the guard-rail benchmarks (test/benchmarks) kept
# regions estimated up to 1.13 times the default kernels measured as fast as
# them; the causal attention reference's backward softmax, estimated at 1.20,
# was 4% slower in training, and declining it brought the step from 20.2 to
# 19.3 ms on GB10.
const REGION_MARGIN = 1.15

"""
The outcome of comparing a graph region, or a proposed region, with the
default kernels for its members: whether to keep it, and the line `explain`
prints.

The region's launches are its implementation's; the default's are those of
the default partition's steps whose outputs are members. A region input that
the default partition inlines must be written by a step of its own when the
region forms (a reduction cut off from the pointwise work that feeds it), so
the region pays for writing and reading it back, unless a region selected
before it already writes the value.
"""
function region_profit(region, implementation, request, defaults, rates, claimed)
    rates === nothing && return (true, nothing)
    kernels = region_cost(implementation, request)
    name = implementation_name(implementation)
    kind = region isa GraphRegion ? "graph region" : "proposal"
    if kernels === nothing
        return (true, "$kind $(region.members) kept: $name gives no estimate")
    end
    covered = [defaults[id] for id in region.members if haskey(defaults, id)]
    if isempty(covered) || any(isnothing, covered)
        return (true, "$kind $(region.members) kept: the default kernels have no estimate")
    end
    nodes = region.program.nodes
    for id in region.inputs
        nodes[id].op isa Pointwise && !haskey(defaults, id) && !claimed[id] || continue
        bytes = sizeof(nodes[id].dtype) * prod(nodes[id].dims; init = 1)
        kernels = [kernels; KernelCost(2bytes, bytes)]
    end
    providers = join(unique(first.(covered)), ", ")
    replaced = reduce(vcat, last.(covered))
    fused = estimated_seconds(kernels, rates)
    unfused = estimated_seconds(replaced, rates)
    keep = fused <= REGION_MARGIN * unfused
    verdict = keep ? "kept" : "declined"
    line = "$kind $(region.members) $verdict by estimate: " *
        "$(cost_summary(name, kernels, rates)) against $(cost_summary(providers, replaced, rates))"
    (keep, line)
end
