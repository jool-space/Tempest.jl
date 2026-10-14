# Residual RMSNorm forward across shapes: the graph-region provider (resident
# or streamed by row count) against the decomposed schedule with cuTile loaded,
# each placed against the memory-traffic bound (see roofline.jl).
#
# Both paths are prepared and checked before anything is timed, and the two are
# measured in alternating order. Device times are per step from one CUDA graph
# holding `repetitions` back-to-back executions, so host submission gaps are
# excluded. *Warm* repeats one input set, which may be served from L2. *Cold*
# rotates through enough prepared copies, each with its own inputs and outputs,
# to cover L2 several times; shapes too small for that report warm only.
# *Replay* is the per-call time of host-issued `replay!` on the warm set.
# Effective bandwidth is minimum traffic over cold time, and is compared with a
# device copy measured in the same run. Compilation is excluded; each figure is
# the median of `samples` samples.
# Run: julia --project=examples examples/research/normalization_schedules_benchmark.jl
include("roofline.jl")

using .Roofline
using Tempest, CUDACore, Statistics
import cuBLAS, cuTile

function residual_normalization(x, residual, weight, bias, epsilon)
    h = x .+ residual
    rstd = inv.(sqrt.(sum(abs2, h; dims = 1) ./ eltype(x)(size(x, 1)) .+ epsilon))
    y = tanh.(h .* rstd .* weight .+ bias)
    (; y, h, rstd)
end

function inputs(rows, columns)
    x = reshape(Float32.(sin.(1:(rows * columns))), rows, columns)
    residual = 0.2f0 .* reverse(x; dims = 1)
    weight = Float32.(range(0.3, 0.8; length = rows))
    bias = reshape(Float32.(range(-0.2, 0.1; length = columns)), 1, columns)
    (x, residual, weight, bias, 0.03f0)
end

device_inputs(args) = map(a -> a isa AbstractArray ? CuArray(a) : a, args)

# `copies` executables with distinct inputs and outputs; the first is checked
# and captured for host replay.
function prepared(flags, args, expected, copies)
    executables = map(1:copies) do _
        prepare_forward(residual_normalization, device_inputs(args)...; flags...)
    end
    ex = first(executables)
    got = execute!(ex)
    for key in keys(expected)
        Array(got[key]) ≈ expected[key] || error("mismatch in $key for $flags")
    end
    capture!(ex)
    replay!(ex)
    schedule = join(filter(!isnothing, map(Tempest.implementation_summary, ex.calls)), "; ")
    (; executables, steps = length(ex.storage.steps), schedule)
end

# One graph of `repetitions` executions, cycling through `executables`.
function batch(executables, repetitions)
    CUDACore.synchronize()
    graph = CUDACore.capture() do
        for i in 1:repetitions
            execute!(executables[mod1(i, length(executables))])
        end
    end
    CUDACore.instantiate(graph)
end

device_sample(executable, repetitions) =
    1e6 * CUDACore.@elapsed(CUDACore.launch(executable)) / repetitions

function replay_sample(ex, repetitions)
    elapsed = CUDACore.@elapsed begin
        for _ in 1:repetitions
            replay!(ex)
        end
    end
    1e6 * elapsed / repetitions
end

function measure(candidates; samples = 7, repetitions = 200)
    warm = map(c -> batch(c.executables[1:1], repetitions), candidates)
    cold = map(c -> batch(c.executables, repetitions), candidates)
    for i in eachindex(candidates)
        CUDACore.launch(warm[i])
        CUDACore.launch(cold[i])
        replay_sample(first(candidates[i].executables), 10)
    end
    CUDACore.synchronize()
    times = [(warm = Float64[], cold = Float64[], replay = Float64[]) for _ in candidates]
    for sample in 1:samples
        order = isodd(sample) ? eachindex(candidates) : reverse(eachindex(candidates))
        for i in order
            push!(times[i].warm, device_sample(warm[i], repetitions))
            push!(times[i].cold, device_sample(cold[i], repetitions))
            push!(times[i].replay, replay_sample(first(candidates[i].executables), repetitions))
        end
    end
    map(t -> map(median, t), times)
end

# Thread blocks per SM of the graph-region launch, the parallelism available.
function blocks_per_sm(candidate, limits)
    ext = Base.get_extension(Tempest, :cuTileExt)
    for call in first(candidate.executables).calls
        call isa Tempest.BuiltImplementation && call.launch isa ext.GraphLaunch || continue
        return prod(ext.graph_grid(call.launch.schedule)) / limits.multiprocessors
    end
    nothing
end

fixed(x, digits = 1) = x === nothing ? "—" : string(round(x; digits))

function run(; shapes = ((256, 32), (259, 32), (8192, 32), (8193, 32), (1024, 1024), (128, 1024), (128, 4096)))
    CUDACore.functional() || error("requires CUDA")
    limits = device_limits()
    println(limits)
    println(
        "shape | traffic MB | copies | graph warm/cold/replay μs, cold GB/s, % copy | ",
        "decomposed warm/cold μs, cold GB/s | blocks/SM | schedule",
    )
    for (rows, columns) in shapes
        args = inputs(rows, columns)
        expected = residual_normalization(args...)
        traffic = minimum_bytes(args..., values(expected)...)
        copies = rotation_count(traffic, limits)
        cold_valid = copies * traffic >= 2 * limits.l2_bytes
        fused = prepared((;), args, expected, cold_valid ? copies : 1)
        decomposed = prepared((; fuse_regions = false), args, expected, cold_valid ? copies : 1)
        f, d = measure((fused, decomposed))
        f_cold = cold_valid ? f.cold : nothing
        d_cold = cold_valid ? d.cold : nothing
        f_rate = cold_valid ? gbps(traffic, f.cold) : nothing
        d_rate = cold_valid ? gbps(traffic, d.cold) : nothing
        share = cold_valid ? 100 * f_rate * 1e9 / limits.copy_bandwidth : nothing
        println(
            "$(rows)×$(columns) | ", fixed(traffic / 1e6, 2), " | ", cold_valid ? copies : "—",
            " | ", fixed(f.warm), "/", fixed(f_cold), "/", fixed(f.replay), ", ",
            fixed(f_rate), ", ", share === nothing ? "—" : "$(fixed(share, 0))%",
            " | ", fixed(d.warm), "/", fixed(d_cold), ", ", fixed(d_rate),
            " | ", fixed(blocks_per_sm(fused, limits), 2), " | ", fused.schedule,
        )
    end
end

run(; shapes = Tuple(Tuple(parse.(Int, split(s, "x"))) for s in split(get(ENV, "SHAPES", "256x32,259x32,8192x32,8193x32,1024x1024,128x1024,128x4096"), ",")))
