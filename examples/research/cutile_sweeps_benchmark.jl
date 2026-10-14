# Forward-only comparison against the pre-region planner with cuTile loaded.
# Compilation and preparation are reported separately from captured replay.
# Run: julia --project=examples examples/research/cutile_sweeps_benchmark.jl
include("cutile_sweeps_tests.jl")
using Statistics

function replay_microseconds(replay; samples = 7, repetitions = 200)
    for _ in 1:10
        replay()
    end
    CUDACore.synchronize()
    samples = map(1:samples) do _
        elapsed = CUDACore.@elapsed begin
            for _ in 1:repetitions
                replay()
            end
        end
        1e6 * elapsed / repetitions
    end
    (minimum = minimum(samples), median = median(samples), maximum = maximum(samples))
end

function run_benchmark()
    CUDACore.functional() || error("This benchmark requires functional CUDA")
    rng = MersenneTwister(20260921)
    extension = Base.get_extension(Tempest, :CUDAExt)
    println((julia = VERSION, cutile = Base.pkgversion(S.ct), device = name(device())))
    for rows in (256, 259, 8192, 8193)
        columns = 32
        inputs = (randn(rng, Float32, rows, columns),
                  0.2f0 .* randn(rng, Float32, rows, columns),
                  randn(rng, Float32, rows))
        device_inputs = CuArray.(inputs)
        program = trace(custom_normalization, inputs...)
        boundary = only(reduction_candidates(program))
        outlined = standalone_program(boundary)
        expected = custom_normalization(inputs...)
        values = Dict(zip(D.tree_leaves(program.output), D.tree_leaves(expected)))
        region_reference = Tuple(values[id] for id in boundary.outputs)

        streamed_preparation = @elapsed streamed = prepare_sweeps(outlined.program, device_inputs...)
        check_outputs(launch!(streamed), region_reference)
        CUDACore.synchronize()
        graph = CUDACore.capture() do
            launch!(streamed)
        end
        executable = CUDACore.instantiate(graph)
        streamed_kernels = count(==(CUDACore.CU_GRAPH_NODE_TYPE_KERNEL), extension.graph_node_types(graph))
        @test streamed_kernels == 1

        current_preparation = @elapsed current = prepare_forward(
            custom_normalization, device_inputs...; fuse_regions = false)
        check_outputs(execute!(current), expected)
        capture!(current)
        check_outputs(replay!(current), expected)
        kernels = count(==(CUDACore.CU_GRAPH_NODE_TYPE_KERNEL),
                        extension.graph_node_types(current.runtime.graph))
        # Finish both compilations before timing either one. Alternate their
        # measurement order so a fixed order does not determine clock/cache state.
        runners = (() -> CUDACore.launch(executable), () -> replay!(current))
        measurements = (Float64[], Float64[])
        for round in 1:7
            order = isodd(round) ? (1, 2) : (2, 1)
            for index in order
                time = replay_microseconds(runners[index]; samples = 1).median
                push!(measurements[index], time)
            end
        end
        summary(times) = (minimum = minimum(times), median = median(times), maximum = maximum(times))
        println((; label = :streamed, rows, columns,
                   preparation_seconds = streamed_preparation,
                   compilation_seconds = streamed.compilation_seconds,
                   kernels = streamed_kernels, output_bytes = sum(sizeof, streamed.outputs),
                   workspace_bytes = 0, microseconds = summary(measurements[1])))
        println((; label = :current_planner, rows, columns,
                   preparation_seconds = current_preparation, kernels,
                   providers = current.providers, value_bytes = sum(sizeof, current.buffers),
                   microseconds = summary(measurements[2])))
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
