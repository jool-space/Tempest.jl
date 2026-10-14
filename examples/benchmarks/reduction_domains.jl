# Matched sum reductions against CUDACore/GPUArrays' public sum! interface.
# Run: julia --project=examples examples/benchmarks/reduction_domains.jl
using Tempest
using CUDACore
using Statistics
import cuBLAS, cuTile

function graph_counts(graph)
    types = Base.get_extension(Tempest, :CUDAExt).graph_node_types(graph)
    (
        kernels = count(==(CUDACore.CU_GRAPH_NODE_TYPE_KERNEL), types),
        copies = count(==(CUDACore.CU_GRAPH_NODE_TYPE_MEMCPY), types),
        allocations = count(==(CUDACore.CU_GRAPH_NODE_TYPE_MEM_ALLOC), types),
        frees = count(==(CUDACore.CU_GRAPH_NODE_TYPE_MEM_FREE), types),
    )
end

function capture_repetitions(run; repetitions)
    run()
    CUDACore.synchronize()
    graph = CUDACore.capture() do
        for _ = 1:repetitions
            run()
        end
    end
    counts = graph_counts(graph)
    # GPUArrays may allocate partial sums inside sum!. Those captured allocation
    # nodes need release before the next graph launch. Include that cost and
    # report the nodes; Tempest itself must have no allocation/free nodes.
    flags =
        counts.allocations == 0 ? 0 : CUDACore.CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
    executable = CUDACore.instantiate(graph, flags)
    (; graph, executable, counts)
end

function reduction_benchmark(; repetitions = 100, rounds = 9)
    CUDACore.functional() || error("This benchmark requires functional CUDA")
    println((
        julia = VERSION,
        cutile = Base.pkgversion(cuTile),
        cudacore = Base.pkgversion(CUDACore),
        device = name(device()),
        repetitions,
        rounds,
    ))
    println(
        "Repeated resident inputs; times include device graph execution, not compilation or host submission per operation.",
    )
    println(
        "Reference allocation/free nodes are reported, not excluded. Copy is a cache/launch sanity check, not a DRAM roofline.",
    )
    for dims in ((64, 97), (259, 32), (512, 1024)), axes in ((1,), (2,), :)
        x = reshape(0.2f0 .+ abs.(sin.(Float32.(1:prod(dims)))), dims)
        input = CuArray(x)
        f(x) = sum(x; dims = axes)
        expected = f(x)
        expected = expected isa Number ? fill(expected) : expected
        preparation_seconds = @elapsed ex = prepare_forward(f, input)
        reference = similar(input, Float32, size(expected))
        copied = similar(input)
        runs = (() -> execute!(ex), () -> sum!(reference, input), () -> copyto!(copied, input))
        # Compile and warm all paths before capturing or measuring any of them.
        for run in runs
            run()
        end
        CUDACore.synchronize()
        @assert isapprox(Array(outputs(ex)), expected; rtol = 4.0f-4, atol = 4.0f-5)
        @assert isapprox(Array(reference), expected; rtol = 4.0f-4, atol = 4.0f-5)
        batches = map(run -> capture_repetitions(run; repetitions), runs)
        @assert batches[1].counts.allocations == batches[1].counts.frees == 0
        for batch in batches
            CUDACore.launch(batch.executable)
        end
        CUDACore.synchronize()
        @assert isapprox(Array(outputs(ex)), expected; rtol = 4.0f-4, atol = 4.0f-5)
        @assert isapprox(Array(reference), expected; rtol = 4.0f-4, atol = 4.0f-5)
        samples = ntuple(_ -> Float64[], length(runs))
        for round = 1:rounds
            order = isodd(round) ? eachindex(runs) : reverse(eachindex(runs))
            for index in order
                # Warm this exact graph before every timed sample, including
                # allocation/release behavior and the device's clock ramp.
                for _ = 1:10
                    CUDACore.launch(batches[index].executable)
                end
                elapsed = CUDACore.@elapsed CUDACore.launch(batches[index].executable)
                push!(samples[index], 1e6 * elapsed / repetitions)
            end
        end
        labels = (:tempest, :gpuarrays_sum!, :copy)
        for index in eachindex(runs)
            times = samples[index]
            println((;
                dims,
                axes,
                implementation = labels[index],
                preparation_seconds = index == 1 ? preparation_seconds : nothing,
                workspace_bytes = index == 1 ? sum(sizeof, ex.scratch; init = 0) : nothing,
                batch_nodes = batches[index].counts,
                microseconds = (minimum(times), median(times), maximum(times)),
            ))
        end
        flush(stdout)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    reduction_benchmark()
end
