# Optional comparison, after the correctness suite. Compilation is excluded.
# julia --project=examples examples/research/tensor_ir_benchmark.jl LIBRARY
include("tensor_ir_training.jl")
using Statistics

function replay_microseconds(ex; samples = 7, repetitions = 200)
    for _ in 1:10
        replay!(ex)
    end
    CUDACore.synchronize()
    times = map(1:samples) do _
        elapsed = CUDACore.@elapsed begin
            for _ in 1:repetitions
                replay!(ex)
            end
        end
        1e6 * elapsed / repetitions
    end
    (minimum = minimum(times), median = median(times), maximum = maximum(times))
end

function benchmark(library_path)
    CUDACore.functional() || error("This benchmark requires functional CUDA")
    compiler = Compiler(library_path)
    rng = MersenneTwister(20260921)
    extension = Base.get_extension(Tempest, :CUDAExt)
    for (rows, columns) in ((64, 8), (256, 32), (4096, 128))
        inputs = sample_inputs(rng, Float32, rows, columns)
        for (label, implementations) in ((:native, ()),
                                         (:tensor_ir, (NormalizationImplementation(compiler),)))
            ex = prepare(loss, CuArray.(inputs)...; implementations)
            @testset "$label $(rows)×$(columns)" begin
                expected_regions = label === :tensor_ir ? 3 : 0
                @test count(==(:tensor_ir), ex.providers) == expected_regions
                check_outputs(execute!(ex), inputs)
                capture!(ex)
                check_outputs(replay!(ex), inputs)
            end
            graph_nodes = extension.graph_node_types(ex.runtime.graph)
            kernels = count(==(CUDACore.CU_GRAPH_NODE_TYPE_KERNEL), graph_nodes)
            println((; label, rows, columns,
                     host_calls = length(ex.calls), kernels, providers = ex.providers,
                     value_bytes = sum(sizeof, ex.buffers; init = 0),
                     microseconds = replay_microseconds(ex)))
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("Pass libtempest_tensor_ir_runtime.so")
    benchmark(only(ARGS))
end
