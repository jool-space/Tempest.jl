# Package loading, checked in fresh Julia processes: the test workers have
# already loaded the GPU packages, so they cannot observe what core use loads
# or what loading a provider late changes.
function run_isolated(name, args...; output = IOBuffer())
    script = joinpath(dirname(@__DIR__), "isolated", name)
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $script $args`
    process = run(pipeline(ignorestatus(cmd); stdout = output, stderr = output))
    success(process) || println(stderr, "--- $name ---\n", String(take!(output)), "\n--- end ---")
    success(process)
end

@testset "Core use in a fresh process" begin
    @test run_isolated("core_loading.jl")
end

@testset "An implementation package registers and activates" begin
    @test run_isolated("out_of_tree.jl")
end

if CUDA_FUNCTIONAL && CUTILE_ENABLED
    @testset "Loading cuTile after preparation" begin
        @test run_isolated("load_order.jl")
    end

    @testset "Candidates follow ranks, not load order" begin
        lists = map((("cuDNN", "cuTile", "CUDACore", "cuBLAS"), ("cuBLAS", "CUDACore", "cuTile", "cuDNN"))) do order
            output = IOBuffer()
            @test run_isolated("registry_order.jl", order...; output)
            last(split(strip(String(take!(output))), "\n"))
        end
        @test lists[1] == lists[2] ==
              "cuDNN (registry) → cutile (registry) → cuda (target) → cuBLAS (registry)"
    end
end
