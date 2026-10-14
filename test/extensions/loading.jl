# Package loading, checked in fresh Julia processes: the test workers have
# already loaded the GPU packages, so they cannot observe what core use loads
# or what loading a provider late changes.
function run_isolated(name)
    script = joinpath(dirname(@__DIR__), "isolated", name)
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) $script`
    output = IOBuffer()
    process = run(pipeline(ignorestatus(cmd); stdout = output, stderr = output))
    success(process) || println(stderr, "--- $name ---\n", String(take!(output)), "\n--- end ---")
    success(process)
end

@testset "Core use in a fresh process" begin
    @test run_isolated("core_loading.jl")
end

if CUDA_FUNCTIONAL && CUTILE_ENABLED
    @testset "Loading cuTile after preparation" begin
        @test run_isolated("load_order.jl")
    end
end
