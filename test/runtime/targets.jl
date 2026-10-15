@testset "Core runtime and placement" begin
    x = reshape(Float32.(1:12), 3, 4)
    f(x) = sum(abs2, x)
    ex = prepare(f, x)
    @test ex.runtime isa Tempest.CPURuntime
    @test isempty(ex.scratch)
    @test execute!(ex)[1][] ≈ f(x)
    @test prepare(f, view(x, :, 1:2)).runtime isa Tempest.CPURuntime
    @test prepare(f, transpose(x)).runtime isa Tempest.CPURuntime
    @test prepare(x -> x * x, 2.0f0).runtime isa Tempest.CPURuntime
    @test prepare(f, x; target = CPUTarget()).runtime isa Tempest.CPURuntime
    @test_throws ArgumentError prepare(f, x; target = :cuda)
    @test_throws ArgumentError CUDATarget(blas_workspace_bytes = -1)
    @test occursin("Target: CPU", sprint(explain, ex))
end
