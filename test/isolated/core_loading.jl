# Run in a fresh process by extensions/loading.jl: core use loads no optional
# package and no extension, and a CUDA target without CUDA is an error.
using Tempest, Test

@testset "Core use loads no optional packages" begin
    x = reshape(Float32.(1:12), 3, 4)
    f(x) = sum(abs2, x)
    ex = prepare(f, x)
    @test execute!(ex)[1][] ≈ f(x)
    @test value_and_gradient(trace(f, x), x)[1] ≈ f(x)
    for extension in (:CUDAExt, :cuTileExt, :cuDNNExt, :ChainRulesCoreExt)
        @test Base.get_extension(Tempest, extension) === nothing
    end
    @test_throws ArgumentError prepare(f, x; target = CUDATarget())
    optional = ("CUDACore", "cuBLAS", "cuTile", "cuDNN", "ChainRulesCore")
    @test !any(id -> id.name in optional, keys(Base.loaded_modules))
end
