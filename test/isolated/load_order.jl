# Run in a fresh process by extensions/loading.jl, with a functional GPU and
# cuTile enabled: loading a provider after preparation preserves the existing
# schedule while enabling that provider for subsequently prepared work.
using Tempest, Test
using CUDACore
import cuBLAS

CUDACore.allowscalar(false)
x = CuArray(Float32[0.2, 0.3, 0.4])
f(x) = sum(abs2, x)
plain = prepare(f, x)
capture!(plain)
import cuTile

@testset "Provider loading after preparation" begin
    @test all(==(:cuda), plain.providers)
    @test :cutile ∉ plain.candidates
    @test Array(replay!(plain)[1])[] ≈ 0.29f0
    tiled = prepare(f, x)
    @test :cutile in tiled.providers
    @test :cutile in tiled.candidates
    capture!(tiled)
    bind!(tiled, 2 .* x)
    @test Array(replay!(tiled)[1])[] ≈ 1.16f0
end
