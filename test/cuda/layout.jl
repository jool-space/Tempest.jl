include("../fixtures/layout.jl")

@testset "CUDA layouts and deterministic pullbacks" begin
    check_layout_cases(CUDATarget(); atol = 2e-4, rtol = 2e-4, gpu = true)
end

@testset "Large repeated gather replay" begin
    x = Float32.(range(-0.5, 0.7; length = 1025))
    inds = [mod(i * 13, 1025) + 1 for i = 1:4097]
    f(x) = sum(abs2, view(x, inds))
    ex = without_cutile(() -> prepare(f, x; target = CUDATarget()))
    capture!(ex)
    expected = f(2x)
    gradient = zeros(Float32, length(x))
    for i in inds
        gradient[i] += 4x[i]
    end
    inds .= 1
    GC.gc() # the runtime must own every device index buffer used by the graph
    bind!(ex, 2x)
    baseline = map(Array, replay!(ex))
    @test baseline[1][] ≈ expected rtol=2e-5
    @test baseline[2] ≈ gradient rtol=2e-5
    for _ = 1:3
        @test map(Array, replay!(ex)) == baseline
    end
end
