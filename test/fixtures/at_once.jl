# Two programs whose plans take a library's call, called at once from tasks
# on threads of their own, round after round, each result as the program
# gives it alone. The test runner's workers have one thread, on which tasks
# take turns and no two calls run at once, so two tasks inside a worker prove
# nothing: the shared handles passed so. cuda/cudnn.jl and cuda/products.jl
# run this in a child process with threads, `julia --threads=3 at_once.jl
# LIBRARY`, LIBRARY `cuDNN` or `cuBLAS`. It prints its failures.
using Tempest, Test, NNlib, CUDACore
import cuBLAS, cuDNN, cuTile
include(joinpath(@__DIR__, "..", "setup.jl"))

# The results of `rounds` rounds that differ from each program's alone: in
# each, both programs called `calls` times, each from a task of its own and
# on its two sets of inputs in turn, the tasks released together.
function wrong_at_once(cs, inputs; rounds, calls, rtol)
    alone = [[Float32.(Array(cs[p](inputs[p][s]...))) for s in 1:2] for p in 1:2]
    sum(1:rounds) do _
        go = Base.Event()
        tasks = map(1:2) do p
            Threads.@spawn begin
                wait(go)
                ys = [(mod1(i, 2), cs[p](inputs[p][mod1(i, 2)]...)) for i in 1:calls]
                count(((s, y),) -> !isapprox(Float32.(Array(y)), alone[p][s]; rtol), ys)
            end
        end
        notify(go)
        sum(fetch, tasks)
    end
end

@test Threads.nthreads() > 1
library = only(ARGS)

if library == "cuDNN"
    include("sdpa.jl")
    @testset "cuDNN attention of two programs at once" begin
        cuDNNExt = Base.get_extension(Tempest, :cuDNNExt)
        half(dims...; scale) = CuArray(BFloat16.(single(data(dims...; scale))))
        operands(p, s) = (half(64, 8, 1024, 2; scale = 0.01 * (p + 2s)), half(64, 2, 1024, 2; scale = 0.02 * (p + 2s)),
            half(64, 2, 1024, 2; scale = 0.03 * (p + 2s)))
        inputs = [[operands(p, s) for s in 1:2] for p in 1:2]
        cs = [compile((q, k, v) -> sdpa(q, k, v; causal = true), inputs[p][1]...; target = T.CUDATarget(T.cuDNNAttention()), capture = false)
            for p in 1:2]
        @test all(c -> count(l -> l isa cuDNNExt.Attention, c.executable.launches) == 1, cs)
        @test wrong_at_once(cs, inputs; rounds = 20, calls = 10, rtol = 1.0e-2) == 0
    end
end

# A product of 64 by 64 summing 16,384 or 131,072, one cuBLAS call, and its
# tanh in a kernel after it, uncaptured and captured. cuBLAS splits the
# longer sum, writing 1.5 MB of the workspace; with a handle per task and
# one workspace per context, 780 of 4000 such results were wrong on GB10,
# silently.
if library == "cuBLAS"
    @testset "cuBLAS products of two programs at once, summing $k, $(capture ? "captured" : "uncaptured")" for k in (1 << 14, 1 << 17),
            capture in (false, true)
        cuBLASExt = Base.get_extension(Tempest, :cuBLASExt)
        operands(p, s) = (CuArray(reshape(Float32.(0.01 .* sin.(p .+ s .* (1:(64k)))), 64, k)), CuArray(reshape(Float32.(cos.(3p .+ s .* (1:(64k)))), k, 64)))
        inputs = [[operands(p, s) for s in 1:2] for p in 1:2]
        cs = [compile((a, b) -> tanh.(a * b), inputs[p][1]...; target = T.CUDATarget(T.cuBLASProducts()), capture) for p in 1:2]
        @test all(c -> count(l -> l isa cuBLASExt.Product, c.executable.launches) == 1, cs)
        @test wrong_at_once(cs, inputs; rounds = 100, calls = 20, rtol = 1.0e-3) == 0
    end
end
