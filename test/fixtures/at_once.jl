# Two programs whose plans take a library's call, called at once from tasks
# on threads of their own, round after round, each result as the program
# gives it alone. The test runner's workers have one thread, on which tasks
# take turns and no two calls run at once: cuda/cudnn.jl runs this in a
# child process with threads, `julia --threads=3 at_once.jl cuDNN`. It
# prints its failures.
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
