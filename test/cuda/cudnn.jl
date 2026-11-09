# Attention on cuDNN: a library whose templates are attention as cuDNN's SDPA
# engines compute it, the scores and softmax in Float32 and the output in the
# operands' type, with or without the log-sum-exp beside it. A program that
# computes attention so, in either half type, with heads grouped or not,
# masked or not, is one cuDNN call, and agrees with the program; one that
# computes something else, or whose arrays the engines cannot read, is
# Tempest's, and so is any on a target that keeps the softmax's two passes.
include("../fixtures/programs.jl")
include("../fixtures/operations.jl")
include("../fixtures/sdpa.jl")

@testset "attention on cuDNN" begin
    cuDNNExt = Base.get_extension(Tempest, :cuDNNExt)
    @test cuDNNExt !== nothing
    called(ex) = [d for d in ex.kernels if d isa cuDNNExt.BuiltAttention]
    half(H, dims...) = H.(single(data(dims...; scale = 0.3)))
    for (H, d, heads, kv, positions, batch, causal) in ((BFloat16, 64, 8, 2, 200, 2, true), (BFloat16, 128, 4, 4, 128, 1, false),
            (Float16, 64, 4, 1, 96, 2, true), (Float16, 32, 2, 2, 64, 3, false))
        q, k, v = half(H, d, heads, positions, batch), half(H, d, kv, positions, batch), half(H, d, kv, positions, batch)
        p = trace((q, k, v) -> sdpa(q, k, v; causal), q, k, v)
        ex = prepare(p, CuArray(q), CuArray(k), CuArray(v); target = T.CUDATarget(T.cuDNNAttention()))
        # Heads and batch are interchangeable to attention, and unmasked the
        # query positions are too: cuDNN is offered each way, and the
        # decision measures them.
        call = only(called(ex)).call
        @test call.causal == causal && call.q.dims[1] == call.k.dims[1] == d && call.k.dims[3] == positions
        @test prod(call.q.dims[2:4]) == heads * positions * batch && call.k.dims[2] * call.k.dims[4] == kv * batch
        @test !causal || call.q.dims[3] == positions
        @test agree(C.leaves(outputs(execute!(ex))), C.leaves(interpret(p, q, k, v)); rtol = 1.0e-2)
    end
    q, k, v = half(BFloat16, 64, 4, 128, 2), half(BFloat16, 64, 2, 128, 2), half(BFloat16, 64, 2, 128, 2)
    claimed(f, args...; target = T.CUDATarget(T.cuDNNAttention())) = !isempty(called(prepare(trace(f, args...), map(CuArray, args)...; target)))
    @test claimed((q, k, v) -> sdpa(q, k, v; causal = true), q, k, v)
    # Attention as the fixture spells it computes the same; with the output
    # kept in Float32 it is another function.
    @test claimed((q, k, v) -> attention(q, k, v; causal = true), q, k, v)
    @test !claimed((q, k, v) -> sdpa(q, k, v; causal = true, round = false), q, k, v)
    # Arrays the engines do not read: Float32, features not a multiple of 8,
    # or keys stored positions first.
    @test !claimed((q, k, v) -> sdpa(q, k, v; causal = true), map(single, (q, k, v))...)
    @test !claimed((q, k, v) -> sdpa(q, k, v; causal = true), half(BFloat16, 20, 4, 64, 1), half(BFloat16, 20, 2, 64, 1), half(BFloat16, 20, 2, 64, 1))
    @test !claimed((q, k, v) -> sdpa(q, permutedims(k, (2, 3, 1, 4)), v; causal = true), q, permutedims(k, (3, 1, 2, 4)), v)
    # The engines rescale the softmax online, which this target forbids.
    @test !claimed((q, k, v) -> sdpa(q, k, v; causal = true), q, k, v; target = T.CUDATarget(T.cuDNNAttention(); online = false))
    # Among cuTile's kernels, measured against them, it computes the same.
    p = trace((q, k, v) -> sdpa(q, k, v; causal = true), q, k, v)
    ex = prepare(p, CuArray(q), CuArray(k), CuArray(v); target = T.CUDATarget(T.cuDNNAttention(), T.cuTileGenerator()))
    @test agree(C.leaves(outputs(execute!(ex))), C.leaves(interpret(p, q, k, v)); rtol = 1.0e-2)
    # With the log-sum-exp: one call, its statistics written where Tempest
    # stores them, a query's heads next to one another, which the engines
    # write through their strides at no cost (58.9 μs with them, 59.6 without,
    # at 64 features, 8 heads, 1024 positions, batch 2, causal).
    for (H, heads, kv, positions, batch, causal) in ((BFloat16, 8, 2, 200, 2, true), (Float16, 4, 4, 128, 1, false))
        q, k, v = half(H, 64, heads, positions, batch), half(H, 64, kv, positions, batch), half(H, 64, kv, positions, batch)
        p = trace((q, k, v) -> sdpa(q, k, v; causal, statistics = true), q, k, v)
        ex = prepare(p, CuArray(q), CuArray(k), CuArray(v); target = T.CUDATarget(T.cuDNNAttention()))
        # Unmasked, the query positions may stand for heads of a group as
        # well: one statistic per query, however the decision binds them.
        call = only(called(ex)).call
        @test call.causal == causal && call.stats !== nothing && prod(call.stats.dims) == heads * positions * batch
        @test agree(C.leaves(outputs(execute!(ex))), C.leaves(interpret(p, q, k, v)); rtol = 1.0e-2)
    end
end

# Two programs whose plans take cuDNN's attention, called at once from tasks
# on two threads, each result as the program gives it alone: each task
# executes on its own handle. On one handle per context, 33 of 1000 results
# were wrong on GB10, silently. In a process of its own, with threads (see
# fixtures/at_once.jl); it prints its failures.
@testset "cuDNN attention of two programs at once" begin
    child = `$(Base.julia_cmd()) --threads=3 --project=$(Base.active_project()) $(joinpath(@__DIR__, "..", "fixtures", "at_once.jl")) cuDNN`
    output = IOBuffer()
    process = run(pipeline(ignorestatus(child); stdout = output, stderr = output))
    success(process) || println(stderr, String(take!(output)))
    @test success(process)
end
