# Attention on cuDNN: a library whose templates are attention as cuDNN's SDPA
# engines compute it, the scores and softmax in Float32 and the output in the
# operands' type, with or without the log-sum-exp beside it. A program that
# computes attention so, in either half type, with heads grouped or not,
# masked or not, is one cuDNN call, and agrees with the program; one that
# computes something else, or whose arrays the engines cannot read, is
# Tempest's, and so is any on a target that keeps the softmax's two passes.
include("../fixtures/programs.jl")
include("../fixtures/operations.jl")

const LABELS = (d = 1, g = 2, h = 3, m = 4, b = 5, n = 6)
# As a model writes it: (features, heads, positions, batch). With
# `statistics`, the log-sum-exp of each query's scores too, (heads,
# positions, batch), as a training forward keeps it.
function sdpa(q, k, v; causal, round = true, statistics = false)
    (; d, g, h, m, b, n) = LABELS
    q5 = reshape(q, size(q, 1), size(q, 2) ÷ size(k, 2), size(k, 2), size(q, 3), size(q, 4))
    s = C.contract(Float32.(k), Float32.(q5), (d, h, n, b), (d, g, h, m, b), (n, g, h, m, b)) .* Float32(inv(sqrt(size(q, 1))))
    causal && (s = ifelse.(reshape(1:size(s, 1), :, 1, 1, 1) .<= reshape(1:size(s, 4), 1, 1, 1, :), s, -Inf32))
    lse = T.logsumexp(s; dims = 1)
    o = C.contract(v, exp.(s .- lse), (d, h, n, b), (n, g, h, m, b), (d, g, h, m, b))
    o = reshape(round ? eltype(q).(o) : o, size(q))
    statistics ? (o, reshape(lse, size(q, 2), size(q, 3), size(q, 4))) : o
end

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
