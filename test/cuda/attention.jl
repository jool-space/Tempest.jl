# Attention on cuTile: generated from its reference as one kernel, its
# gradient as FlashAttention's backward, and decode over a cache read up to
# lengths the device holds; and a layer's gradient whose sums recompute the
# product they read.

# Attention generated from its reference: one kernel, a fold over keys whose
# scores and logsumexp are recomputed in it, online or in two passes.
const GENERATED_ATTENTION = [
    ("causal", (16, 2, 2, 200, 2), true),
    ("causal, grouped heads", (16, 4, 2, 200, 2), true),
    ("grouped heads", (16, 4, 2, 200, 2), false),
    ("causal, odd sizes", (32, 3, 1, 70, 3), true),
    ("one head", (8, 1, 1, 133, 1), false),
]

# The folds a program's streaming plan recomputes where they are read, as
# numbered in the graph `prepare` plans for `target`.
streaming(p, target) = C.recomputable_folds(C.permits_rewrites(target) ? C.rewrite(C.lower(p)) : C.lower(p))

@testset "attention generated on cuTile" begin
    @testset "$name" for (name, (D, H, KV, S, B), causal) in GENERATED_ATTENTION
        q, k, v = single(data(D, H, S, B; scale = 0.1)), single(data(D, KV, S, B; scale = 0.2)), single(data(D, KV, S, B; scale = 0.3))
        p = trace((q, k, v) -> Ops.attention(q, k, v; causal), q, k, v)
        for online in (true, false)
            target = T.CUDATarget(cuTile; online)
            ex = prepare(p, map(CuArray, (q, k, v))...; target, recompute = streaming(p, target))
            kernel = only(ex.kernels)
            @test !(kernel isa C.FloorKernels)
            @test occursin("online softmax", string(kernel)) == online
            # A causal mask is a formula: the steps it rules out are
            # skipped, where the queries have a tile axis of their own.
            # Grouped heads may share it with the group, which the
            # formula's bound over a tile cannot see through.
            (!causal || H == KV) && @test occursin("masked steps skipped", string(kernel)) == causal
            @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, q, k, v)]; rtol = 1.0e-4)
        end
    end
    # Softmax spelled with a maximum and a sum: rewritten into the same one
    # kernel, at full precision and in TF32.
    q, k, v = single(data(16, 2, 70, 2; scale = 0.1)), single(data(16, 2, 70, 2; scale = 0.2)), single(data(16, 2, 70, 2; scale = 0.3))
    p = trace(q, k, v) do q, k, v
        s = C.contract(k, q, (1, 2, 3, 4), (1, 2, 5, 4), (3, 2, 5, 4)) .* 0.25f0
        s = ifelse.(reshape(1:size(s, 1), :, 1, 1) .<= reshape(1:size(s, 3), 1, 1, :), s, -Inf32)
        e = exp.(s .- maximum(s; dims = 1))
        C.contract(v, e ./ sum(e; dims = 1), (1, 2, 3, 4), (3, 2, 5, 4), (1, 2, 5, 4))
    end
    for precision in (:full, :tf32)
        target = T.CUDATarget(cuTile; precision)
        ex = prepare(p, map(CuArray, (q, k, v))...; target, recompute = streaming(p, target))
        @test occursin("online softmax", string(only(ex.kernels)))
        @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, q, k, v)]; rtol = precision === :tf32 ? 1.0e-2 : 1.0e-4)
    end
    # In half precision the rewrite reads the exponential through the
    # product's conversion to Float32: one kernel, online, the weights
    # rounded to BFloat16 for the tensor cores.
    qh, kh, vh = map(x -> BFloat16.(single(x)), (data(16, 2, 70, 2; scale = 0.1), data(16, 2, 70, 2; scale = 0.2), data(16, 2, 70, 2; scale = 0.3)))
    p = trace((q, k, v) -> Ops.attention(q, k, v; causal = true), qh, kh, vh)
    ex = prepare(p, map(CuArray, (qh, kh, vh))...; target = T.CUDATarget(cuTile), recompute = streaming(p, T.CUDATarget(cuTile)))
    @test occursin("online softmax", string(only(ex.kernels)))
    @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, qh, kh, vh)]; rtol = 1.0e-2)
    # The reference's gradient: every product generated, also those that
    # sum over key heads shared by one query head each.
    q, k, v = single(data(16, 2, 70, 2; scale = 0.1)), single(data(16, 2, 70, 2; scale = 0.2)), single(data(16, 2, 70, 2; scale = 0.3))
    d = differentiate(trace((q, k, v) -> sum(abs2, Ops.attention(q, k, v; causal = true)), q, k, v))
    ex = prepare(d, map(CuArray, (q, k, v))...; target = T.CUDATarget(cuTile))
    @test all(d -> !(d isa C.FloorKernels), ex.kernels)
    @test agree(filter(!isnothing, C.leaves(outputs(execute!(ex)))), filter(!isnothing, C.leaves(interpret(d, q, k, v))); rtol = 1.0e-4)
    # Its streaming plan, as FlashAttention's backward: the scores and
    # their gradient recomputed in every kernel that reads them, the
    # log-sum-exp in a sweep over keys of its own, and nothing the size of
    # the scores stored. The key gradients sum over the queries after each
    # key and the query gradients over the keys before each query, in one
    # sweep, each fold stepping where its mask leaves it work.
    ex = prepare(d, map(CuArray, (q, k, v))...; target = T.CUDATarget(cuTile), recompute = streaming(d, T.CUDATarget(cuTile)))
    @test all(d -> !(d isa C.FloorKernels), ex.kernels)
    @test all(v -> !ex.plan.stored[v] || prod(ex.plan.graph.values[v].dims) < 70 * 70, eachindex(ex.plan.graph.values))
    @test any(d -> occursin("folds stepping by their masks", string(d)), ex.kernels)
    @test agree(filter(!isnothing, C.leaves(outputs(execute!(ex)))), filter(!isnothing, C.leaves(interpret(d, q, k, v))); rtol = 1.0e-4)
    # With grouped heads, the products into the key and value gradients sum
    # over the query heads of a group and the queries, which no strided
    # axis spans: the queries are tiled and the heads swept in turn.
    g = single(data(16, 4, 70, 2; scale = 0.1))
    d = differentiate(trace((q, k, v) -> sum(abs2, Ops.attention(q, k, v; causal = true)), g, k, v))
    ex = prepare(d, map(CuArray, (g, k, v))...; target = T.CUDATarget(cuTile))
    @test all(d -> !(d isa C.FloorKernels), ex.kernels)
    @test any(d -> occursin("in turn", string(d)), ex.kernels)
    @test agree(filter(!isnothing, C.leaves(outputs(execute!(ex)))), filter(!isnothing, C.leaves(interpret(d, g, k, v))); rtol = 1.0e-4)
    # A residual is part of the one kernel, also with a key head per query
    # head, where the reference groups heads along an axis of extent one.
    x = single(data(16, 2, 70, 2; scale = 0.5))
    p = trace((q, k, v, x) -> Ops.attention(q, k, v; causal = true) .+ x, q, k, v, x)
    ex = prepare(p, map(CuArray, (q, k, v, x))...; target = T.CUDATarget(cuTile), recompute = streaming(p, T.CUDATarget(cuTile)))
    @test length(ex.kernels) == 1 && !(only(ex.kernels) isa C.FloorKernels)
    @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, q, k, v, x)]; rtol = 1.0e-4)
end

# A layer's gradient whose sums recompute the hidden product they read:
# generated as folds that recompute it, stepping at a product's depths.
@testset "sums that recompute a product, generated" begin
    x, w1, b1 = single(data(64, 256; scale = 0.5)), single(data(256, 64; scale = 0.05)), single(data(256; scale = 0.1))
    w2, b2 = single(data(64, 256; scale = 0.05)), single(data(64; scale = 0.1))
    d = differentiate(trace((x, w1, b1, w2, b2) -> sum(abs2, w2 * tanh.(w1 * x .+ b1) .+ b2), x, w1, b1, w2, b2))
    target = T.CUDATarget(cuTile)
    ex = prepare(d, map(CuArray, (x, w1, b1, w2, b2))...; target, recompute = streaming(d, target))
    @test all(k -> !(k isa C.FloorKernels), ex.kernels)
    @test agree(filter(!isnothing, C.leaves(outputs(execute!(ex)))), filter(!isnothing, C.leaves(interpret(d, x, w1, b1, w2, b2))); rtol = 1.0e-4)
end

# Decode attention: one query per head over a cache of keys, each
# sequence's read up to a length the device holds, with query heads
# grouped over key heads as Vallmo's cache holds them, (features,
# positions, key heads, sequences). The length is a mask whose formula
# reads it, and the sweep over keys stops where it ends, whether the
# scores are recomputed in it or stored before.
@testset "attention up to lengths the device holds" begin
    function decode(q, k, v, len)
        s = Einops.@einsum(k, q, "d n h b, d (g h) b -> n g h b", h = size(k, 3)) .* 0.25f0
        s = ifelse.((Int32(1):Int32(size(k, 2))) .<= reshape(len, 1, 1, 1, :), s, -Inf32)
        Einops.@einsum(v, exp.(s .- T.logsumexp(s; dims = 1)), "d n h b, n g h b -> d (g h) b")
    end
    q, k, v = single(data(16, 8, 3; scale = 0.1)), single(data(16, 200, 2, 3; scale = 0.2)), single(data(16, 200, 2, 3; scale = 0.3))
    n = Int32[37, 200, 1]
    p = trace(decode, q, k, v, n)
    target = T.CUDATarget(cuTile)
    for recompute in ((), streaming(p, target))
        ex = prepare(p, map(CuArray, (q, k, v, n))...; target, recompute)
        @test any(k -> occursin("masked steps skipped", string(k)), ex.kernels)
        @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, q, k, v, n)]; rtol = 1.0e-4)
        # Another length is the same program.
        bind!(ex, 4 => CuArray(Int32[200, 5, 64]))
        @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, q, k, v, Int32[200, 5, 64])]; rtol = 1.0e-4)
    end
    # Its claims are timed at the cache's length, so that the sweep runs
    # whole, as it is priced, whatever lengths it is prepared with.
    graph = C.lower(p)
    planned = C.plan(graph)
    d = C.Decider(target, graph; measure = false)
    foreach(g -> C.buffers!(d, planned, g), planned.groups)
    @test Array(d.buffers[only(i for i in graph.arguments if graph.values[i].eltype === Int32)]) == fill(Int32(200), 3)
    # Scores stored with their mask, as a step whose layers share one mask
    # stores them: a block whose positions all lie past the length sweeps
    # no features and stores minus infinity.
    scored(q, k, v, len) = (s = Einops.@einsum(k, q, "d n h b, d (g h) b -> n g h b", h = size(k, 3)) .* 0.25f0;
        s = ifelse.((Int32(1):Int32(size(k, 2))) .<= reshape(len, 1, 1, 1, :), s, -Inf32);
        (Einops.@einsum(v, exp.(s .- T.logsumexp(s; dims = 1)), "d n h b, n g h b -> d (g h) b"), s))
    q3, k3, v3 = single(data(64, 8, 3; scale = 0.1)), single(data(64, 200, 2, 3; scale = 0.2)), single(data(64, 200, 2, 3; scale = 0.3))
    p = trace(scored, q3, k3, v3, n)
    ex = withenv(() -> prepare(p, map(CuArray, (q3, k3, v3, n))...; target), "TEMPEST_PRODUCT_TILE" => "64,4,16")
    @test any(k -> occursin("masked tiles skipped", string(k)), ex.kernels)
    for m in (n, Int32[200, 5, 64])
        bind!(ex, 4 => CuArray(m))
        (o, s), (o′, s′) = map(Array, C.leaves(outputs(execute!(ex)))), interpret(p, q3, k3, v3, m)
        @test agree([o], [o′]; rtol = 1.0e-4)
        @test isinf.(s) == isinf.(s′) && agree([s[isfinite.(s)]], [s′[isfinite.(s′)]]; rtol = 1.0e-4)
    end
    # Over a BFloat16s.BFloat16 cache, the probabilities rounded to it
    # before their product, as Vallmo's decode rounds them: the product
    # reads the length only through their mask.
    half(q, k, v, len) = (s = Einops.@einsum(k, q, "d n h b, d (g h) b -> n g h b", h = size(k, 3)) .* 0.25f0;
        s = ifelse.((Int32(1):Int32(size(k, 2))) .<= reshape(len, 1, 1, 1, :), s, -Inf32);
        Einops.@einsum(v, BFloat16s.BFloat16.(exp.(s .- T.logsumexp(s; dims = 1))), "d n h b, n g h b -> d (g h) b"))
    qh, kh, vh = BFloat16s.BFloat16.(q), BFloat16s.BFloat16.(k), BFloat16s.BFloat16.(v)
    p = trace(half, qh, kh, vh, n)
    ex = prepare(p, map(CuArray, (qh, kh, vh, n))...; target)
    @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, qh, kh, vh, n)]; rtol = 1.0e-2)
    # Scores whose query a mask in memory chooses lane by lane, as a
    # rotation's stored mask does, stepping 128 deep along the features:
    # tileiras 13.4 gets the product wrong when the mask is loaded as Bool,
    # so it is loaded as bytes.
    chosen(q, k, m) = Einops.@einsum(k, BFloat16s.BFloat16.(ifelse.(m, q, -q)), "d n h b, d (g h) b -> n g h b", h = size(k, 3))
    q2, k2 = single(data(256, 8, 1; scale = 0.1)), BFloat16s.BFloat16.(single(data(256, 128, 2, 1; scale = 0.2)))
    m = isodd.(reshape(1:length(q2), size(q2)) .÷ 3)
    p = trace(chosen, q2, k2, m)
    ex = withenv(() -> prepare(p, map(CuArray, (q2, k2, m))...; target), "TEMPEST_PRODUCT_TILE" => "128,4,128")
    @test occursin("tile 128×4×128", string(only(ex.kernels)))
    @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, q2, k2, m)]; rtol = 1.0e-2)
    # With a key head per query head, the scores of one query are a product
    # of one row, no product of tiles: its sweep over keys is a fold that
    # recomputes them, generated as one.
    k1, v1 = single(data(16, 200, 8, 3; scale = 0.2)), single(data(16, 200, 8, 3; scale = 0.3))
    p = trace(decode, q, k1, v1, n)
    ex = prepare(p, map(CuArray, (q, k1, v1, n))...; target, recompute = streaming(p, target))
    @test !(only(ex.kernels) isa C.FloorKernels)
    @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, q, k1, v1, n)]; rtol = 1.0e-4)
end
