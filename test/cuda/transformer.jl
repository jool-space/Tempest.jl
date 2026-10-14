# A small GPT training step from examples/gpt.jl: batched products, logsumexp,
# runtime token indices, gathers inside permutations, packed and split
# reductions, residual dropout, and Adam state with an advancing seed.
# The CUDA schedule, on whichever providers are loaded, is checked against the
# CPU schedule.
include(joinpath(EXAMPLES, "gpt.jl"))

@testset "Transformer training step on CUDA matches CPU" begin
    G = GPTExample
    text = repeat("the quick brown fox jumps over the lazy dog; ", 40)
    path = tempname()
    write(path, text)
    corpus = G.load_corpus(path)
    cfg = (;
        vocabulary = length(corpus.chars),
        hidden = 32,
        heads = 4,
        kv_heads = 2,
        head_dim = 8,
        ffn = 64,
        layers = 2,
        T = 16,
        B = 4,
        dropout = 0.1,
    )
    rng = G.MersenneTwister(1)
    params = G.parameters(rng, cfg)
    state = (
        first = G.map_leaves(zero, params),
        second = G.map_leaves(zero, params),
        step = 0.0f0,
        seed = Int32[3],
    )
    c = G.constants(cfg)
    batches = [G.batch_arrays(G.get_batch(rng, corpus.data, cfg.T, cfg.B)..., cfg.vocabulary) for _ = 1:4]
    step_loss(p, s, b, c) = G.loss(p, b, c, cfg; seed = s.seed)
    gradients = differentiate(trace(step_loss, params, state, batches[1], c); wrt = (1,))
    step = with_update(gradients) do value, grads, p, s, b, c
        next_params, next_state = G.adam(p, s, only(grads))
        (next_params, next_state, value)
    end
    cpu = prepare(step, params, state, batches[1], c; donate = (1, 2))
    gpu = prepare(step, params, state, batches[1], c; target = CUDATarget(), donate = (1, 2))
    @test occursin("call batched_mul [cuBLAS]", sprint(explain, gpu))
    @test occursin("call logsumexp [cuda]", sprint(explain, gpu))
    capture!(gpu)
    for batch in batches
        bind_argument!(cpu, 3, batch)
        bind_argument!(gpu, 3, batch)
        expected = execute!(cpu)
        actual = replay!(gpu)
        @test Array(actual[3])[] ≈ expected[3][] rtol = 1e-4
        for (x, y) in zip(Tempest.tree_leaves(actual[1]), Tempest.tree_leaves(expected[1]))
            @test Array(x) ≈ y rtol = 1e-3 atol = 1e-5
        end
    end
    @test Array(outputs(gpu)[2].step)[] == 4
    @test Array(outputs(gpu)[2].seed) == Int32[7]
end
