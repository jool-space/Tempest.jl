# Plans compute what programs compute: every program and its differentiated
# form, planned and run kernel by kernel on the CPU, agrees with the program
# interpreter.
include("../fixtures/programs.jl")

@testset "plans" begin
    @testset "$name" for (name, f, args) in [PROGRAMS; REPEATED_WRITES]
        for p in (trace(f, args...), differentiate(trace(f, args...)))
            planned = C.plan(C.lower(p))
            @test agree(C.execute(planned, args...), filter(!isnothing, C.leaves(interpret(p, args...))))
        end
    end
end

# Folds recomputed where they are read, as plans for cuTile recompute them:
# the same results, and attention's scores and logsumexp inside the one
# kernel of the sum over keys.
@testset "plans recomputing folds" begin
    @testset "$name" for (name, f, args) in [PROGRAMS; REPEATED_WRITES]
        for p in (trace(f, args...), differentiate(trace(f, args...)))
            planned = recomputing(C.lower(p))
            @test agree(C.execute(planned, args...), filter(!isnothing, C.leaves(interpret(p, args...))))
        end
    end
    q, k, v = data(4, 4, 6, 2; scale = 0.3), data(4, 2, 6, 2; scale = 0.2), data(4, 2, 6, 2; scale = 0.1)
    p = trace((q, k, v) -> Ops.attention(q, k, v; causal = true), q, k, v)
    planned = recomputing(C.lower(p))
    @test length(planned.groups) == 1
    @test length(C.recomputed_in(only(planned.groups))) == 2
    @test agree(C.execute(planned, q, k, v), [interpret(p, q, k, v)])
end

# A fold that recomputes another is recomputed only where its reader computes
# it once per block. Here a sum over keys recomputing the scores is read by a
# sum over queries, along which it moves, as attention's query gradient is by
# the gradient of the weights before it: it is stored, and its kernel
# recomputes the scores.
@testset "a fold recomputed inside a sweep recomputes no other" begin
    q, k, x = data(4, 8), data(4, 8), data(6, 8)
    function f(q, k, x)
        e = exp.(C.contract(k, q, (1, 2), (1, 3), (2, 3)))
        d = C.contract(k, e, (1, 2), (2, 3), (1, 3))
        C.contract(x, d, (1, 3), (2, 3), (1, 2))
    end
    p = trace(f, q, k, x)
    planned = recomputing(C.lower(p))
    @test all(g -> C.recomputed_in(g) == C.recomputed_in(g; nested = false), planned.groups)
    @test count(g -> !isempty(C.recomputed_in(g)), planned.groups) == 1
    @test agree(C.execute(planned, q, k, x), [interpret(p, q, k, x)])
end

# A broadcast that indexes an array by elements, `getindex.(Ref(s), i, j)`,
# selects an element at each position of the broadcast, as Base's does: a
# gather by the linear index the elements name together, whose shape is
# theirs. A block's scale read at `k ÷ 32` is one.
@testset "a broadcast getindex through a Ref" begin
    x, s, v = data(64, 6), data(2, 6), data(64)
    k, m = reshape(0:63, :, 1), reshape(1:6, 1, :)
    scaled(x, s) = x .* getindex.(Ref(s), k .÷ 32 .+ 1, m)
    @testset "$name" for (name, f, args) in [
        ("by two indices", scaled, (x, s)),
        ("by a linear index", (x, s) -> x .* getindex.(Ref(s), k .÷ 32 .+ 1 .+ 2 .* (m .- 1)), (x, s)),
        ("a block-scaled product", (x, s, v) -> scaled(x, s)' * v, (x, s, v)),
    ]
        p = trace(f, args...)
        @test interpret(p, args...) ≈ f(args...)
        for q in (p, differentiate(trace((args...) -> sum(abs2, f(args...)), args...)))
            @test agree(C.execute(C.plan(C.lower(q)), args...), filter(!isnothing, C.leaves(interpret(q, args...))))
        end
        @test outputs(execute!(prepare(p, args...))) ≈ f(args...)
    end
end
