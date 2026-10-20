# NNlib's functions whose bodies do not trace are methods on Tempest's traced
# arrays, written in its primitives: they mean what the same computations
# spelled in primitives mean, differentiate alike, and plan as they do.
using NNlib

results(ex) = filter(!isnothing, C.leaves(outputs(execute!(ex))))
gradients(p, args...) = filter(!isnothing, C.leaves(interpret(differentiate(p), args...)))

const NNLIB = [
    ("softmax", (x, t) -> sum(NNlib.softmax(x; dims = 1) .* t),
     (x, t) -> sum(exp.(x .- T.logsumexp(x; dims = 1)) .* t), (data(8, 6), data(8, 6; scale = 0.7))),
    ("softmax over rows", (x, t) -> sum(NNlib.softmax(x; dims = 2) .* t),
     (x, t) -> sum(exp.(x .- T.logsumexp(x; dims = 2)) .* t), (data(8, 6), data(8, 6; scale = 0.7))),
    ("logsoftmax", (x, t) -> sum(NNlib.logsoftmax(x) .* t),
     (x, t) -> sum((x .- T.logsumexp(x; dims = 1)) .* t), (data(8, 6), data(8, 6; scale = 0.7))),
    ("batched_mul", (a, b) -> sum(abs2, NNlib.batched_mul(a, b)),
     (a, b) -> sum(abs2, T.batched_mul(a, b)), (data(3, 4, 2), data(4, 5, 2; scale = 0.3))),
    ("batched_mul of transposes", (a, b) -> sum(abs2, NNlib.batched_mul(NNlib.batched_transpose(b), NNlib.batched_adjoint(a))),
     (a, b) -> sum(abs2, T.batched_mul(permutedims(b, (2, 1, 3)), permutedims(a, (2, 1, 3)))), (data(3, 4, 2), data(4, 5, 2; scale = 0.3))),
    ("batched_mul sharing a batch", (a, b) -> sum(abs2, NNlib.batched_mul(a, b)),
     (a, b) -> sum(abs2, T.batched_mul(a .* ones(1, 1, 2), b)), (data(3, 4, 1), data(4, 5, 2; scale = 0.3))),
]

@testset "NNlib on traced arrays" begin
    @testset "$name" for (name, f, spelled, args) in NNLIB
        p = trace(f, args...)
        @test interpret(p, args...) ≈ f(args...)
        @test all(map(≈, gradients(p, args...), gradients(trace(spelled, args...), args...)))
        for program in (p, differentiate(p))
            @test agree(results(prepare(program, args...)), filter(!isnothing, C.leaves(interpret(program, args...))))
        end
    end
    # A traced factor beside a Julia array, in either order, is a product with
    # a constant.
    a, b = data(3, 4, 2), data(4, 5, 2; scale = 0.3)
    @test interpret(trace(x -> NNlib.batched_mul(x, b), a), a) ≈ NNlib.batched_mul(a, b)
    @test interpret(trace(x -> NNlib.batched_mul(a, x), b), b) ≈ NNlib.batched_mul(a, b)
    # The methods on traced arrays add no ambiguity.
    for f in (NNlib.softmax, NNlib.logsoftmax, NNlib.batched_mul)
        ms = collect(methods(f))
        @test !any(Base.isambiguous(m, n) for m in ms for n in ms if m !== n)
    end
end

if CUDA_FUNCTIONAL
    close(a, b; rtol) = all(map((x, y) -> isapprox(ndims(x) == 0 ? Array(x)[] : Array(x), y; rtol), a, b))

    @testset "NNlib on CUDA" begin
        @testset "$name" for (name, f, spelled, args) in NNLIB
            args = map(a -> Float32.(a), args)
            ex = prepare(trace(f, args...), map(CuArray, args)...)
            @test close(C.leaves(outputs(execute!(ex))), [f(args...)]; rtol = 1.0e-4)
        end
        # Attention written with NNlib's softmax is one generated kernel, as
        # it is with the log-sum-exp.
        scores(q, k) = NNlib.batched_mul(NNlib.batched_transpose(k), q) .* (1 / sqrt(Float32(size(q, 1))))
        spellings = [(q, k, v) -> NNlib.batched_mul(v, NNlib.softmax(scores(q, k); dims = 1)),
            (q, k, v) -> (s = scores(q, k); NNlib.batched_mul(v, exp.(s .- T.logsumexp(s; dims = 1))))]
        q, k, v = (Float32.(data(16, 200, 3; scale)) for scale in (0.1, 0.2, 0.3))
        for f in spellings
            ex = prepare(trace(f, q, k, v), map(CuArray, (q, k, v))...; target = T.CUDATarget(cuTile))
            kernel = only(ex.kernels)
            @test !(kernel isa C.FloorKernels) && occursin("online softmax", string(kernel))
            @test close(C.leaves(outputs(execute!(ex))), [f(q, k, v)]; rtol = 1.0e-4)
        end
    end
end
