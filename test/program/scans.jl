# Base's scans trace as folds that keep the axes they fold along, and compute
# what eager Julia does in its order: a vector's cumulative sum pairwise,
# every other scan in sequence. A linear recurrence is a scan of pairs. A
# cumulative sum's derivative is a cumulative sum the other way, and any
# other scan's a linear recurrence the other way.
include("../fixtures/programs.jl")

@testset "scans" begin
    x, v = data(8, 6), data(300)
    a, b = 0.9 .+ 0.1 .* data(300; scale = 0.3), data(300; scale = 0.7)
    same(f, args...) = interpret(trace(f, args...), args...) == f(args...)
    scans = [(cumsum, (v,)), (x -> cumsum(x; dims = 1), (x,)), (x -> cumprod(x; dims = 2), (x,)), (cumprod, (v,)),
        (x -> accumulate(max, x; dims = 1), (x,)), (x -> accumulate(min, x), (x,)), (recurrence, (a, b))]
    for (f, args) in scans
        @test same(f, args...)
        @test same(f, map(single, args)...)
    end
    # An `init` other than a builtin fold's identity combines after the scan,
    # as after a fold.
    @test interpret(trace(x -> accumulate(+, x; dims = 2, init = 1.0), x), x) ≈ accumulate(+, x; dims = 2, init = 1.0)
    # A function given to `accumulate` claims no law: Base combines it in
    # sequence, from an `init` that need be no identity, as its own example
    # divides, and so does the scan. A `GenericFold` claims the laws.
    forward(f, args...) = [n.op.fold for n in trace(f, args...).nodes if n.op isa C.Accumulate]
    @test same(x -> accumulate(/, abs.(x) .+ 1; dims = 1, init = 100), x)
    @test same(x -> running_average(x; dims = 2), x)
    @test same((a, b) -> accumulate(RECURRENCE, tuple.(a, b)), a, b)
    @test only(forward(x -> running_average(x; dims = 2), x)) isa IR.InSequence
    @test only(forward(recurrence, a, b)) isa IR.InSequence
    @test only(forward((a, b) -> accumulate(RECURRENCE, tuple.(a, b)), a, b)) isa IR.GenericFold
    @test accumulate(RECURRENCE, tuple.(a, b)) == recurrence(a, b)
    @test occursin("needs `init`, the state the scan starts from", message(() -> trace(x -> accumulate((p, q) -> p * q + q, x; dims = 1), x)))
    backward(f, args...) = [n.op.fold for n in differentiate(trace(f, args...)).nodes if n.op isa C.Accumulate && n.op.reverse]
    @test backward(x -> sum(cumsum(x; dims = 1) .* x), x) == [IR.SumFold()]
    @test backward(x -> sum(cumprod(x; dims = 1) .* x), x) == [C.Recurrence{1}()]
    @test backward((a, b) -> sum(abs2, recurrence(a, b)), a, b) == [C.Recurrence{2}()]
    @test backward(x -> sum(running_average(x; dims = 2) .* x), x) == [C.Recurrence{1}()]
end
