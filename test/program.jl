# The program level: traced programs agree with plain Julia, and gradients of
# differentiated programs agree with finite differences.
include("fixtures/programs.jl")

function finite_difference(f, args, k)
    x = args[k]
    if x isa Number
        e = 1.0e-6
        plus = Base.setindex(args, x + e, k)
        minus = Base.setindex(args, x - e, k)
        return (f(plus...) - f(minus...)) / 2e
    end
    g = similar(x)
    for i in eachindex(x)
        e = 1.0e-6 * max(1.0, abs(x[i]))
        xp = copy(x); xp[i] += e
        xm = copy(x); xm[i] -= e
        g[i] = (f(Base.setindex(args, xp, k)...) - f(Base.setindex(args, xm, k)...)) / 2e
    end
    g
end

@testset "program level" begin
    @testset "$name" for (name, f, args) in [PROGRAMS; REPEATED_WRITES]
        p = trace(f, args...)
        @test interpret(p, args...) ≈ f(args...)
        d = differentiate(p)
        # Every node is read by a result, or is an argument.
        read = C.live(d.nodes, d.results)
        @test all(id -> read[id] || id in d.arguments, eachindex(d.nodes))
        loss, gradients... = interpret(d, args...)
        @test loss ≈ f(args...)
        for k in eachindex(args)
            args[k] isa Union{AbstractFloat, AbstractArray{<:AbstractFloat}} || continue
            @test isapprox(gradients[k], finite_difference(f, args, k); rtol = 1.0e-5, atol = 1.0e-7)
        end
    end
    @test any(n -> n.op isa C.Map && n.op.f === leaky, trace(x -> leaky.(x), data(3)).nodes)
    @test !any(n -> n.op isa C.Map && n.op.f === swish, trace(x -> swish.(x), data(3)).nodes)
    # The adjoint of a selection by a constant permutation is a selection by
    # its inverse, not a scatter.
    d = differentiate(trace(x -> sum(abs2, x[[3, 4, 1, 2], :] .* data(4, 1)), data(4, 6)))
    @test !any(n -> n.op isa C.Scatter, d.nodes)
    @test any(n -> n.op isa C.Scatter, differentiate(trace(x -> sum(abs2, x[[3, 4, 1, 1], :]), data(4, 6))).nodes)
    # What a scatter adds at one position is summed as a sum fold sums it: 4096
    # BFloat16 gradients of 0.01 in Float32, rounded once to 41, not 4.
    E, t, g = BFloat16s.BFloat16.(fill(0.5f0, 4, 3)), Int32.(fill(2, 4096)), BFloat16s.BFloat16.(fill(0.01f0, 4, 4096))
    dE = interpret(differentiate(trace((E, t, g) -> sum(E[:, t] .* g), E, t, g)), E, t, g)[2]
    @test dE[:, 2] == fill(BFloat16s.BFloat16(41), 4)
end

# A gradient in traced code is the differentiated program inlined where it is
# called; what the function closes over is a constant to it, and an argument
# passed twice gets a gradient for each place.
@testset "gradients in traced code" begin
    same(a, b) = a === nothing ? b === nothing : isapprox(a, b; rtol = 1.0e-12)
    @testset "$name" for (name, f, args) in [PROGRAMS; REPEATED_WRITES]
        expected = C.leaves(interpret(differentiate(trace(f, args...)), args...))
        @test all(map(same, expected, C.leaves(interpret(trace((xs...) -> T.gradient(f, xs...), args...), args...))))
        @test all(map(same, expected, C.leaves(T.gradient(f, args...))))
    end
    x, y = data(8, 6), data(8, 6; scale = 0.3)
    loss, dx = interpret(trace((x, y) -> T.gradient(a -> sum(a .* y .* x), x), x, y), x, y)
    @test loss ≈ sum(x .* y .* x) && dx ≈ y .* x
    loss, da, db = interpret(trace(x -> T.gradient((a, b) -> sum(a .* b .* b), x, x), x), x)
    @test da ≈ x .* x && db ≈ 2 .* x .* x
    # Structured arguments, and Julia arrays among them.
    w, b = data(8, 8; scale = 0.2), data(8)
    loss, g = interpret(trace((w, x) -> T.gradient(q -> sum(abs2, q.w * x .+ q.b), (; w, b)), w, x), w, x)
    @test g.w ≈ 2 .* (w * x .+ b) * x' && g.b ≈ vec(sum(2 .* (w * x .+ b); dims = 2))
end

# A shift the loss does not depend on gets no cotangent. The log-sum-exp
# written out is the same for every `m` constant along its sum, so the rule of
# the maximum, which compares each element with it, never runs, and the
# gradient is the named log-sum-exp's; read elsewhere too, the maximum gets
# its cotangent from there. A softmax of `x .+ b` does not depend on `b` at
# all: its gradient is zero, not the rounding of terms that cancel.
@testset "a shift the loss does not depend on" begin
    x, w, b, t = data(8, 6), data(1, 6), data(1, 6; scale = 0.3), data(8, 6; scale = 0.7)
    compares(p) = any(n -> n.op isa C.Map && n.op.f === (==), p.nodes)
    d = differentiate(trace(x -> sum(written_logsumexp(x; dims = 1) .* w), x))
    @test !compares(d)
    @test last(interpret(d, x)) ≈ last(T.gradient(x -> sum(T.logsumexp(x; dims = 1) .* w), x))
    @test compares(differentiate(trace(x -> (m = maximum(x; dims = 1); sum(m .+ log.(sum(exp.(x .- m); dims = 1))) + sum(m)), x)))
    @test T.gradient((x, b) -> sum(softmax_columns(x .+ b) .* t), x, b)[3] == zero(b)
end

# Base's scans trace as folds that keep the axes they fold along, and compute
# what eager Julia does in its order: a vector's cumulative sum pairwise,
# every other scan in sequence. A linear recurrence is a scan of pairs. A
# cumulative sum's derivative is a cumulative sum the other way, and any
# other scan's a linear recurrence the other way.
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

# Extremes and where they are as Base finds them (`EXTREMES` and `FOUND` in
# fixtures/programs.jl), and the derivative: the cotangent at the element
# found.
@testset "extremes" begin
    @testset "$(eltype(x)) $i" for (i, x) in enumerate(EXTREMES), f in FOUND
        @test isequal(found(interpret(trace(f, x), x)), found(f(x)))
    end
    x = Float64[1 5 2; 3 -5 7; 0 4 -1; 2 2 9]
    _, g = interpret(differentiate(trace(x -> sum(findmax(x; dims = 1)[1] .* [1.0 2.0 3.0]) + sum(abs2, x[argmin(x)]), x)), x)
    @test g == [0 2 0; 1 -10 0; 0 0 0; 0 0 3]
end

# Writes into arrays a program allocates are functional updates: each write
# makes a new version of the array, later reads see the latest version, reads
# before a write keep what they read, and layouts share their array's memory.
@testset "functional updates" begin
    x, v = data(8, 6), data(3, 6; scale = 0.3)
    same(f, args...) = interpret(trace(f, args...), args...) == f(args...)
    @test same(x -> (y = similar(x); y .= 1; y[3, 2] = x[1]; y[CartesianIndex(4, 5)] = 7; y[11] = -1; y[:, 6] = x[:, 1]; y), x)
    @test same((x, v) -> (y = copy(x); y[2:4, :] = v; y[end, :] .= 2; y), x, v)
    @test same(x -> (y = fill!(similar(x, 10, 8), 0); y[1:2:8, end:-2:1] = x[1:4, 1:4]; y), x)
    @test same(x -> (y = fill!(similar(x, 7), 0.5); copyto!(y, 2, x, 5, 3); y), x)
    @test same(x -> (y = similar(x, 6, 8); permutedims!(y, x, (2, 1)); map!(abs, y, y)), x)
    @test same(x -> (r = sum(x; dims = 1); r ./= size(x, 1); r), x)
    @test same(x -> (y = copy(x); a = sum(y); y[1, 1] = 100; (a, sum(y))), x)
    # Views, reshapes and transposes read the writes into their array, and
    # write into it.
    @test same(x -> (y = copy(x); w = view(y, 2:3, :); y .= 0; w .+ 1), x)
    @test same(x -> (y = zero(x); view(y, :, 2:3) .= x[:, 1:2]; reshape(y, 6, 8)[1, :] .= 5; transpose(y)[2, 1] = 9; y), x)
    @test same(x -> (y = copy(x); w = view(y, 2:7, :); view(w, 1:2, 3)[2] = -1; (y, w)), x)
    @test same(x -> (y = copy(x); w = PermutedDimsArray(y, (2, 1)); y .= 0; w[2, 1] = 3; (y, w .+ 1)), x)
    # `similar` holds zeros until written; Julia leaves them undefined.
    @test interpret(trace(x -> similar(x, 3, 2), x), x) == zeros(3, 2)
    # A copy is the array's contents now, in memory of its own.
    @test same(x -> (y = 2 .* x; z = copy(y); y[1] = 0; (y, z)), x)
    # Writes at positions arrays of integers select, constant or traced, and
    # at a traced integer: where two land at one position, the interpreter's
    # is the last, as Base's.
    @test same(x -> (y = copy(x); y[[1, 3, 3]] = [1, 2, 3]; y), x)
    @test same((x, i) -> (y = copy(x); y[i, 2:3] .= 7; y[i[2], 1] = -1; view(y, :, 4)[i] .= x[i, 5]; y), x, Int32[5, 2, 5])
    # A traced mask selects a number of elements known only when the program runs.
    @test_throws ArgumentError trace((x, m) -> (y = copy(x); y[m] .= 0; y), x, x .> 0)
end

# The program's inputs, and anything that shares their memory, are never
# written: a program is a function of its inputs.
@testset "inputs are not written" begin
    x = data(8, 6)
    message(f) = try trace(f, x); "" catch e sprint(showerror, e) end
    for f in (x -> (x[1] = 0; x), x -> (x .= 2; x), x -> (fill!(x, 1); x), x -> (copyto!(x, ones(8, 6)); x),
              x -> (view(x, :, 1) .= 0; x), x -> (vec(x)[2] = 1; x), x -> (x'[1, 2] = 3; x), x -> (reshape(x, 6, 8) ./= 2; x))
        @test startswith(message(f), "ArgumentError: mutation of the program's inputs is not supported")
    end
    # A copy of an input may be written.
    @test interpret(trace(x -> (y = copy(x); y[1] = 0; y), x), x) == (y = copy(x); y[1] = 0; y)
    # A gradient's function never writes its arguments, traced or run eagerly.
    writes(p) = (p[1] = 0; sum(p))
    @test startswith(message(x -> T.gradient(writes, x)), "ArgumentError: mutation of the program's inputs is not supported")
    @test_throws ArgumentError T.gradient(writes, x)
end

# A Julia array a function closes over is a constant factor of the products it
# enters, and the gradient flows to the traced factor, through writes too.
@testset "products with a Julia matrix closed over" begin
    W, x = data(5, 8; scale = 0.2), data(8, 6)
    dx = W' * ones(5, 6)
    loss, g = interpret(differentiate(trace(p -> sum(W * p), x)), x)
    @test loss ≈ sum(W * x) && g ≈ dx
    written(p) = (y = copy(p); y[1, :] .= 0; sum(W * y))
    masked = dx .* (axes(x, 1) .!= 1)
    @test last(interpret(differentiate(trace(written, x)), x)) ≈ masked
    # Tempest.gradient, eagerly and where it is called in traced code.
    loss, g = T.gradient(p -> sum(W * p), x)
    @test loss ≈ sum(W * x) && g ≈ dx
    loss, g = interpret(trace(x -> T.gradient(p -> sum(W * p), x), x), x)
    @test loss ≈ sum(W * x) && g ≈ dx
    @test last(interpret(trace(x -> T.gradient(written, x), x), x)) ≈ masked
end
