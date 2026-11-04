# The program level: traced programs agree with plain Julia, and gradients of
# differentiated programs agree with finite differences.
include("../fixtures/programs.jl")

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
            args[k] isa AbstractArray{<:AbstractFloat} || continue
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

# A broadcast of numbers and arrays of no dimensions is a number, as in Julia,
# and with an array of one element it is that array. An array of no
# dimensions written in place stays one, as do Base's arithmetic on it, `map`
# of it and Statistics' variance along a dimension; an input of no dimensions
# is an array, and so is its gradient.
twice(s::Number) = 2s

@testset "broadcasts of no dimensions" begin
    x, y, v = fill(2.0), fill(0.5), [3.0]
    @testset "$name" for (name, f) in [
        ("with a number", (x, y, v) -> x .* 3.0),
        ("of two arrays", (x, y, v) -> x .+ y),
        ("fused", (x, y, v) -> exp.(x .* y) .+ 1.0),
        ("of the identity", (x, y, v) -> identity.(x)),
        ("with an array of one element", (x, y, v) -> x .+ v),
        ("read as a number", (x, y, v) -> (x .* y) isa Number),
        ("taken by a method for numbers", (x, y, v) -> twice(x .* y)),
        ("broadcast with a vector", (x, y, v) -> (s = x .* y; s .* v)),
        ("summed", (x, y, v) -> sum(x .* y)),
        ("written in place", (x, y, v) -> (z = similar(x); z .= x .* y; z)),
        ("mapped", (x, y, v) -> map(abs, x)),
        ("summed along a dimension", (x, y, v) -> sum(abs2, x; dims = 1)),
        ("in Base's arithmetic", (x, y, v) -> (x + y, 2x, -x)),
        ("in a variance", (x, y, v) -> (Statistics.var(x), Statistics.var(x; dims = 1), Statistics.std(x; dims = 1))),
    ]
        expected = f(copy(x), y, v)
        traced = interpret(trace(f, x, y, v), copy(x), y, v)
        @test isequal(traced, expected) && typeof(traced) == typeof(expected)
    end
    @test T.withgradient(x -> x .* 3.0, x) == (val = 6.0, grad = (fill(3.0),))
    @test T.gradient((x, a) -> sum(x .* a), x, 3.0) == (fill(3.0), nothing)
    loss, (g,) = interpret(trace(x -> T.withgradient(x -> exp.(x .* y), x), x), x)
    @test loss ≈ exp(1.0) && g isa Array{Float64, 0} && g[] ≈ 0.5 * exp(1.0)
    # Keeping such a result an array records the broadcast alone, not a
    # number and a copy of it back into an array.
    @test length(trace(x -> map(sin, x), x).nodes) == 2
    @test length(trace(x -> sum(sin, x), x).nodes) == 3
    @test length(trace((x, y) -> x + y, x, y).nodes) == 3
end

# A gradient in traced code is the differentiated program inlined where it is
# called; what the function closes over is a constant to it, and an argument
# passed twice gets a gradient for each place.
@testset "gradients in traced code" begin
    same(a, b) = a === nothing ? b === nothing : isapprox(a, b; rtol = 1.0e-12)
    @testset "$name" for (name, f, args) in [PROGRAMS; REPEATED_WRITES]
        expected = C.leaves(interpret(differentiate(trace(f, args...)), args...))
        @test all(map(same, expected, C.leaves(interpret(trace((xs...) -> T.withgradient(f, xs...), args...), args...))))
        @test all(map(same, expected, C.leaves(T.withgradient(f, args...))))
    end
    x, y = data(8, 6), data(8, 6; scale = 0.3)
    loss, (dx,) = interpret(trace((x, y) -> T.withgradient(a -> sum(a .* y .* x), x), x, y), x, y)
    @test loss ≈ sum(x .* y .* x) && dx ≈ y .* x
    loss, (da, db) = interpret(trace(x -> T.withgradient((a, b) -> sum(a .* b .* b), x, x), x), x)
    @test da ≈ x .* x && db ≈ 2 .* x .* x
    # Structured arguments, and Julia arrays among them.
    w, b = data(8, 8; scale = 0.2), data(8)
    loss, (g,) = interpret(trace((w, x) -> T.withgradient(q -> sum(abs2, q.w * x .+ q.b), (; w, b)), w, x), w, x)
    @test g.w ≈ 2 .* (w * x .+ b) * x' && g.b ≈ vec(sum(2 .* (w * x .+ b); dims = 2))
    # Arguments none of which is traced, as a layer without weights has, and a
    # function closing over traced values: the gradient is the program's.
    loss, (g,) = interpret(trace(x -> T.withgradient(p -> p.k * sum(abs2, x), (; k = 2)), x), x)
    @test loss ≈ 2 * sum(abs2, x) && g == (; k = nothing)
    # A loss no argument reaches, run eagerly, is a constant with no gradient.
    @test T.withgradient(a -> 1.5, x) == (val = 1.5, grad = (zero(x),))
end

# A stopped gradient is its input in value and a constant to every derivative:
# the gradient flows through the input's other uses, a straight-through
# rounding is the identity in derivative, a struct is stopped leaf by leaf, a
# derivative of a derivative keeps the stop, and the lowered program holds no
# value for it.
@testset "stopped gradients" begin
    x = data(8, 6)
    @test T.stopgrad(x) === x
    @test last(T.gradient(x -> sum(x .* T.stopgrad(x)), x)) ≈ x
    @test last(T.gradient(x -> sum(T.stopgrad(x)), x)) == zero(x)
    @test last(T.gradient(x -> sum(x) * T.stopgrad(sum(x)), x)) ≈ fill(sum(x), size(x))
    d = differentiate(trace(x -> sum(abs2, x .+ T.stopgrad(round.(x) .- x)), x))
    loss, g = interpret(d, x)
    @test loss ≈ sum(abs2, round.(x)) && g ≈ 2 .* round.(x)
    graph = C.lower(d)
    @test !any(v -> v.definition isa C.Pointwise && v.definition.f === T.stopgrad, graph.values)
    @test agree(C.interpret(graph, x), (loss, g))
    g = only(interpret(trace(x -> T.gradient(a -> sum(a .* T.stopgrad((; s = a)).s), x), x), x))
    @test g ≈ x
    inner(a) = sum(a .^ 3 .* T.stopgrad(a))
    @test last(T.gradient(x -> sum(last(T.gradient(inner, x))), x)) ≈ 6 .* x .^ 2
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
    @test T.gradient((x, b) -> sum(softmax_columns(x .+ b) .* t), x, b)[2] == zero(b)
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
    loss, (g,) = T.withgradient(p -> sum(W * p), x)
    @test loss ≈ sum(W * x) && g ≈ dx
    loss, (g,) = interpret(trace(x -> T.withgradient(p -> sum(W * p), x), x), x)
    @test loss ≈ sum(W * x) && g ≈ dx
    @test last(interpret(trace(x -> T.gradient(written, x), x), x)) ≈ masked
end

# `x log y`, zero where `x` is: the branch guards the product against `0 log 0`,
# and its derivative in `x` there, none, is not the function's, `log y`.
xlogy(x, y) = ifelse(iszero(x), zero(x * log(y)), x * log(y))
T.scalar_partials(::typeof(xlogy), z, x, y) = (log(y), x / y)

@testset "stated derivatives" begin
    # A function whose code traces takes a statement of its derivative where
    # it is broadcast, as a primitive takes it for its rule.
    x, y = [0.0, 0.5, 2.0], [0.3, 1.5, 2.0]
    gx, gy = T.gradient((x, y) -> sum(xlogy.(x, y)), x, y)
    @test gx ≈ log.(y) && gy ≈ x ./ y
    @test interpret(trace((x, y) -> xlogy.(x, y), x, y), x, y) == xlogy.(x, y)
    @test only(T.gradient(x -> sum(leaky.(x)), [-1.0, 2.0])) ≈ [0.01, 1.0]
    # A constant partial is in the cotangent's type: `1 / 3` of a quotient by
    # an integer is a Float64, but a Float32 gradient stays one.
    @test only(T.gradient(x -> sum(x ./ 3), Float32[1, 2])) isa Vector{Float32}
    @test all(n -> n.eltype !== Float64, differentiate(trace(x -> sum(x ./ 3), Float32[1, 2])).nodes)
end
