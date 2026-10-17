# The program level: traced programs agree with plain Julia, and gradients of
# differentiated programs agree with finite differences.
using Test
include("cases.jl")

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
    @testset "$name" for (name, f, args) in PROGRAMS
        p = trace(f, args...)
        @test interpret(p, args...) ≈ f(args...)
        d = differentiate(p)
        # Every node is read by a result, or is a call's result beside one.
        read = T.live(d.nodes, d.results)
        @test all(id -> read[id] || id in d.arguments || d.nodes[id].op isa T.Call, eachindex(d.nodes))
        loss, gradients... = interpret(d, args...)
        @test loss ≈ f(args...)
        for k in eachindex(args)
            args[k] isa Union{AbstractFloat, AbstractArray{<:AbstractFloat}} || continue
            @test isapprox(gradients[k], finite_difference(f, args, k); rtol = 1.0e-5, atol = 1.0e-7)
        end
    end
    @test any(n -> n.op isa T.Map && n.op.f === leaky, trace(x -> leaky.(x), data(3)).nodes)
    @test !any(n -> n.op isa T.Map && n.op.f === swish, trace(x -> swish.(x), data(3)).nodes)
end

# Writes into arrays a program allocates are functional updates: each write
# makes a new version of the array, later reads see the latest version, reads
# before a write keep what they read, and layouts share their array's memory.
@testset "functional updates" begin
    x, v = data(8, 6), data(3, 6; scale = 0.3)
    same(f, args...) = interpret(trace(f, args...), args...) == f(args...)
    @test same(x -> (y = similar(x); y .= 1; y[3, 2] = x[1]; y[CartesianIndex(4, 5)] = 7; y[11] = -1; y[:, 6] = x[:, 1]; y), x)
    @test same((x, v) -> (y = copy(x); y[2:4, :] = v; y[end, :] .= 2; y), x, v)
    @test same(x -> (y = zeros(1) .* similar(x, 10, 8); y[1:2:8, end:-2:1] = x[1:4, 1:4]; y), x)
    @test same(x -> (y = fill!(similar(x, 7), 0.5); copyto!(y, 2, x, 5, 3); y), x)
    @test same(x -> (y = similar(x, 6, 8); permutedims!(y, x, (2, 1)); map!(abs, y, y)), x)
    @test same(x -> (r = sum(x; dims = 1); r ./= size(x, 1); r), x)
    @test same(x -> (y = copy(x); a = sum(y); y[1, 1] = 100; (a, sum(y))), x)
    # Views, reshapes and transposes read the writes into their array, and
    # write into it.
    @test same(x -> (y = copy(x); w = view(y, 2:3, :); y .= 0; w .+ 1), x)
    @test same(x -> (y = zero(x); view(y, :, 2:3) .= x[:, 1:2]; reshape(y, 6, 8)[1, :] .= 5; transpose(y)[2, 1] = 9; y), x)
    @test same(x -> (y = copy(x); w = view(y, 2:7, :); view(w, 1:2, 3)[2] = -1; (y, w)), x)
    # `similar` holds zeros until written; Julia leaves them undefined.
    @test interpret(trace(x -> similar(x, 3, 2), x), x) == zeros(3, 2)
    # A copy is the array's contents now, in memory of its own.
    @test same(x -> (y = 2 .* x; z = copy(y); y[1] = 0; (y, z)), x)
    # Writes at positions an array selects are not functional updates yet.
    @test_throws ArgumentError trace(x -> (y = copy(x); y[[1, 3, 3]] = [1, 2, 3]; y), x)
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
    # Tempest.gradient, eagerly and where it is called in traced code, once the
    # runtime's training-step API is here.
    if isdefined(T, :gradient)
        loss, g = T.gradient(p -> sum(W * p), x)
        @test loss ≈ sum(W * x) && g ≈ dx
        loss, g = interpret(trace(x -> T.gradient(p -> sum(W * p), x), x), x)
        @test loss ≈ sum(W * x) && g ≈ dx
        @test last(interpret(trace(x -> T.gradient(written, x), x), x)) ≈ masked
    end
end
