# `compile`: a function traced and prepared, called like the function, with
# structs among its arguments and integers as constants.

struct Dense{W, B, F}
    w::W
    b::B
    act::F
end
(d::Dense)(x) = d.act.(d.w * x .+ d.b)

struct Fixed
    w::Matrix{Float64}
end

loss(m, x) = sum(abs2, m(x))

function finite_difference_weights(m, x)
    g = similar(m.w)
    for i in eachindex(m.w)
        e = 1.0e-6
        plus, minus = copy(m.w), copy(m.w)
        plus[i] += e
        minus[i] -= e
        g[i] = (loss(Dense(plus, m.b, m.act), x) - loss(Dense(minus, m.b, m.act), x)) / 2e
    end
    g
end

@testset "compiled functions" begin
    c = compile(x -> (sum(x), 2 .* x), data(4))
    a, b = c([1.0, 2, 3, 4]), c([10.0, 20, 30, 40])
    # Results are the caller's: a number as a number, an array apart from the
    # buffers the next call overwrites.
    @test a == (10.0, [2.0, 4.0, 6.0, 8.0]) && b == (100.0, [20.0, 40.0, 60.0, 80.0])
    @test compile(x -> (; total = sum(x), twice = 2 .* x), data(3))([1.0, 2, 3]) == (total = 6.0, twice = [2.0, 4.0, 6.0])
    @test occursin("1 kernel", sprint(show, compile(sum, data(4))))
    @test occursin("kernel 1", sprint(explain, compile(sum, data(4))))
end

@testset "a call's arguments match the compiled ones" begin
    c = compile(x -> 2 .* x, data(4))
    @test occursin("argument 1 is a Float64 array of size (5,), not a Float64 array of size (4,)", message(() -> c(data(5))))
    @test occursin("argument 1 is a Float32 array", message(() -> c(Float32.(data(4)))))
    @test occursin("it takes 1 argument, not 2", message(() -> c(data(4), data(4))))
end

@testset "integers are constants" begin
    p = trace((x, n) -> reshape(x, n, :), data(4), 2)
    @test length(p.arguments) == 1
    @test interpret(p, [1.0, 2, 3, 4], 2) == [1.0 3.0; 2.0 4.0]
    c = compile((x, n) -> reshape(x, n, :), data(4), 2)
    @test c([1.0, 2, 3, 4], 2) == [1.0 3.0; 2.0 4.0]
    @test occursin("argument 2 is the constant 4, not the constant 2", message(() -> c(data(4), 4)))
    # Floating-point numbers stay inputs.
    s = compile((x, a) -> a .* x, data(4), 2.0)
    @test s([1.0, 1, 1, 1], 3.0) == [3.0, 3.0, 3.0, 3.0]
end

@testset "structs among the arguments" begin
    m, x = Dense(data(3, 4; scale = 0.3), data(3), tanh), data(4; scale = 0.5)
    other = Dense(data(3, 4; scale = 0.7), data(3; scale = 0.2), tanh)
    # The arrays inside are inputs, so the compiled model takes new weights.
    p = trace((m, x) -> m(x), m, x)
    @test length(p.arguments) == 3
    @test interpret(p, other, x) ≈ other(x)
    c = compile((m, x) -> m(x), m, x)
    @test c(other, x) ≈ other(x)
    @test occursin("argument 1.act is the constant sin, not the constant tanh", message(() -> c(Dense(m.w, m.b, sin), x)))
    # A struct's gradient is a named tuple of its fields'.
    g = compile((m, x) -> Tempest.gradient(loss, m, x), m, x)
    value, gm, gx = g(other, x)
    reference = interpret(differentiate(trace(loss, other, x)), other, x)
    @test value ≈ loss(other, x)
    @test gm isa NamedTuple{(:w, :b, :act)} && gm.act === nothing
    @test gm.w ≈ reference[2].w && gm.b ≈ reference[2].b && gx ≈ reference[3]
    @test isapprox(gm.w, finite_difference_weights(other, x); rtol = 1.0e-5)
    # A struct whose fields cannot hold traced arrays says so.
    @test occursin("need types that are parameters", message(() -> trace(f -> sum(f.w), Fixed(data(2, 2)))))
end

if CUDA_FUNCTIONAL
    @testset "compiled on CUDA" begin
        m, x = Dense(single(data(64, 32; scale = 0.3)), single(data(64)), tanh), single(data(32, 16; scale = 0.5))
        c = compile((m, x) -> (loss(m, x), m(x)), Dense(CuArray(m.w), CuArray(m.b), tanh), CuArray(x))
        value, y = c(Dense(CuArray(m.w), CuArray(m.b), tanh), CuArray(x))
        @test value isa Float32 && isapprox(value, loss(m, x); rtol = 1.0e-4)
        @test y isa CuArray && isapprox(Array(y), m(x); rtol = 1.0e-4)
        # Each call replays one captured graph, unless told to launch kernel by
        # kernel; new values are copied into the buffers the graph reads.
        @test c.executable.captured !== nothing
        other, z = Dense(single(data(64, 32; scale = 0.7)), single(data(64; scale = 0.2)), tanh), single(data(32, 16; scale = 0.9))
        value2, y2 = c(Dense(CuArray(other.w), CuArray(other.b), tanh), CuArray(z))
        @test isapprox(value2, loss(other, z); rtol = 1.0e-4) && isapprox(Array(y2), other(z); rtol = 1.0e-4)
        launched = compile((m, x) -> (loss(m, x), m(x)), Dense(CuArray(m.w), CuArray(m.b), tanh), CuArray(x); capture = false)
        @test launched.executable.captured === nothing
        @test launched(Dense(CuArray(m.w), CuArray(m.b), tanh), CuArray(x))[1] == value
    end
end
