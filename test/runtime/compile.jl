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

# A layer that keeps running statistics, mutable as Flux's normalisations are.
mutable struct Running{V}
    scale::V
    mean::V
end

# A layer whose constructor checks its weights, with a parameter that is no
# field's type.
struct Checked{kind, W}
    weights::W
    function Checked{kind}(weights) where {kind}
        all(>(0), weights) || throw(DomainError(weights, "the weights must be positive"))
        new{kind, typeof(weights)}(weights)
    end
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
    # A number comes back as a number, an array of no dimensions as one.
    @test compile(sum, data(1))([3.0]) === 3.0
    @test compile(x -> reshape(x, ()), data(1))([3.0]) == fill(3.0)
    @test occursin("1 kernel", sprint(show, compile(sum, data(4))))
    @test occursin("kernel 1", sprint(explain, compile(sum, data(4))))
end

@testset "a call's arguments match the compiled ones" begin
    c = compile(x -> 2 .* x, data(4))
    @test occursin("argument 1 is a Float64 array of size (5,), not a Float64 array of size (4,)", message(() -> c(data(5))))
    @test occursin("argument 1 is a Float32 array", message(() -> c(Float32.(data(4)))))
    @test occursin("it takes 1 argument, not 2", message(() -> c(data(4), data(4))))
end

@testset "numbers are constants" begin
    p = trace((x, n) -> reshape(x, n, :), data(4), 2)
    @test length(p.arguments) == 1
    @test interpret(p, [1.0, 2, 3, 4], 2) == [1.0 3.0; 2.0 4.0]
    c = compile((x, n) -> reshape(x, n, :), data(4), 2)
    @test c([1.0, 2, 3, 4], 2) == [1.0 3.0; 2.0 4.0]
    @test occursin("argument 2 is the constant 4, not the constant 2", message(() -> c(data(4), 4)))
    # So are floating-point numbers, as configuration is; one that varies
    # between calls is an array.
    s = compile((x, a) -> a .* x, data(4), 2.0)
    @test length(s.executable.program.arguments) == 1 && s([1.0, 1, 1, 1], 2.0) == [2.0, 2.0, 2.0, 2.0]
    @test occursin("argument 2 is the constant 3.0, not the constant 2.0", message(() -> s([1.0, 1, 1, 1], 3.0)))
    v = compile((x, a) -> a .* x, data(4), fill(2.0))
    @test v([1.0, 1, 1, 1], fill(3.0)) == [3.0, 3.0, 3.0, 3.0]
    # A gradient has nothing for a number, as for an integer, and one for an
    # array of no dimensions.
    @test T.gradient((x, a) -> sum(a .* x), [1.0, 2.0], 2.0)[3] === nothing
    @test T.gradient((x, a) -> sum(a .* x), [1.0, 2.0], fill(2.0))[3] == fill(3.0)
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

@testset "structs are built without their constructors" begin
    # Tracing builds each struct anew around traced fields, as `new` would: a
    # constructor that checks its weights, which traced weights cannot
    # answer, does not run. Its parameters other than fields' types are kept.
    layer = Checked{:scale}([1.0, 2.0, 3.0])
    c = compile((l, x) -> l.weights .* x, layer, data(3))
    @test c(Checked{:scale}([2.0, 2.0, 2.0]), [1.0, 2, 3]) == [2.0, 4.0, 6.0]
    @test compile(l -> (l, sum(l.weights)), layer)(layer)[1] isa Checked{:scale}
end

@testset "constants among the results" begin
    # A struct among the results keeps its leaves that are no arrays or
    # numbers, a layer's function say, as they are.
    m, x = Dense(data(3, 4), data(3), tanh), data(4)
    c = compile(m -> Dense(2 .* m.w, m.b, m.act), m)
    @test c(m).act === tanh && c(m).w ≈ 2 .* m.w
    # A number is a constant among the results as among the arguments.
    window = (size = (2, 3), rate = 0.5, x = data(4))
    r = compile(w -> (; w.size, w.rate, x = 2 .* w.x), window)(window)
    @test r.size === (2, 3) && r.rate === 0.5 && r.x == 2 .* window.x
    # A constant array of numbers is a result like a computed one.
    @test compile(x -> (x, [1.0, 2.0]), x)(x)[2] == [1.0, 2.0]
    # A closure over traced arrays, or an array of them, would hand them out.
    @test occursin("may hold traced values", message(() -> trace(x -> (y -> y .+ x), x)))
    @test occursin("may hold traced values", message(() -> trace(x -> Any[x, x], x)))
end

@testset "mutable structs among the arguments" begin
    # A mutable struct's arrays are inputs, as an immutable one's are: the
    # compiled function takes another's, and a copy is no other argument.
    m, x = Running(data(3), data(3)), data(3)
    p = trace((m, x) -> m.scale .* (x .- m.mean), m, x)
    @test length(p.arguments) == 3
    c = compile((m, x) -> m.scale .* (x .- m.mean), m, x)
    other = Running(data(3; scale = 2.0), data(3; scale = 0.5))
    @test c(other, x) ≈ other.scale .* (x .- other.mean)
    @test c(deepcopy(m), x) ≈ m.scale .* (x .- m.mean)
    _, gm, _ = compile((m, x) -> Tempest.gradient((m, x) -> sum(m.scale .* x), m, x), m, x)(m, x)
    @test gm isa NamedTuple{(:scale, :mean)} && gm.scale ≈ x
    # The traced code may write the struct it was given, as a layer updates
    # its running statistics: the result is a struct of its own, and the
    # caller's is as it was.
    function update(m, x)
        m.mean = 0.9 .* m.mean .+ 0.1 .* x
        m
    end
    before = copy(m.mean)
    after = compile(update, m, x)(m, x)
    @test after isa Running && after !== m && after.mean ≈ 0.9 .* before .+ 0.1 .* x
    @test m.mean == before
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
