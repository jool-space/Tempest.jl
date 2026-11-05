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

# Globals a compiled function may read by name, as a script's or the REPL's.
held_weights = data(3, 4)
held_layer = Dense(data(3, 4), data(3), tanh)
const held_table = data(4)
held_rate = 0.5
held_step(x) = Tempest.gradient(m -> loss(m, x), held_layer)
held_total = data(4)
held_read(x) = held_layer(x)
held_add!(x) = (global held_total = held_total .+ x; held_total)
held_shift(x) = held_total .+ sum(x)

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
    @test compile(x -> 2 .* x, fill(1.0))(fill(3.0)) === 6.0
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
    @test T.gradient((x, a) -> sum(a .* x), [1.0, 2.0], 2.0)[2] === nothing
    @test T.gradient((x, a) -> sum(a .* x), [1.0, 2.0], fill(2.0))[2] == fill(3.0)
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
    g = compile((m, x) -> Tempest.withgradient(loss, m, x), m, x)
    value, (gm, gx) = g(other, x)
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
    # A closure is a struct of what it captures, built anew around results.
    shift = compile(x -> (y -> y .+ 2 .* x), x)(x)
    @test shift([1.0, 1.0, 1.0, 1.0]) ≈ 1 .+ 2 .* x
    # An array of traced arrays would hand them out.
    @test occursin("may hold traced values", message(() -> trace(x -> Any[x, x], x)))
end

@testset "the function's arrays are inputs" begin
    # `compile` traces the call `f(args...)` with `f` an argument like the
    # others: what a closure captures and a callable struct's fields are read
    # at each call, as the untraced function reads them.
    w, x = data(3, 4; scale = 0.3), data(4)
    c = compile(x -> w * x, x)
    @test length(c.executable.program.arguments) == 2
    w .= 2 .* w
    @test c(x) ≈ w * x
    # A captured variable that is assigned again is held in a box, whose
    # contents are read at each call.
    v = data(4)
    scaled = x -> v .* x
    boxed = compile(scaled, x)
    v = data(4; scale = 2.0)
    @test fieldtype(typeof(scaled), :v) === Core.Box && boxed(x) ≈ v .* x
    # A layer compiles as itself, and a closure over a model as over arrays.
    m = Dense(data(3, 4; scale = 0.3), data(3), tanh)
    layer, step = compile(m, x), compile(x -> Tempest.withgradient(loss, m, x), x)
    m.w .= 0.5 .* m.w
    @test layer(x) ≈ m(x)
    value, (gm, gx) = step(x)
    @test value ≈ loss(m, x) && isapprox(gm.w, finite_difference_weights(m, x); rtol = 1.0e-5)
    # What the program writes into an array of `f`'s lands in it, as running
    # statistics land in a layer's.
    function normalized(m, x)
        μ = vec(sum(x; dims = 2)) ./ size(x, 2)
        m.mean .= 0.9 .* m.mean .+ 0.1 .* μ
        m.scale .* (x .- μ)
    end
    r, y = Running(data(3), data(3; scale = 0.5)), data(3, 4)
    theirs = deepcopy(r)
    c = compile(y -> normalized(r, y), y)
    @test c(y) ≈ normalized(theirs, y) && r.mean ≈ theirs.mean
    # `f`'s own leaves are matched at each call, as the arguments are.
    push!(v, 1.0)
    @test occursin("f.v.contents is a Float64 array of size (5,), not a Float64 array of size (4,)", message(() -> boxed(x)))
end

@testset "globals the function reads" begin
    x = data(4)
    # A global holding arrays would be kept as it was when traced, `const` or
    # not: refused, saying what to do instead.
    @test occursin("reads the global held_weights, which holds arrays", message(() -> compile(x -> held_weights * x, x)))
    @test occursin("reads the global held_layer", message(() -> compile(x -> held_layer(x), x)))
    @test occursin("reads the global held_table", message(() -> compile(x -> held_table .* x, x)))
    # Passed as an argument, it is an input.
    @test compile((w, x) -> w * x, held_weights, x)(held_weights, x) ≈ held_weights * x
    # A number is a constant, and one not declared `const` is checked at each call.
    c = compile(x -> held_rate .* x, x)
    @test c(x) ≈ 0.5 .* x
    global held_rate = 0.25
    @test occursin("the global held_rate is 0.25, not 0.5 as when traced", message(() -> c(x)))
    global held_rate = 0.5
    # Base's globals are its code's.
    @test compile(x -> π .* x, x)(x) ≈ π .* x
    # What a function the compiled one calls reads is seen by its arrays, one
    # broadcast with a traced number too.
    @test occursin("reads the global held_layer", message(() -> compile(x -> held_step(x), x)))
    @test occursin("reads the global held_total", message(() -> compile(x -> held_shift(x), x)))
    # Passed as an argument too, the global is still read by name: the argument is to be read instead.
    @test occursin("held_layer is an argument too", message(() -> compile((m, x) -> m(x) .+ held_layer(x), held_layer, x)))
    @test occursin("held_layer is an argument too", message(() -> compile((m, x) -> m(x) .+ held_read(x), held_layer, x)))
    # A global it assigns while traced holds stand-ins: refused, to be assigned again.
    total = held_total
    @test occursin("assigned the global held_total while traced", message(() -> compile(x -> held_add!(x), x)))
    @test held_total isa Tempest.Compiler.TracedArray
    global held_total = total
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
    gm, _ = compile((m, x) -> Tempest.gradient((m, x) -> sum(m.scale .* x), m, x), m, x)(m, x)
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

# Results that are inputs as they were before the program wrote them, against
# the functions run untraced: the results, and the caller's arrays after.
host(a) = a isa Number ? a : Array(a)
function check_before_write(device; options...)
    stepped = x -> (old = copy(x); x .= x .- 0.1f0 .* x; old)
    overwritten = x -> (y = copy(x); x .= 1; (y, sum(x)))
    reshaped = x -> (y = reshape(copy(x), 1, :); x[2] = 0; (y, copy(y)))
    for f in (stepped, overwritten, reshaped)
        c = compile(f, device(single(data(3))); options...)
        # Each call on new contents: the copy is made on every call.
        for scale in (1, 2)
            mine, theirs = device(single(data(3; scale))), single(data(3; scale))
            @test all(map((a, b) -> host(a) ≈ b, C.leaves(c(mine)), C.leaves(f(theirs)))) && Array(mine) ≈ theirs
        end
    end
    @test occursin("input 1 copied apart before the kernels", sprint(explain, compile(stepped, device(single(data(3)))).executable))
end

@testset "writes into the arguments" begin
    # What the traced code writes into an argument's arrays, a layer's running
    # statistics say, the caller's arrays hold after each call, as they would
    # after the code ran untraced; reads before the write keep what was given.
    function normalized(m, x)
        μ = vec(sum(x; dims = 2)) ./ size(x, 2)
        m.mean .= 0.9 .* m.mean .+ 0.1 .* μ
        m.scale .* (x .- μ)
    end
    m, x = Running(data(3), data(3; scale = 0.5)), data(3, 4)
    c = compile(normalized, m, x)
    for _ in 1:2
        mine, theirs = deepcopy(m), deepcopy(m)
        @test c(mine, x) ≈ normalized(theirs, x) && mine.mean ≈ theirs.mean && mine.scale == theirs.scale
        m = theirs
    end
    # Also through a gradient, whose value and gradient are at the arrays as
    # they were given.
    step(m, x) = Tempest.withgradient((m, x) -> sum(abs2, normalized(m, x)), m, x)
    g = compile(step, m, x)
    mine, theirs = deepcopy(m), deepcopy(m)
    value, (gm, gx) = g(mine, x)
    @test value ≈ sum(abs2, normalized(deepcopy(m), x)) && gm.mean === nothing || all(iszero, gm.mean)
    normalized(theirs, x)
    @test mine.mean ≈ theirs.mean
    # A result may be what an input held before the program wrote it, as an
    # optimizer's step returns a parameter's old value: the input is copied
    # apart before the kernels, on each call.
    check_before_write(single)
end

# How many arguments the kernels read where the caller has them.
pointed(c) = (ex = c.executable; count(k -> ex.pointable[k] !== nothing && ex.pointed[k] !== ex.arguments[k], eachindex(ex.pointable)))

@testset "arguments read where the caller has them" begin
    # An argument the program only reads is copied into the program's storage
    # on its first call, and read where the caller has it once the caller
    # brings the same array twice in a row: what the caller writes into it
    # after, the next call reads.
    m, x = Dense(data(3, 4), data(3), tanh), data(4, 2)
    c = compile((m, x) -> m(x), m, x)
    @test c(m, x) ≈ m(x) && pointed(c) == 0
    @test c(m, x) ≈ m(x) && pointed(c) == 3
    x .*= 2
    m.w[1] = 7
    @test c(m, x) ≈ m(x) && pointed(c) == 3
    # A new array is copied, as an array that moves every call is.
    y = data(4, 2; scale = 0.3)
    @test c(m, y) ≈ m(y) && pointed(c) == 2
    @test c(m, y) ≈ m(y) && pointed(c) == 3
    # `bind!` copies, and the kernels read the program's storage again.
    ex = c.executable
    bind!(ex, c.f, m, x)
    @test execute!(ex) === ex && outputs(ex) ≈ m(x) && pointed(c) == 0
    # An argument the program writes is copied into its storage and back on
    # every call, the same array or not; the others are read where they are.
    function normalized(m, x)
        μ = vec(sum(x; dims = 2)) ./ size(x, 2)
        m.mean .= 0.9 .* m.mean .+ 0.1 .* μ
        m.scale .* (x .- μ)
    end
    mine, theirs, x = Running(data(3), data(3; scale = 0.5)), Running(data(3), data(3; scale = 0.5)), data(3, 4)
    n = compile(normalized, mine, x)
    for _ in 1:3
        @test n(mine, x) ≈ normalized(theirs, x) && mine.mean ≈ theirs.mean
    end
    @test pointed(n) == 2
    # A result that is an input is the caller's array copied, never pointed at.
    same = compile(x -> (x, 2 .* x), x)
    before = copy(x)
    a, = same(x)
    b, = same(x)
    x .= 0
    @test a == b == before && pointed(same) == 0
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
        # Inputs as they were before the program wrote them, copied apart
        # within the capture and launched kernel by kernel.
        check_before_write(CuArray)
        check_before_write(CuArray; capture = false)
    end

    @testset "arguments read where the caller has them, on CUDA" begin
        # The captured graph is recorded again when its kernels are pointed
        # at the caller's arrays, and when they are pointed back.
        m, x = Dense(CuArray(single(data(64, 32; scale = 0.3))), CuArray(single(data(64))), tanh), CuArray(single(data(32, 16; scale = 0.5)))
        reference(m, x) = Array(m.act.(m.w * x .+ m.b))
        for capture in (true, false)
            c = compile((m, x) -> m(x), m, x; capture)
            @test isapprox(Array(c(m, x)), reference(m, x); rtol = 1.0e-5) && pointed(c) == 0
            @test isapprox(Array(c(m, x)), reference(m, x); rtol = 1.0e-5) && pointed(c) == 3
            x .*= 2
            @test isapprox(Array(c(m, x)), reference(m, x); rtol = 1.0e-5) && pointed(c) == 3
            z = copy(x) .+ 1
            @test isapprox(Array(c(m, z)), reference(m, z); rtol = 1.0e-5) && pointed(c) == 2
            @test isapprox(Array(c(m, z)), reference(m, z); rtol = 1.0e-5) && pointed(c) == 3
            # An array the kernels cannot read where it is, one element into
            # another, so not at the alignment they assume, is copied.
            y = CuArray{Float32}(undef, length(x) + 1)
            unaligned = unsafe_wrap(CuArray, pointer(y) + sizeof(Float32), size(x))
            copyto!(unaligned, x)
            for _ in 1:2
                @test isapprox(Array(c(m, unaligned)), reference(m, x); rtol = 1.0e-5) && pointed(c) == 2
            end
        end
    end
end
