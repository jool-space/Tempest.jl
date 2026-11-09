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

# An array whose memory Tempest cannot see: no strides, no parent, only Base's `dataids`.
struct Opaque{T} <: AbstractVector{T}
    data::Vector{T}
end
Base.size(x::Opaque) = size(x.data)
Base.getindex(x::Opaque, i::Int) = x.data[i]
Base.dataids(x::Opaque) = Base.dataids(x.data)

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
    # A struct is matched by its kind, a named tuple by its names, and an
    # array by its element type and size, wherever it is.
    @test occursin("argument 1 is a named tuple of w, b, act, not a Dense", message(() -> c((w = m.w, b = m.b, act = tanh), x)))
    @test occursin("argument 1.w is a Float32 array of size (3, 4), not a Float64 array of size (3, 4)",
        message(() -> c(Dense(Float32.(m.w), m.b, tanh), x)))
    t = compile((m, x) -> m.act.(m.w * x .+ m.b), (w = m.w, b = m.b, act = tanh), x)
    @test occursin("argument 1 is a Dense, not a named tuple of w, b, act", message(() -> t(m, x)))
    @test occursin("argument 1 is a named tuple of b, w, act, not a named tuple of w, b, act", message(() -> t((b = m.b, w = m.w, act = tanh), x)))
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
    # A matrix of traced arrays would hand them out.
    @test occursin("may hold traced values", message(() -> trace(x -> reshape(Any[x, x], 1, 2), x)))
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

@testset "arrays inside an argument that is not looked inside" begin
    # A dictionary or an array of other things than numbers that is no
    # vector is a constant of the program, and so are the arrays it holds:
    # refused, naming the argument they were reached through, though a global
    # holds them too.
    x = data(4)
    table = Dict(:w => held_table)
    @test occursin("argument 1, a Dict{Symbol, Vector{Float64}} with 1 entry, holds arrays",
        message(() -> compile((d, x) -> d[:w] .* x, table, x)))
    grid = reshape([held_table, data(4)], 1, 2)
    @test occursin("argument 1.grid, a 1×2 Matrix{Vector{Float64}}, holds arrays",
        message(() -> compile((m, x) -> m.grid[1] .* x, (; grid), x)))
    @test occursin("f.table, a Dict", message(() -> compile(x -> table[:w] .* x, x)))
    @test occursin("Hold them in tuples, named tuples, structs or vectors", message(() -> compile((d, x) -> d[:w] .* x, table, x)))
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

@testset "vectors among the arguments" begin
    # A vector of layers, as a checkpoint's tree holds them, is looked inside
    # as a tuple is: each layer's arrays are inputs, and its length is the
    # program's.
    forward(model, x) = foldl((x, layer) -> tanh.(layer.w * x .+ layer.b), model.layers; init = x)
    layer(scale) = (w = data(3, 3; scale), b = data(3; scale))
    model, x = (; layers = [layer(0.3), layer(0.7)]), data(3, 2)
    other = (; layers = [layer(0.5), layer(0.2)])
    p = trace(forward, model, x)
    @test length(p.arguments) == 5
    @test interpret(p, other, x) ≈ forward(other, x)
    c = compile(forward, model, x)
    @test c(other, x) ≈ forward(other, x)
    @test occursin("argument 1.layers is a vector of 3, not a vector of 2", message(() -> c((; layers = [layer(0.1), layer(0.2), layer(0.3)]), x)))
    @test occursin("argument 1.layers is a tuple of 2, not a vector of 2", message(() -> c((; layers = (layer(0.1), layer(0.2))), x)))
    @test occursin("argument 1.layers[2].w is a Float64 array of size (3, 4), not a Float64 array of size (3, 3)",
        message(() -> c((; layers = [layer(0.1), (w = data(3, 4), b = data(3))]), x)))
    # So is a vector of structs, and a vector of arrays.
    stack = [Dense(data(3, 3; scale = 0.3), data(3), tanh), Dense(data(3, 3; scale = 0.6), data(3), tanh)]
    @test compile((ds, x) -> ds[2](ds[1](x)), stack, x)(stack, x) ≈ stack[2](stack[1](x))
    arrays = [data(3), data(3; scale = 0.5)]
    vs = compile((vs, x) -> vs[1] .* x .+ vs[2], arrays, x)
    @test vs(arrays, x) ≈ arrays[1] .* x .+ arrays[2]
    # A vector of numbers is an array, not a vector of arrays.
    @test occursin("argument 1 is a Float64 array of size (2,), not a vector of 2", message(() -> vs([1.0, 2.0], x)))
    # A vector among the results is built as `map` builds one, and a vector's
    # gradient is a vector of its elements'.
    twice = compile(x -> [x, 2 .* x], x)(x)
    @test twice isa Vector{Matrix{Float64}} && twice == [x, 2 .* x]
    g = compile((m, x) -> Tempest.gradient(m -> sum(forward(m, x)), m), model, x)(model, x)[1]
    reference = interpret(differentiate(trace(m -> sum(forward(m, x)), model)), model)[2]
    @test g.layers isa Vector && length(g.layers) == 2 && all(k -> g.layers[k].w ≈ reference.layers[k].w, 1:2)
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

@testset "arrays sharing memory with one the function writes" begin
    # The program reads each array apart, where the function itself would
    # read what it wrote: a call with arrays that may share memory with one
    # it writes throws, naming both.
    f!(a, b) = (a .= 2 .* a; sum(b))
    c = compile(f!, data(2), data(2; scale = 0.5))
    u = data(2)
    @test occursin("argument 2 shares memory with argument 1, which f! writes", message(() -> c(u, u)))
    @test occursin("argument 1 shares memory with argument 2", message(() -> compile((b, a) -> f!(a, b), data(2), data(2))(u, u)))
    m = Running(data(3), data(3))
    @test occursin("argument 2 shares memory with argument 1.mean", message(() -> compile((m, x) -> (m.mean .+= x; sum(m.scale)), m, data(3))(m, m.mean)))
    # Memory is shared through a view, reversed or not, a reshape, a
    # transpose and a view by indices, of the array or of its parent, and
    # through an array whose memory Tempest cannot see, by Base's `dataids`.
    long = data(4)
    @test occursin("argument 2 shares memory with argument 1", message(() -> c(view(long, 1:2), view(long, 2:3))))
    @test occursin("argument 2 shares memory with argument 1", message(() -> c(u, view(u, 2:-1:1))))
    @test occursin("argument 2 shares memory with argument 1", message(() -> c(view(u, [2, 1]), u)))
    g = compile((a, b) -> (a .= 2 .* a; sum(b)), data(4), data(2, 2))
    @test occursin("argument 2 shares memory with argument 1", message(() -> g(long, reshape(long, 2, 2))))
    @test occursin("argument 2 shares memory with argument 1", message(() -> g(long, transpose(reshape(long, 2, 2)))))
    @test occursin("argument 2 shares memory with argument 1", message(() -> c(u, Opaque(u))))
    # Arrays apart run as the function does, two parts of one array apart
    # too, though they lie between each other as two rows do, and arrays it
    # only reads may share memory.
    rows = permutedims([data(2) data(2; scale = 0.5)])
    @test c(view(rows, 1, :), view(rows, 2, :)) ≈ sum(data(2; scale = 0.5)) && rows[1, :] == 2 .* data(2) && rows[2, :] == data(2; scale = 0.5)
    v, w = data(2), data(2; scale = 0.5)
    @test c(v, w) == f!(data(2), data(2; scale = 0.5)) && v == 2 .* data(2)
    both = [data(2) data(2; scale = 0.5)]
    @test c(view(both, :, 1), view(both, :, 2)) ≈ sum(data(2; scale = 0.5)) && both[:, 1] == 2 .* data(2)
    @test compile((a, b) -> sum(a) + sum(b), data(2), data(2))(u, u) == 2 * sum(u)
end

# How many arguments the kernels read where the caller has them.
pointed(c) = (ex = c.executable; count(k -> ex.pointable[k] !== nothing && ex.pointed[k] !== ex.arguments[k], eachindex(ex.pointable)))

@testset "arguments read where the caller has them" begin
    # An argument the program only reads is read where the caller has it:
    # what the caller writes into it after, the next call reads.
    m, x = Dense(data(3, 4), data(3), tanh), data(4, 2)
    c = compile((m, x) -> m(x), m, x)
    ex = c.executable
    @test c(m, x) ≈ m(x) && pointed(c) == 3
    x .*= 2
    m.w[1] = 7
    @test c(m, x) ≈ m(x) && pointed(c) == 3
    # Another array is read where it is too.
    y = data(4, 2; scale = 0.3)
    @test c(m, y) ≈ m(y) && pointed(c) == 3 && any(p -> p === y, ex.pointed)
    # `bind!` copies, and the kernels read the program's storage again.
    bind!(ex, c.f, m, x)
    @test execute!(ex) === ex && outputs(ex) ≈ m(x) && pointed(c) == 0
    # An argument whose graph had to be recorded again to read elsewhere
    # waits to be brought twice in a row: the layer's arrays, brought on the
    # call before, are read where they are, a new array is copied, and read
    # where it is on the next call.
    ex.settles .= true
    w = copy(y)
    @test c(m, w) ≈ m(w) && pointed(c) == 2
    @test c(m, w) ≈ m(w) && pointed(c) == 3
    # An argument the program writes is written where the caller has it too.
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
    @test pointed(n) == 3
    # A result that is an input is over the caller's array, and copied out of it.
    same = compile(x -> (x, 2 .* x), x)
    before = copy(x)
    a, = same(x)
    b, = same(x)
    x .= 0
    @test a == b == before && pointed(same) == 1
end

# Functions that write their arguments: a cache at positions the device
# holds, a state overwritten whole, and a ring shifted within itself, whose
# reads and writes overlap, so that its result is staged and copied in.
cache_put!(K, x, positions) = (K[:, positions, :] .= x; sum(x))
state_over!(S, x) = (S .= tanh.(S .+ x); nothing)
ring_shift!(ring, x) = (ring .= hcat(ring[:, 2:end], x); sum(ring))

# Each compiled on `device`'s arrays and called three times on each of two
# sets of arrays, in turn, against the function run eagerly on copies: the
# results, and every argument after the call. The written argument is the
# program's storage, the caller's array itself, and the program holds no
# buffer of its own for it; an `Array` passed to a function compiled for a
# device is copied in and back, through a buffer made for it then.
function check_written_in_place(device; options...)
    arguments = (
        cache_put! => s -> (data(4, 6, 2; scale = s), data(4, 2, 2; scale = 2s), Int32[5, 2]),
        state_over! => s -> (data(3, 4; scale = s), data(3, 4; scale = 2s)),
        ring_shift! => s -> (data(3, 5; scale = s), data(3; scale = 2s)))
    for (f, make) in arguments
        on(s) = map(x -> device(single(x)), make(s))
        c = compile(f, on(0.1)...; options...)
        ex = c.executable
        @test all(isnothing, ex.arguments)
        sets = (on(0.3), on(0.7))
        for _ in 1:3, mine in sets
            theirs = map(host, mine)
            r, e = c(mine...), f(theirs...)
            @test (r === nothing ? e === nothing : r ≈ e) && all(map((a, b) -> host(a) ≈ b, mine, theirs))
            @test ex.pointed[1] === mine[1]
        end
        @test all(isnothing, ex.arguments) && !any(ex.settles)
        @test occursin("input 1 is the caller's array", sprint(explain, c))
        mine = map(host, on(0.5))
        theirs = deepcopy(mine)
        r, e = c(mine...), f(theirs...)
        @test (r === nothing ? e === nothing : r ≈ e) && mine[1] ≈ theirs[1]
        @test all(isnothing, ex.arguments) == (device === identity)
    end
    @test occursin("staged, then copied into input 1", sprint(explain, compile(ring_shift!, device(single(data(3, 5))), device(single(data(3))))))
end

@testset "writes land in the caller's arrays" begin
    check_written_in_place(identity)
end

@testset "the level beneath binds a call's arrays as compile does" begin
    # `point!` binds the caller's arrays both ways: one the kernels can
    # write where it is they write there, one they cannot, a view say, is
    # copied in, and each run copies what the program wrote back into it.
    # `bind!` copies in alone.
    f!(s, x) = (s .= 0.5 .* s .+ x; sum(s))
    s, x = data(2), data(2; scale = 0.5)
    ex = prepare(trace(f!, s, x), s, x)
    both = [data(2) data(2)]
    mine = view(both, :, 1)
    C.point!(ex, mine, x)
    @test execute!(ex) === ex && mine ≈ 0.5 .* data(2) .+ x
    execute!(ex)
    @test mine ≈ 0.25 .* data(2) .+ 1.5 .* x && both[:, 2] == data(2)
    bind!(ex, data(2), x)
    execute!(ex)
    @test mine ≈ 0.25 .* data(2) .+ 1.5 .* x
    C.point!(ex, s, x)
    execute!(ex)
    @test s ≈ 0.5 .* data(2) .+ x && ex.pointed[1] === s
end

# A recurrence no one kernel holds, which runs a step at a time.
function settled(W, u)
    for _ in 1:8
        u = tanh.(W' * (W * u)) .+ u
    end
    sum(abs2, u)
end

@testset "an argument's storage: the caller's array, or a buffer made when one is copied" begin
    add(a, b) = a .+ b
    a, b = data(6), data(6; scale = 0.5)
    c = compile(add, a, b)
    ex = c.executable
    # Built on the caller's arrays, the program holds no buffer of its own
    # for them: its bytes are its result's.
    @test all(isnothing, ex.arguments) && C.footprint(ex).bytes == sizeof(a)
    @test c(a, b) == a .+ b && all(isnothing, ex.arguments)
    # An array the kernels cannot read where it is, a stepped view, is
    # copied into a buffer made then, and kept for the next.
    m = data(12)
    @test c(view(m, 1:2:12), b) == m[1:2:12] .+ b && ex.arguments[1] isa Vector{Float64} && ex.arguments[2] === nothing
    own = ex.arguments[1]
    @test c(view(m, 2:2:12), b) == m[2:2:12] .+ b && ex.arguments[1] === own && C.footprint(ex).bytes == 2 * sizeof(a)
    @test c(a, b) == a .+ b && ex.pointed[1] === a
    # The level beneath copies into a buffer for each argument from the start.
    e = prepare(trace(add, a, b), a, b)
    @test all(x -> x isa Vector{Float64}, e.arguments) && C.footprint(e).bytes == 3 * sizeof(a)
    bind!(e, b, a)
    @test outputs(execute!(e)) == b .+ a && e.arguments[1] == b
    # A compiled program holds the arrays it was compiled on until a call
    # brings others, and those of its last call after.
    function compiled_and_called()
        x, y = data(6), data(6)
        c = compile(add, x, y)
        @test c(a, b) == a .+ b
        c, WeakRef(x), WeakRef(y)
    end
    c, x, y = compiled_and_called()
    foreach(_ -> GC.gc(true), 1:3)
    @test x.value === nothing && y.value === nothing && c.executable.pointed[1] === a
    # A recurrence run a step at a time cannot be pointed elsewhere: each
    # argument has a buffer of the program's own, its storage for good, and
    # a call's arrays are copied into it, as the level beneath copies.
    W, u = data(4, 4; scale = 0.2), data(4, 3)
    r = compile(settled, W, u)
    ex = r.executable
    @test any(k -> k isa C.SteppedKernels, ex.kernels)
    @test all(isnothing, ex.pointable) && all(x -> x isa Vector{Float64}, ex.arguments)
    @test r(W, u) ≈ settled(W, u) && r(2 .* W, u) ≈ settled(2 .* W, u)
end

# A small LSTM whose training step writes its parameters and Adam's moments
# in place, for a search behind the first result to give another plan.
σ(x) = 1 / (1 + exp(-x))
function lstm_cell(p, (h, c), x)
    i, f = σ.(p.Wi * x .+ p.Ui * h .+ p.bi), σ.(p.Wf * x .+ p.Uf * h .+ p.bf)
    g, o = tanh.(p.Wg * x .+ p.Ug * h .+ p.bg), σ.(p.Wo * x .+ p.Uo * h .+ p.bo)
    c = f .* c .+ i .* g
    (o .* tanh.(c), c)
end
lstm(p, xs, h0) = Tuple(map(first, accumulate((s, x) -> lstm_cell(p, s, x), eachslice(xs; dims = 2); init = (h0, h0))))
lstm_weights() = (; (Symbol(:W, k) => data(3, 2; scale = 0.3) for k in "ifgo")..., (Symbol(:U, k) => data(3, 3; scale = 0.3) for k in "ifgo")...,
    (Symbol(:b, k) => data(3; scale = 0.1) for k in "ifgo")...)
function adam!(p, m, v, xs, h0)
    l, (g,) = Tempest.withgradient(q -> sum(h -> sum(abs2, h), lstm(q, xs, h0)), p)
    for (w, mw, vw, d) in zip(values(p), values(m), values(v), values(g))
        mw .= 0.9 .* mw .+ 0.1 .* d
        vw .= 0.999 .* vw .+ 0.001 .* d .^ 2
        w .-= 0.01 .* mw ./ (sqrt.(vw) .+ 1.0e-8)
    end
    l
end

# The plan the search behind a first result gives on the CPU, which times
# nothing: the program prepared again with every fold that can be recomputed
# where it is read recomputed, which is another plan of it.
function other_plan(c, args...)
    program = c.executable.program
    folds = C.recomputable_folds(C.rewrite(C.lower(program); rounded = C.rounded_backward(T.CPUTarget())))
    timer -> prepare(program, c.f, args...; recompute = folds)
end

# `c` searching behind its calls as `work(timer)` does.
searching(c, work) = C.Searching(c, C.Search(work))

# A target whose timings take a millisecond, noting how many ran at once.
struct Clock <: C.Target end
const TIMING_NOW, TIMING_MOST = Threads.Atomic{Int}(0), Threads.Atomic{Int}(0)
function C.time_launch(::Clock, launch)
    Threads.atomic_max!(TIMING_MOST, Threads.atomic_add!(TIMING_NOW, 1) + 1)
    sleep(0.005)
    launch()
    Threads.atomic_sub!(TIMING_NOW, 1)
    1.0e-3
end

@testset "timings one at a time" begin
    # From several tasks at once, each timing waits for the one before.
    ran = Threads.Atomic{Int}(0)
    foreach(wait, [Threads.@spawn(C.timed(Clock(), () -> Threads.atomic_add!(ran, 1))) for _ in 1:8])
    @test ran[] == 8 && TIMING_MOST[] == 1
end

@testset "a search behind the first result" begin
    # A target that times nothing has nothing to search.
    c = compile(x -> 2 .* x, data(4); search = :after, measure = :close)
    @test c isa T.Compiled && wait(c) === c && c(data(4)) == 2 .* data(4)
    @test_throws ArgumentError compile(x -> 2 .* x, data(4); search = :never)
    # The calls take the other plan between two of them, from two tasks
    # calling until the search has ended, each call a step of the LSTM: a
    # recurrence, its parameters and Adam's moments replaced in place. After
    # as many steps they are those of the first plan run throughout, and so
    # are the losses, in whatever order the tasks took them.
    xs, h0 = data(2, 5; scale = 0.5), data(3; scale = 0.1)
    p, m, v = lstm_weights(), map(zero, lstm_weights()), map(zero, lstm_weights())
    c = compile(adam!, p, m, v, xs, h0)
    first_plan = c.executable
    losses = [c(p, m, v, xs, h0) for _ in 1:3]
    s = searching(c, other_plan(c, p, m, v, xs, h0))
    search = s.search
    function calling()
        ls = Float64[]
        while !C.ended(search) || length(ls) < 3
            push!(ls, s(p, m, v, xs, h0))
            sleep(0.02)
        end
        ls
    end
    tasks = [Threads.@spawn(calling()) for _ in 1:2]
    foreach(t -> append!(losses, fetch(t)), tasks)
    @test wait(s) === s && s.search === nothing
    append!(losses, [s(p, m, v, xs, h0) for _ in 1:3])
    plan = s.compiled.executable
    @test plan !== first_plan && length(plan.kernels) != length(first_plan.kernels)
    @test s.compiled.lock === c.lock && first_plan.captured === nothing
    q, mq, vq = lstm_weights(), map(zero, lstm_weights()), map(zero, lstm_weights())
    one_plan = compile(adam!, q, mq, vq, xs, h0)
    alone = [one_plan(q, mq, vq, xs, h0) for _ in eachindex(losses)]
    @test isapprox(sort(losses), sort(alone); rtol = 1.0e-10)
    @test all(map((a, b) -> isapprox(a, b; rtol = 1.0e-10), values(p), values(q)))
    @test all(map((a, b) -> isapprox(a, b; rtol = 1.0e-10), values(m), values(mq)))
    @test all(map((a, b) -> isapprox(a, b; rtol = 1.0e-10), values(v), values(vq)))
    r, mr, vr = lstm_weights(), map(zero, lstm_weights()), map(zero, lstm_weights())
    @test isapprox(alone[1:3], [adam!(r, mr, vr, xs, h0) for _ in 1:3]; rtol = 1.0e-12)
    # A search that gives the plan the calls run changes nothing.
    c = compile(adam!, p, m, v, xs, h0)
    s = searching(c, timer -> prepare(c.executable.program, adam!, p, m, v, xs, h0))
    @test wait(s).compiled === c
    # A search that fails leaves the program its plan, with a warning.
    c = compile(adam!, p, m, v, xs, h0)
    @test_logs (:warn, r"search behind the first result failed") wait(s = searching(c, timer -> error("no plan")))
    @test s.compiled === c
    @test isapprox(s(p, m, v, xs, h0), adam!(q, mq, vq, xs, h0); rtol = 1.0e-10)
end

@testset "a search's timings, taken by the calls" begin
    xs, h0 = data(2, 5; scale = 0.5), data(3; scale = 0.1)
    p, m, v = lstm_weights(), map(zero, lstm_weights()), map(zero, lstm_weights())
    q, mq, vq = lstm_weights(), map(zero, lstm_weights()), map(zero, lstm_weights())
    # The search times nothing while the program is not called; each call
    # takes one of its timings after its own work, on the calling task, and
    # `wait` takes the rest.
    c = compile(adam!, p, m, v, xs, h0)
    timers = Task[]
    seconds = Float64[]
    s = searching(c, timer -> (append!(seconds, [timer(Clock(), () -> push!(timers, current_task())) for _ in 1:3]); nothing))
    asked = s.search.asked
    @test timedwait(() -> isready(asked), 60) === :ok && isempty(timers)
    @test isapprox(s(p, m, v, xs, h0), adam!(q, mq, vq, xs, h0); rtol = 1.0e-10)
    @test timers == [current_task()]
    @test wait(s) === s && timers == fill(current_task(), 3) && seconds == fill(1.0e-3, 3)
    # A timing that throws fails the search, not the call that took it.
    c = compile(adam!, p, m, v, xs, h0)
    @test_logs (:warn, r"search behind the first result failed") begin
        s = searching(c, timer -> timer(Clock(), () -> error("a claim that throws")))
        asked = s.search.asked
        @test timedwait(() -> isready(asked), 60) === :ok
        @test isapprox(s(p, m, v, xs, h0), adam!(q, mq, vq, xs, h0); rtol = 1.0e-10)
        wait(s)
    end
    @test s.compiled === c && isapprox(s(p, m, v, xs, h0), adam!(q, mq, vq, xs, h0); rtol = 1.0e-10)
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

    @testset "writes land in the caller's arrays, on CUDA" begin
        # A captured graph's kernels and copies are changed in place where
        # they read and write an argument, not recorded again.
        check_written_in_place(CuArray)
        check_written_in_place(CuArray; capture = false)
        # Arrays that share memory with one the function writes are refused
        # on the device as on the CPU: the kernels would read what they write.
        u = CuArray(single(data(4)))
        c = compile((a, b) -> (a .= 2 .* a; sum(b)), CuArray(single(data(4))), CuArray(single(data(4))))
        @test occursin("argument 2 shares memory with argument 1", message(() -> c(u, u)))
        # A stepped view of a host array is copied in, and what the program
        # wrote copied back through an `Array`.
        step!(s, x) = (s .= 0.5f0 .* s .+ x; sum(s))
        c = compile(step!, CuArray(single(data(4))), CuArray(single(data(4))))
        b, x = single(data(8)), single(data(4; scale = 0.5))
        before = copy(b)
        expected = 0.5f0 .* before[1:2:8] .+ x
        @test c(view(b, 1:2:8), x) ≈ sum(expected) && b[1:2:8] ≈ expected && b[2:2:8] == before[2:2:8]
    end

    @testset "timings one at a time, on CUDA" begin
        # A preparation's timings, of claims no record here holds, wait while
        # another holds the timings' lock, recording nothing.
        record = C.record_path(T.CUDATarget())
        lines() = isfile(record) ? countlines(record) : 0
        waiting() = !isempty(C.TIMINGS.cond_wait.waitq)
        z = CuArray(single(data(48, 80; scale = 0.1)))
        before = lines()
        t = lock(C.TIMINGS) do
            t = Threads.@spawn prepare(trace(x -> sum(abs2, 1.5f0 .* x; dims = 1), z), z; measure = :all)
            @test timedwait(waiting, 300) === :ok
            @test lines() == before
            t
        end
        @test isapprox(Array(outputs(execute!(fetch(t)))), sum(abs2, 1.5f0 .* Array(z); dims = 1); rtol = 1.0e-4)
        @test lines() > before
    end

    @testset "a search behind the first result, on CUDA" begin
        # A program with no matrix product, whose plans share no library
        # handle between the search's task and the calls': cuBLAS keeps one
        # handle and one workspace per context, which two tasks' products race on.
        w, x = single(data(64, 32; scale = 0.3)), single(data(64, 32; scale = 0.5))
        dw, dx = CuArray(w), CuArray(x)
        f = (w, x) -> (sum(abs2, tanh.(w .* x .+ 1.0f0); dims = 1), tanh.(w .* x .+ 1.0f0))
        agrees((value, y)) = isapprox(Array(value), f(w, x)[1]; rtol = 1.0e-4) && isapprox(Array(y), f(w, x)[2]; rtol = 1.0e-4)
        s = compile(f, dw, dx; search = :after, measure = :all)
        @test s isa T.Searching && agrees(s(dw, dx))
        # A task calling, and synchronizing the device after each call, until
        # the search ends: the calls take its timings, and nothing the search
        # does beside them forbids the synchronization, as a capture would.
        search = s.search
        calls = Threads.@spawn begin
            results = Bool[]
            while !C.ended(search) || isempty(results)
                push!(results, agrees(s(dw, dx)))
                CUDACore.device_synchronize()
            end
            results
        end
        @test all(fetch(calls)) && C.ended(search) && !istaskfailed(search.task)
        @test wait(s) === s && agrees(s(dw, dx)) && s.compiled.executable.captured !== nothing
        # Compiled again, in this process or another, a program whose search
        # measured close calls replays the decisions it settled: its first
        # result is the measured plan, and there is nothing to search behind
        # it. (A preparation measuring nothing replays those of one measuring
        # close calls, not every option.)
        s = compile(f, dw, dx; search = :after, measure = :close)
        @test s isa T.Searching && wait(s) === s
        again = compile(f, dw, dx; search = :after, measure = :close)
        @test again isa T.Compiled && C.same_choices(again.executable, s.compiled.executable) && agrees(again(dw, dx))
        # The first result times nothing, and the search times only in the
        # calls' time, a claim each, and in `wait`, which takes what is left
        # and ends it: a program no other here prepares, its options in no
        # record yet.
        record = C.record_path(T.CUDATarget())
        lines() = isfile(record) ? countlines(record) : 0
        z = CuArray(single(data(48, 80; scale = 0.1)))
        before = lines()
        s = compile(x -> sum(abs2, 2.5f0 .* x; dims = 2), z; search = :after, measure = :all)
        search = s.search
        @test timedwait(() -> isready(search.asked), 300) === :ok && lines() == before
        expected = sum(abs2, 2.5f0 .* Array(z); dims = 2)
        @test isapprox(Array(s(z)), expected; rtol = 1.0e-4)
        @test timedwait(() -> lines() > before, 60) === :ok && lines() == before + 1
        @test isapprox(Array(wait(s)(z)), expected; rtol = 1.0e-4)
        @test s.search === nothing && C.ended(search) && !istaskfailed(search.task) && !isready(search.asked)
    end

    @testset "arguments read where the caller has them, on CUDA" begin
        # The captured graph's kernels are changed in place where they read
        # an argument pointed elsewhere, not recorded again.
        m, x = Dense(CuArray(single(data(64, 32; scale = 0.3))), CuArray(single(data(64))), tanh), CuArray(single(data(32, 16; scale = 0.5)))
        reference(m, x) = Array(m.act.(m.w * x .+ m.b))
        for capture in (true, false)
            c = compile((m, x) -> m(x), m, x; capture)
            ex = c.executable
            @test isapprox(Array(c(m, x)), reference(m, x); rtol = 1.0e-5) && pointed(c) == 3
            x .*= 2
            @test isapprox(Array(c(m, x)), reference(m, x); rtol = 1.0e-5) && pointed(c) == 3
            for _ in 1:3
                z = copy(x) .+ 1
                @test isapprox(Array(c(m, z)), reference(m, z); rtol = 1.0e-5) && pointed(c) == 3
            end
            @test !any(ex.settles)
            # An array the kernels cannot read where it is, one element into
            # another, so not at the alignment they assume, is copied.
            y = CuArray{Float32}(undef, length(x) + 1)
            unaligned = unsafe_wrap(CuArray, pointer(y) + sizeof(Float32), size(x))
            copyto!(unaligned, x)
            for _ in 1:2
                @test isapprox(Array(c(m, unaligned)), reference(m, x); rtol = 1.0e-5) && pointed(c) == 2
            end
            # Every execution a replay runs reads where the arguments are.
            if capture
                capture!(ex; repetitions = 3)
                for _ in 1:2
                    z = copy(x) .- 1
                    @test isapprox(Array(c(m, z)), reference(m, z); rtol = 1.0e-5) && pointed(c) == 3
                end
                @test !any(ex.settles)
            end
        end
    end
end
