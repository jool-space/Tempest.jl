# A handle stands for a number only on its tape, so no handle is ever a
# program's data. A host array of traced numbers or arrays, as a
# comprehension, `map` over a range, `collect`, `[a, b]` or `vcat` make, is
# trace-time organisation of what it holds, as a tuple is: returned, it comes
# back as Julia's run returns it, an array on the host; Julia's own methods
# over its elements trace as they run, a write into a traced array among
# them (below). An operation on arrays it meets, a broadcast with traced
# arrays, a product, an index, refuses it by a message that says what to
# write, and so do a function the program calls and a result that hold a
# handle where nothing rebuilds them.
@testset "host containers of traced values" begin
    x, W = 3 .+ data(5, 3), data(4, 3; scale = 0.3)
    cols(x) = [sum(x[:, b]) for b in axes(x, 2)]
    handles(p) = any(n -> C.holds_handle(n.op), p.nodes)
    held = [
        (cols, (x,)),
        (x -> [sum(x[i, j:j]) for i in 1:2, j in 1:3], (x,)),
        (x -> map(b -> sum(x[:, b]), 1:3), (x,)),
        (x -> collect(sum(x[:, b]) for b in 1:3), (x,)),
        (x -> [sum(x), maximum(x), minimum(x)], (x,)),
        (x -> vcat(sum(x), maximum(x)), (x,)),
        (x -> hcat(sum(x), maximum(x)), (x,)),
        (x -> [cols(x)...], (x,)),
        (x -> collect((sum(x), maximum(x))), (x,)),
        (x -> cols(x) .+ [1.0, 2.0, 3.0], (x,)),
        (x -> Float32.(cols(x)), (x,)),
        (x -> reshape(cols(x), 1, :), (x,)),
        (x -> [x[:, b] .* i for i in 1:2, b in 1:3], (x,)),
        (x -> (sum(cols(x)), maximum(cols(x)), dot(cols(x), cols(x))), (x,)),
    ]
    for (f, args) in held
        p = trace(f, args...)
        @test !handles(p)
        got, eager = interpret(p, args...), f(args...)
        @test typeof(got) == typeof(eager)
        @test agree(C.leaves(got), C.leaves(eager); rtol = 1.0e-12)
    end
    compiled = T.compile(cols, single(x); target = T.CPUTarget(), capture = false)
    @test compiled(single(x)) isa Vector{Float32}
    @test compiled(single(x)) ≈ cols(single(x))

    refused = [
        x -> x .* cols(x)',
        x -> x[1:2, :] .* (sum(x), maximum(x)),
        x -> (y = copy(x); y[1, :] .= cols(x); y),
        x -> x[[argmax(x[:, 1]), argmin(x[:, 2])], 1],
        x -> W * cols(x),
    ]
    for f in refused
        @test occursin("a host array of traced numbers", message(() -> trace(f, x)))
    end
    @test occursin("would be recorded as data", message(() -> trace(x -> (v = cols(x); f = (a, b) -> a * v[b]; f.(x, (1:3)')), x)))
    s(x) = sum(x[:, 1])
    for f in (x -> complex(s(x), s(x)), x -> Dict(:s => s(x)))
        @test occursin("holding traced values is none of them", message(() -> trace(f, x)))
    end
    # A host array of plain numbers is a constant, and a host value that
    # holds no handle a constant result.
    @test count(n -> n.op isa C.Constant && n.op.value isa Array, trace(x -> x .* [1.0, 2.0, 3.0]', x).nodes) == 1
    @test interpret(trace(x -> (sum(x), Dict(:a => 1)), x), x)[2] == Dict(:a => 1)
end

# A traced number stands for a number of its type wherever one may be, but
# where Julia needs a plain one: a branch, a place typed for plain numbers,
# a hash. Julia finds each where it is taken, and the trace says what
# happened and what to write, whatever Julia tripped on. A host array of
# traced numbers written into a traced array is written as Base's loop
# writes it, each number at its position.
struct Held
    v::Float64
end

@testset "traced numbers where plain ones are needed" begin
    x = 3 .+ data(5, 3)
    s(x) = sum(x[:, 1])
    cols(x) = [sum(x[:, b]) for b in axes(x, 2)]
    stored = [
        x -> [s(x), 1.0],
        x -> [s(x) maximum(x); minimum(x) 1.0],
        x -> Float64[s(x), maximum(x)],
        x -> push!(Float64[], s(x)),
        x -> append!(Float64[], [s(x), maximum(x)]),
        x -> (v = zeros(3); v[2] = s(x); v),
        x -> fill!(Vector{Float64}(undef, 3), s(x)),
        x -> Ref{Float64}(s(x))[],
        x -> Held(s(x)).v,
        x -> (d = Dict{Symbol, Float64}(); d[:a] = s(x); d[:a]),
    ]
    for f in stored
        @test occursin("is stored where Julia needs a plain Float64", message(() -> trace(f, x)))
    end
    for f in (x -> length(Set([s(x)])), x -> Dict(s(x) => 1), x -> s(x) in Set([1.0]))
        @test occursin("is hashed", message(() -> trace(f, x)))
    end
    @test occursin("branches on a value", message(() -> trace(x -> s(x) in [1.0, 2.0], x)))
    @test occursin("would be recorded as data", message(() -> trace(x -> (t = s(x); reduce((p, q) -> p + t * q, x; init = 0.0)), x)))
    # Written element by element, as Base writes them; planned as one write.
    for f in (x -> (y = copy(x); y[1, :] = cols(x); y), x -> copyto!(similar(x, 3), cols(x)))
        @test agree((interpret(trace(f, x), x),), (f(x),); rtol = 1.0e-12)
    end
end
