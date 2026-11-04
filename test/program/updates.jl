# Writes at the program level: into the arrays a program allocates, as
# functional updates, and into its inputs, as writes into the caller's arrays.

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
    # An array the function allocates itself holds numbers, not the
    # program's values; one allocated from a traced array does.
    @test_throws ArgumentError trace(x -> (y = zeros(size(x)); y .= 2 .* x; y), x)
    @test same(x -> (y = fill!(similar(x, size(x, 2)), 0); y .= x[1, :]; y), x)
end

# A write into an input, or into anything that shares its memory, is a write
# into the caller's array: what the input holds when the program ends is a
# result that replaces it, and the interpreter leaves it in the caller's array,
# as the code does untraced.
@testset "writes into inputs" begin
    x = data(8, 6)
    function same(f, args...)
        mine, theirs = deepcopy(args), deepcopy(args)
        interpret(trace(f, args...), mine...) == f(theirs...) && mine == theirs
    end
    for f in (x -> (x[1] = 0; x), x -> (x .= 2; x), x -> (fill!(x, 1); x), x -> (copyto!(x, ones(8, 6)); x),
              x -> (view(x, :, 1) .= 0; x), x -> (vec(x)[2] = 1; x), x -> (x'[1, 2] = 3; x), x -> (reshape(x, 6, 8) ./= 2; x))
        @test same(f, x)
    end
    # Reads before a write keep the contents the program was given; the
    # written contents are a result after those `f` returns.
    @test same(x -> (s = sum(x); x .= 0.9 .* x .+ 0.1; (s, sum(x))), x)
    @test same(x -> (y = copy(x); x[2] = 0; y), x)
    p = trace((m, x) -> (m .= 0.9 .* m .+ 0.1 .* vec(sum(x; dims = 2)); sum(abs2, x)), data(8), x)
    @test p.writes == [2 => 1] && p.output_structure == 1
    # A gradient's function writes into its arguments, traced or run eagerly:
    # what it writes lands in the argument, and the gradient is at the
    # contents it was given.
    writes(p) = (p[1] = 0; sum(p))
    @test same(x -> T.gradient(writes, x), x)
    y = copy(x)
    @test only(T.gradient(writes, y)) == (g = ones(8, 6); g[1] = 0; g) && y == (z = copy(x); z[1] = 0; z)
    statistics(m, x) = (m .= 0.9 .* m .+ 0.1 .* vec(sum(x; dims = 2)); sum(abs2, x))
    m = data(8; scale = 0.3)
    mine = copy(m)
    value, (g,) = interpret(trace((m, x) -> T.withgradient(x -> statistics(m, x), x), m, x), mine, x)
    @test value ≈ sum(abs2, x) && g ≈ 2 .* x && mine ≈ 0.9 .* m .+ 0.1 .* vec(sum(x; dims = 2))
    # A constant of the program cannot change.
    @test occursin("a constant of the program", message(() -> trace(x -> T.gradient(w -> (w .= 0; sum(x)), data(3)), x)))
end
