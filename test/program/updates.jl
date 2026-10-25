# Writes at the program level: into the arrays a program allocates, as
# functional updates, and never into its inputs.

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
