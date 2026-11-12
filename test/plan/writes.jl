# Writes alike, joined: a loop writing each slice of an axis at positions of
# its own is one write over the slices, where no two land at one position.

# A cache (features, positions, heads, sequences) written at each sequence's
# own positions, a sequence at a time, as a transformer's step appends its
# keys; the sequences `b` at `step·b - shift`.
function appended(K, k, p; step = 1, shift = 0)
    K = copy(K)
    for b in axes(k, 4)
        K[:, p[:, b], :, step * b - shift] .= permutedims(k[:, :, :, b], (1, 3, 2))
    end
    K
end

keyed(g) = count(v -> v.definition isa C.Keyed, g.values)
joined(g) = count(((_, note),) -> occursin("writes alike", note), g.rewrites)

@testset "writes alike but for a position each lands apart at are one" begin
    for B in (1, 2, 3, 5)
        K, k = data(3, 8, 2, B), data(3, 2, 2, B; scale = 0.3)
        p = Int32.(mod1.([3, 6] .+ 2 .* (1:B)', 8))
        loss(K, k, p) = sum(abs2, sin.(appended(K, k, p)))
        for program in (trace(appended, K, k, p), differentiate(trace(loss, K, k, p)))
            g = C.joined_writes(C.lower(program))
            @test agree(C.execute(recomputing(g), K, k, p), filter(!isnothing, C.leaves(interpret(program, K, k, p))))
        end
        # From three sequences the trace rolls them, a map and one write
        # (see `joined_write`); two the index level joins.
        g = C.joined_writes(C.lower(trace(appended, K, k, p)))
        @test keyed(g) == 1 && joined(g) == (B == 2)
        @test prod(only(C.plan(g).groups).domain) == 3 * 2 * 2 * B
    end
    K, k, p = data(3, 8, 2, 6), data(3, 2, 2, 3; scale = 0.3), Int32[1 4 2; 2 5 3]
    # Sequences a step apart other than one, and the writes of the gradient,
    # each along the sequences too.
    g = C.joined_writes(C.lower(trace((K, k, p) -> appended(K, k, p; step = 2, shift = 1), K, k, p)))
    @test keyed(g) == 1
    @test agree(C.execute(recomputing(g), K, k, p), [appended(K, k, p; step = 2, shift = 1)])
    # Elements each write converts to the cache's type, a value per slice,
    # are stacked into one value over the slices, which the one write reads.
    Ks = single(K[:, :, :, 1:3])
    g = C.joined_writes(C.lower(trace(appended, Ks, k, p)))
    @test keyed(g) == 1 && length(C.plan(g).groups) == 1
    @test C.execute(recomputing(g), Ks, k, p) == [appended(Ks, k, p)]
    # Writes into one sequence, where a later one may land where an earlier
    # did, stay in their order, as does a write whose contents something
    # reads before the next.
    into_one(K, k, p) = (K = copy(K); foreach(b -> K[:, p[:, b], :, 1] .= permutedims(k[:, :, :, b], (1, 3, 2)), 1:3); K)
    function read_between(K, k, p)
        K = copy(K)
        s = sum(K)
        for b in 1:3
            K[:, p[:, b], :, b] .= permutedims(k[:, :, :, b], (1, 3, 2)) .* s
            s = sum(K)
        end
        K
    end
    for f in (into_one, read_between)
        g = C.joined_writes(C.lower(trace(f, K, k, p)))
        @test keyed(g) == 3 && joined(g) == 0
        @test agree(C.execute(recomputing(g), K, k, p), [f(K, k, p)])
    end
    # So the step a model writes its cache in, a sequence at a time or in
    # one statement over Cartesian indices, is one kernel's write, in place.
    cartesian(K, k, p) = (K = copy(K); K[:, :, CartesianIndex.(p, reshape(axes(p, 2), 1, :))] .= k; K)
    Kc = permutedims(K[:, :, :, 1:3], (1, 3, 2, 4))
    for (f, K) in ((appended, K[:, :, :, 1:3]), (cartesian, Kc))
        ex = prepare(trace(f, K, k, p), copy(K), k, p; target = T.CPUTarget(), replaces = (1 => 1,))
        @test count("overwrite by key", sprint(explain, ex)) == 1 && length(ex.storage.overwrites) == 1
        @test outputs(execute!(ex)) == f(K, k, p)
    end
end
