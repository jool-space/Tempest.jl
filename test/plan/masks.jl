# A value masked by a formula over its positions carries the formula and the
# number it is where the mask fails, through the maps of it: the planner
# reads it, so that a kernel sweeping the value skips the tiles the mask
# rules out.
@testset "masked values carry their formulas" begin
    s = single(data(6, 6))
    causal(s) = ifelse.(reshape(1:size(s, 1), :, 1) .<= reshape(1:size(s, 2), 1, :), s, -Inf32)
    g = C.lower(trace(s -> exp.(causal(s) .- T.logsumexp(causal(s); dims = 1)), s))
    found = filter(!isnothing, [C.masked(g, v) for v in eachindex(g.values)])
    # The masked scores are minus infinity where the mask fails, and their
    # probabilities zero.
    @test any(((m, x),) -> x == -Inf32, found)
    @test any(((m, x),) -> x == 0.0f0, found)
    # Keys at most the query: keys − queries ≤ 0, over the value's axes.
    @test all(((m, x),) -> m == [IR.add(IR.var(1), IR.scale(IR.var(2), -1), IR.Domain((6, 6)))], found)
    # A fold of a masked value is no masked value, nor is a product of one
    # and something unmasked, which the mask makes infinite, not a number.
    @test all(v -> C.masked(g, v) === nothing, [v for v in eachindex(g.values) if g.values[v].definition isa C.Reduction])
    h = C.lower(trace(s -> causal(s) .* s, s))
    @test C.masked(h, length(h.values)) === nothing
    # A sum of one and something unmasked is minus infinity there.
    h = C.lower(trace(s -> causal(s) .+ s, s))
    @test C.masked(h, length(h.values)) !== nothing && C.masked(h, length(h.values))[2] == -Inf32
    # A position compared with an integer the program reads, a length per
    # column as decode holds on the device, masks too: the formula reads the
    # length, keys − length ≤ 0.
    upto(s, n) = ifelse.(reshape(Int32(1):Int32(size(s, 1)), :, 1) .<= reshape(n, 1, :), s, -Inf32)
    n = Int32[2, 5, 6, 1, 3, 4]
    h = C.lower(trace((s, n) -> exp.(upto(s, n) .- T.logsumexp(upto(s, n); dims = 1)), s, n))
    found = filter(!isnothing, [C.masked(h, v) for v in eachindex(h.values)])
    d = IR.Domain((6, 6))
    length_read = IR.held_formula(h.arguments[2], (IR.var(2),), Int32, d)
    @test !isempty(found) && all(((m, x),) -> m == [IR.add(IR.add(IR.var(1), 1, d), IR.scale(length_read, -1), d)], found)
    @test any(((m, x),) -> x == 0.0f0, found)
    # A length of 64 bits is no formula: its range would overflow one's.
    h = C.lower(trace((s, n) -> upto(s, n), s, Int64.(n)))
    @test C.masked(h, length(h.values)) === nothing
    # A window, keys at most the query and fewer than 3 before it: a position
    # compared with an integer, queries − keys − 2 ≤ 0 beside the causal mask.
    i, j = reshape(1:6, :, 1), reshape(1:6, 1, :)
    h = C.lower(trace(s -> ifelse.((i .<= j) .& (j .- i .< 3), s, -Inf32), s))
    window = [IR.add(IR.var(1), IR.scale(IR.var(2), -1), d), IR.add(IR.add(IR.scale(IR.var(1), -1), IR.var(2), d), -2, d)]
    @test C.masked(h, length(h.values)) !== nothing && Set(first(C.masked(h, length(h.values)))) == Set(window)
    # A conjunction with a condition that is no formula, keys of the query's
    # document, fails wherever its causal part fails: it masks by that part.
    doc = [1, 1, 1, 2, 2, 2]
    h = C.lower(trace((s, doc) -> ifelse.((i .<= j) .& (reshape(doc, :, 1) .== reshape(doc, 1, :)), s, -Inf32), s, doc))
    @test C.masked(h, length(h.values)) !== nothing && first(C.masked(h, length(h.values))) == [window[1]]
end
