# Programs over empty arrays, which the interpreter runs: a map of no
# elements gives none, and a fold over an axis of no positions gives its
# identity at every position it keeps. On the device every one fails while
# it is prepared, at each measure level: cuTile's generator sizes its tiles
# by `nextpow(2, extent)`, which takes no zero (`default_tiles`), and the
# cover then looks for the ends of a domain of no positions
# (`holding_bound`). Whether a group of no positions is left to the floor or
# planned before any implementation is asked is not decided yet.

@testset "gap: a domain of extent zero, closed once a group of no positions is planned without a generator's tiles" begin
    @testset "$name" for (name, f, x) in (
        ("a map", x -> x .+ 1.0f0, zeros(Float32, 0)),
        ("a masked map", x -> ifelse.(Int32.(1:length(x)) .< Int32(3), x, 0.0f0), zeros(Float32, 0)),
        ("a fold over no positions", x -> sum(x; dims = 1), zeros(Float32, 0, 3)),
    )
        p = trace(f, x)
        @test_broken Array(only(C.leaves(outputs(execute!(prepare(p, CuArray(x))))))) == interpret(p, x)
    end
end
