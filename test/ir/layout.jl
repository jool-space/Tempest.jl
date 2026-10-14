include("../fixtures/layout.jl")

@testset "Layout composition and slice pullbacks" begin
    check_layout_cases(CPUTarget())
    x = reshape(collect(1.0:24.0), 3, 4, 2)
    ex = prepare(x -> sum(reshape(view(x, 1:2, :, :), 2, 8)), x)
    # The legal reshape/view commute becomes one affine read, without a copy.
    primal = ex.storage.program.primal
    for (id, n) in enumerate(primal.nodes)
        if !(Tempest.is_layout(n))
            continue
        end
        @test !haskey(ex.storage.slots, id)
        @test ex.storage.layouts[id][2] isa Tempest.AffineMap
    end
    @test occursin("(alias)", sprint(explain, ex))
    # The gather is a fixed snapshot, not a mutable captured host input.
    inds = [3, 1, 3]
    f(x) = sum(view(x, inds, :, :))
    ex = prepare(f, x)
    expected = f(x)
    inds .= 2
    @test execute!(ex)[1][] == expected
    for f in (x -> sum(view(x, 0:2, :, :)), x -> sum(x[:, 1:5, :]))
        @test_throws BoundsError trace(f, x)
    end
    @test_throws ArgumentError trace(x -> sum(view(x, [true, false, true], :, :)), x)
    @test_throws ArgumentError trace(x -> sum(permutedims(x, (1, 1, 3))), x)
    @test_throws ArgumentError trace(x -> sum(reshape(view(x, 1:0, :, :), 0, :)), x)
end
