# Each transform is also applied to ordinary Julia arrays and LinearIndices.
# Weighting by output position exposes incorrect reorderings that sum(abs2, x)
# alone would miss. Gathers include repetitions to exercise additive pullbacks.
const layout_transforms = (
    x -> view(x, 1:2, :, :),
    x -> x[Base.OneTo(2), :, :],
    x -> view(x, 3:-1:1, 1:2:4, :),
    x -> reshape(view(x, 1:2, :, :), 2, 8),
    x -> reshape(view(x, 1:2, :, :), 4, 4),
    x -> reshape(view(x, 1:2, :, :), :),
    x -> view(reshape(x, 6, 4), 2:2:6, 2:4),
    x -> reshape(view(reshape(x, 6, 4), 2:2:6, 2:4), 1, 9),
    x -> permutedims(view(x, 1:2, :, :), (3, 1, 2)),
    x -> PermutedDimsArray(view(x, 1:2, :, :), (2, 3, 1)),
    x -> view(permutedims(x, (3, 1, 2)), :, 2, 2:4),
    x -> view(reshape(permutedims(x, (3, 1, 2)), 4, 6), 2:4, 2:2:6),
    x -> view(view(x, [3, 1, 3], :, :), [3, 1, 1], 2:4, :),
    x -> reshape(permutedims(view(x, [3, 1, 3], :, :), (2, 3, 1)), 8, 3),
    x -> view(x, 2, 2:4, 1),
    x -> view(x, CartesianIndex(2, 3), :),
    x -> view(x, 3:2:21),
    x -> x[[8, 1, 8, 24, 1]],
    x -> view(x, 2, 3, 1),
    x -> view(x, :, :, :, 1),
    x -> permutedims(permutedims(x, (3, 1, 2)), (2, 3, 1)),
)

function check_layout_cases(target; atol = 2e-7, rtol = 2e-6, gpu = false)
    T = gpu ? Float32 : Float64
    x = reshape(T.(range(-0.6, 0.7; length = 24)), 3, 4, 2)
    for transform in layout_transforms
        y = transform(x)
        w = T.(range(0.5; step = 0.03, length = length(y)))
        f = (x, w) -> let y=transform(x)
            sum(abs2, y .* reshape(w, size(y)))
        end
        reference = value_and_gradient(trace(f, x, w), x, w)
        if !gpu
            for (i, g) in enumerate(reference[2])
                @test g ≈ finite_difference(f, (x, w), i) atol=atol rtol=rtol
            end
        end
        for reuse in (true, false)
            ex = compare_prepared(f, x, w; target, reuse, atol, rtol)
            @test Array(outputs(ex)[1])[] ≈ f(x, w) atol=atol rtol=rtol
            if gpu
                capture!(ex)
                bind!(ex, 2x, w)
                got = replay!(ex)
                ref = value_and_gradient(ex.storage.program, 2x, w)
                @test Array(got[1])[] ≈ f(2x, w) atol=atol rtol=rtol
                @test Array(got[2]) ≈ ref[2][1] atol=atol rtol=rtol
            end
        end
        # Check the forward mapping directly against Julia's indexing semantics.
        p = trace(x -> sum(transform(x)), x)
        storage = Tempest.plan(differentiate(p))
        id = only(p.nodes[p.output].inputs)
        root, mapping = storage.layouts[id]
        @test root == only(p.arguments)
        expected = transform(reshape(collect(1:24), size(x)))
        @test [Tempest.map_index(mapping, i) for i = 1:length(expected)] == vec(collect(expected))
    end
    for f in (
        x -> sum(view(x, 1:0, :, :)),
        x -> sum(reshape(view(x, 1:0, :, :), 0, 8)),
        x -> sum(sum(view(x, 1:0, :, :); dims = 1)),
        x -> sum(sum(view(x, 1:0, :, :); dims = 2)),
        x -> sum(view(x, Int[], :, :)),
        x -> sum(abs2, sum(view(x, :, :, 1); dims = 1)),
        x -> sum(abs2, sum(view(x, :, :, 1) .+ view(x, 1, 1, 1)[]; dims = 1)),
        x -> sum(abs2, view(x, 1:2, :, :)) + sum(abs2, view(x, 2:3, :, :)),
    )
        ex = compare_prepared(f, x; target, atol, rtol)
        @test Array(outputs(ex)[1])[] ≈ f(x) atol=atol rtol=rtol
        if gpu
            capture!(ex)
            bind!(ex, 2x)
            @test Array(replay!(ex)[1])[] ≈ f(2x) atol=atol rtol=rtol
        else
            @test Array(outputs(ex)[2]) ≈ finite_difference(f, (x,), 1) atol=atol rtol=rtol
        end
    end
    # A mapped primal can itself be a returned gradient. Materialize at that
    # boundary and keep its storage alive through subsequent backward steps.
    w = ones(T, 2, 4, 2)
    returned = (x, w) -> sum(view(sin.(x), 1:2, :, :) .* w)
    for reuse in (true, false)
        ex = compare_prepared(returned, x, w; target, reuse, atol, rtol)
        @test Array(outputs(ex)[3]) ≈ view(sin.(x), 1:2, :, :) atol=atol rtol=rtol
        @test haskey(ex.storage.slots, ex.storage.program.results[3])
        if gpu
            capture!(ex)
            bind!(ex, 2x, w)
            replay!(ex)
            @test Array(outputs(ex)[3]) ≈ view(sin.(2x), 1:2, :, :) atol=atol rtol=rtol
        end
    end
    # A scalar read from a larger-than-one root must still broadcast as scalar
    # when a cuTile column reduction consumes a different dense array.
    b = T[0.4, 0.7]
    scalar_column = (x, b) -> sum(abs2, sum(view(x, :, :, 1) .+ view(b, 1)[]; dims = 1))
    ex = compare_prepared(scalar_column, x, b; target, atol, rtol)
    @test Array(outputs(ex)[1])[] ≈ scalar_column(x, b) atol=atol rtol=rtol
    if gpu
        capture!(ex)
        bind!(ex, 2x, 2b)
        @test Array(replay!(ex)[1])[] ≈ scalar_column(2x, 2b) atol=atol rtol=rtol
    end
    # General mapped matrices are explicitly copied before BLAS. Both operands
    # and their reverse paths must preserve logical order through those copies.
    a = reshape(T.(range(-0.3, 0.5; length = 24)), 6, 4)
    b = reshape(T.(range(-0.2, 0.4; length = 12)), 4, 3)
    f = (a, b) -> sum(abs2, view(a, 1:2:5, :) * b)
    for reuse in (true, false)
        ex = compare_prepared(f, a, b; target, reuse, atol, rtol)
        @test any(
            id -> Tempest.is_layout(ex.storage.program.nodes[id]),
            keys(ex.storage.slots),
        )
        if gpu
            capture!(ex)
            bind!(ex, 2a, b)
            @test Array(replay!(ex)[1])[] ≈ f(2a, b) atol=atol rtol=rtol
        else
            @test Array(outputs(ex)[2]) ≈ finite_difference(f, (a, b), 1) atol=atol rtol=rtol
        end
    end
end

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
