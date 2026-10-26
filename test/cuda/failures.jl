# Claims that fail to build: cuTile cannot lower `exp10` or `log10` (Base's
# branch on a comparison of its argument), so a group that applies one is
# planned again without the claim. The floor runs the whole group where that
# is cheapest, its fold split among the device's lanes; where the rest of the
# group is worth a claim, the map is stored apart and the floor computes it
# alone. `explain` names the claim that failed, and prices what runs.

# The floor's kernels of a prepared program.
floors(ex) = [k for d in ex.kernels if d isa C.FloorKernels for k in d.kernels]

@testset "a fold of a map cuTile cannot lower" begin
    x = 0.1f0 .+ 0.8f0 .* single(abs.(data(1024, 512)) ./ 2)
    @testset "$f" for f in (exp10, log10, expm1)
        p = trace(x -> sum(f.(x)), x)
        ex = prepare(p, CuArray(x))
        @test isapprox(Array(only(C.leaves(outputs(execute!(ex)))))[], interpret(p, x); rtol = 1.0e-4)
        f === expm1 && continue
        # A fold of a map is cheapest on the floor whole, split among lanes
        # rather than one thread's.
        x1 = only(ex.decisions)
        @test x1.choice.implementation === nothing && !isempty(x1.failed)
        @test all(k -> k.split !== nothing && k.split.parts * k.split.lanes >= 1024, floors(ex))
        # `explain` says the claim failed, and gives the floor's estimate.
        text = sprint(explain, ex)
        @test occursin("a claim of cuTileGenerator failed to build", text)
        @test occursin(string("(estimated ", round(1.0e6 * x1.choice.estimate; sigdigits = 3), " μs"), text)
        @test x1.choice.estimate < 1.0e-3
    end
end

@testset "the derivative of a fold of such a map" begin
    x = single(data(1024, 512; scale = 0.3)) ./ 4
    p = differentiate(trace(x -> sum(exp10.(x)), x))
    ex = prepare(p, CuArray(x))
    @test agree(filter(!isnothing, C.leaves(outputs(execute!(ex)))), filter(!isnothing, C.leaves(interpret(p, x))); rtol = 1.0e-4)
    # No fold is left to one thread.
    @test all(k -> k.split !== nothing || k.items > 1, floors(ex))
end

@testset "such a map stored apart from a product" begin
    w, x = single(data(1024, 1024; scale = 0.01)), single(data(1024, 1024; scale = 0.3)) ./ 4
    p = trace((w, x) -> w * log10.(1 .+ x .^ 2), w, x)
    ex = prepare(p, CuArray(w), CuArray(x))
    @test isapprox(Array(only(C.leaves(outputs(execute!(ex))))), interpret(p, w, x); rtol = 1.0e-3)
    # The map is the floor's, alone, and a package computes the product.
    stored = [c.proposal for c in ex.considered if c.accepted && c.proposal isa C.Store]
    @test length(stored) == 1
    @test any(d -> d isa C.FloorKernels, ex.kernels) && any(d -> !(d isa C.FloorKernels), ex.kernels)
    @test occursin("is stored apart", sprint(explain, ex))
    @test occursin("a claim of cuTileGenerator failed to build", sprint(explain, ex))
end
