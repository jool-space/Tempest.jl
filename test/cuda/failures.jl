# Functions cuTile cannot lower: `exp10` or `log10` (Base's branch on a
# comparison of its argument). The generator is asked before it claims
# (`lowers`), compiling a trial kernel once per machine, so no claim of a group
# that applies one fails to build. The floor runs the whole group where that
# is cheapest, its fold split among the device's lanes; where the rest of the
# group is worth a claim, the map is stored apart and the floor computes it
# alone. `explain` prices what runs.

# The floor's kernels of a prepared program.
floors(ex) = [k for d in ex.kernels if d isa C.FloorKernels for k in d.kernels]

@testset "a fold of a map cuTile cannot lower" begin
    x = 0.1f0 .+ 0.8f0 .* single(abs.(data(1024, 512)) ./ 2)
    @testset "$f" for f in (exp10, log10, expm1)
        p = trace(x -> sum(f.(x)), x)
        ex = prepare(p, CuArray(x))
        @test isapprox(Array(only(C.leaves(outputs(execute!(ex)))))[], interpret(p, x); rtol = 1.0e-4)
        x1 = only(ex.decisions)
        @test isempty(x1.failed)
        f === expm1 && continue
        # A fold of a map is cheapest on the floor whole, split among lanes
        # rather than one thread's.
        @test x1.choice.implementation === nothing
        @test all(k -> k.split !== nothing && k.split.parts * k.split.lanes >= 1024, floors(ex))
        # `explain` gives the floor's estimate.
        text = sprint(explain, ex)
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
    @test all(d -> d === nothing || isempty(d.failed), ex.decisions)
end

# A preparation that builds nothing asks the generator nothing it would compile
# to answer: it claims what it cannot tell it lowers, here a function that
# branches on its argument, which is recorded whole.
@testset "a preparation that builds nothing compiles no trial" begin
    x = single(data(256, 64))
    p = trace(x -> sum((y -> y > 0.5f0 ? y : 2y).(x)), x)
    before = C.cache_counts()
    ex = prepare(p, CuArray(x); build = false)
    @test C.cache_counts() == before
    @test only(ex.decisions).choice.implementation !== nothing
    ex = prepare(p, CuArray(x))
    @test isempty(only(ex.decisions).failed)
    @test isapprox(Array(only(C.leaves(outputs(execute!(ex)))))[], interpret(p, x); rtol = 1.0e-4)
end
