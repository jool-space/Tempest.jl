# Arrays of no elements. A value of none is the one array of its shape and
# type, a constant, and a fold over none what it starts from, so that nothing
# a lowered program computes ranges over an axis of extent zero, and no kernel
# is asked for one; the program gives what Julia gives, an extreme of none
# Base's refusal among it. On the device in cuda/empty.jl.
include("../fixtures/empty.jl")

# The values of `graph` computed over an axis of extent zero, a fold's
# iteration among them.
function computed_over_none(graph)
    over(v) = 0 in v.dims || v.definition isa Union{C.Reduction, C.Keyed} && 0 in v.definition.iteration
    [v for v in graph.values if over(v) && !(v.definition isa Union{C.Input, C.ConstantArray})]
end

@testset "a program over no elements gives what Julia does" begin
    @testset "$name" for (name, f, args) in OF_NONE
        p = trace(f, args...)
        @test isempty(computed_over_none(C.lower(p)))
        @test as_julia(map(on_host, C.leaves(outputs(execute!(prepare(p, args...))))), listed(f(args...)))
    end
end

@testset "a derivative with respect to no elements" begin
    f(x, y) = sum(abs2, x) + sum(abs2, y) + sum(x * y')
    x, y = nothing32(0), counted32(3)
    r = T.withgradient(f, x, y)
    @test r.val == f(x, y)
    @test r.grad[1] isa Vector{Float32} && isempty(r.grad[1]) && r.grad[2] == 2 .* y
end

@testset "an extreme of none is Base's" begin
    # Refused without `init`, along an axis of none as over all, and where it
    # is found; an axis of some of an array of none is reduced to none.
    refused = [
        (x -> maximum(x; dims = 1), nothing32(0, 3)),
        (x -> minimum(x), nothing32(0)),
        (x -> minimum(abs, x), nothing32(0)),
        (x -> findmax(x; dims = 1), nothing32(0, 3)),
        (x -> argmin(x), nothing32(0)),
    ]
    for (f, x) in refused
        @test_throws ArgumentError f(x)
        @test_throws ArgumentError trace(f, x)
    end
    x = nothing32(0, 3)
    p = trace(x -> maximum(x; dims = 2), x)
    @test only(C.leaves(outputs(execute!(prepare(p, x))))) == maximum(x; dims = 2)
end
