# The index level computes what the program level computes: every program and
# its differentiated form, lowered and interpreted value by value, agrees with
# the program interpreter, and layout leaves no trace among the values.
using Test
include("cases.jl")

flat(x) = filter(!isnothing, T.leaves(x))
agree(a, b) = all(map((x, y) -> isapprox(x, y; rtol = 1.0e-10, atol = 1.0e-12), a, b))

@testset "index level" begin
    @testset "$name" for (name, f, args) in PROGRAMS
        for p in (trace(f, args...), differentiate(trace(f, args...)))
            graph = T.lower(p)
            @test agree(T.interpret(graph, args...), flat(interpret(p, args...)))
            # Only a scatter by runtime indices is opaque; one by integers and
            # ranges is a select of a read through a formula.
            @test all(v -> !(v.definition isa T.Opaque) || v.definition.node.op isa T.Scatter &&
                           any(i -> i isa T.IndexInput, v.definition.node.op.indices), graph.values)
            # Filling out to a larger shape is a formula, not a value. (A
            # call's argument is materialized for an implementation that
            # claims the call; nothing else reads it.)
            users = T.readers(graph)
            @test !any(k -> graph.values[k].definition isa T.Pointwise && graph.values[k].definition.f === identity &&
                            !isempty(users[k]), eachindex(graph.values))
        end
    end
end

# Simplified divisions and remainders agree with `fld` and `mod` everywhere on
# their domain, and a small term below a common factor drops out.
@testset "index formulas" begin
    d = T.Domain((4, 16, 3))
    e = T.add(T.var(1), T.mul(T.var(2), 4, d), d) # i1 + 4·i2, with i1 < 4
    @test T.fdiv(e, 32, d) == T.fdiv(T.var(2), 8, d)
    @test T.fmod(e, 32, d) == T.add(T.var(1), T.mul(T.fmod(T.var(2), 8, d), 4, d), d)
    points = Iterators.product(0:3, 0:15, 0:2)
    for a in -3:6, b in (0, 2, 4, 8, 12), c in (0, 1, 6), k0 in -5:7, k in (2, 3, 4, 6, 8, 12, 32)
        e = T.add(T.add(T.mul(T.var(1), a, d), T.mul(T.var(2), b, d), d), T.add(T.mul(T.var(3), c, d), k0, d), d)
        q, r = T.fdiv(e, k, d), T.fmod(e, k, d)
        @test all(x -> T.evaluate_index(q, x) == fld(T.evaluate_index(e, x), k) &&
                       T.evaluate_index(r, x) == mod(T.evaluate_index(e, x), k), points)
        # The two recombine to the expression they came from.
        @test T.add(T.mul(q, k, d), r, d) == e
    end
end

# An index out of range is an error in the interpreters.
@testset "gathers out of range" begin
    x = data(8, 6)
    p = trace((x, i) -> x[:, i], x, [1, 2])
    @test_throws BoundsError interpret(p, x, [1, 7])
    @test_throws BoundsError T.interpret(T.lower(p), x, [1, 7])
end
