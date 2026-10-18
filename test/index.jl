# The index level computes what the program level computes: every program and
# its differentiated form, lowered and interpreted value by value, agrees with
# the program interpreter, and layout leaves no trace among the values.
include("fixtures/programs.jl")

flat(x) = filter(!isnothing, C.leaves(x))

@testset "index level" begin
    @testset "$name" for (name, f, args) in PROGRAMS
        for p in (trace(f, args...), differentiate(trace(f, args...)))
            graph = C.lower(p)
            @test agree(C.interpret(graph, args...), flat(interpret(p, args...)))
            # Only a scatter by runtime indices is opaque; one by integers and
            # ranges is a select of a read through a formula.
            @test all(v -> !(v.definition isa C.Opaque) || v.definition.node.op isa C.Scatter &&
                           any(i -> i isa C.IndexInput, v.definition.node.op.indices), graph.values)
            # Filling out to a larger shape is a formula, not a value. (A
            # call's argument is materialized for an implementation that
            # claims the call; nothing else reads it.)
            users = C.readers(graph)
            @test !any(k -> graph.values[k].definition isa C.Pointwise && graph.values[k].definition.f === identity &&
                            !isempty(users[k]), eachindex(graph.values))
        end
    end
end

# Simplified divisions and remainders agree with `fld` and `mod` everywhere on
# their domain, and a small term below a common factor drops out.
@testset "index formulas" begin
    d = C.Domain((4, 16, 3))
    e = C.add(C.var(1), C.mul(C.var(2), 4, d), d) # i1 + 4·i2, with i1 < 4
    @test C.fdiv(e, 32, d) == C.fdiv(C.var(2), 8, d)
    @test C.fmod(e, 32, d) == C.add(C.var(1), C.mul(C.fmod(C.var(2), 8, d), 4, d), d)
    points = Iterators.product(0:3, 0:15, 0:2)
    for a in -3:6, b in (0, 2, 4, 8, 12), c in (0, 1, 6), k0 in -5:7, k in (2, 3, 4, 6, 8, 12, 32)
        e = C.add(C.add(C.mul(C.var(1), a, d), C.mul(C.var(2), b, d), d), C.add(C.mul(C.var(3), c, d), k0, d), d)
        q, r = C.fdiv(e, k, d), C.fmod(e, k, d)
        @test all(x -> C.evaluate_index(q, x) == fld(C.evaluate_index(e, x), k) &&
                       C.evaluate_index(r, x) == mod(C.evaluate_index(e, x), k), points)
        # The two recombine to the expression they came from.
        @test C.add(C.mul(q, k, d), r, d) == e
    end
    # Digits of a number recombine into it, also when nested divisions have
    # composed: (i mod 32) + 32·((i ÷ 32) mod 2) + 64·(i ÷ 64) is i.
    d = C.Domain((128,))
    i = C.var(1)
    digits = [C.fmod(i, 32, d), C.mul(C.fmod(C.fdiv(i, 32, d), 2, d), 32, d), C.mul(C.fdiv(i, 64, d), 64, d)]
    @test foldl((a, b) -> C.add(a, b, d), digits) == i
    @test C.is_affine(foldl((a, b) -> C.add(a, b, d), reverse(digits)))
end

# An index out of range is an error in the interpreters.
@testset "gathers out of range" begin
    x = data(8, 6)
    p = trace((x, i) -> x[:, i], x, [1, 2])
    @test_throws BoundsError interpret(p, x, [1, 7])
    @test_throws BoundsError C.interpret(C.lower(p), x, [1, 7])
end
