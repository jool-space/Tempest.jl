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
            @test all(v -> !(v.definition isa T.Opaque) || v.definition.node.op isa T.Scatter, graph.values)
        end
    end
end
