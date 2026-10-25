# The index level computes what the program level computes: every program and
# its differentiated form, lowered and interpreted value by value, agrees with
# the program interpreter, and layout leaves no trace among the values.
include("../fixtures/programs.jl")

flat(x) = filter(!isnothing, C.leaves(x))

@testset "index level" begin
    @testset "$name" for (name, f, args) in [PROGRAMS; REPEATED_WRITES]
        for p in (trace(f, args...), differentiate(trace(f, args...)))
            graph = C.lower(p)
            @test agree(C.interpret(graph, args...), flat(interpret(p, args...)))
            # A scatter by index arrays is a sum by key, and a write an
            # overwrite by key into the contents written over; by integers and
            # ranges, either is a select of a read through a formula.
            @test all(graph.values) do v
                d = v.definition
                !(d isa C.Keyed) || any(e -> !isempty(IR.indirect_values(e)), d.key) &&
                    (d.fold isa IR.SumFold && d.into === nothing || d.fold isa IR.Overwrite && d.into !== nothing)
            end
            # Filling out to a larger shape is a formula, not a value. (A
            # call's argument is materialized for an implementation that
            # claims the call; nothing else reads it.)
            users = C.readers(graph)
            @test !any(k -> graph.values[k].definition isa C.Pointwise && graph.values[k].definition.f === identity &&
                            !isempty(users[k]), eachindex(graph.values))
        end
    end
end

# An index out of range is an error in the interpreters.
@testset "gathers out of range" begin
    x = data(8, 6)
    p = trace((x, i) -> x[:, i], x, [1, 2])
    @test_throws BoundsError interpret(p, x, [1, 7])
    @test_throws BoundsError C.interpret(C.lower(p), x, [1, 7])
end
