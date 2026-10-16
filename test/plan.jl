# Plans compute what programs compute: every program and its differentiated
# form, planned and run kernel by kernel on the CPU, agrees with the program
# interpreter.
using Test
include("cases.jl")

agree(a, b) = all(map((x, y) -> isapprox(x, y; rtol = 1.0e-10, atol = 1.0e-12), a, b))

@testset "plans" begin
    @testset "$name" for (name, f, args) in PROGRAMS
        for p in (trace(f, args...), differentiate(trace(f, args...)))
            planned = T.plan(T.lower(p))
            @test agree(T.execute(planned, args...), filter(!isnothing, T.leaves(interpret(p, args...))))
        end
    end
end
