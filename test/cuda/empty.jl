# Programs over arrays of no elements on the device, as in index/empty.jl on
# the CPU: nothing computed ranges over an axis of extent zero, so no
# implementation is asked for a kernel over one, and arrays of none are bound
# and returned as any others.
include("../fixtures/empty.jl")

@testset "a program over no elements gives what Julia does, on the device" begin
    @testset "$name" for (name, f, args) in OF_NONE
        p = trace(f, args...)
        results = C.leaves(outputs(execute!(prepare(p, map(CuArray, args)...))))
        @test all(r -> r isa CuArray, results)
        @test as_julia(map(on_host, results), listed(f(args...)))
    end
end

@testset "a derivative with respect to no elements, on the device" begin
    f(x, y) = sum(abs2, x) + sum(abs2, y) + sum(x * y')
    x, y = nothing32(0), counted32(3)
    p = differentiate(trace(f, x, y))
    results = C.leaves(outputs(execute!(prepare(p, CuArray(x), CuArray(y)))))
    @test all(r -> r isa CuArray, results)
    @test as_julia(map(on_host, results), map(on_host, C.leaves(outputs(execute!(prepare(p, x, y))))))
    @test any(isempty, results)
end
