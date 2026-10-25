# Prepared programs compute what programs compute, through generated kernels,
# on the CPU (and on CUDA when it is loaded and functional).
include("../fixtures/programs.jl")

function check_prepared(device, expected_target; rtol = 1.0e-10, precision = identity)
    @testset "$name" for (name, f, args) in PROGRAMS
        args = map(precision, args)
        for p in (trace(f, args...), differentiate(trace(f, args...)))
            reference = filter(!isnothing, C.leaves(interpret(p, args...)))
            ex = prepare(p, map(device, args)...)
            @test ex.target == expected_target
            got = filter(!isnothing, C.leaves(outputs(execute!(ex))))
            @test agree(got, reference; rtol)
        end
    end
end

@testset "prepared on the CPU" begin
    check_prepared(identity, T.CPUTarget())
end

if CUDA_FUNCTIONAL
    @testset "prepared on CUDA" begin
        check_prepared(x -> x isa AbstractArray ? CuArray(x) : x, T.CUDATarget())
    end
    @testset "prepared on CUDA with cuTile, Float32" begin
        check_prepared(x -> x isa AbstractArray ? CuArray(x) : x, T.CUDATarget(); rtol = 2.0e-4, precision = single)
    end
end
