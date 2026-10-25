# The floor, the scalar kernels every target runs: writes where elements land
# at one position, extremes as Base finds them, a BFloat16 scatter summed in
# Float32, and folds recomputed where they are read, on the CPU and on CUDA.
include("../fixtures/programs.jl")

# Writes where elements land at one position, on the CPU's floor: in order,
# the last landing, as the interpreter places them.
@testset "repeated writes on the CPU floor" begin
    @testset "$name" for (name, f, args) in REPEATED_WRITES
        for p in (trace(f, args...), differentiate(trace(f, args...)))
            @test agree(filter(!isnothing, C.leaves(outputs(execute!(prepare(p, args...))))),
                filter(!isnothing, C.leaves(interpret(p, args...))); rtol = 1.0e-10)
        end
    end
end

# Extremes prepared on the CPU find what Base finds, exactly (`EXTREMES` and
# `FOUND` in fixtures/programs.jl).
@testset "extremes on the CPU" begin
    @testset "$(eltype(x)) $i" for (i, x) in enumerate(EXTREMES), f in FOUND
        @test isequal(found(outputs(execute!(prepare(trace(f, x), x)))), found(f(x)))
    end
end

# A BFloat16 scatter on the floor sums in Float32 and rounds once, as the
# reference does: 41 from 4096 gradients of 0.01 at one position, not 4.
@testset "a BFloat16 scatter on the floor" begin
    E, t, g = BFloat16s.BFloat16.(fill(0.5f0, 4, 3)), Int32.(fill(2, 4096)), BFloat16s.BFloat16.(fill(0.01f0, 4, 4096))
    p = differentiate(trace((E, t, g) -> sum(E[:, t] .* g), E, t, g))
    @test outputs(execute!(prepare(p, E, t, g)))[2] == interpret(p, E, t, g)[2]
end

# The floor evaluates recomputed folds too: an implementation that asks for
# them and leaves the kernels to the floor.
struct RecomputeOnTheFloor <: C.Implementation end
C.recomputes_folds(::RecomputeOnTheFloor, target) = true
C.claims(::RecomputeOnTheFloor, target, g) = isempty(C.recomputed_in(g)) ? () : (:floor,)
C.build_claim(::RecomputeOnTheFloor, claim, target, g, buffers) = error("left to the floor")

function with_recomputing_floor(f)
    C.register!(@__MODULE__, RecomputeOnTheFloor())
    try
        f()
    finally
        C.register!(@__MODULE__)
    end
end

attention_cases = [
    ("causal attention, grouped heads", (q, k, v) -> Ops.attention(q, k, v; causal = true),
     (data(4, 4, 9, 2; scale = 0.3), data(4, 2, 9, 2; scale = 0.2), data(4, 2, 9, 2; scale = 0.1))),
    ("attention", (q, k, v) -> Ops.attention(q, k, v), (data(8, 2, 7, 3; scale = 0.3), data(8, 2, 7, 3; scale = 0.2), data(8, 2, 7, 3; scale = 0.1))),
]

@testset "recomputed folds on the CPU floor" begin
    with_recomputing_floor() do
        @testset "$name" for (name, f, args) in attention_cases
            p = trace(f, args...)
            ex = prepare(p, args...)
            @test !isempty(C.recomputed_in(only(ex.plan.groups)))
            @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, args...)])
        end
    end
end

if CUDA_FUNCTIONAL
    @testset "recomputed folds on the CUDA floor" begin
        with_recomputing_floor() do
            @testset "$name" for (name, f, args) in attention_cases
                p = trace(f, args...)
                ex = prepare(p, map(CuArray, args)...; target = T.CUDATarget(@__MODULE__))
                @test all(d -> d isa C.FloorKernels, ex.kernels)
                @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, args...)]; rtol = 1.0e-8)
            end
        end
    end
end
