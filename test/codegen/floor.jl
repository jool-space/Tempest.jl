# The floor, the scalar kernels every target runs: writes where elements land
# at one position, extremes as Base finds them, a BFloat16 scatter summed in
# Float32, folds recomputed where they are read, and folds split among lanes,
# on the CPU and on CUDA.
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
# them and builds the floor's kernels, the group's reference.
struct RecomputeOnTheFloor <: C.Implementation end
C.recomputes_folds(::RecomputeOnTheFloor, target) = true
C.claims(::RecomputeOnTheFloor, target, g) = isempty(C.recomputed_in(g)) ? () : (:floor,)
C.build_claim(::RecomputeOnTheFloor, claim, target, g, buffers) = g.reference(target, buffers)

attention_cases = [
    ("causal attention, grouped heads", (q, k, v) -> Ops.attention(q, k, v; causal = true),
     (data(4, 4, 9, 2; scale = 0.3), data(4, 2, 9, 2; scale = 0.2), data(4, 2, 9, 2; scale = 0.1))),
    ("attention", (q, k, v) -> Ops.attention(q, k, v), (data(8, 2, 7, 3; scale = 0.3), data(8, 2, 7, 3; scale = 0.2), data(8, 2, 7, 3; scale = 0.1))),
]

@testset "recomputed folds on the CPU floor" begin
    @testset "$name" for (name, f, args) in attention_cases
        p = trace(f, args...)
        ex = prepare(p, args...; target = T.CPUTarget(RecomputeOnTheFloor()))
        @test !isempty(C.recomputed_in(only(ex.plan.groups)))
        @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, args...)])
    end
end

if CUDA_FUNCTIONAL
    @testset "recomputed folds on the CUDA floor" begin
        @testset "$name" for (name, f, args) in attention_cases
            p = trace(f, args...)
            ex = prepare(p, map(CuArray, args)...; target = T.CUDATarget(RecomputeOnTheFloor()))
            @test all(d -> d isa C.FloorKernels, ex.kernels)
            @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, args...)]; rtol = 1.0e-8)
        end
    end
end

# Folds split among lanes, where items are too few to keep a device busy (see
# `Split`): every kind of fold, its parts' states merged, with the outputs
# that read it and its derivative. On the CPU through a target whose floor
# splits each item's positions into three parts, one lane each; on CUDA's
# floor alone, as it splits them.
struct ThreeParts <: C.Target end
C.allocate(::ThreeParts, T, n) = Vector{T}(undef, n)
C.upload(::ThreeParts, x) = x
C.device_name(::ThreeParts) = "cpu"
C.floor_launch(::ThreeParts, k::C.ScalarKernel, args) = C.floor_launch(C.CPUTarget(), k, args)
C.floor_split(::ThreeParts, items, positions, adjacent) = positions > 1 ? C.Split(3, 1) : nothing

# CUDA with no implementation, so that the floor runs every kernel.
floor_alone() = T.CUDATarget(:full, true, ())

split_cases = [
    ("a sum", x -> sum(abs2, x), (data(300, 200),)),
    ("sums down columns", x -> sum(x; dims = 1), (data(300, 200),)),
    ("sums along rows", x -> sum(x; dims = 2), (data(300, 200),)),
    ("a maximum", x -> maximum(x), (data(300, 200),)),
    ("minima along rows", x -> minimum(x; dims = 2), (data(300, 200),)),
    ("products down columns", x -> prod(1 .+ x ./ 100; dims = 1), (data(300, 200),)),
    ("a log-sum-exp", x -> T.logsumexp(x), (data(300, 200),)),
    ("softmax down columns", softmax_columns, (data(300, 200),)),
    ("softmax along rows", softmax_rows, (data(300, 200),)),
    ("layer norm", layernorm, (data(300, 200), data(300; scale = 0.1))),
    ("a fold of the user's", x -> reduce(VARIANCE, x; dims = 1), (data(300, 200),)),
    ("extremes and where they are", x -> (m = findmax(x; dims = 2); (m[1], getindex.(m[2], 2))), (data(300, 200),)),
    ("a Float16 sum", x -> sum(x), (Float16.(data(300, 200)),)),
    ("a sum of a map, and its derivative", x -> sum(expm1.(x ./ 4)), (data(300, 200),)),
    ("RMSNorm's loss", (x, w) -> sum(abs2, rmsnorm(x, w)), (data(300, 200), data(300; scale = 0.1))),
]

split_programs(f, args) = f(args...) isa Number ? (trace(f, args...), differentiate(trace(f, args...))) : (trace(f, args...),)
rtol_of(args) = any(x -> x isa AbstractArray{Float16}, args) ? 1.0e-3 : 1.0e-8

@testset "folds split among lanes on the CPU" begin
    @testset "$name" for (name, f, args) in split_cases
        for p in split_programs(f, args)
            ex = prepare(p, args...; target = ThreeParts())
            @test any(k -> k isa C.FloorKernels && any(s -> s.split !== nothing, k.kernels), ex.kernels)
            @test agree(filter(!isnothing, C.leaves(outputs(execute!(ex)))), filter(!isnothing, C.leaves(interpret(p, args...)));
                rtol = rtol_of(args))
        end
    end
end

if CUDA_FUNCTIONAL
    @testset "folds split among lanes on CUDA" begin
        @testset "$name" for (name, f, args) in split_cases
            for p in split_programs(f, args)
                ex = prepare(p, map(CuArray, args)...; target = floor_alone())
                @test all(d -> d isa C.FloorKernels, ex.kernels)
                @test any(k -> any(s -> s.split !== nothing, k.kernels), ex.kernels)
                @test agree(filter(!isnothing, C.leaves(outputs(execute!(ex)))), filter(!isnothing, C.leaves(interpret(p, args...)));
                    rtol = rtol_of(args))
            end
        end
    end

    # A full reduction is dealt among the device's lanes, its parts many
    # blocks, not one thread; a few items' neighbours in memory side by side.
    @testset "a fold of few items fills the device" begin
        x = data(2048, 512)
        ex = prepare(trace(sum, x), CuArray(x); target = floor_alone())
        s = first(only(ex.kernels).kernels).split
        @test s.parts * s.lanes >= 1024 && s.side == 1
        ex = prepare(trace(x -> sum(x; dims = 2), x), CuArray(x); target = floor_alone())
        @test first(only(ex.kernels).kernels).split.side == 32
        @test isapprox(Array(outputs(execute!(ex))), sum(x; dims = 2); rtol = 1.0e-10)
    end
end
