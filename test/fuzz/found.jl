# Faults the randomized checker (run.jl) found that nothing has closed yet,
# each a seed's case shrunk (repro.jl SEED --shrink) and then by hand, under a
# testset naming what closes it. runtests.jl skips this directory; run it
# alone, under the GPU lock where there is a GPU:
#
#     julia --project=test test/fuzz/found.jl
#
# A fault closed becomes a test of its own beside the code it exercises, and
# leaves this file.
import cuBLAS, cuTile
using Tempest, CUDACore, Test, Random, LinearAlgebra
import Tempest as T, Tempest.Compiler as C

# The one number a case returns, as its plan computes it on the CPU.
prepared(f, args...) = only(C.leaves(C.outputs(C.execute!(C.prepare(C.trace(f, args...), args...)))))
runs(f, args...) = try
    prepared(f, args...)
    true
catch
    false
end

σ(x) = 1 / (1 + exp(-x))

# Seed 106 (loop), as shrunk; seed 234 shrinks to the like, its scan of one
# state read by the other's update.
function scanning_step(a1)
    s1 = a1
    acc = []
    for t in 3:-1:1
        z1 = σ.(s1)
        z2 = cumsum(z1; dims = 1) .* 0.3
        s1 = tanh.(z2)
        push!(acc, s1)
    end
    sum(acc[end])
end

# Seed 1040 (loop, drawn with --big), as shrunk.
function softmax_step(xs, a1, a2, U)
    s1 = a1
    s2 = a2
    for t in 2:4
        x = xs[:, :, t]
        z1 = s1 .* s2
        z2 = (U * x) .+ z1
        z4 = exp.(z2 .- maximum(z2; dims = 1))
        s2 = 0.5 .* s2 .+ 0.5 .* tanh.(z4)
    end
    sum(s2) * 0.3
end

@testset "found by the randomized checker" begin
    # Seeds 106 and 234 (loop), on the CPU: refused by name, "a recurrence
    # whose step scans, scatters or writes is not supported yet", a scan in
    # the cycle of states as TRIALS counts it.
    @testset "a recurrence whose step scans a state prepares" begin
        a = 0.5 .* randn(Xoshiro(106010101), 1, 1)
        @test_broken runs(scanning_step, a) && prepared(scanning_step, a) ≈ scanning_step(a)
    end

    # Seed 1040 (loop), on the CPU and on CUDA: `riding_sums` offers a sum of
    # the gradient that `prepare` then refuses to make ride, "a recurrence
    # with sums riding its steps runs only a step at a time".
    @testset "every sum riding_sums offers rides" begin
        args = (randn(Xoshiro(1040013682), 2, 1, 5), 0.5 .* randn(Xoshiro(1040012903), 1, 1),
            0.5 .* randn(Xoshiro(1040012904), 1, 1), 0.09 .* randn(Xoshiro(1040003422), 1, 2))
        q = C.trace((xs...) -> T.withgradient(softmax_step, xs...), args...)
        sums = C.riding_sums(C.prepare(q, args...).plan)
        rides(s) = try
            C.prepare(q, args...; ride = [s])
            true
        catch
            false
        end
        @test !isempty(sums)
        @test_broken all(rides, sums)
    end

    # Seeds 1052 and 1060 (scan, drawn with --cuda), on CUDA in Float32 only:
    # a scan along an axis of one element, `accumulate(max, …)` as drawn and
    # `cumsum` alike, of a 1×1 or a 3×1 matrix, fails to plan, BoundsError on
    # an empty Vector{Union{}}; the CPU's plan runs it, and so does CUDA's
    # along an axis of three.
    @testset "a scan along an axis of one element prepares on CUDA" begin
        if CUDACore.functional()
            x = Float32.(randn(Xoshiro(1060003517), 1, 1))
            f(x) = sum(abs2, accumulate(max, x; dims = 2)) * 0.1f0
            g(x) = sum(abs2, cumsum(x; dims = 2))
            on_device(f, x) = try
                Array(only(C.leaves(C.outputs(C.execute!(C.prepare(C.trace(f, x), CuArray(x); measure = :none))))))[]
            catch
                nothing
            end
            @test_broken on_device(f, x) ≈ f(x)
            @test_broken on_device(g, x) ≈ g(x)
        end
    end
end
