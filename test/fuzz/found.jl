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

@testset "found by the randomized checker" begin
    # Seeds 106 and 234 (loop), on the CPU: refused by name, "a recurrence
    # whose step scans, scatters or writes is not supported yet", a scan in
    # the cycle of states as TRIALS counts it. A recurrence's group is a term
    # for each member, which the CPU's `run_recurrence!`, the floor's
    # estimate and `verify` start from, and a scan has no term of an
    # element; `step_graph` copies only maps and folds. What closes it: a
    # recurrence group that runs only a step at a time, its scan a member
    # without a term; `recurrence_places` placing a scan, every axis its
    # own; `step_graph` copying the scan with the steps' axis dropped; and
    # the cover offering it no claim and no floor, the CPU running its
    # step's plan as the stepped run does.
    @testset "a recurrence whose step scans a state prepares" begin
        a = 0.5 .* randn(Xoshiro(106010101), 1, 1)
        @test_broken runs(scanning_step, a) && prepared(scanning_step, a) ≈ scanning_step(a)
    end
end
