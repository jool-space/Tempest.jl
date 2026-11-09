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

# Each fault's case covers, in a directory of the run's own.
include(joinpath(@__DIR__, "..", "fixtures", "stores.jl"))
run_stores!()
ENV["TEMPEST_REPLAY"] = "false"

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

# Seed 6452 (loop), as shrunk; seed 6466 shrinks to the like, reversed.
function two_shapes(a1)
    s1 = a1
    r = sum(s1) * 0.1
    for t in 1:3
        z1 = s1 .* s1
        z4 = z1 .* r
        s1 = 0.5 .* s1 .+ 0.5 .* tanh.(z4)
        r = 0.5 * r + 0.01 * sum(abs2, z4)
    end
    r
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
    # Seeds 6452 and 6466 (loop), on the CPU: refused by name, "a recurrence
    # whose states differ in shape or direction is not supported yet", a
    # vector state beside a number, as TRIALS counts it. A recurrence's
    # group is over the first state's domain, each state placed at its
    # identity there (`recurrence_places`), and the CPU's
    # `run_recurrence!`, the floor, the claims, `verify` and the stepped
    # run's copies after the steps size a step by the first state's
    # storage. `step_graph` already slices each state by its own shape.
    # What closes it: `recurrence_places` placing a state of fewer axes
    # along those it shares with the first, the steps' among them, as a
    # fold's result is placed, written by one item of the rest; and every
    # reader of the group sizing each state's step by its own storage.
    @testset "a recurrence whose states differ in shape prepares" begin
        a = 0.5 .* randn(Xoshiro(6452029139), 2)
        @test_broken runs(two_shapes, a) && prepared(two_shapes, a) ≈ two_shapes(a)
    end
end
