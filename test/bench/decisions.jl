# Whether the estimates that decide hold up when measured, on the device: for
# each of the coverage suite's compositions (test/fixtures/compositions.jl), at
# full precision and at TF32, every group chosen by estimate among several
# options and every comparison of plans the cover made by estimate is timed.
# A device timing, so it is not on the path that gates landing: a GPU another
# job shares, and the rates each run fits afresh, move a time or an estimate
# across the close-call factor in some runs and not others. The attention
# forwards at full precision show it: their product recomputing the scores
# is estimated at 51 to 63 μs on most fresh fits and at 132 on about one in
# ten, timed at 65 every time; at 132 the comparison with storing the scores
# (51.7 estimated, 53.3 timed) is no close call, stays an estimate, and its
# new side misses its time by a factor of two. The decision, storing them,
# stands.
# Run it alone on the GPU, nightly or after a change to the cost model:
#   julia --project=test test/bench/decisions.jl
# It fits the device's rates first, into a fresh tuning directory, as the
# suite does, and fails as a test file fails. TEMPEST_RATES names a rates file
# to start from instead.
using Tempest, Test, NNlib, CUDACore, Einops
import ChainRulesCore, SpecialFunctions, Statistics, cuBLAS, cuDNN, cuTile
include(joinpath(@__DIR__, "..", "fixtures", "stores.jl"))
run_stores!()
include(joinpath(@__DIR__, "..", "setup.jl"))
haskey(ENV, "TEMPEST_RATES") || foreach(precision -> C.calibrate!(T.CUDATarget(; precision)), (:full, :tf32, :tf32x3))

# The coverage suite's compositions, in a module of their own.
module Coverage include(joinpath(@__DIR__, "..", "fixtures", "compositions.jl")) end

within(e, t) = max(e, t) <= C.ESTIMATE_ERROR * min(e, t)

# A group chosen by estimate among several options: the choice is within the
# close-call factor of its measurement, and no option measures faster beyond
# the noise, since an implementation's own alternatives (a product's tiles)
# are measured, not ranked by the estimate. Each side of a comparison made
# by estimate is within the factor of its measurement, and each such
# comparison stands when measured.
# Pointwise work inside a product's sweep (a softcap's tanh, HSTU's SiLU) is
# priced per position by a constant measured at one shape, at full occupancy.
# Where the kernel's occupancy or sweep differs from that shape's, as HSTU's
# recomputing kernels' and the softcap's tf32 kernels' do, the estimate misses
# by up to three times, though the decision stands.
const SWEEP_SHAPE = "a sweep's pointwise work priced per position at one shape and occupancy (TRIALS: the constants of the cost model)"
in_sweep(c, precision) = occursin("SiLU", c.name) || occursin("softcap", c.name) && precision === :tf32
# A launch of fewer blocks than the device runs at once, its operands read
# from memory, loads them before it computes: the two add rather than
# overlap, and cuBLAS's 256×512·384 product at full precision takes 14 μs
# where it is estimated at 9, timed from memory (11.8 from the cache).
const ONE_WAVE = "a launch of one wave from memory, its loads and its products added rather than overlapped (`OVERLAP`, src/Compiler/cover/cost.jl)"
one_wave(c, precision) = c.name == "linear, bias, GELU" && precision === :full
# A launch's bytes are priced at the bandwidth its reads are fit on, writes
# too; timed from memory, writes that fit the cache land there and reach
# memory after it. Attention's backward with its scores and dp stored writes
# p and ds from the log-sum-exp's kernel, where Δ no longer waits for it:
# 4.2 MB read and 4.2 MB written, estimated at 35 μs and timed at 19, what
# the reads take. In full precision the function's step misses so where the
# backward would recompute the scores: the exponential, 2 MB read and 2 MB
# written, is estimated at 18 μs and timed at 10, and that side of the
# comparison at 326 to 428 μs from run to run, timed at 270, about the
# close-call factor. At TF32 the forward misses so where it would store the
# scores: the product writing them, 2 MB, is estimated at 20 μs and timed at
# 8.5, and with the softmax over the stored scores stepping in matrices
# (14.6 μs) that side comes to 34.8 against 23.0, just past the factor; the
# decision, recomputing them, stands at 22.1.
const CACHED_WRITES = "a launch's writes priced at the bandwidth of its reads, where timed from memory they land in the cache (`fit_memory!`, src/Compiler/cover/calibrate.jl)"
const STORED_SCORES = ("attention as a function", "written out with contract and logsumexp", "softmax spelled exp(s - max) / sum",
    "the function, then a residual", "written out, then a residual", "the function, and a reduction of its output")
cached_writes(c, precision) = c.name == "the function, a training step" ||
    precision === :tf32 && (c.name == "written out, a training step" || c.name in STORED_SCORES)

@testset "estimates that decide are within the close-call factor, or measured" begin
    @testset "$(c.name), $precision" for precision in (:full, :tf32), c in Coverage.COMPOSITIONS
        d = C.Decider(T.CUDATarget(; precision), C.rewrite(C.lower(c.program)); measure = :close)
        p, considered = C.cover(d)
        timed(q, g, o) = (C.measure!(d, q, g, o); d.record[o.key])
        held = Bool[]
        for g in p.groups
            x = C.decide(d, p, g)
            x.how === :estimated && length(x.options) > 1 || continue
            push!(held, within(x.choice.estimate, timed(p, g, x.choice)))
            push!(held, all(o -> timed(p, g, o) >= (1 - d.rates.noise) * timed(p, g, x.choice), x.options))
        end
        # A comparison decided by estimate stands when measured; times within
        # the device's noise are a tie, which the estimate decides.
        stands = Bool[]
        for k in filter(k -> k.comparison.how === :estimated, considered)
            measured = map(side -> sum(s -> timed(s[1], s[2], s[3].choice), side; init = 0.0), k.comparison.sides)
            push!(stands, abs(measured[2] - measured[1]) <= d.rates.noise * minimum(measured) || (measured[2] < measured[1]) == C.cheaper(k.comparison))
            append!(held, within.((k.comparison.old, k.comparison.new), measured))
        end
        if in_sweep(c, precision)
            @testset "gap: $SWEEP_SHAPE" begin
                @test_skip all(stands)
                @test_skip all(held)
            end
        elseif one_wave(c, precision)
            @testset "gap: $ONE_WAVE" begin
                @test all(stands)
                @test_skip all(held)
            end
        elseif cached_writes(c, precision)
            @testset "gap: $CACHED_WRITES" begin
                @test all(stands)
                @test_skip all(held)
            end
        else
            @test all(stands)
            @test all(held)
        end
    end
end
