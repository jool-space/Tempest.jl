# For one commit: the cover that prices proposals alike once and takes them
# together chooses what the cover that tries each alone chose, on every
# program and composition, and compares every proposal alike.
include("fixtures/programs.jl")
module Coverage include(joinpath(@__DIR__, "fixtures", "compositions.jl")) end

stack_layer(x, (w, b)) = tanh.(w * x .+ b)
stack_loss(x, layers) = sum(abs2, foldl(stack_layer, layers; init = x))

if CUDA_FUNCTIONAL
    @testset "proposals alike, taken together, agree with each tried alone" begin
        seen(cs) = sort([string(c.proposal, " ", c.accepted, " ", c.comparison.old, " ", c.comparison.new) for c in cs])
        programs = Any[c.program for c in Coverage.COMPOSITIONS]
        for (_, f, args) in PROGRAMS
            p = trace(f, map(single, args)...)
            push!(programs, p, differentiate(p))
        end
        for L in (1, 3, 8)
            layers = Tuple((single(data(64, 64; scale = 0.1k)), single(data(64; scale = 0.2k))) for k in 1:L)
            push!(programs, differentiate(trace(stack_loss, single(data(64, 16)), layers)))
        end
        for p in programs, target in (T.CUDATarget(), T.CUDATarget(; precision = :tf32))
            graph = C.rewrite(C.lower(p))
            a = C.whole_cover(C.Decider(target, graph; measure = :none))
            b = C.cover(C.Decider(target, graph; measure = :none))
            @test sprint(explain, a[1]) == sprint(explain, b[1]) && seen(a[2]) == seen(b[2])
        end
    end
end
