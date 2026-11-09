# What tasks on threads of their own do at once: two programs whose plans
# take a library's call, called round after round, each result as the
# program gives it alone (`cuDNN`, `cuBLAS`); and Tempest's own (`Tempest`):
# timings, a compiled program's calls while its search changes its plan, and
# preparations. The test runner's workers have one thread, on which tasks
# take turns and no two run at once, so two tasks inside a worker prove
# nothing: the shared handles passed so. A test runs a case in a child
# process with threads, `julia --threads=3 at_once.jl CASE`, through
# `at_once` (see setup.jl). It prints its failures.
using Tempest, Test, NNlib, CUDACore
import cuBLAS, cuDNN, cuTile
include(joinpath(@__DIR__, "..", "setup.jl"))

# The results of `rounds` rounds that differ from each program's alone: in
# each, both programs called `calls` times, each from a task of its own and
# on its two sets of inputs in turn, the tasks released together. A result
# is compared leaf by leaf, on the host, in Float32.
host(y) = [Float32.(on_host(x)) for x in C.leaves(y)]
function wrong_at_once(cs, inputs; rounds, calls, rtol)
    alone = [[host(cs[p](inputs[p][s]...)) for s in 1:2] for p in 1:2]
    sum(1:rounds) do _
        go = Base.Event()
        tasks = map(1:2) do p
            Threads.@spawn begin
                wait(go)
                ys = [(mod1(i, 2), cs[p](inputs[p][mod1(i, 2)]...)) for i in 1:calls]
                count(((s, y),) -> !all(map((a, b) -> isapprox(a, b; rtol), host(y), alone[p][s])), ys)
            end
        end
        notify(go)
        sum(fetch, tasks)
    end
end

@test Threads.nthreads() > 1
library = only(ARGS)

if library == "cuDNN"
    include("sdpa.jl")
    @testset "cuDNN attention of two programs at once" begin
        cuDNNExt = Base.get_extension(Tempest, :cuDNNExt)
        half(dims...; scale) = CuArray(BFloat16.(single(data(dims...; scale))))
        operands(p, s) = (half(64, 8, 1024, 2; scale = 0.01 * (p + 2s)), half(64, 2, 1024, 2; scale = 0.02 * (p + 2s)),
            half(64, 2, 1024, 2; scale = 0.03 * (p + 2s)))
        inputs = [[operands(p, s) for s in 1:2] for p in 1:2]
        cs = [compile((q, k, v) -> sdpa(q, k, v; causal = true), inputs[p][1]...; target = T.CUDATarget(T.cuDNNAttention()), capture = false)
            for p in 1:2]
        @test all(c -> count(l -> l isa cuDNNExt.Attention, c.executable.launches) == 1, cs)
        @test wrong_at_once(cs, inputs; rounds = 20, calls = 10, rtol = 1.0e-2) == 0
    end
end

# A product of 64 by 64 summing 16,384 or 131,072, one cuBLAS call, and its
# tanh in a kernel after it, uncaptured and captured. cuBLAS splits the
# longer sum, writing 1.5 MB of the workspace; with a handle per task and
# one workspace per context, 780 of 4000 such results were wrong on GB10,
# silently.
if library == "cuBLAS"
    @testset "cuBLAS products of two programs at once, summing $k, $(capture ? "captured" : "uncaptured")" for k in (1 << 14, 1 << 17),
            capture in (false, true)
        cuBLASExt = Base.get_extension(Tempest, :cuBLASExt)
        operands(p, s) = (CuArray(reshape(Float32.(0.01 .* sin.(p .+ s .* (1:(64k)))), 64, k)), CuArray(reshape(Float32.(cos.(3p .+ s .* (1:(64k)))), k, 64)))
        inputs = [[operands(p, s) for s in 1:2] for p in 1:2]
        cs = [compile((a, b) -> tanh.(a * b), inputs[p][1]...; target = T.CUDATarget(T.cuBLASProducts()), capture) for p in 1:2]
        @test all(c -> count(l -> l isa cuBLASExt.Product, c.executable.launches) == 1, cs)
        @test wrong_at_once(cs, inputs; rounds = 100, calls = 20, rtol = 1.0e-3) == 0
    end
end

if library == "Tempest"
    include("searching.jl")
    @testset "timings one at a time, from threads" begin
        # Each timing waits for the one before, whichever thread asks.
        ran = Threads.Atomic{Int}(0)
        foreach(wait, [Threads.@spawn(C.timed(Clock(), () -> Threads.atomic_add!(ran, 1))) for _ in 1:8])
        @test ran[] == 8 && TIMING_MOST[] == 1
    end

    @testset "a program called from two threads while its search changes its plan" begin
        # The calls run one at a time, under the program's lock, though they
        # share its buffers; the plan changes between two of them, while the
        # other waits. Each result is the program's alone on its inputs: the
        # LSTM's loss and gradient, on two sets of inputs in turn.
        gradient(p, xs, h0) = Tempest.withgradient(q -> sum(h -> sum(abs2, h), lstm(q, xs, h0)), p)
        inputs = [(map(w -> k .* w, lstm_weights()), data(2, 5; scale = 0.5k), data(3; scale = 0.1k)) for k in 1:2]
        c = compile(gradient, inputs[1]...)
        s = searching(c, other_plan(c, inputs[1]...))
        search = s.search
        wrong = 0
        while !C.ended(search)
            wrong += wrong_at_once([s, s], [inputs, inputs]; rounds = 1, calls = 10, rtol = 1.0e-5)
        end
        wrong += wrong_at_once([s, s], [inputs, inputs]; rounds = 5, calls = 10, rtol = 1.0e-5)
        @test wrong == 0 && !istaskfailed(search.task)
        @test s.search === nothing && length(s.compiled.executable.kernels) != length(c.executable.kernels)
    end

    @testset "preparations on two threads at once" begin
        # Each plans as it would alone, though both intern terms and match the
        # library as they plan: the tables are each preparation's own, and
        # what the extensions fill as they go is locked. The LSTM's step and
        # a layer's gradient, at once, three times over.
        xs, h0 = data(2, 5; scale = 0.5), data(3; scale = 0.1)
        p, m, v = lstm_weights(), map(zero, lstm_weights()), map(zero, lstm_weights())
        w, b, x = data(6, 4; scale = 0.3), data(6), data(4, 3; scale = 0.5)
        layer(w, b) = sum(abs2, tanh.(w * x .+ b))
        step = trace(adam!, p, m, v, xs, h0)
        gradient = trace((w, b) -> Tempest.withgradient(layer, w, b), w, b)
        alone = (sprint(explain, prepare(step, p, m, v, xs, h0)), sprint(explain, prepare(gradient, w, b)))
        for _ in 1:3
            tasks = (Threads.@spawn(prepare(step, p, m, v, xs, h0)), Threads.@spawn(prepare(gradient, w, b)))
            @test map(t -> sprint(explain, fetch(t)), tasks) == alone
        end
        # The tables only spare a preparation time: emptied again and again
        # from another thread as it plans, its plan is the same. Were a
        # term's identity to matter, as it would for a term holding a vector,
        # whose `===` is the vector's identity rather than its contents',
        # this would plan otherwise or fail.
        s = C.setup(step; target = T.CPUTarget())
        done = Threads.Atomic{Bool}(false)
        emptying = Threads.@spawn while !done[]
            foreach(((lock, table),) -> @lock(lock, empty!(table)), s.tables.interned.tables)
            @lock C.MATCHES_LOCK empty!(s.tables.matches)
            yield()
        end
        emptied = sprint(explain, C.prepared(s, C.decide!(s), p, m, v, xs, h0))
        done[] = true
        wait(emptying)
        @test emptied == alone[1]
    end

    CUDA_FUNCTIONAL && @testset "a program called from one thread while its search runs on another, on CUDA" begin
        # A program with no matrix product, whose plans share no library
        # handle between the search and the calls. The calls take the
        # search's timings, and the device is synchronized after each: nothing
        # the search does beside them forbids it, as a capture would.
        w, x = single(data(64, 32; scale = 0.3)), single(data(64, 32; scale = 0.5))
        dw, dx = CuArray(w), CuArray(x)
        f = (w, x) -> (sum(abs2, tanh.(w .* x .+ 1.0f0); dims = 1), tanh.(w .* x .+ 1.0f0))
        agrees((value, y)) = isapprox(Array(value), f(w, x)[1]; rtol = 1.0e-4) && isapprox(Array(y), f(w, x)[2]; rtol = 1.0e-4)
        s = compile(f, dw, dx; search = :after, measure = :all)
        search = s.search
        calls = Threads.@spawn begin
            results = Bool[]
            while !C.ended(search) || isempty(results)
                push!(results, agrees(s(dw, dx)))
                CUDACore.device_synchronize()
            end
            results
        end
        @test all(fetch(calls)) && !istaskfailed(search.task)
        @test wait(s) === s && agrees(s(dw, dx))
    end
end
