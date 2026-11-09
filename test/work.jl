# The compiler's own work, counted and marked (src/Compiler/work.jl): off, a
# count, a mark or an event costs nothing and allocates nothing; tallied, a
# preparation's counts are the same each time and its phases are named, a
# phase nested in itself counted once; an observer hears each phase begin
# and end, from every task, and what it throws does not take a preparation
# down.

# A gradient: a fold, the maps that read it, and their gradients.
tallied_program(x) = differentiate(trace(x -> sum(abs2, tanh.(x) .- sum(x; dims = 1) ./ size(x, 1)), x))

# A stack of `n` layers' gradients, each layer scaled by a constant of its
# own: no two are alike, so that on the CUDA target the cover tries a
# proposal in each, and compares those an implementation claims.
function stacked(n)
    ws = ntuple(k -> single(data(16, 16; scale = 1 + k / 8)) ./ 8, n)
    x = single(data(16, 8))
    function loss(ws, x)
        h = x
        for (k, w) in enumerate(ws)
            h = tanh.(w * h) .* Float32(1 + k / 64)
        end
        sum(abs2, h)
    end
    trace((ws, x) -> T.withgradient(v -> loss(v, x), ws), ws, x), (ws, x)
end

@testset "off, the work is neither counted nor heard" begin
    @test !C.TALLYING[] && C.OBSERVER[] === nothing
    count_off() = (C.tally!(:plans); C.tally!(:terms, 3); nothing)
    mark_off() = (C.lap!(:x, C.mark()); C.observe(:kernel, 1); C.phase(() -> nothing, :x))
    count_off(); mark_off()
    @test (@allocated count_off()) == 0
    @test (@allocated mark_off()) == 0
    foreach(c -> c[] = 0, values(C.COUNTS))
    x = data(16, 8)
    prepare(tallied_program(x), x)
    @test all(c -> c[] == 0, values(C.COUNTS))
end

@testset "a preparation's work, tallied" begin
    x = data(16, 8)
    p = tallied_program(x)
    ex, counts, phases = C.tallied(() -> prepare(p, x; build = false))
    names = first.(phases)
    @test issubset([:lower, :rewrite, :partition, :decider, :planned, :cover, :cover_start, :cover_trials, :cover_loop,
        :search, :decisions, :storage, :verify], names)
    @test all(((_, (s, b, g)),) -> s >= 0 && b >= 0 && g >= 0, phases)
    @test counts[:plans] >= 1 && counts[:groupings] >= counts[:plans]
    @test counts[:tries] > 0 && counts[:walked] > 0 && counts[:terms] > 0 && counts[:options] > 0
    # The counts are the same each time, the trials planned side by side or
    # not: the decisions kept are let go first, which it would replay.
    rm(joinpath(C.tuning_directory(), "decisions"); recursive = true, force = true)
    again, counted, _ = C.tallied(() -> prepare(p, x; build = false))
    @test counted == counts
    # Replayed, its work is the replay's: no trial planned, no cover.
    replayed, counted, phases = C.tallied(() -> prepare(p, x; build = false, verify = false))
    @test :replay in first.(phases) && !(:cover in first.(phases)) && counted[:trials] == 0
    @test sprint(explain, again) == sprint(explain, ex) == sprint(explain, replayed)
    @test !C.TALLYING[]
    @test_throws ArgumentError C.tallied(() -> C.tallied(() -> nothing))
    @test !C.TALLYING[]
    # The program above forms one kernel on the CUDA target, with nothing to
    # propose.
    if CUDA_FUNCTIONAL
        q, args = stacked(2)
        _, counts, phases = C.tallied(() -> prepare(q, args...; target = T.CUDATarget(), build = false))
        @test :cover_pricing in first.(phases) && counts[:trials] >= 1 && counts[:compared] >= 1
    end
end

@testset "from several tasks, nothing is lost" begin
    heard = Dict{Tuple{Symbol, Any}, Int}()
    lock = ReentrantLock()
    C.OBSERVER[] = (name, what) -> @lock lock (heard[(name, what)] = get(heard, (name, what), 0) + 1)
    try
        _, counts, phases = C.tallied() do
            tasks = map(1:8) do _
                Threads.@spawn for _ in 1:500
                    C.tally!(:plans)
                    C.phase(() -> C.tally!(:terms, 2), :inner)
                    C.observe(:kernel, :key)
                end
            end
            foreach(wait, tasks)
        end
        @test counts[:plans] == 4000 && counts[:terms] == 8000
        @test only(phases)[1] === :inner
        @test heard[(:inner, :begin)] == heard[(:inner, :end)] == 4000 && heard[(:kernel, :key)] == 4000
    finally
        C.OBSERVER[] = nothing
    end
end

@testset "a phase nested in itself spends once" begin
    # Through a task it spawns, as a trial's: the outer phase's seconds are
    # within the wall's, and the sections of the inner one spend nothing.
    inside() = (clock = C.mark(); sleep(0.02); C.lap!(:within, clock))
    function nested()
        clock = C.mark()
        sleep(0.02)
        C.lap!(:section, clock)
        C.phase(() -> fetch(Threads.@spawn(C.phase(inside, :outer))), :inner)
    end
    t = time_ns()
    _, _, phases = C.tallied(() -> C.phase(nested, :outer))
    wall = 1.0e-9 * (time_ns() - t)
    seconds = Dict(name => s for (name, (s, _, _)) in phases)
    @test sort!(collect(keys(seconds))) == [:inner, :outer, :section]
    @test seconds[:inner] <= seconds[:outer] <= wall
end

@testset "an observer that throws is not heard from" begin
    x = data(16, 8)
    p = tallied_program(x)
    C.OBSERVER[] = (name, what) -> error("the observer fails")
    ex = try
        prepare(p, x)
    finally
        C.OBSERVER[] = nothing
    end
    @test agree(filter(!isnothing, C.leaves(outputs(execute!(ex)))), filter(!isnothing, C.leaves(interpret(p, x))))
end
