# The compiler's own work, counted and marked (src/Compiler/work.jl): off, a
# count, a mark or an event costs nothing and allocates nothing; tallied, a
# preparation's counts are the same each time and its phases are named; an
# observer hears each phase begin and end, from every task, and what it
# throws does not take a preparation down.

# A gradient, so that the cover has proposals to try.
tallied_program(x) = differentiate(trace(x -> sum(abs2, tanh.(x) .- sum(x; dims = 1) ./ size(x, 1)), x))

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
    # The counts are the same each time, the trials planned side by side or not.
    again, counted, _ = C.tallied(() -> prepare(p, x; build = false))
    @test counted == counts
    @test sprint(explain, again) == sprint(explain, ex) == sprint(explain, prepare(p, x; build = false))
    @test !C.TALLYING[]
    @test_throws ArgumentError C.tallied(() -> C.tallied(() -> nothing))
    @test !C.TALLYING[]
    if CUDA_FUNCTIONAL
        _, counts, phases = C.tallied(() -> prepare(p, x; target = T.CUDATarget(), build = false))
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
