# The compiler's own work, counted and marked (src/Compiler/work.jl): off, a
# count, a mark or an event costs nothing and allocates nothing; tallied, a
# preparation's counts are the same each time and its phases are named, a
# phase nested in itself counted once, and the values the cover's trials try
# grow with the trials alone; an observer hears each phase begin and end,
# from every task, and what it throws does not take a preparation down.

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
    # not, the decisions kept not replayed.
    again, counted, _ = withenv("TEMPEST_REPLAY" => "false") do
        C.tallied(() -> prepare(p, x; build = false))
    end
    @test counted == counts
    # Replayed, and not crosschecked, its work is the replay's: no trial
    # planned, no cover.
    replayed, counted, phases = withenv("TEMPEST_CROSSCHECK" => "false") do
        C.tallied(() -> prepare(p, x; build = false))
    end
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

# The gradient of an RK4 of `n` steps of a two-layer field: each step's
# kernels alike but for its constants, so that the cover tries proposals in
# each, on the CUDA target as on the CPU.
function rk4(n)
    ps = (single(data(64, 32)) ./ 8, single(data(64, 1)) ./ 8, single(data(32, 64)) ./ 8, single(data(32, 1)) ./ 8)
    u0 = single(data(32, 32))
    function loss(ps)
        w1, b1, w2, b2 = ps
        f(v) = w2 * tanh.(w1 * v .+ b1) .+ b2
        u = u0
        for k in 1:n
            dt = Float32(1 + k / 64) / n
            k1 = f(u)
            k2 = f(u .+ (dt / 2) .* k1)
            k3 = f(u .+ (dt / 2) .* k2)
            k4 = f(u .+ dt .* k3)
            u = u .+ (dt / 6) .* (k1 .+ 2.0f0 .* k2 .+ 2.0f0 .* k3 .+ k4)
        end
        sum(abs2, u)
    end
    trace(ps -> T.withgradient(loss, ps), ps), (ps,)
end

@testset "a trial forms again only the kernels its proposals reach" begin
    # A trial formed near the plan it changes takes as they were the kernels
    # its proposals do not reach: of the kernels its passes hold, about 20
    # taken for each formed again, at 4 steps as at 8, with the test's
    # packages loaded; planning each trial whole took none. This is the
    # guard. Twice the steps, twice the trials: the values tried, which grew
    # four times planned whole (127,841 at 4 steps, 521,467 at 8), grow 2.9
    # times (9,780 and 28,602), a trial that recomputes a fold forming again
    # the kernels after it that looked at whether it was stored. The ratio is
    # the weaker check of the two, its margin to planning whole thin. Not
    # crosschecked, which plans each trial whole as well.
    function tried(n)
        q, args = rk4(n)
        _, counts, _ = withenv("TEMPEST_REPLAY" => "false", "TEMPEST_CROSSCHECK" => "false") do
            C.tallied(() -> prepare(q, args...; target = T.CUDATarget(precision = :tf32), build = false))
        end
        counts
    end
    four, eight = tried(4), tried(8)
    @test four[:reused] > 9 * four[:formed] && eight[:reused] > 9 * eight[:formed]
    @test eight[:tries] < 3.5 * four[:tries]
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
