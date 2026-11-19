# The tables of answers a session keeps (src/Compiler/session.jl): every
# table-like constant of Tempest's modules and its extensions loaded is a
# session table or is named, with why, as not one; `forget_session!` empties
# every session table, and a preparation after it, or during it, makes the
# same plan from the same answers found again.

# A layer's gradient, whose product an implementation claims on CUDA.
layer_program(w, x, b) = differentiate(trace((w, x, b) -> sum(abs2, tanh.(w * x .+ b)), w, x, b))

function plan_and_outputs(p, args, target)
    ex = prepare(p, args...; target, measure = false)
    sprint(T.explain, ex), map(on_host, filter(!isnothing, C.leaves(outputs(execute!(ex)))))
end

session_tables() = [name => t for (name, t) in C.tempest_globals() if t isa C.SessionTable]

@testset "every table-like constant is a session table or named as not one" begin
    found = C.tempest_globals()
    named = first.(C.NOT_SESSION_TABLES)
    unnamed = [name for (name, x) in found if !(x isa C.SessionTable) && !(name in named)]
    @test unnamed == String[]
    misnamed = [name for (name, x) in found if x isa C.SessionTable && name in named]
    @test misnamed == String[]
    # A name kept of a module loaded is found, so that the list does not
    # outlive what it names.
    loaded = Set(join(fullname(m), '.') for m in C.tempest_modules())
    stale = [name for name in named if first(rsplit(name, '.'; limit = 2)) in loaded && !any(==(name) ∘ first, found)]
    @test stale == String[]
    # The compiler's own: the kernel cache's four and the library's two.
    @test count(((name, _),) -> startswith(name, "Tempest.Compiler."), session_tables()) >= 6
end

@testset "forgotten, every answer is found again and the plans are the same" begin
    w, x, b = single(data(8, 16)) ./ 4, single(data(16, 12)), single(data(8))
    p = layer_program(w, x, b)
    targets = Any[(T.CPUTarget(), (w, x, b))]
    CUDA_FUNCTIONAL && push!(targets, (T.CUDATarget(), map(CuArray, (w, x, b))))
    for (target, args) in targets
        before = plan_and_outputs(p, args, target)
        C.forget_session!()
        @test all(((_, t),) -> length(t) == 0, session_tables())
        @test plan_and_outputs(p, args, target) == before
        if target isa T.CUDATarget
            # A preparation on the CPU reads none of the tables, which are
            # the kernel cache's, the library's and the CUDA extensions'; on
            # CUDA the one just made after forgetting found its kernels and
            # cuTile's lowerings again, so forgetting is tested here.
            filled = Set(name for (name, t) in session_tables() if length(t) > 0)
            @test issubset(["Tempest.Compiler.RESOLVED_KERNELS", "cuTileExt.LOWERED", "cuTileExt.FUNCTIONS"], filled)
        end
        # Forgotten again and again while a preparation reads the tables,
        # each answer it loses is found again, the same.
        preparing = Threads.@spawn [plan_and_outputs(p, args, target) for _ in 1:3]
        while !istaskdone(preparing)
            C.forget_session!()
            yield()
        end
        @test all(==(before), fetch(preparing))
    end
end

# A session table's key holds everything its answer was derived from that can
# differ between two questions in one session (see `SessionTable`), and its
# answers are of its session alone.

@testset "a package image holds no answers for the session that loads it" begin
    # A process of its own, freshly loaded: what the precompile workloads
    # found as they precompiled, in another process, is none of its answers.
    script = """
        using Tempest, CUDACore, cuTile, cuBLAS, NNlib
        C = Tempest.Compiler
        print(join([name for (name, t) in C.tempest_globals() if t isa C.SessionTable && length(t) > 0], " "))
        """
    output = readchomp(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`)
    @test isempty(split(output))
    # A table of another session holds none of its answers here.
    t = C.SessionTable{Int, Int}()
    get!(t, 1, 2)
    t.session[] = C.SESSION[] + 1
    @test get(t, 1, nothing) === nothing && length(t) == 0
end

# A scalar function the tracer records whole, its method taking only Float32,
# which cuTile lowers.
act(x::Float32) = 2.0f0 * x
scaled(x) = sum(act.(x) .* x; dims = 1)

function scaled_run(x, target)
    p = trace(scaled, x)
    ex = prepare(p, target isa T.CUDATarget ? CuArray(x) : x; target)
    got = only(map(on_host, filter(!isnothing, C.leaves(outputs(execute!(ex))))))
    (; got, expected = only(filter(!isnothing, C.leaves(interpret(p, x)))), views = C.group_views(ex))
end

# The definitions the keys of answers about a plan's groups hold.
named_code(views) = unique!(reduce(vcat, [C.definitions(C.Implement.code(C.Implement.structure(g))) for g in views]))

@testset "a function redefined in the session is other code" begin
    x = single(data(256, 64)) ./ 4
    if CUDA_FUNCTIONAL
        first = Base.invokelatest(scaled_run, x, T.CUDATarget())
        @test first.got ≈ first.expected
        @eval act(x::Float32) = 3.0f0 * x
        # Prepared again in the session, its kernels are not those of the
        # definition before.
        second = Base.invokelatest(scaled_run, x, T.CUDATarget())
        @test second.got ≈ second.expected
        @test !(second.got ≈ first.got)
        @eval act(x::Float32) = 2.0f0 * x
    end
    before = Base.invokelatest(scaled_run, x, T.CPUTarget())
    named = named_code(before.views)
    @test length(named) == 1
    # Code of packages is named by its object alone.
    @test C.definitions(Any[sin, +, Float32, IR.SumFold()]) == String[]
    @eval act(x::Float32) = 3.0f0 * x
    after = Base.invokelatest(scaled_run, x, T.CPUTarget())
    @test after.got ≈ after.expected
    @test length(named_code(after.views)) == 1 && named_code(after.views) != named
end

if CUDA_FUNCTIONAL
    @testset "a preparation that compiles is not handed what one that compiles nothing found" begin
        # cuTile has no lowering of Base's `exp10`, which only a preparation
        # that compiles finds out (`Implement.compiling`).
        x = CuArray(single(data(256, 64)) ./ 4)
        p = trace(x -> sum(exp10.(x) .* x; dims = 1), x)
        target = T.CUDATarget()
        C.forget_session!()
        fresh = sprint(T.explain, prepare(p, x; target))
        C.forget_session!()
        prepare(p, x; target, build = false)
        ex = @test_logs min_level = Base.CoreLogging.Warn prepare(p, x; target)
        @test sprint(T.explain, ex) == fresh
    end

end
