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

# A developed package of the test's own, loaded from its source, whose scalar
# function a program applies: as a user's model package is, under Revise.
const PACKAGE = joinpath(mktempdir(), "Redefined")
write_package(factor) = write(joinpath(PACKAGE, "src", "Redefined.jl"), """
    module Redefined
    __precompile__(false)
    act(x::Float32) = $(factor)f0 * x
    end
    """)
mkpath(joinpath(PACKAGE, "src"))
write(joinpath(PACKAGE, "Project.toml"), "name = \"Redefined\"\nuuid = \"$(Base.UUID(0x6f1a77e2_5c2b_4e0e_9d1e_3b8f6a4c2d10))\"\n")
write_package(2)
push!(LOAD_PATH, dirname(PACKAGE))
const Redefined = Base.require(@__MODULE__, :Redefined)
scaled_by_package(x) = sum(Redefined.act.(x) .* x; dims = 1)

# The package's function redefined as Revise redefines it, its file written
# and evaluated in its module, or by an `@eval` into it, its file unchanged.
function redefine(factor; file)
    file && write_package(factor)
    Core.eval(Redefined, :(act(x::Float32) = $(Float32(factor)) * x))
end

function scaled_run(f, x, target)
    p = trace(f, x)
    ex = prepare(p, target isa T.CUDATarget ? CuArray(x) : x; target)
    got = only(map(on_host, filter(!isnothing, C.leaves(outputs(execute!(ex))))))
    (; got, expected = only(filter(!isnothing, C.leaves(interpret(p, x)))), views = C.group_views(ex))
end

# The definitions the keys of answers about a plan's groups hold.
named_code(views) = unique!(reduce(vcat, [C.definitions(C.Implement.code(C.Implement.structure(g))) for g in views]))

# A global code reads, assigned in the session.
divisor = 6

@testset "a session ends where code is defined that is not Tempest's own" begin
    before = C.session()
    t = C.SessionTable{Int, Int}()
    get!(t, 1, 2)
    # A kernel's function, which every build defines, leaves it standing.
    C.generated(:(function (x) x + 1 end))
    @test C.session() == before && get(t, 1, nothing) == 2
    # A definition in a package's module, as Revise or an `@eval` makes,
    # ends it, and the table's answers with it.
    Core.eval(Redefined, :(unrelated(x) = x))
    @test C.session() != before && get(t, 1, nothing) === nothing
    # A global assigned is no definition: what code reads of it is named in
    # the keys (`definitions`).
    during = C.session()
    global divisor = 5
    @test C.session() == during
    global divisor = 6
end

# The library's tables name the code a template is, with the globals it
# reads, which change in a session without ending it.
# Functions are defined before a session is looked at: a function defined,
# as a closure written at the top level is where it is reached, ends it.
divided_mean = C.Template(x -> sum(x; dims = 1) ./ divisor; inputs = (X = (:r, :c),), outputs = (:Y,))
mean_of_six(x) = sum(x; dims = 1) ./ 6
bound(t, f, a) = [get(C.match_template(t, 1, T.CPUTarget(), g), 1, nothing) for g in C.group_views(C.plan(C.rewrite(C.lower(trace(f, a)))))]

@testset "the library's tables name the code a template reads" begin
    a = data(6, 4)
    during = C.session()
    @test only(bound(divided_mean, mean_of_six, a)) !== nothing
    global divisor = 5
    @test all(isnothing, bound(divided_mean, mean_of_six, a))
    @test C.session() == during
    global divisor = 6
end

@testset "a function redefined in the session is other code" begin
    x = single(data(256, 64)) ./ 4
    if CUDA_FUNCTIONAL
        first = Base.invokelatest(scaled_run, scaled, x, T.CUDATarget())
        @test first.got ≈ first.expected
        @eval act(x::Float32) = 3.0f0 * x
        # Prepared again in the session, its kernels are not those of the
        # definition before.
        second = Base.invokelatest(scaled_run, scaled, x, T.CUDATarget())
        @test second.got ≈ second.expected
        @test !(second.got ≈ first.got)
        @eval act(x::Float32) = 2.0f0 * x
        # A package's function, its file edited and evaluated as Revise does:
        # the session's answers are asked again and its kernel compiled anew.
        first = Base.invokelatest(scaled_run, scaled_by_package, x, T.CUDATarget())
        @test first.got ≈ first.expected
        redefine(3; file = true)
        second = Base.invokelatest(scaled_run, scaled_by_package, x, T.CUDATarget())
        @test second.got ≈ second.expected
        @test !(second.got ≈ first.got)
        redefine(2; file = true)
    end
    before = Base.invokelatest(scaled_run, scaled, x, T.CPUTarget())
    named = named_code(before.views)
    @test length(named) == 1
    # Code of packages is named by its object alone.
    @test C.definitions(Any[sin, +, Float32, IR.SumFold()]) == String[]
    @eval act(x::Float32) = 3.0f0 * x
    after = Base.invokelatest(scaled_run, scaled, x, T.CPUTarget())
    @test after.got ≈ after.expected
    @test length(named_code(after.views)) == 1 && named_code(after.views) != named
    @eval act(x::Float32) = 2.0f0 * x
    for file in (true, false)
        before = Base.invokelatest(scaled_run, scaled_by_package, x, T.CPUTarget())
        redefine(3; file)
        after = Base.invokelatest(scaled_run, scaled_by_package, x, T.CPUTarget())
        @test after.got ≈ after.expected && !(after.got ≈ before.got)
        redefine(2; file)
    end
end

# The session asks again once a package's code is defined anew, but the
# kernel store answers by the package's identity, the hash of its files,
# which an `@eval` leaves as it was: the kernel compiled before is loaded
# again. Closed by the store's key naming the code the process has loaded.
if CUDA_FUNCTIONAL
    @testset "a package's function redefined, its files unchanged, runs as redefined once the store's key names the code loaded" begin
        x = single(data(256, 64)) ./ 4
        first = Base.invokelatest(scaled_run, scaled_by_package, x, T.CUDATarget())
        @test first.got ≈ first.expected
        redefine(3; file = false)
        second = Base.invokelatest(scaled_run, scaled_by_package, x, T.CUDATarget())
        @test_broken second.got ≈ second.expected
        redefine(2; file = false)
    end
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

    @testset "cuTile's tables name the architecture and the code their answers were derived from" begin
        X = Base.get_extension(Tempest, :cuTileExt)
        arch = X.ct.device_sm_arch()
        table_keys(t) = @lock t.lock collect(keys(C.answers(t)))
        x = single(data(256, 64)) ./ 4
        # The keys are read as the preparation leaves them: the closures
        # below, defined where they are reached, end the session.
        C.forget_session!()
        Base.invokelatest(scaled_run, scaled, x, T.CUDATarget())
        lowered, said, emitted = table_keys(X.LOWERED), table_keys(X.SAID), table_keys(X.EMITTED)
        named = C.definitions((act,))
        @test !isempty(named)
        # Whether cuTile lowers `act`: by its definition and the architecture
        # its trial compiled for.
        @test any(k -> k[1] === act && k[2] == named && k[5] == arch, lowered)
        # What the generator says of a group, and the kernels it emits: by
        # the code the group's functions are and, compiling, the architecture.
        @test all(k -> k[4][3] && k[4][4] == arch, said) && any(k -> issubset(named, k[4][2]), said)
        @test !isempty(emitted) && all(k -> k[1][3] && k[1][4] == arch, emitted)
        # A preparation that compiles nothing reads no architecture.
        C.forget_session!()
        prepare(trace(scaled, x), CuArray(x); target = T.CUDATarget(), build = false)
        said = table_keys(X.SAID)
        @test !isempty(said) && all(k -> !k[4][3] && k[4][4] === nothing, said)
    end

    @testset "groups alike but numbered in different orders share their kernels, each computing its own" begin
        # Layers alike but for the numbers of their values, which the program
        # numbers in different orders: arguments permuted, and operands of a
        # commutative product read on either side.
        X = Base.get_extension(Tempest, :cuTileExt)
        layer_a(w, s, t, h) = tanh.((w * h) .* s .+ t)
        layer_b(w, s, t, h) = tanh.(s .* (w * h) .+ t)
        function layers(s2, w1, t3, x, w3, s1, t2, w2, s3, t1)
            h = layer_a(w1, s1, t1, x)
            h = layer_b(w2, s2, t2, h)
            h = layer_a(w3, s3, t3, h)
            sum(abs2, h)
        end
        n, m = 64, 48
        args = map(((rows, scale),) -> single(data(n, rows == :n ? n : m; scale)) ./ 4,
            ((:m, 2), (:n, 3), (:m, 5), (:m, 7), (:n, 11), (:m, 13), (:m, 17), (:n, 19), (:m, 23), (:m, 29)))
        for p in (trace(layers, args...), differentiate(trace(layers, args...)))
            C.forget_session!()
            ex = prepare(p, map(CuArray, args)...; target = T.CUDATarget())
            got = map(on_host, filter(!isnothing, C.leaves(outputs(execute!(ex)))))
            expected = filter(!isnothing, C.leaves(interpret(p, args...)))
            @test all(isapprox.(got, expected; rtol = 1.0e-4))
            tiled = count(d -> d !== nothing && d.choice.implementation isa X.cuTileGenerator, ex.decisions)
            @test 0 < length(X.EMITTED) < tiled
        end
    end
end
