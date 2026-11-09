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
