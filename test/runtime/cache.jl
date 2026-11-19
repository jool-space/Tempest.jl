# Content keys, the tuning record and the precompile workload: keys name
# content and are the same in every session, and the record lives in
# Tempest's scratch space unless the environment names a directory. That a
# kernel compiles once per key and loads from disk after is in cuda/cache.jl.
include("../fixtures/programs.jl")

# Code outside any package, identified by its code.
module Elsewhere
scale(x) = 2x
end

closure(s) = x -> s * x

@testset "content keys" begin
    # Generated names count up through a session; keys number them instead.
    a, b = gensym(:t), gensym(:t)
    @test C.content_key(:(for $a in 1:2; f($a); end)) == C.content_key(:(for $b in 1:2; f($b); end))
    @test C.content_key(:(x + 1)) != C.content_key(:(x + 2))
    @test C.content_key(1.0f0) != C.content_key(1.0)
    # A generated function is its source.
    f, g, h = map(n -> C.generated(:(function (x) x + $n end)), (1, 1, 2))
    @test C.content_key(f) == C.content_key(g) != C.content_key(h)
    @test_throws ErrorException C.content_key(Core.eval(C.Generated, :(function (x) x end)))
    # Package code is identified by its package; other code by its code, and
    # a closure by what it captures too.
    @test occursin("Tempest = sources", C.canonical_text(T.logsumexp))
    @test occursin("Base = julia", C.canonical_text(sin))
    before = C.content_key(Elsewhere.scale)
    Core.eval(Elsewhere, :(scale(x) = 3x))
    @test C.content_key(Elsewhere.scale) != before
    @test C.content_key(closure(1.0f0)) != C.content_key(closure(2.0f0))
    # A checked key names Tempest's own code without its identity, which an
    # edit anywhere in Tempest changes; the kernel is checked by what it uses.
    @test occursin("Tempest = checked by dependencies", C.canonical_text(T.logsumexp; checked = true))
    @test occursin("Base = julia", C.canonical_text(T.logsumexp, sin; checked = true))
    # Before Julia 1.12 there are no edges to check by, and the key is whole.
    @test (C.content_key(T.logsumexp; checked = true) == C.content_key(T.logsumexp)) == !C.EDGES
end

# Code a key does not name. Each is a stale kernel and a wrong result with no
# error: a new process loads from the store the binary of the code before the
# change, or the session keeps the kernel it compiled before it.
struct Half end
Base.:*(::Half, x::Float32) = 0.5f0 * x
halve(x::Float32) = Half() * x

struct Unit end
Base.Float32(::Unit) = 1.0f0
in_units(x::Float32) = x * Float32(Unit())

doubled(x) = 2x

@testset "a key names all the code its kernel is compiled from" begin
    # A method of a package's function written outside any package, for a
    # type of its own: the key names the function, which is the package's.
    before = C.content_key(halve)
    @eval Base.:*(::Half, x::Float32) = 0.25f0 * x
    @test_broken C.content_key(halve) != before
    before = C.content_key(in_units)
    @eval Base.Float32(::Unit) = 2.0f0
    @test_broken C.content_key(in_units) != before
    # A package's function redefined in the session, as Revise does when a
    # developed package's file is edited: a package is identified once.
    Core.eval(LinearAlgebra, :(redefined_in_session(x) = 2x))
    before = C.content_key(LinearAlgebra.redefined_in_session)
    Core.eval(LinearAlgebra, :(redefined_in_session(x) = 3x))
    @test_broken C.content_key(LinearAlgebra.redefined_in_session) != before
    # A function outside any package as a parameter of a package's type is
    # named by its code, as a session table's key names it.
    before = C.definitions(Any[Val(doubled)])
    @test !isempty(before)
    @eval doubled(x) = 3x
    @test C.definitions(Any[Val(doubled)]) != before
    @test isempty(C.definitions(Any[Val(sin), Val(:name), Val(2)]))
end

@testset "what a kernel depends on of Tempest's code" begin
    # The methods of Tempest's that inference reached, from the edges it keeps:
    # here a fold's observe, through a function outside Tempest that calls it.
    observed(s) = IR.fold_observe(IR.SumFold(), s)
    observed((1.0f0,))
    mi = only(methods(observed)).specializations
    mi = mi isa Core.MethodInstance ? mi : first(filter(!isnothing, collect(mi)))
    bytes = C.dependencies(mi.cache, m -> isdefined(m, :cache) ? m.cache : nothing)
    if !C.EDGES
        @test isempty(bytes)
    else
        records = C.deserialize(IOBuffer(bytes))
        @test any(r -> r[1] === :method && r[2] <: Tuple{typeof(IR.fold_observe), Vararg}, records)
        @test all(r -> r[1] !== :method || C.own_code(which(r[2]).module), records)
        @test C.current(bytes)
        # Changed code, a record that cannot be read back, and none.
        kind, sig, _ = first(filter(r -> r[1] === :method, records))
        io = IOBuffer(); C.serialize(io, [(kind, sig, "other code")])
        @test !C.current(take!(io))
        @test !C.current(UInt8[1, 2, 3])
    end
    @test C.current(UInt8[])
end

@testset "where the tuning record lives" begin
    # In Tempest's scratch space, unless the environment names a directory.
    scratch(depot) = joinpath(depot, "scratchspaces", string(Base.PkgId(Tempest).uuid), "tuning")
    withenv("TEMPEST_CACHE" => nothing) do
        @test C.tuning_directory() == scratch(first(DEPOT_PATH)) && isdir(C.tuning_directory())
    end
    elsewhere = mktempdir()
    withenv("TEMPEST_CACHE" => elsewhere) do
        @test C.tuning_directory() == joinpath(elsewhere, "tuning")
        @test dirname(C.record_path(T.CPUTarget())) == joinpath(elsewhere, "tuning")
    end
    # The path is found when Tempest loads, not baked in when it precompiles:
    # a process with another first depot keeps its record there.
    depot = mktempdir()
    script = "using Tempest; print(Tempest.Compiler.tuning_directory())"
    path = withenv("JULIA_DEPOT_PATH" => join([depot; DEPOT_PATH], Sys.iswindows() ? ';' : ':'), "TEMPEST_CACHE" => nothing) do
        readchomp(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`)
    end
    @test path == scratch(depot)
end

# A target on a device of a name whose kernels a toolchain of a version builds.
struct BuiltBy <: C.Target
    version::String
end
C.device_name(::BuiltBy) = "Some GPU"
C.toolchain(t::BuiltBy) = ["Some compiler" => t.version]

@testset "a record per device and toolchain" begin
    # A kernel's time is of what built it: one toolchain's times are not read
    # for another's kernels, and each keeps its own rates. A target stating
    # none names its files by its device alone.
    withenv("TEMPEST_CACHE" => mktempdir()) do
        old, new = BuiltBy("13.4"), BuiltBy("13.5")
        @test C.record_path(old) != C.record_path(new) && C.rates_path(old) != C.rates_path(new)
        @test basename(C.record_path(new)) == "Some_GPU_Some_compiler_13.5.cold.tsv"
        @test basename(C.record_path(T.CPUTarget())) == "cpu.cold.tsv"
        graph = C.lower(trace(x -> 2 .* x, [1.0f0]))
        C.save_record(C.record_path(old), "a kernel", 1.0e-6)
        @test C.Decider(old, graph).record["a kernel"] == 1.0e-6
        @test !haskey(C.Decider(new, graph).record, "a kernel")
    end
end

# A target on a device of a name and an identity, for the rates shipped.
struct NamedDevice <: C.Target
    name::String
    identity::Vector{Pair{String, Float64}}
end
C.device_name(t::NamedDevice) = t.name
C.device_identity(t::NamedDevice) = t.identity

@testset "rates shipped for a device" begin
    # Read for the device a file is named for, of the identity it states;
    # not for a device of another name, nor one of that name and another
    # identity, as a product of fewer multiprocessors.
    dir = mktempdir()
    identity = ["device bandwidth" => 2.7e11, "device cache" => 2.5e7, "device blocks" => 48.0]
    function ship(basis)
        open(joinpath(dir, "Some_GPU.cold.rates.tsv"), "w") do io
            println(io, "timing basis\t", basis)
            foreach(((k, v),) -> println(io, k, '\t', v), identity)
            println(io, "cuTile memory latency\t1.5e-6")
        end
    end
    # Nor one timed on another basis than the one Tempest times on.
    ship(C.TIMING_BASIS + 1)
    @test isempty(C.shipped_rates(NamedDevice("Some GPU", identity), dir))
    ship(C.TIMING_BASIS)
    @test C.shipped_rates(NamedDevice("Some GPU", identity), dir)["cuTile memory latency"] == 1.5e-6
    @test isempty(C.shipped_rates(NamedDevice("Other GPU", identity), dir))
    @test isempty(C.shipped_rates(NamedDevice("Some GPU", ["device bandwidth" => 2.7e11, "device cache" => 2.5e7, "device blocks" => 20.0]), dir))
    @test isempty(C.shipped_rates(T.CPUTarget(), dir))
    # Tempest ships the GB10's. A decider prices on them, under the rates
    # fitted on the machine itself; one pricing on nominal rates reads none,
    # and `calibrate!`, given the machine's alone, fits what its file lacks.
    file = C.load_record(joinpath(C.shipped_directory(), "NVIDIA_GB10.cold.rates.tsv"))
    gb10 = NamedDevice("NVIDIA GB10", [k => v for (k, v) in file if startswith(k, "device ")])
    shipped = C.shipped_rates(gb10)
    @test haskey(shipped, "cuTile tf32 rate") && haskey(shipped, "applied swept tf32 exp")
    graph = C.lower(trace(x -> 2 .* x, [1.0f0]))
    withenv("TEMPEST_CACHE" => mktempdir()) do
        C.save_record(C.rates_path(gb10), "cuTile tf32 rate", 1.0e12)
        @test C.Decider(gb10, graph).calibration["cuTile tf32 rate"] == 1.0e12
        @test C.Decider(gb10, graph).calibration["cuBLAS tf32 rate"] == shipped["cuBLAS tf32 rate"]
        @test isempty(C.known_rates(gb10, :dry))
    end
end

@testset "a fit outside what its rates can be" begin
    # Kept where every rate of the fit lies in its domain; reported where
    # one does not, and the stand-in kept for all of them, or nothing.
    graph = C.lower(trace(x -> 2 .* x, [1.0f0]))
    withenv("TEMPEST_CACHE" => mktempdir()) do
        d = C.Decider(T.CPUTarget(), graph)
        @test C.keep_fit!(d, ("a latency", "a rate"), (1.0e-6, 1.0e12), (C.SECONDS, C.RATE), (0.0, 1.0)) == (1.0e-6, 1.0e12)
        @test d.calibration["a rate"] == 1.0e12
        @test (@test_logs (:warn, r"a recompute failed") C.keep_fit!(d, ("a recompute",), (-1.07e13,), (C.RATE,), (1.0e12,))) == (1.0e12,)
        @test d.calibration["a recompute"] == 1.0e12
        @test (@test_logs (:warn,) C.keep_fit!(d, ("a bandwidth",), (2 * d.rates.bandwidth,), (C.bandwidths(d),), (d.rates.bandwidth,))) == (d.rates.bandwidth,)
        @test (@test_logs (:warn,) C.keep_fit!(d, ("a step",), (NaN,), (C.SECONDS,), nothing)) === nothing
        @test !haskey(d.calibration, "a step")
    end
end

# What Tempest compiles while it precompiles, which development sessions skip.
@testset "the precompile workload" begin
    @test T.workload() === nothing
end

# The extensions' workloads prepare for CUDA where a package is built, which
# may be where no device is visible: a preparation that builds nothing, for a
# device named and of arguments on the host, asks the driver nothing. In a
# process of its own, which sees no device.
@testset "a preparation that builds nothing needs no device" begin
    script = """
        using Tempest, CUDACore
        import cuBLAS, cuDNN, cuTile
        f(w, x) = sum(abs2, tanh.(w * x))
        w, x = Float32.(reshape(sin.(1:32), 4, 8)), Float32.(reshape(cos.(1:48), 8, 6))
        program = Tempest.Compiler.trace((xs...) -> Tempest.withgradient(f, xs...), w, x)
        for precision in (:full, :tf32)
            ex = Tempest.Compiler.prepare(program, w, x; target = Tempest.CUDATarget(; device = 0, precision), build = false)
            explain(devnull, ex)
        end
        """
    child = addenv(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`, "CUDA_VISIBLE_DEVICES" => "")
    output = IOBuffer()
    @test success(run(pipeline(ignorestatus(child); stdout = output, stderr = output))) || (print(String(take!(output))); false)
end
