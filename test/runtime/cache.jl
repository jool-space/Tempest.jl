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
    @test occursin("Tempest = image", C.canonical_text(T.logsumexp))
    @test occursin("Base = image", C.canonical_text(sin))
    before = C.content_key(Elsewhere.scale)
    Core.eval(Elsewhere, :(scale(x) = 3x))
    @test C.content_key(Elsewhere.scale) != before
    @test C.content_key(closure(1.0f0)) != C.content_key(closure(2.0f0))
    # A checked key names Tempest's own code without its identity, which an
    # edit anywhere in Tempest changes; the kernel is checked by what it uses.
    @test occursin("Tempest = checked by dependencies", C.canonical_text(T.logsumexp; checked = true))
    @test occursin("Base = image", C.canonical_text(T.logsumexp, sin; checked = true))
    @test C.content_key(T.logsumexp; checked = true) != C.content_key(T.logsumexp)
end

# Code a key does not name, since only inference finds it: a stored kernel
# whose compilation reached it loads only while what it reached holds. Each
# change here, unchecked, is a stale kernel and a wrong result with no error.
struct Half end
Base.:*(::Half, x::Float32) = 0.5f0 * x
halve(x::Float32) = Half() * x

struct Unit end
Base.Float32(::Unit) = 1.0f0
in_units(x::Float32) = x * Float32(Unit())

struct Third end
Base.:*(::Third, x::Real) = x / 3
thirds(x::Float32) = Third() * x

doubled(x) = 2x

# What the host's compilation of `f` for `args` used, read as a kernel's is.
function used(f, args...)
    f(args...)
    mi = only(Base.method_instances(f, map(typeof, args), Base.get_world_counter()))
    C.dependencies(mi.cache, m -> isdefined(m, :cache) ? m.cache : nothing, nothing)
end

# Checked in the latest world, as a preparation after a definition checks.
holding(dependencies) = Base.invokelatest(all, C.holds, dependencies)

@testset "a stored kernel loads only while what its compilation used holds" begin
    if !C.EDGES
        # Without edges there is nothing to check a kernel by: none is kept.
        @test isempty(used(halve, 1.0f0))
        @test C.load_kernel(C.content_key("a kernel")) === nothing
    else
        # A method of a package's function written outside any package, for a
        # type of its own, redefined.
        before = used(halve, 1.0f0)
        @test holding(before)
        @eval Base.:*(::Half, x::Float32) = 0.25f0 * x
        @test !holding(before)
        before = used(in_units, 1.0f0)
        @eval Base.Float32(::Unit) = 2.0f0
        @test !holding(before)
        # A method more specific than the one a call matched, added while that
        # one is unchanged; and deleted again.
        before = used(thirds, 1.0f0)
        @eval Base.:*(::Third, x::Float32) = 0.0f0
        @test !holding(before)
        after = used(thirds, 1.0f0)
        Base.delete_method(which(*, (Third, Float32)))
        @test !holding(after) && holding(before)
        # A method a script evaluates into an installed package's module is
        # identified by its code, not by the package.
        Core.eval(LinearAlgebra, :(evaluated_here(x::Float32) = 2x))
        twice(x) = LinearAlgebra.evaluated_here(x)
        before = used(twice, 1.0f0)
        @test !occursin("package", C.method_identity(which(LinearAlgebra.evaluated_here, (Float32,))))
        @test occursin("package", C.method_identity(which(sin, (Float32,))))
        Core.eval(LinearAlgebra, :(evaluated_here(x::Float32) = 3x))
        @test !holding(before)
        # What is read back holds as it did; what cannot be read back does not.
        dependency = first(used(thirds, 1.0f0))
        io = IOBuffer()
        C.serialize(io, (dependency.kind, dependency.table, dependency.named, dependency.found))
        @test C.read_holds(take!(io)) && !C.read_holds(UInt8[1, 2, 3])
    end
end

struct Shape
    side::Float32
end
area(s::Shape) = s.side * s.side
invoked_third(x::Float32) = invoke(*, Tuple{Third, Real}, Third(), x)
module Constants
const K = 2.0f0
end
read_constant(x::Float32) = Constants.K * x
through_constant(x::Float32) = read_constant(x)

@testset "each kind of dependency holds until what it names changes" begin
    if C.EDGES
        # An `invoke`, the method it names redefined.
        before = used(invoked_third, 1.0f0)
        @test any(d -> d.kind === :invoke, before) && holding(before)
        @eval Base.:*(::Third, x::Real) = x / 4
        @test !holding(before)
        # A global of another module, a constant redefined: read by a method
        # the kernel calls, written with that method's code.
        before = used(through_constant, 1.0f0)
        @test any(d -> d.kind === :global, before) && holding(before)
        Core.eval(Constants, :(const K = 3.0f0))
        @test !holding(before)
        # A type a method was specialized on, redefined mutable: read back by
        # its name, as another process reads it.
        dependency = only(filter(d -> d.kind === :type && d.named === Shape, used(area, Shape(2.0f0))))
        bytes = C.written(dependency)
        @test C.read_holds(bytes)
        @eval mutable struct Shape
            side::Float32
        end
        @test !Base.invokelatest(C.read_holds, bytes)
    end
end

@testset "code no image identifies is written with what it reads" begin
    # A constant a method reads of its own module, to which inference keeps
    # no edge, is written with the method's code where no image names it.
    Core.eval(LinearAlgebra, :(const TEXT_SLOPE = 2.0f0))
    Core.eval(LinearAlgebra, :(text_slope(x::Float32) = TEXT_SLOPE * x))
    method = which(LinearAlgebra.text_slope, (Float32,))
    before = C.method_text(method)
    Core.eval(LinearAlgebra, :(const TEXT_SLOPE = 3.0f0))
    @test Base.invokelatest(C.method_text, method) != before
    # A generated function by its generator's code.
    @eval @generated generated_twice(x) = :(2x)
    before = C.method_text(only(methods(generated_twice)))
    @eval @generated generated_twice(x) = :(3x)
    @test C.method_text(only(methods(generated_twice))) != before
    # Values by their content: an enumeration's by its bytes, a dictionary's by
    # its contents, and a pointer by its address in this session alone.
    @test C.content_key(RoundingMode{:Up}()) != C.content_key(RoundingMode{:Down}())
    @test C.content_key(Base.MPFR.MPFRRoundUp) != C.content_key(Base.MPFR.MPFRRoundDown)
    @test C.content_key(Dict(1 => 2.0f0)) != C.content_key(Dict(1 => 3.0f0))
    @test occursin("in session", C.canonical_text(Ptr{Cvoid}(1)))
    @test C.content_key(Core.svec(1)) != C.content_key(Core.svec(2))
    @test C.content_key(-NaN32) != C.content_key(NaN32)
end

@testset "a global is known by its image only while the image holds it" begin
    # A method an image holds, reading a constant of its own module, to which
    # inference keeps no edge: the constant redefined, the image no longer
    # holds the binding as it is.
    push!(LOAD_PATH, joinpath(@__DIR__, "..", "fixtures", "packages"))
    ImageConstant = Base.require(@__MODULE__, :ImageConstant)
    through_image = Core.eval(@__MODULE__, :(x::Float32 -> $ImageConstant.act(x)))
    before = Base.invokelatest(used, through_image, 1.0f0)
    @test !C.EDGES || any(d -> d.kind === :global && d.named == GlobalRef(ImageConstant, :SLOPE), before)
    Core.eval(ImageConstant, :(const SLOPE = 3.0f0))
    @test !C.EDGES || !holding(before)
end

# Constants of Tempest's own code, which a checked key leaves out, read by its
# methods: each changed changes the key or the dependencies, through immutable
# structure to its leaves, a function or a type named by a constant too.
@testset "a constant is known by its content" begin
    # From Julia 1.12, where a kernel is stored with its dependencies and a
    # constant may be redefined to a value of another type. Before, no kernel
    # is stored, and a checked key leaves Tempest's code to dependencies no
    # kernel has.
    if C.EDGES
        cases = [(:((:a, 2.0f0)), :((:a, 3.0f0)), name -> :($name[2] * x)),
            (:((kind = :a, k = 2.0f0)), :((kind = :a, k = 3.0f0)), name -> :($name.k * x)),
            (:((Float32, 2.0f0)), :((Float32, 3.0f0)), name -> :($name[2] * x)),
            (:sin, :cos, name -> :($name(x))),
            (:Float32, :Float16, name -> :(Float32($name(x) / 3)))]
        for (i, (before, after, body)) in enumerate(cases)
            name, reader = Symbol(:OWN_CONSTANT_, i), Symbol(:reads_own_constant_, i)
            Core.eval(IR, :(const $name = $before))
            Core.eval(IR, :($reader(x::Float32) = $(body(name))))
            through = Core.eval(@__MODULE__, :(x::Float32 -> IR.$reader(x)))
            key = Base.invokelatest(C.content_key, through; checked = true)
            found = Base.invokelatest(used, through, 1.0f0)
            Core.eval(IR, :(const $name = $after))
            @test Base.invokelatest(C.content_key, through; checked = true) != key || !holding(found)
        end
        # A table, read through an accessor declared foldable, which inference
        # folds into code as it would a constant's value.
        Core.eval(IR, :(const OWN_TABLE = Dict(:k => 2.0f0)))
        Core.eval(IR, :(Base.@assume_effects :foldable reads_own_table(k::Symbol) = OWN_TABLE[k]))
        through = Core.eval(@__MODULE__, :(x::Float32 -> IR.reads_own_table(:k) * x))
        key = Base.invokelatest(C.content_key, through; checked = true)
        found = Base.invokelatest(used, through, 1.0f0)
        Core.eval(IR, :(const OWN_TABLE = Dict(:k => 3.0f0)))
        @test Base.invokelatest(C.content_key, through; checked = true) != key || !holding(found)
    end
end

@testset "waiting on a generator's callees" begin
    # A generated function is written by its generator's own code, not by the
    # code that generator calls. Within a session Julia does not generate the
    # function again when a callee changes, so the kernel and the host are
    # stale alike there. A new process does generate it again, and its stored
    # kernel still holds where a callee changed in code a key leaves out:
    # Tempest's own, which a checked key leaves to the dependencies (its one
    # generated function, `built`, calls no helper today), and a method
    # evaluated into a module an image identifies. Open there, not closed by
    # Julia's behaviour; closed by writing a generated function with the code
    # its generator calls.
    Core.eval(IR, :(generator_helper() = :(2x)))
    Core.eval(IR, :(@generated generated_by_helper(x) = generator_helper()))
    method = only(methods(IR.generated_by_helper))
    before = C.method_text(method)
    Core.eval(IR, :(generator_helper() = :(3x)))
    @test_broken Base.invokelatest(C.method_text, method) != before
end

# A top-level anonymous function's type is named by a counter the process
# keeps, so that one source evaluated again, or after a line making another
# closure above it, has another key, and its kernels compile again: in a REPL,
# a notebook, or a script that gained a closure. Time, not a wrong result.
@testset "waiting on an anonymous function named by its code, not its number" begin
    source = :((x -> 2x, y -> y + 1))
    first = Core.eval(@__MODULE__, source)
    Core.eval(@__MODULE__, :(z -> z - 1))
    second = Core.eval(@__MODULE__, source)
    @test_broken C.content_key(first; checked = true) == C.content_key(second; checked = true)
end

@testset "a package's function redefined in the session is checked again" begin
    # As Revise redefines a developed package's function: the session ends,
    # and what the kernel's compilation used is checked again, the method
    # redefined being no image's and identified by its code.
    Core.eval(LinearAlgebra, :(redefined_in_session(x::Float32) = 2x))
    through(x) = LinearAlgebra.redefined_in_session(x)
    during = C.session()
    before = used(through, 1.0f0)
    Core.eval(LinearAlgebra, :(redefined_in_session(x::Float32) = 3x))
    @test C.session() != during
    @test !C.EDGES || !holding(before)
end

@testset "a value among a type's parameters is named by its code" begin
    # A function outside any package as a parameter of a package's type is
    # named by its code, as a session table's key names it.
    before = C.definitions(Any[Val(doubled)])
    @test !isempty(before)
    @eval doubled(x) = 3x
    @test C.definitions(Any[Val(doubled)]) != before
    @test isempty(C.definitions(Any[Val(sin), Val(:name), Val(2)]))
end

@testset "what a kernel uses of Tempest's code" begin
    # Tempest's methods that inference reached are identified by their code,
    # so that an edit elsewhere in Tempest leaves the kernel loading: here a
    # fold's observe, through a function outside Tempest that calls it.
    observed(s) = IR.fold_observe(IR.SumFold(), s)
    dependencies = used(observed, (1.0f0,))
    if C.EDGES
        call = only(filter(d -> d.kind === :call && d.named <: Tuple{typeof(IR.fold_observe), Vararg}, dependencies))
        @test call.found == C.method_text(which(call.named))
        @test holding(dependencies)
        @test !C.holds(C.Dependency(call.kind, call.table, call.named, "other code"))
    end
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
