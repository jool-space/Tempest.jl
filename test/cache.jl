# The kernel cache: keys name content and are the same in every session; a
# kernel compiles once per key and loads from disk after.
include("fixtures/programs.jl")

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

# What Tempest compiles while it precompiles, which development sessions skip.
@testset "the precompile workload" begin
    @test T.workload() === nothing
end

if CUDA_FUNCTIONAL
    @testset "kernel cache" begin
        # Launch parameters are those cuTile's code generation gives the kernel.
        ext = Base.get_extension(Tempest, :cuTileExt)
        y = CuArray{Float32}(undef, 16, 8)
        tiles = (C.generated(:(function (y, arrays, index) y end)), (cuTile.TileArray(y),))
        for argument in (cuTile.TileArray(y), cuTile.Constant(64), 1.5f0, tiles, (cuTile.Constant(true), 2.0f0))
            expressions, types = Any[], Any[]
            cuTile.flatten_params!(expressions, types, typeof(argument), :argument)
            @test map(typeof, ext.flatten!(Any[], argument)) == types
        end
        # So are the groups of arrays that overlap.
        whole, part, other = cuTile.TileArray(y), cuTile.TileArray(view(y, :, 3:4)), cuTile.TileArray(CuArray{Float32}(undef, 4))
        for arguments in ((whole, other), (whole, other, part), (other, tiles, whole), ())
            @test ext.alias_groups(arguments) == cuTile.alias_groups(arguments...)
        end

        # Kernels compile once and load after, from a store this test starts
        # empty, in a process of its own; it prints its failures.
        mktempdir() do store
            child = `$(Base.julia_cmd()) --project=$(Base.active_project()) $(joinpath(@__DIR__, "fixtures", "kernel_cache.jl"))`
            output = IOBuffer()
            process = run(pipeline(ignorestatus(addenv(child, "JULIA_OBJCACHE_PATH" => store)); stdout = output, stderr = output))
            success(process) || println(stderr, String(take!(output)))
            @test success(process)
        end

        # Captured graphs are destroyed before the modules they launch are
        # unloaded: graphs that outlive their programs and are collected
        # after them, and a graph alive when the process exits.
        script = """
            using Tempest, Tempest.Compiler, CUDACore, cuTile
            x = CUDACore.CuArray(Float32.(reshape(1:4096, 64, 64)))
            p = trace(x -> sum(abs2, x * x; dims = 1), x)
            captured() = (ex = prepare(p, x; target = Tempest.CUDATarget(precision = :tf32)); replay!(capture!(ex)); ex.captured)
            graphs = [captured() for _ in 1:3]
            CUDACore.synchronize()
            foreach(_ -> GC.gc(true), 1:3)
            graphs = nothing
            foreach(_ -> GC.gc(true), 1:3)
            ex = prepare(p, x; target = Tempest.CUDATarget(precision = :tf32))
            replay!(capture!(ex))
            CUDACore.synchronize()
            """
        @test success(pipeline(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`; stderr = devnull))
    end
end
