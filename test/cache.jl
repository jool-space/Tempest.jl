# The kernel cache: keys name content and are the same in every session; a
# kernel compiles once per key and loads from disk after.
using Test
include("cases.jl")

# Code outside any package, identified by its code.
module Elsewhere
scale(x) = 2x
end

closure(s) = x -> s * x

@testset "content keys" begin
    # Generated names count up through a session; keys number them instead.
    a, b = gensym(:t), gensym(:t)
    @test T.content_key(:(for $a in 1:2; f($a); end)) == T.content_key(:(for $b in 1:2; f($b); end))
    @test T.content_key(:(x + 1)) != T.content_key(:(x + 2))
    @test T.content_key(1.0f0) != T.content_key(1.0)
    # A generated function is its source.
    f, g, h = map(n -> T.generated(:(function (x) x + $n end)), (1, 1, 2))
    @test T.content_key(f) == T.content_key(g) != T.content_key(h)
    @test_throws ErrorException T.content_key(Core.eval(T.Generated, :(function (x) x end)))
    # Package code is identified by its package; other code by its code, and
    # a closure by what it captures too.
    @test occursin("Tempest = sources", T.canonical_text(T.logsumexp))
    @test occursin("Base = julia", T.canonical_text(sin))
    before = T.content_key(Elsewhere.scale)
    Core.eval(Elsewhere, :(scale(x) = 3x))
    @test T.content_key(Elsewhere.scale) != before
    @test T.content_key(closure(1.0f0)) != T.content_key(closure(2.0f0))
end

@testset "where the caches live" begin
    # In Tempest's scratch spaces, unless the environment names a directory.
    scratch(depot, key) = joinpath(depot, "scratchspaces", string(Base.PkgId(Tempest).uuid), key)
    withenv("TEMPEST_KERNEL_CACHE" => nothing, "TEMPEST_CACHE" => nothing) do
        @test T.kernel_cache() == scratch(first(DEPOT_PATH), "kernels") && isdir(T.kernel_cache())
        @test T.tuning_directory() == scratch(first(DEPOT_PATH), "tuning") && isdir(T.tuning_directory())
    end
    withenv("TEMPEST_KERNEL_CACHE" => "0") do
        @test T.kernel_cache() === nothing
    end
    elsewhere = mktempdir()
    withenv("TEMPEST_KERNEL_CACHE" => elsewhere, "TEMPEST_CACHE" => elsewhere) do
        @test T.kernel_cache() == elsewhere
        @test T.tuning_directory() == joinpath(elsewhere, "tuning")
        @test dirname(T.record_path(T.CPUTarget())) == joinpath(elsewhere, "tuning")
    end
    # The paths are found when Tempest loads, not baked in when it
    # precompiles: a process with another first depot keeps its caches there.
    depot = mktempdir()
    script = "using Tempest; print(Tempest.kernel_cache(), '\\n', Tempest.tuning_directory())"
    paths = withenv("JULIA_DEPOT_PATH" => join([depot; DEPOT_PATH], Sys.iswindows() ? ';' : ':'),
        "TEMPEST_KERNEL_CACHE" => nothing, "TEMPEST_CACHE" => nothing) do
        readchomp(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`)
    end
    @test split(paths, '\n') == [scratch(depot, "kernels"), scratch(depot, "tuning")]
end

@testset "the kernel cache's size" begin
    # Beyond its limit the cache removes the entries used least recently.
    withenv("TEMPEST_KERNEL_CACHE" => mktempdir(), "TEMPEST_KERNEL_CACHE_LIMIT" => "3000") do
        keys = [T.content_key(n) for n in 1:4]
        binary = zeros(UInt8, 990) # entries of 1000 bytes, the name and its length included
        kept() = map(key -> isfile(T.entry_path(T.kernel_cache(), key)), keys)
        # Modification times tell entries apart only so finely.
        for key in keys[1:3]
            T.store_kernel(key, "kernel", binary)
            sleep(0.05)
        end
        @test kept() == [true, true, true, false]
        # A hit marks its entry used, so the second entry goes first.
        @test T.load_kernel(keys[1]) == ("kernel", binary)
        sleep(0.05)
        T.store_kernel(keys[4], "kernel", binary)
        @test kept() == [true, false, true, true]
        @test T.load_kernel(keys[2]) === nothing
    end
    for (limit, bytes) in (nothing => 1 << 30, "2048" => 2048, "64K" => 64 << 10, "1g" => 1 << 30)
        withenv(() -> @test(T.kernel_cache_limit() == bytes), "TEMPEST_KERNEL_CACHE_LIMIT" => limit)
    end
    withenv(() -> @test_throws(ArgumentError, T.kernel_cache_limit()), "TEMPEST_KERNEL_CACHE_LIMIT" => "lots")
end

using CUDACore, cuTile
if CUDACore.functional()
    @testset "kernel cache" begin
        # Launch parameters are those cuTile's code generation gives the kernel.
        ext = Base.get_extension(Tempest, :cuTileExt)
        y = CuArray{Float32}(undef, 16, 8)
        epilogue = ext.Epilogue(T.generated(:(function (y, arrays, index) y end)), (ext.tile_array(y, (16, 8)),))
        for argument in (ext.tile_array(y, (16, 8)), cuTile.Constant(64), 1.5f0, epilogue, (cuTile.Constant(true), 2.0f0))
            expressions, types = Any[], Any[]
            cuTile.flatten_params!(expressions, types, typeof(argument), :argument)
            @test map(typeof, ext.flatten!(Any[], argument)) == types
        end
        # So are the groups of arrays that overlap.
        whole, part, other = ext.tile_array(y, (16, 8)), cuTile.TileArray(view(y, :, 3:4)), ext.tile_array(CuArray{Float32}(undef, 4), (4,))
        for arguments in ((whole, other), (whole, other, part), (other, epilogue, whole), ())
            @test ext.alias_groups(arguments) == cuTile.alias_groups(arguments...)
        end

        previous = get(ENV, "TEMPEST_KERNEL_CACHE", nothing)
        ENV["TEMPEST_KERNEL_CACHE"] = mktempdir()
        try
            single(x) = Float32.(x)
            w, x, b = single(data(96, 80; scale = 0.1)), single(data(80, 112)), single(data(96))
            q, k, v = single(data(16, 4, 64, 2; scale = 0.1)), single(data(16, 2, 64, 2; scale = 0.2)), single(data(16, 2, 64, 2; scale = 0.3))
            sums = T.GenericFold(init = T -> (zero(T),), combine = (a, b) -> (a[1] + b[1],))
            programs = [
                ("a product with an epilogue", trace((w, x, b) -> gelu.(w * x .+ b), w, x, b), (w, x, b), T.CUDATarget()),
                ("training, TF32", differentiate(trace((w, x, b) -> sum(abs2, gelu.(w * x .+ b)), w, x, b)), (w, x, b),
                    T.CUDATarget(precision = :tf32)),
                ("attention claims", differentiate(trace((q, k, v) -> sum(abs2, T.attention(q, k, v; causal = true)), q, k, v)),
                    (q, k, v), T.CUDATarget(precision = :tf32)),
                ("the floor", trace(x -> T.fold(sums, x; dims = 1), x), (x,), T.CUDATarget()),
            ]
            host(r) = (a = Array(r); ndims(a) == 0 ? a[] : a)
            @testset "$name" for (name, p, args, target) in programs
                results = map(1:2) do pass
                    before = T.cache_counts()
                    ex = prepare(p, map(CuArray, args)...; target)
                    after = T.cache_counts()
                    (after.loaded - before.loaded, after.compiled - before.compiled,
                        map(host, filter(!isnothing, T.leaves(execute!(ex)))), ex.kernels)
                end
                (loaded, compiled, first, kernels), (reloaded, recompiled, second, _) = results
                @test compiled > 0 && recompiled == 0 && reloaded == loaded + compiled
                @test isequal(first, second)
                @test all(isapprox.(first, filter(!isnothing, T.leaves(interpret(p, args...))); rtol = 1.0e-2))
                name == "the floor" && @test all(d -> d isa T.FloorKernels, kernels)
            end
        finally
            previous === nothing ? delete!(ENV, "TEMPEST_KERNEL_CACHE") : (ENV["TEMPEST_KERNEL_CACHE"] = previous)
        end

        # A process that captured launches into a graph exits cleanly: modules
        # are made on the calling thread, so the graph is destroyed first.
        script = """
            using Tempest, CUDACore, cuTile
            x = CUDACore.CuArray(Float32.(reshape(1:4096, 64, 64)))
            ex = prepare(trace(x -> sum(abs2, x * x; dims = 1), x), x; target = Tempest.CUDATarget(precision = :tf32))
            execute!(ex)
            graph = CUDACore.instantiate(CUDACore.capture(() -> foreach(l -> l(), ex.launches)))
            CUDACore.launch(graph)
            CUDACore.synchronize()
            """
        @test success(pipeline(`$(Base.julia_cmd()) --project=$(Base.active_project()) -e $script`; stderr = devnull))
    end
end
