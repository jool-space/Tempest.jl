# The kernel cache on CUDA: a kernel compiles once per key and loads from disk
# after, launched with the parameters cuTile's code generation gives it, and a
# captured graph is destroyed before the modules it launches are unloaded.
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
        child = `$(Base.julia_cmd()) --project=$(Base.active_project()) $(joinpath(@__DIR__, "..", "fixtures", "kernel_cache.jl"))`
        output = IOBuffer()
        process = run(pipeline(ignorestatus(addenv(child, "JULIA_OBJCACHE_PATH" => store)); stdout = output, stderr = output))
        success(process) || println(stderr, String(take!(output)))
        @test success(process)
    end

    # Captured graphs are destroyed before the modules they launch are
    # unloaded: graphs that outlive their programs and are collected
    # after them, and a graph alive when the process exits.
    script = """
        using Tempest, CUDACore, cuTile
        using Tempest.Compiler: trace, differentiate, interpret, prepare, bind!, execute!, capture!, replay!, outputs
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
