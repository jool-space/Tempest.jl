# The cover on CUDA: claims timed on buffers of their own, choices measured and
# kept in the tuning record, a search that tries the variants of the option
# that measures best, and the implementations a target names.
include("../fixtures/programs.jl")

# Claims are timed on buffers of their own, which a kernel reads as data:
# an integer array is an index there, and zero is outside every axis. The
# buffers hold integers spread over the axes they index, so that timing
# every option of a gather (the floor too, with bounds checked as under
# `Pkg.test`) reads in bounds, and a gather or a scatter reads or writes
# as many places as it has indices: the arguments here are indices, and
# no two of an argument's entries are alike.
@testset "timed on trial buffers, a gather reads in bounds" begin
    @testset "$name" for (name, f, args) in filter(((_, _, args),) -> any(a -> a isa AbstractArray{<:Integer}, args), PROGRAMS)
        for p in (trace(f, args...), differentiate(trace(f, args...)))
            graph = C.lower(p)
            planned = C.plan(graph)
            d = C.Decider(T.CUDATarget(), graph; measure = false)
            foreach(g -> C.buffers!(d, planned, g), planned.groups)
            indices = [b for (v, b) in zip(graph.values, d.buffers) if b !== nothing && v.eltype <: Integer && !(v.definition isa C.ConstantArray)]
            @test !isempty(indices) && all(b -> all(>=(1), Array(b)), indices)
            held = [d.buffers[a] for a in graph.arguments if graph.values[a].eltype <: Integer && d.buffers[a] !== nothing]
            @test !isempty(held) && all(b -> allunique(Array(b)), held)
            reference = filter(!isnothing, C.leaves(interpret(p, args...)))
            ex = prepare(p, map(x -> x isa AbstractArray ? CuArray(x) : x, args)...; target = T.CUDATarget(), measure = :all)
            @test agree(filter(!isnothing, C.leaves(outputs(execute!(ex)))), reference)
        end
    end
end

@testset "choosing between claims" begin
    # Half-precision products cuBLAS claims and cuTile claims under each
    # of its tiles, every option measured at preparation and kept in the
    # tuning record.
    a, b = Float16.(data(1024, 256; scale = 0.1)), Float16.(data(256, 1024))
    p = trace((a, b) -> a * b, a, b)
    target = T.CUDATarget()
    prepared() = prepare(p, CuArray(a), CuArray(b); target, measure = :all)
    ex = prepared()
    x = only(ex.decisions)
    @test x.how === :measured && length(x.options) >= 2
    @test length(unique(C.package_of(o.implementation) for o in x.options)) == 2
    @test occursin("chosen by measurement", sprint(explain, ex))
    path = C.record_path(target)
    lines = readlines(path)
    @test length(lines) >= 2
    # Given the record, preparing again measures nothing and chooses alike.
    again = prepared()
    @test readlines(path) == lines && only(again.decisions).choice.key == x.choice.key
    # A measurement overrides the estimate only beyond the noise margin.
    rewrite(f) = write(path, join((string(k, '\t', f(k, parse(Float64, t))) for (k, t) in split.(lines, '\t')), '\n') * '\n')
    other = first(o for o in x.options if o.key != x.choice.key)
    rewrite((k, t) -> k == other.key ? 1.0e-7 : 1.0e-5)
    @test only(prepared().decisions).choice.key == other.key
    rewrite((k, t) -> 1.0e-5)
    @test only(prepared().decisions).choice.key == first(x.options).key
    @test isapprox(Array(only(C.leaves(outputs(execute!(prepared()))))), interpret(p, a, b); rtol = 2 * eps(Float16))
end

@testset "the search is staged" begin
    # The generator claims a product under every output tile at the
    # default depth, and the cover tries the variants of the one that
    # measures best: its other depths, and its sum split the other way. A
    # close-call search tries fewer tiles than measuring every option
    # does, and chooses alike each time from the record. A product with
    # an epilogue, on a target of cuTile alone, at full precision:
    # depths 16 and 32, and a grid of few output tiles, so that the sum
    # splits.
    a, b, c = single(data(256, 4096; scale = 0.1)), single(data(4096, 256)), single(data(256))
    p = trace((a, b, c) -> gelu.(a * b .+ c), a, b, c)
    target = T.CUDATarget(cuTile)
    dir = mktempdir()
    rates = C.rates_path(target)
    isfile(rates) && (mkpath(joinpath(dir, "tuning")); cp(rates, joinpath(dir, "tuning", basename(rates))))
    ext = Base.get_extension(Tempest, :cuTileExt)
    tile(o) = (s = o.claim.schedule; (s.tiling[1].tile, s.tiling[3].tile, s.tiling[2].tile, s.splits))
    withenv("TEMPEST_CACHE" => dir) do
        prepared(measure) = only(x for x in prepare(p, CuArray(a), CuArray(b), CuArray(c); target, measure).decisions
            if x !== nothing && x.choice.implementation isa ext.cuTileGenerator && x.choice.claim.schedule.sketch === :product)
        staged = prepared(:close)
        tiles = map(tile, staged.options)
        @test staged.how === :measured
        # Every output tile at depth 16, the default, and the variants of
        # the best: at least its other depth and its split, fewer than all.
        @test length(unique(t[1:2] for t in tiles if t[3] == 16)) == 4
        @test length(tiles) >= 6
        again = prepared(:close)
        @test again.choice.key == staged.choice.key && length(again.options) == length(staged.options)
        # Every option: the four output tiles at both depths, split and
        # not, but the tile of 32 output blocks, which does not split.
        every = prepared(:all)
        @test length(staged.options) < length(every.options) == 14
        @test all(t -> t[1:3] in [(tm, tn, tk) for (tm, tn) in ((64, 64), (128, 64), (64, 128), (32, 64)) for tk in (16, 32)],
            map(tile, every.options))
    end
end

@testset "a matrix-vector product searches its step along the sum" begin
    # A matrix stored along its sum times one column is the streamed
    # reduction of the sum, and the cover tries its longer steps. Rows
    # enough for 64 blocks, so that the sum streams rather than splits.
    w, x = single(data(512, 4096; scale = 0.1)), single(data(512, 1))
    p = trace((w, x) -> Einops.@einsum(w, x, "k m, k n -> m n"), w, x)
    target = T.CUDATarget(cuTile)
    dir = mktempdir()
    rates = C.rates_path(target)
    isfile(rates) && (mkpath(joinpath(dir, "tuning")); cp(rates, joinpath(dir, "tuning", basename(rates))))
    ext = Base.get_extension(Tempest, :cuTileExt)
    step(o) = (s = o.claim.schedule; s.tiling[ext.sweep_axis(s)].tile)
    withenv("TEMPEST_CACHE" => dir) do
        ex = prepare(p, CuArray(w), CuArray(x); target)
        d = only(d for d in ex.decisions if d !== nothing && d.choice.implementation isa ext.cuTileGenerator)
        @test sort([step(o) for o in d.options if o.implementation isa ext.cuTileGenerator]) == [32, 64, 128]
        @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, w, x)]; rtol = 1.0e-4)
    end
end

@testset "implementation sets" begin
    w, x, b = Float32.(data(96, 80; scale = 0.1)), Float32.(data(80, 112)), Float32.(data(96))
    p = trace((w, x, b) -> gelu.(w * x .+ b), w, x, b)
    # Loading a package makes its implementations available; a target
    # names the ones it uses, and the floor is always there.
    @test C.used_packages(T.CUDATarget()) == [cuBLAS, cuTile]
    ex = prepare(p, CuArray(w), CuArray(x), CuArray(b); target = T.CUDATarget(cuTile))
    @test ex.packages == [cuTile]
    @test occursin("implementations from cuTile, and the floor", sprint(explain, ex))
    @test_throws ArgumentError prepare(p, CuArray(w), CuArray(x), CuArray(b); target = T.CUDATarget(Test))
end
