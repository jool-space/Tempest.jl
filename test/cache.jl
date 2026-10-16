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
    end
end
