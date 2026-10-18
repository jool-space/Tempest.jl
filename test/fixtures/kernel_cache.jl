# The kernel cache from an empty store: cache.jl runs this in a child process
# with `JULIA_OBJCACHE_PATH` naming a fresh directory, since a process opens its
# object cache once. Each program compiles its kernels when first prepared and
# loads them all when prepared again, with the same results. The second pass
# reads what the first wrote, which the store before Julia 1.14 commits at once.
using Tempest, Test, NNlib, CUDACore
import cuBLAS, cuTile
include(joinpath(@__DIR__, "..", "setup.jl"))

w, x, b = single(data(96, 80; scale = 0.1)), single(data(80, 112)), single(data(96))
q, k, v = single(data(16, 4, 64, 2; scale = 0.1)), single(data(16, 2, 64, 2; scale = 0.2)), single(data(16, 2, 64, 2; scale = 0.3))
sums = T.GenericFold(init = T -> (zero(T),), combine = (a, b) -> (a[1] + b[1],))
programs = [
    ("a product with an epilogue", trace((w, x, b) -> gelu.(w * x .+ b), w, x, b), (w, x, b), T.CUDATarget()),
    ("training, TF32", differentiate(trace((w, x, b) -> sum(abs2, gelu.(w * x .+ b)), w, x, b)), (w, x, b),
        T.CUDATarget(precision = :tf32)),
    ("attention claims", differentiate(trace((q, k, v) -> sum(abs2, Ops.attention(q, k, v; causal = true)), q, k, v)),
        (q, k, v), T.CUDATarget(precision = :tf32)),
    # Float64 elements, which tiles do not take.
    ("the floor", trace(x -> T.fold(sums, x; dims = 1), data(80, 112)), (data(80, 112),), T.CUDATarget()),
]
@testset "$name" for (name, p, args, target) in programs
    results = map(1:2) do pass
        before = C.cache_counts()
        ex = prepare(p, map(CuArray, args)...; target)
        after = C.cache_counts()
        (after.loaded - before.loaded, after.compiled - before.compiled,
            map(on_host, filter(!isnothing, C.leaves(outputs(execute!(ex))))), ex.kernels)
    end
    (loaded, compiled, first, kernels), (reloaded, recompiled, second, _) = results
    # The first pass may also compile the side of a close call it measures and
    # does not keep.
    @test compiled > 0 && recompiled == 0 && 0 < reloaded <= loaded + compiled
    @test isequal(first, second)
    @test all(isapprox.(first, filter(!isnothing, C.leaves(interpret(p, args...))); rtol = 1.0e-2))
    name == "the floor" && @test all(d -> d isa C.FloorKernels, kernels)
end
