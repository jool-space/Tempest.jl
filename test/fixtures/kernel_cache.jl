# The kernel cache from an empty store: cuda/cache.jl runs this in a child
# process with `JULIA_OBJCACHE_PATH` naming a fresh directory, since a process
# opens its object cache once. Each program compiles its kernels when first
# prepared and loads them all when prepared again, with the same results.
# Prepared again in the same session it loads nothing, since the session keeps
# what it resolved: the second pass forgets the kernels the generator emitted
# and the functions it loaded (`EMITTED`, `FUNCTIONS`), so that each kernel is
# emitted again and resolved by its content. The first and last passes forget
# what the session keeps (`forget_session!`), as a new process starts without
# it: the last reads what the first wrote (the store before Julia 1.14 commits
# at once), and checks each kernel by what its compilation used.
using Tempest, Test, NNlib, CUDACore
import cuBLAS, cuTile
include(joinpath(@__DIR__, "..", "setup.jl"))
X = Base.get_extension(Tempest, :cuTileExt)

w, x, b = single(data(96, 80; scale = 0.1)), single(data(80, 112)), single(data(96))
q, k, v = single(data(16, 4, 64, 2; scale = 0.1)), single(data(16, 2, 64, 2; scale = 0.2)), single(data(16, 2, 64, 2; scale = 0.3))
sums = T.GenericFold(init = T -> (zero(T),), combine = (a, b) -> (a[1] + b[1],))
programs = [
    ("a product with an epilogue", trace((w, x, b) -> gelu.(w * x .+ b), w, x, b), (w, x, b), T.CUDATarget()),
    ("training, TF32", differentiate(trace((w, x, b) -> sum(abs2, gelu.(w * x .+ b)), w, x, b)), (w, x, b),
        T.CUDATarget(precision = :tf32)),
    ("attention training step", differentiate(trace((q, k, v) -> sum(abs2, Ops.attention(q, k, v; causal = true)), q, k, v)),
        (q, k, v), T.CUDATarget(precision = :tf32)),
    # Float64 elements, which tiles do not take.
    ("the floor", trace(x -> reduce(sums, x; dims = 1), data(80, 112)), (data(80, 112),), T.CUDATarget()),
]
@testset "$name" for (name, p, args, target) in programs
    results = map(1:3) do pass
        pass in (1, 3) && C.forget_session!()
        pass == 2 && (empty!(X.EMITTED); empty!(X.FUNCTIONS))
        before = C.cache_counts()
        ex = prepare(p, map(CuArray, args)...; target)
        after = C.cache_counts()
        (after.loaded - before.loaded, after.compiled - before.compiled,
            map(on_host, filter(!isnothing, C.leaves(outputs(execute!(ex))))), ex.kernels)
    end
    (loaded, compiled, first, kernels), (resident, _, middle, _), (reloaded, recompiled, second, _) = results
    # The first pass may also compile the side of a close call it measures and
    # does not keep.
    # Before Julia 1.12 nothing loads from the store.
    if C.EDGES
        @test compiled > 0 && recompiled == 0 && 0 < reloaded <= loaded + compiled
    else
        @test compiled > 0 && reloaded == 0
    end
    @test resident == 0 && isequal(first, middle)
    @test isequal(first, second)
    @test all(isapprox.(first, filter(!isnothing, C.leaves(interpret(p, args...))); rtol = 1.0e-2))
    name == "the floor" && @test all(d -> d isa C.FloorKernels, kernels)
end

# A kernel that failed to compile fails again, with the compiler's message,
# while what its inference used holds; not once it does not, nor where that
# could not be read. Before Julia 1.12 none is kept.
@testset "a failure kept" begin
    key = C.content_key("a kernel that fails")
    @test C.load_kernel(key) === nothing
    C.store_failure(key, "the compiler's message"; dependencies = C.Dependency[])
    C.EDGES ? (@test_throws "the compiler's message" C.load_kernel(key)) : (@test C.load_kernel(key) === nothing)
    changed = C.content_key("a kernel that fails with code changed since")
    C.store_failure(changed, "the compiler's message"; dependencies = [C.Dependency(:call, nothing, Tuple{typeof(sin), Float32}, "other code")])
    @test C.load_kernel(changed) === nothing
    unread = C.content_key("a kernel whose inference failed")
    C.store_failure(unread, "the compiler's message"; dependencies = nothing)
    @test C.load_kernel(unread) === nothing
end
