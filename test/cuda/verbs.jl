include("../fixtures/verbs.jl")

struct WaveCUDA <: Implementation end
Tempest.implementation_name(::WaveCUDA) = :wave_cuda

function Tempest.support(::WaveCUDA, r::CallRequest)
    if !(r.call.f in (pair_wave, wave_back))
        return Unsupported("not a wave operation")
    end
    if !(r.target isa CUDATarget)
        return Unsupported("CUDA only")
    end
    if !(all(s -> s.dtype === Float32, r.inputs))
        return Unsupported("Float32 only")
    end
    Supported()
end

function wave_forward_kernel(s, c, x, n)
    i = (CUDACore.blockIdx().x - 1) * CUDACore.blockDim().x + CUDACore.threadIdx().x
    if i <= n
        s[i], c[i] = sincos(x[i])
    end
    nothing
end

function wave_backward_kernel(dx, gs, gc, s, c, n)
    i = (CUDACore.blockIdx().x - 1) * CUDACore.blockDim().x + CUDACore.threadIdx().x
    if i <= n
        dx[i] = gs[i] * c[i] - gc[i] * s[i]
    end
    nothing
end

# These are ordinary destination-writing launch wrappers. Preparing them compiles
# the kernel and binds actual device arrays, before capture.
function wave_forward_call(out, x::CuArray)
    n = length(x)
    kernel = CUDACore.@cuda launch=false wave_forward_kernel(out.s, out.c, x, n)
    () -> begin
        if n != 0
            kernel(out.s, out.c, x, n; threads = 256, blocks = cld(n, 256))
        end
        nothing
    end
end

function wave_backward_call(out::CuArray, gs, gc, s, c)
    n = length(out)
    kernel = CUDACore.@cuda launch=false wave_backward_kernel(out, gs, gc, s, c, n)
    () -> begin
        if n != 0
            kernel(out, gs, gc, s, c, n; threads = 256, blocks = cld(n, 256))
        end
        nothing
    end
end

function Tempest.build_implementation(::WaveCUDA, r::CallRequest, runtime, out, inputs)
    launch = if r.call.f === pair_wave
        wave_forward_call(out, only(inputs))
    else
        wave_backward_call(out, inputs...)
    end
    BuiltImplementation(launch, (out, inputs), 0)
end

@testset "Custom semantic CUDA kernels in training" begin
    x = reshape(Float32.(range(-0.5, 0.7; length = 513)), 27, 19)
    function loss(x)
        y = pair_wave(x)
        sum(abs2, y.s .+ 2.0f0 .* y.c)
    end
    for reuse in (false, true)
        ex = prepare(loss, CuArray(x); reuse, implementations = (WaveCUDA(),))
        @test count(==(:wave_cuda), ex.providers) == 2
        expected = execute!(prepare(loss, x))
        @test all(
            isapprox(Array(a), b; rtol = 2e-5, atol = 2e-5) for
            (a, b) in zip(execute!(ex), expected)
        )
        capture!(ex)
        for scale in (0.7f0, 1.3f0)
            bind!(ex, scale * x)
            GC.gc()
            actual = replay!(ex)
            expected = execute!(prepare(loss, scale * x))
            @test all(
                isapprox(Array(a), b; rtol = 2e-5, atol = 2e-5) for
                (a, b) in zip(actual, expected)
            )
        end
    end
    function f(x)
        y = pair_wave(view(x, 1:2:27, :))
        (; y, alias = y.s, total = sum(y.c))
    end
    ex = prepare_forward(f, CuArray(x); implementations = (WaveCUDA(),))
    @test count(==(:wave_cuda), ex.providers) == 1
    result = execute!(ex)
    @test result.alias === result.y.s
    @test Array(result.y.s) ≈ sin.(view(x, 1:2:27, :))
    capture!(ex)
    bind!(ex, 2x)
    @test Array(replay!(ex).y.c) ≈ cos.(view(2x, 1:2:27, :))
    ex = prepare_forward(
        x -> pair_wave(transpose(x)),
        CuArray(x);
        implementations = (WaveCUDA(),),
    )
    @test Array(execute!(ex).s) ≈ sin.(transpose(x))
    # Unused sibling outputs still have distinct destination storage.
    ex = prepare_forward(x -> pair_wave(x).s, CuArray(x); implementations = (WaveCUDA(),))
    @test Array(execute!(ex)) ≈ sin.(x)
    s = only(ex.storage.steps)
    @test length(unique(ex.storage.slots[i] for i in s.outputs)) == 2
end

@testset "Built-in verbs on CUDA" begin
    target = CUDATarget()
    for (ta, tb) in ((false, false), (true, false), (false, true), (true, true))
        a = randn(Float32, ta ? (3, 4, 5) : (4, 3, 5))
        b = randn(Float32, tb ? (6, 3, 5) : (3, 6, 5))
        f(a, b) = sum(abs2, Tempest.batched_mul(a, b; transpose_a = ta, transpose_b = tb))
        ex = compare_prepared(f, a, b; target, atol = 1e-3, rtol = 1e-4)
        @test occursin("call batched_mul [cuBLAS]", sprint(explain, ex))
    end
    x = 30 .* randn(Float32, 70, 33)
    ex = compare_prepared(x -> sum(abs2, Tempest.logsumexp(x)), x; target, atol = 1e-1, rtol = 1e-4)
    @test occursin("call logsumexp [cuda]", sprint(explain, ex))
    compare_prepared(x -> sum(abs2, exp.(x .- Tempest.logsumexp(x))), x; target, atol = 1e-3, rtol = 1e-4)
    # Masked columns: -Inf entries contribute nothing; the gradient stays finite.
    mask = Float32[k <= q ? 0 : -Inf for k = 1:8, q = 1:8]
    y = randn(Float32, 8, 8)
    ex = compare_prepared((y, m) -> sum(abs2, exp.((y .+ m) .- Tempest.logsumexp(y .+ m))), y, mask; target, atol = 1e-4, rtol = 1e-4)
    @test all(isfinite, Array(outputs(ex)[2]))
    # Other axes fall back to the traced reference, a stable fold.
    ex = prepare_forward(x -> Tempest.logsumexp(x; dims = 2), CuArray(x))
    @test Array(execute!(ex)) ≈ Tempest.logsumexp(x; dims = 2) rtol = 1e-5
    @test occursin("expanded reference", sprint(explain, ex))
end

@testset "Runtime integer indices on CUDA" begin
    for (rows, columns, shape) in ((5, 7, (2, 3)), (300, 147, (64, 16)), (40, 1000, (600,)))
        table = reshape(Float32.(sin.(1:(rows * columns))), rows, columns)
        indices = Int32.(mod1.(1:prod(shape), columns) .* 7 .% columns .+ 1)
        indices = reshape(indices, shape)
        f(t, i) = sum(abs2, sin.(t[:, i]))
        d = differentiate(trace(f, table, indices); wrt = (1,))
        expected = value_and_gradient(d, table, indices)
        ex = prepare(d, table, indices; target = CUDATarget())
        got = execute!(ex)
        @test Array(got[1])[] ≈ expected[1] rtol = 1e-4
        @test Array(got[2]) ≈ expected[2][1] rtol = 1e-4 atol = 1e-5
        @test Array(execute!(ex)[2]) == Array(execute!(ex)[2])  # deterministic accumulation
        x = reshape(Float32.(cos.(1:(columns * prod(shape)))), columns, shape...)
        g(x, r) = sum(abs2, Tempest.logsumexp(x) .- Tempest.take_rows(x, r))
        q = differentiate(trace(g, x, indices); wrt = (1,))
        expected = value_and_gradient(q, x, indices)
        got = execute!(prepare(q, x, indices; target = CUDATarget()))
        @test Array(got[1])[] ≈ expected[1] rtol = 1e-4
        @test Array(got[2]) ≈ expected[2][1] rtol = 1e-4 atol = 1e-6
    end
    # Out-of-range indices read as NaN on the device instead of faulting.
    ex = prepare_forward((t, i) -> t[:, i], ones(Float32, 2, 3), Int32[1, 4]; target = CUDATarget())
    got = Array(execute!(ex))
    @test got[:, 1] == [1, 1] && all(isnan, got[:, 2])
end

@testset "Dropout on CUDA matches the CPU mask" begin
    x = reshape(Float32.(sin.(1:(64 * 129))), 64, 129)
    seed = Int32[11]
    f(x, s) = sum(abs2, Tempest.dropout(tanh.(x), s; rate = 0.3, stream = 2))
    d = differentiate(trace(f, x, seed); wrt = (1,))
    cpu = execute!(prepare(d, x, seed))
    gpu = execute!(prepare(d, x, seed; target = CUDATarget()))
    @test Array(gpu[1])[] ≈ cpu[1][] rtol = 1e-5
    @test Array(gpu[2]) ≈ cpu[2] rtol = 1e-5
    # A captured step advances an integer seed held as donated state.
    step = with_update(d) do loss, gradients, x, s
        (x .- 0.1f0 .* only(gradients), s .+ Int32(1), loss)
    end
    ex = prepare(step, x, seed; target = CUDATarget(), donate = (1, 2))
    capture!(ex)
    losses = [Array(replay!(ex)[3])[] for _ = 1:3]
    @test Array(outputs(ex)[2]) == Int32[14]
    @test allunique(losses)
end
