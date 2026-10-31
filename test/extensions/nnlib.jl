# NNlib's functions whose bodies do not trace are methods on Tempest's traced
# arrays, written in its primitives: they mean what the same computations
# spelled in primitives mean, differentiate alike, and plan as they do.
using NNlib

results(ex) = filter(!isnothing, C.leaves(outputs(execute!(ex))))
gradients(p, args...) = filter(!isnothing, C.leaves(interpret(differentiate(p), args...)))
batched(a, b) = C.contract(a, b, (1, 2, 4), (2, 3, 4), (1, 3, 4))

const NNLIB = [
    ("softmax", (x, t) -> sum(NNlib.softmax(x; dims = 1) .* t),
     (x, t) -> sum(exp.(x .- T.logsumexp(x; dims = 1)) .* t), (data(8, 6), data(8, 6; scale = 0.7))),
    ("softmax over rows", (x, t) -> sum(NNlib.softmax(x; dims = 2) .* t),
     (x, t) -> sum(exp.(x .- T.logsumexp(x; dims = 2)) .* t), (data(8, 6), data(8, 6; scale = 0.7))),
    ("logsoftmax", (x, t) -> sum(NNlib.logsoftmax(x) .* t),
     (x, t) -> sum((x .- T.logsumexp(x; dims = 1)) .* t), (data(8, 6), data(8, 6; scale = 0.7))),
    ("softplus", x -> sum(NNlib.softplus.(x)),
     x -> sum(max.(x, 0) .+ log1p.(exp.(.-abs.(x)))), (data(8, 6; scale = 3.0),)),
    ("logσ and mish", x -> sum(NNlib.logσ.(x) .+ NNlib.mish.(x)),
     x -> sum(.-(max.(.-x, 0) .+ log1p.(exp.(.-abs.(x)))) .+ x .* tanh.(max.(x, 0) .+ log1p.(exp.(.-abs.(x))))), (data(8, 6; scale = 3.0),)),
    ("batched_mul", (a, b) -> sum(abs2, NNlib.batched_mul(a, b)),
     (a, b) -> sum(abs2, batched(a, b)), (data(3, 4, 2), data(4, 5, 2; scale = 0.3))),
    ("batched_mul of transposes", (a, b) -> sum(abs2, NNlib.batched_mul(NNlib.batched_transpose(b), NNlib.batched_adjoint(a))),
     (a, b) -> sum(abs2, batched(permutedims(b, (2, 1, 3)), permutedims(a, (2, 1, 3)))), (data(3, 4, 2), data(4, 5, 2; scale = 0.3))),
    ("batched_mul sharing a batch", (a, b) -> sum(abs2, NNlib.batched_mul(a, b)),
     (a, b) -> sum(abs2, batched(a .* ones(1, 1, 2), b)), (data(3, 4, 1), data(4, 5, 2; scale = 0.3))),
]

@testset "NNlib on traced arrays" begin
    @testset "$name" for (name, f, spelled, args) in NNLIB
        p = trace(f, args...)
        @test interpret(p, args...) ≈ f(args...)
        @test all(map(≈, gradients(p, args...), gradients(trace(spelled, args...), args...)))
        for program in (p, differentiate(p))
            @test agree(results(prepare(program, args...)), filter(!isnothing, C.leaves(interpret(program, args...))))
        end
    end
    # A traced factor beside a Julia array, in either order, is a product with
    # a constant.
    a, b = data(3, 4, 2), data(4, 5, 2; scale = 0.3)
    @test interpret(trace(x -> NNlib.batched_mul(x, b), a), a) ≈ NNlib.batched_mul(a, b)
    @test interpret(trace(x -> NNlib.batched_mul(a, x), b), b) ≈ NNlib.batched_mul(a, b)
    # Inside a gradient, as NNlib's `within_gradient` says of Zygote's.
    x = data(4)
    @test interpret(trace(x -> NNlib.within_gradient(x) ? 2 .* x : x, x), x) == x
    @test interpret(trace(x -> T.gradient(x -> sum(NNlib.within_gradient(x) ? 2 .* x : x), x), x), x)[2] == fill(2.0, 4)
    @test interpret(trace(x -> T.gradient(x -> sum(x), x)[1] + sum(NNlib.within_gradient(x) ? 2 .* x : x), x), x) ≈ 2 * sum(x)
    # A traced softplus's derivative at zero is one half, as NNlib's is.
    @test interpret(differentiate(trace(x -> sum(NNlib.softplus.(x)), [0.0])), [0.0])[2] == [0.5]
    # The methods on traced arrays add no ambiguity.
    for f in (NNlib.softmax, NNlib.logsoftmax, NNlib.batched_mul, NNlib.softplus)
        ms = collect(methods(f))
        @test !any(Base.isambiguous(m, n) for m in ms for n in ms if m !== n)
    end
end

# Pooling and convolution, against NNlib's loops on Julia arrays: the values,
# and the gradients NNlib's own backward functions give.
const POOLS = [
    ("max 2×2", NNlib.maxpool, NNlib.∇maxpool, (6, 6, 2, 2), (2, 2), (;), (;)),
    ("max 3×3, stride 2, padded", NNlib.maxpool, NNlib.∇maxpool, (7, 6, 2, 2), (3, 3), (; stride = 2, padding = 1), (;)),
    ("max 2, ceil mode", NNlib.maxpool, NNlib.∇maxpool, (7, 3, 2), (2,), (; ceil_mode = true), (;)),
    ("mean 2×2", NNlib.meanpool, NNlib.∇meanpool, (6, 6, 2, 2), (2, 2), (;), (;)),
    ("mean 3×3, stride 2, padded", NNlib.meanpool, NNlib.∇meanpool, (7, 6, 2, 2), (3, 3), (; stride = 2, padding = 1), (;)),
    ("mean, padding uncounted", NNlib.meanpool, NNlib.∇meanpool, (7, 6, 2, 2), (3, 3), (; stride = 2, padding = 1),
     (; count_include_pad = false)),
    ("mean over everything", NNlib.meanpool, NNlib.∇meanpool, (6, 5, 3, 2), (6, 5), (;), (;)),
]

const CONVS = [
    ("3×3", (;), (6, 6, 2, 2), (3, 3, 2, 4)),
    ("3×3, stride 2, padded", (; stride = 2, padding = 1), (7, 6, 2, 2), (3, 3, 2, 4)),
    ("3×3, dilated", (; dilation = 2), (8, 7, 2, 1), (3, 3, 2, 3)),
    ("3×3, not flipped", (; flipkernel = true), (6, 6, 2, 2), (3, 3, 2, 4)),
    ("3×3, two groups", (; groups = 2, padding = 1), (6, 6, 4, 2), (3, 3, 2, 6)),
    ("1-D", (; padding = (2, 0)), (10, 2, 2), (3, 2, 4)),
]

@testset "pooling and convolution as NNlib computes them" begin
    @testset "pool $name" for (name, pool, backward, dims, window, options, keywords) in POOLS
        x = data(dims...; scale = 2.0)
        pdims = NNlib.PoolDims(x, window; options...)
        y = pool(x, pdims; keywords...)
        t = data(size(y)...; scale = 0.5)
        @test interpret(trace(x -> pool(x, pdims; keywords...), x), x) ≈ y
        @test interpret(differentiate(trace(x -> sum(pool(x, pdims; keywords...) .* t), x)), x)[2] ≈ backward(t, y, x, pdims; keywords...)
    end
    @testset "convolution $name" for (name, options, xdims, wdims) in CONVS
        x, w = data(xdims...), data(wdims...; scale = 0.5)
        cdims = NNlib.DenseConvDims(x, w; options...)
        y = NNlib.conv(x, w, cdims)
        t = data(size(y)...; scale = 0.5)
        @test interpret(trace((x, w) -> NNlib.conv(x, w, cdims), x, w), x, w) ≈ y
        _, dx, dw = interpret(differentiate(trace((x, w) -> sum(NNlib.conv(x, w, cdims) .* t), x, w)), x, w)
        @test dx ≈ NNlib.∇conv_data(t, w, cdims) && dw ≈ NNlib.∇conv_filter(x, t, cdims)
        # The transposed convolution, the data's gradient as a function of the
        # output's: its own gradients are a convolution and the filter's.
        u = data(size(x)...; scale = 0.5)
        @test interpret(trace((t, w) -> NNlib.∇conv_data(t, w, cdims), t, w), t, w) ≈ NNlib.∇conv_data(t, w, cdims)
        _, dt, dw = interpret(differentiate(trace((t, w) -> sum(NNlib.∇conv_data(t, w, cdims) .* u), t, w)), t, w)
        @test dt ≈ NNlib.conv(u, w, cdims) && dw ≈ NNlib.∇conv_filter(u, t, cdims)
    end
    @testset "depthwise convolution" begin
        x, w = data(6, 6, 2, 2), data(3, 3, 2, 2; scale = 0.5)
        cdims = NNlib.DepthwiseConvDims(x, w; padding = 1)
        y = NNlib.depthwiseconv(x, w, cdims)
        t = data(size(y)...; scale = 0.5)
        @test interpret(trace((x, w) -> NNlib.depthwiseconv(x, w, cdims), x, w), x, w) ≈ y
        _, dx, dw = interpret(differentiate(trace((x, w) -> sum(NNlib.depthwiseconv(x, w, cdims) .* t), x, w)), x, w)
        @test dx ≈ NNlib.∇depthwiseconv_data(t, w, cdims) && dw ≈ NNlib.∇depthwiseconv_filter(x, t, cdims)
    end
    @testset "linear upsampling, corners $(corners ? "" : "not ")aligned" for corners in (true, false)
        x = data(4, 3, 2, 2)
        y = NNlib.upsample_bilinear(x; size = (7, 6), align_corners = corners)
        t = data(size(y)...; scale = 0.5)
        @test interpret(trace(x -> NNlib.upsample_bilinear(x; size = (7, 6), align_corners = corners), x), x) ≈ y
        g = interpret(differentiate(trace(x -> sum(NNlib.upsample_bilinear(x; size = (7, 6), align_corners = corners) .* t), x)), x)[2]
        @test g ≈ NNlib.∇upsample_bilinear(t; size = (4, 3), align_corners = corners)
        @test interpret(trace(x -> NNlib.upsample_linear(x, 2), data(5, 2, 1)), data(5, 2, 1)) ≈ NNlib.upsample_linear(data(5, 2, 1), 2)
    end
    for f in (NNlib.maxpool, NNlib.meanpool, NNlib.conv, NNlib.depthwiseconv, NNlib.∇conv_data, NNlib.upsample_linear)
        ms = collect(methods(f))
        @test !any(Base.isambiguous(m, n) for m in ms for n in ms if m !== n)
    end
end

if CUDA_FUNCTIONAL
    close(a, b; rtol) = all(map((x, y) -> isapprox(ndims(x) == 0 ? Array(x)[] : Array(x), y; rtol), a, b))

    @testset "NNlib on CUDA" begin
        @testset "$name" for (name, f, spelled, args) in NNLIB
            args = map(a -> Float32.(a), args)
            ex = prepare(trace(f, args...), map(CuArray, args)...)
            @test close(C.leaves(outputs(execute!(ex))), [f(args...)]; rtol = 1.0e-4)
        end
        # A convolution and a pooling of it run as kernels too.
        x, w = Float32.(data(8, 8, 2, 2)), Float32.(data(3, 3, 2, 4; scale = 0.5))
        cdims, pdims = NNlib.DenseConvDims(x, w; padding = 1), NNlib.PoolDims((8, 8, 4, 2), (2, 2))
        layer = (x, w) -> NNlib.maxpool(NNlib.conv(x, w, cdims), pdims)
        ex = prepare(trace(layer, x, w), CuArray(x), CuArray(w))
        @test close(C.leaves(outputs(execute!(ex))), [layer(x, w)]; rtol = 1.0e-4)
        # Attention written with NNlib's softmax is one generated kernel, as
        # it is with the log-sum-exp.
        scores(q, k) = NNlib.batched_mul(NNlib.batched_transpose(k), q) .* (1 / sqrt(Float32(size(q, 1))))
        spellings = [(q, k, v) -> NNlib.batched_mul(v, NNlib.softmax(scores(q, k); dims = 1)),
            (q, k, v) -> (s = scores(q, k); NNlib.batched_mul(v, exp.(s .- T.logsumexp(s; dims = 1))))]
        q, k, v = (Float32.(data(16, 200, 3; scale)) for scale in (0.1, 0.2, 0.3))
        for f in spellings
            p = trace(f, q, k, v)
            ex = prepare(p, map(CuArray, (q, k, v))...; target = T.CUDATarget(T.cuTileGenerator()), recompute = C.recomputable_folds(C.rewrite(C.lower(p))))
            kernel = only(ex.kernels)
            @test !(kernel isa C.FloorKernels) && occursin("online softmax", string(kernel))
            @test close(C.leaves(outputs(execute!(ex))), [f(q, k, v)]; rtol = 1.0e-4)
        end
    end
end
