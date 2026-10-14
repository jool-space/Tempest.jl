include("../fixtures/frontend.jl")

@testset "CUDA array frontend, mapped broadcasts, and axis reductions" begin
    arr(dims, s) = reshape(s .* sin.(Float32.(1:prod(dims))), dims)
    x = arr((3, 5, 2), 0.4f0)
    cases = (
        ((x, b) -> sum(abs2, x .+ b), (x, arr((1, 5), 0.1f0))),
        ((x, b) -> sum(abs2, x .* b), (x, arr((3,), 0.2f0))),
        ((x, b) -> sum(abs2, x .+ b'), (arr((3, 5), 0.4f0), arr((5,), 0.2f0))),
        (x -> sum(abs2, sum(x; dims = 1)), (x,)),
        (x -> sum(abs2, sum(x; dims = 2)), (x,)),
        (x -> sum(abs2, sum(x; dims = (1, 3))), (x,)),
        (x -> sum(abs2, sum(x; dims = (1, 2, 3))), (x,)),
        (x -> sum(abs2, sum(x; dims = 4)), (x,)),
        (x -> sum(sin.(reshape(reshape(x, 6, 5)', 3, 10))), (x,)),
        (x -> sum(abs2, dropdims(sum(x; dims = (1, 3)); dims = (1, 3))), (x,)),
        (
            (x, b, c) -> sum(@. tanh(x + b + c) ^ 2),
            (arr((3, 5), 0.4f0), arr((3,), 0.2f0), arr((1, 5), 0.1f0)),
        ),
        ((x, z) -> sum(abs2, x .* z) / z[], (x, fill(1.2f0))),
        (
            (x, b, c) -> sum(@. x * b * c),
            (arr((3, 5), 0.4f0), arr((3,), 0.2f0), arr((1, 5), 0.1f0)),
        ),
        (
            (x, b, c) -> sum(triple.(x, b, c)),
            (arr((3, 5), 0.4f0), arr((3,), 0.2f0), arr((1, 5), 0.1f0)),
        ),
        (let power=3
            x -> sum(x .^ power)
        end, (x,)),
    )
    for (f, args) in cases
        for reuse in (true, false)
            ex = compare_prepared(
                f,
                args...;
                target = CUDATarget(),
                reuse,
                atol = 2e-5,
                rtol = 2e-5,
            )
            capture!(ex)
            for _ = 1:2
                replay!(ex)
                @test Array(outputs(ex)[1])[] ≈ f(args...) atol=2e-5 rtol=2e-5
            end
        end
    end
    # Mask after the mapped function: exp(0) is nonzero in padded lanes.
    for rows in (1, 2, 255, 256, 257)
        xx = arr((rows, 5), 0.4f0)
        ex = compare_prepared(
            x -> sum(sum(exp.(0 * x); dims = 1)),
            xx;
            target = CUDATarget(),
            atol = 2e-4,
            rtol = 2e-5,
        )
        axes_steps = findall(s -> s.region isa Tempest.AxisReductionRegion, ex.storage.steps)
        expected = Base.get_extension(Tempest, :cuTileExt) === nothing ? :cuda : :cutile
        @test all(i -> ex.providers[i] === expected, axes_steps)
    end
    # The axis reduction traverses multiple tiles per output for a long dimension.
    for dims in ((257, 7), (7, 257))
        xx = arr(dims, 0.4f0)
        for axis in (1, 2)
            compare_prepared(
                x -> sum(abs2, sum(x; dims = axis)),
                xx;
                target = CUDATarget(),
                atol = 2e-4,
                rtol = 2e-5,
            )
        end
    end
    for dims in ((3, 5), (257, 7))
        args = (arr(dims, 0.4f0), fill(1.1f0, dims[1]), arr((dims[1],), 0.1f0))
        ex = compare_prepared(
            norm_loss,
            args...;
            target = CUDATarget(),
            atol = 5e-4,
            rtol = 2e-4,
        )
        @test :cuda in ex.providers
        capture!(ex)
        pointers = map(pointer, ex.buffers)
        for scale in (1.0f0, 0.3f0, -0.8f0)
            updated = (scale * args[1], args[2], args[3])
            bind!(ex, updated...)
            got = replay!(ex)
            ref = value_and_gradient(ex.storage.program, updated...)
            @test Array(got[1])[] ≈ ref[1] atol=5e-4 rtol=2e-4
            for (g, r) in zip(got[2:end], ref[2])
                @test Array(g) ≈ r atol=5e-4 rtol=2e-4
            end
            @test map(pointer, ex.buffers) == pointers
        end
    end
end

@testset "Captured runtime scalar and zero-dimensional array inputs" begin
    f(x::AbstractMatrix, scale::Real) = sum(abs2, tanh.(x * scale)) / scale
    x = reshape(Float32.(range(-0.5, 0.6; length = 35)), 5, 7)
    ex = compare_prepared(f, x, 1.3f0; target = CUDATarget(), atol = 2e-5, rtol = 2e-5)
    capture!(ex)
    for scale in (0.5f0, 1.3f0, 2.1f0)
        bind!(ex, x, scale)
        got = replay!(ex)
        ref = value_and_gradient(ex.storage.program, x, scale)
        @test Array(got[1])[] ≈ ref[1] atol=2e-5 rtol=2e-5
        @test Array(got[2]) ≈ ref[2][1] atol=2e-5 rtol=2e-5
        @test Array(got[3])[] ≈ ref[2][2] atol=2e-5 rtol=2e-5
    end
    f0(x, z::AbstractArray{T, 0}) where {T} = sum(abs2, x .* z) / z[]
    ex0 = compare_prepared(f0, x, fill(1.2f0); target = CUDATarget(), atol = 2e-5, rtol = 2e-5)
    capture!(ex0)
    for scale in (0.5f0, 1.3f0, 2.1f0)
        bind!(ex0, CuArray(x), CuArray(fill(scale)))
        got = replay!(ex0)
        ref = value_and_gradient(ex0.storage.program, x, fill(scale))
        @test Array(got[1])[] ≈ ref[1] atol=2e-5 rtol=2e-5
        @test Array(got[3]) ≈ ref[2][2] atol=2e-5 rtol=2e-5
    end
    CUDACore.synchronize()
end
