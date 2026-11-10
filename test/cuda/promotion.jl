# Numbers of several types in a map on cuTile's tiles compute as Julia
# computes them, in their promotion: a number wider than a tile of its kind
# widens the tile, as the program's interpreter and the CPU's kernels do, and
# a narrow one leaves it as it is.
const BF = BFloat16s.BFloat16

const PROMOTED = [
    ("a BFloat16 array times a Float32 number, plus an integer", w -> 1 .+ w .* 1.0f0, (BF.(data(256; scale = 0.3)),)),
    ("a Float16 array plus a Float32 number", w -> w .+ 1.0f-3, (Float16.(data(256; scale = 0.3)),)),
    ("a BFloat16 array times a Float32 array", (w, x) -> w .* x .+ 1, (BF.(data(256; scale = 0.3)), Float32.(data(256; scale = 0.7)))),
    ("an Int32 array plus an Int64 number", a -> a .+ 2^40, (Int32.(round.(100 .* data(256))),)),
    ("a Float32 array times a BFloat16 number", x -> x .* BF(0.3) .+ BF(1), (Float32.(data(256; scale = 0.3)),)),
]

@testset "numbers of several types in a map on cuTile" begin
    @testset "$name" for (name, f, args) in PROMOTED
        p = trace(f, args...)
        ex = prepare(p, map(CuArray, args)...; target = T.CUDATarget(T.cuTileGenerator()))
        @test all(d -> !(d isa C.FloorKernels), ex.kernels)
        # Within a rounding of Float32's, which contracts a product and a sum
        # where the host rounds each, near zero too: a rounding to BFloat16
        # or Float16 is a thousand times that.
        @test all(isapprox.(Array(outputs(execute!(ex))), f(args...); rtol = 1.0e-6, atol = 1.0e-6))
    end
end
