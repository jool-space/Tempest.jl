# Reproducer for scalar FMA formation inside a tile broadcast. The derivative
# of x + x^3 is ordinary Julia, but cuTile may reject its generated FMA types.
using CUDACore
import cuTile as ct

function derivative_kernel(out, x, tile::Int)
    values = ct.load(x, (ct.bid(1),), (tile,); padding_mode = ct.PaddingMode.Zero)
    result = (v -> 1 + 3v * v).(values)
    ct.store(out, (ct.bid(1),), result)
    return
end

function main()
    CUDACore.functional() || error("this probe requires a functional CUDA device")
    x = CuArray(Float32.(1:16))
    out = similar(x)
    println("cuTile version: ", Base.pkgversion(ct))
    CUDACore.@cuda backend=ct blocks=1 derivative_kernel(out, x, ct.Constant(16))
    @assert Array(out) ≈ 1 .+ 3 .* Array(x) .^ 2
    println("Scalar derivative compiled and matched its reference")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
