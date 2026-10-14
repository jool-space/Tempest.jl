# Optional standalone reproducers for toolchain issues encountered during the
# first experiment. Not correctness tests for Tempest and not expected to fail
# forever. Run after changing cuTile/compiler versions to revisit workarounds.
using CUDACore
import cuTile as ct

function indexed_read(out, a, n::Int32, tile::Int)
    base = ct.Tile((ct.bid(1) - Int32(1)) * Int32(tile))
    indices = ct.broadcast_to(base, (tile,)) .+ ct.arange(tile)
    data = ct.gather(a, indices; mask = indices .<= n, padding_value = 0.0f0)
    ct.store(out, (ct.bid(1),), data)
    return
end

function mixed_shape_fma(out, a, tile::Int)
    x = ct.load(a, (ct.bid(1),), (tile,); padding_mode = ct.PaddingMode.Zero)
    y = (z -> one(z) - tanh(z) ^ 2).(x)
    ct.store(out, (ct.bid(1),), y)
    return
end

x=CuArray(Float32.(1:133));
y=CuArray{Float32}(undef, 256)
CUDACore.@cuda backend=ct blocks=1 indexed_read(y, x, Int32(133), ct.Constant(256))
got=Array(y)
println("Indexed read correct: ", got[1:133] == Float32.(1:133))
println("First lanes: ", got[1:5], "; expected ", Float32.(1:5))

try
    CUDACore.@cuda backend=ct blocks=1 mixed_shape_fma(y, x, ct.Constant(256))
    CUDACore.synchronize()
    println("Mixed-shape FMA compiled; compare numerics before removing workaround.")
catch err
    showerror(stdout, err)
    println()
end
