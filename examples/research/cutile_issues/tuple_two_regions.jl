# cuTile 1.0.0: a kernel that reads an array passed inside a tuple argument from
# two sibling regions (here two loops) fails in tileiras with
#
#   'cuda_tile.make_tensor_view' op operand #0 must be 0D tile of Pointer type
#   values, but got '!cuda_tile.tile<f32>'
#
# The same kernel with separate array parameters compiles. Prints `true` when
# the kernel compiles and its result is correct.
# Run: julia --project=examples examples/research/cutile_issues/tuple_two_regions.jl
using CUDACore
import cuTile as ct

function center_columns(arrays, blocks::Int32)
    column = ct.bid(1)
    total = zeros(Float32, (16, 1))
    for block in Int32(1):blocks
        total = total .+ ct.load(arrays[2], (block, column), (16, 1))
    end
    mean = sum(total) / Float32(16 * blocks)
    for block in Int32(1):blocks
        values = ct.load(arrays[2], (block, column), (16, 1))
        ct.store(arrays[1], (block, column), values .- mean)
    end
    return
end

input = CuArray(rand(Float32, 64, 8))
output = CuArray(zeros(Float32, 64, 8))
println((cuTile = pkgversion(ct), source = pathof(ct)))
CUDACore.@cuda backend=ct blocks=8 center_columns((output, input), Int32(4))
host = Array(input)
println(Array(output) ≈ host .- sum(host; dims = 1) ./ 64)
