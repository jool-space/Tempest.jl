# Run with: julia --project=examples examples/research/cutile_streaming_abi_probe.jl
# Research diagnostic for cuTile 1.0.0: streamed loads from tuple kernel arguments
# can produce invalid Tile IR. A flat entry wrapper can reconstruct those tuples
# inside the kernel. This checks the workaround without requiring the bug to stay.
module CuTileStreamingABIProbe

using CUDACore
import cuTile as ct
using Random
using Test

@inline function center_columns!(outputs, inputs, rows::Int32, tile::Int)
    column = ct.bid(1)
    blocks = cld(rows, Int32(tile))
    accumulated = zeros(Float32, tile)
    for block = Int32(1):blocks
        values = reshape(ct.load(inputs[1], (block, column), (tile, 1);
                                 padding_mode = ct.PaddingMode.Zero), (tile,))
        offset = ct.Tile((block - Int32(1)) * Int32(tile))
        indices = ct.broadcast_to(offset, (tile,)) .+ ct.arange(tile)
        accumulated = accumulated .+ ifelse.(indices .<= rows, values, 0f0)
    end
    mean = sum(accumulated) / Float32(rows)
    for block = Int32(1):blocks
        values = ct.load(inputs[1], (block, column), (tile, 1);
                         padding_mode = ct.PaddingMode.Zero)
        ct.store(outputs[1], (block, column), values .- mean)
    end
    return
end

# The mathematical body and tuple access remain identical. Only the kernel entry
# changes: each array is now a separate parameter instead of a field of a tuple.
function flat_entry!(output, input, rows, tile)
    center_columns!((output,), (input,), rows, tile)
    return
end

function check_result(output, expected)
    actual = Array(output)
    @test all(isfinite, actual)
    @test actual ≈ expected rtol=2f-5 atol=2f-6
    maximum(abs.(actual .- expected))
end

function run_case(rows)
    host = randn(MersenneTwister(rows), Float32, rows, 7)
    expected = host .- sum(host; dims = 1) ./ rows
    input = CuArray(host)
    output = similar(input)
    tile = ct.Constant(256)
    device_input, device_output = ct.cuTileconvert(input), ct.cuTileconvert(output)

    # Catch compilation only. If a newer compiler accepts the tuple entry, its
    # numerical result must meet the same checks as the flat entry.
    tuple_kernel = try
        CUDACore.@cuda backend=ct launch=false center_columns!(
            (output,), (input,), Int32(rows), tile,
        )
    catch exception
        println("Tuple entry did not compile for $rows rows:")
        showerror(stdout, exception)
        println()
        nothing
    end
    if tuple_kernel !== nothing
        tuple_kernel((device_output,), (device_input,), Int32(rows), tile; blocks=7)
        CUDACore.synchronize()
        println((entry = :tuple, rows, maximum_error = check_result(output, expected)))
    end

    flat_kernel = CUDACore.@cuda backend=ct launch=false flat_entry!(
        output, input, Int32(rows), tile,
    )
    flat_kernel(device_output, device_input, Int32(rows), tile; blocks=7)
    CUDACore.synchronize()
    println((entry = :flat, rows, maximum_error = check_result(output, expected)))
end

function run_probe()
    if !CUDACore.functional()
        println("Skipping cuTile ABI probe: CUDACore.functional() is false.")
        return
    end
    println((julia = VERSION, cuTile = pkgversion(ct), source = pathof(ct)))
    @testset "cuTile streamed argument ABI" begin
        for rows in (259, 8193)
            run_case(rows)
        end
    end
end

end # module

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    CuTileStreamingABIProbe.run_probe()
end
