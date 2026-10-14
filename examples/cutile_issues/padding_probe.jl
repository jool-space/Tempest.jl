# Run: julia --project=examples examples/cutile_issues/padding_probe.jl
# Reproduce an eager cuTile mapreduce tail issue without changing its package.
# This is a diagnostic: it prints matches and mismatches rather than expecting
# an upstream bug to remain present in future releases.
using CUDACore
import cuTile

function padding_probe()
    CUDACore.functional() || error("This probe requires a functional CUDA device")
    println((julia = VERSION, cutile = Base.pkgversion(cuTile), source = pathof(cuTile)))
    for rows in (8, 259)
        input = zeros(Float32, rows, 7)
        expected = mapreduce(exp, +, input; dims = 1)
        result = mapreduce(exp, +, cuTile.Tiled(CuArray(input)); dims = 1)
        actual = Array(result)
        println((; rows, expected = vec(expected), actual = vec(actual),
                   correct = actual == expected))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    padding_probe()
end
