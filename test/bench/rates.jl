# The rates Tempest ships for the device this runs on (see
# `Tempest.Compiler.shipped_rates`): every package's lines and memories and
# every scalar function's price, at each precision, fitted afresh in a tuning
# directory of their own, then written with the device's identity where the
# package keeps them. Run it alone on the GPU, with every kernel package the
# tests load, whenever calibration's code or what `calibrate!` fits changes,
# or for a device Tempest does not know yet:
#   julia --project=test test/bench/rates.jl
# With `--keep`, the rates the device's shipped file holds are kept, and only
# what it lacks is fitted: when calibration fits a rate it did not, the
# others stay as they were, and so do the plans they price.
using Tempest, CUDACore
import cuBLAS, cuDNN, cuTile
const C = Tempest.Compiler

include(joinpath(@__DIR__, "..", "fixtures", "stores.jl"))
run_stores!()
target = Tempest.CUDATarget()
shipped = joinpath(C.shipped_directory(), C.device_file_name(target, ".cold.rates.tsv"))
if "--keep" in ARGS
    isempty(C.shipped_rates(target)) && error("no rates shipped for this device to keep")
    header = ["timing basis"; map(first, C.device_identity(target))]
    kept = filter(line -> !(first(split(line, '\t')) in header), readlines(shipped))
    mkpath(dirname(C.rates_path(target)))
    write(C.rates_path(target), join(kept .* '\n'))
end
foreach(precision -> C.calibrate!(Tempest.CUDATarget(; precision)), (:full, :tf32, :tf32x3))
mkpath(dirname(shipped))
open(shipped, "w") do io
    println(io, "timing basis\t", C.TIMING_BASIS)
    foreach(((key, value),) -> println(io, key, '\t', value), C.device_identity(target))
    write(io, read(C.rates_path(target)))
end
println("wrote ", shipped)
