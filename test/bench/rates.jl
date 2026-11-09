# The rates Tempest ships for the device this runs on (see
# `Tempest.Compiler.shipped_rates`): every package's lines and memories and
# every scalar function's price, at each precision, fitted afresh in a tuning
# directory of their own, then written with the device's identity where the
# package keeps them. Run it alone on the GPU, with every kernel package the
# tests load, whenever calibration's code or what `calibrate!` fits changes,
# or for a device Tempest does not know yet:
#   julia --project=test test/bench/rates.jl
using Tempest, CUDACore
import cuBLAS, cuDNN, cuTile
const C = Tempest.Compiler

ENV["TEMPEST_CACHE"] = mktempdir()
target = Tempest.CUDATarget()
foreach(precision -> C.calibrate!(Tempest.CUDATarget(; precision)), (:full, :tf32, :tf32x3))
shipped = joinpath(C.shipped_directory(), basename(C.rates_path(target)))
mkpath(dirname(shipped))
open(shipped, "w") do io
    println(io, "timing basis\t", C.TIMING_BASIS)
    foreach(((key, value),) -> println(io, key, '\t', value), C.device_identity(target))
    write(io, read(C.rates_path(target)))
end
println("wrote ", shipped)
