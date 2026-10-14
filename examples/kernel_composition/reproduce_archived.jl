# Direct eager launch, without Tempest or an adapter. This isolates compatibility
# of the archived kernel with the installed cuTile compiler.
# julia --project=examples examples/kernel_composition/reproduce_archived.jl /path/to/tile-kernels
using CUDACore

length(ARGS) == 1 || error("pass the archived tile-kernels directory")
ENV["TEMPEST_RMS_SOURCE"] = abspath(only(ARGS))
include("archived_library.jl")
CUDACore.functional() || error("CUDA is unavailable")

x = CuArray(reshape(sin.(Float32.(1:35)), 7, 5))
weight = CUDACore.ones(Float32, 7)
y = similar(x)
checkpoints = (; Rstd = similar(weight, 5))
CUDACore.@sync ArchivedRMSKernels.rms_norm!(y, x, weight; eps = 0.03f0, checkpoints)
println(Array(y))
