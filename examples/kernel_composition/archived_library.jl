# Adapter to an existing kernel library. Set TEMPEST_RMS_SOURCE to the archived
# tile-kernels directory. Only these three source files are loaded; its obsolete
# package environment and AD extensions are not needed or changed.
module ArchivedRMSKernels

using CUDACore
import cuTile
import cuTile as ct
using cuTile: Constant, TFloat32, BFloat16

macro cutile(args...)
    esc(:(CUDACore.@cuda backend=$cuTile $(args...)))
end

const source = ENV["TEMPEST_RMS_SOURCE"]
include(joinpath(source, "src", "utils.jl"))
include(joinpath(source, "src", "buffers.jl"))
include(joinpath(source, "src", "norm", "rms_norm.jl"))

end
