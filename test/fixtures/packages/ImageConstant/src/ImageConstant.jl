# A package loaded from its image, as it precompiles, whose functions read
# constants of its own module: code_changed.jl redefines `SLOPE`, and sets
# `CONFIG`'s field, mutable state a kernel's code reads, as session.jl does.
module ImageConstant
const SLOPE = 2.0f0
act(x::Float32) = SLOPE * x
mutable struct Config
    slope::Float32
end
const CONFIG = Config(2.0f0)
configured(x::Float32) = CONFIG.slope * x
end
