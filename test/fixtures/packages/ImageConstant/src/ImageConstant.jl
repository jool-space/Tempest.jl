# A package loaded from its image, as it precompiles, whose function reads a
# constant of its own module: code_changed.jl redefines the constant.
module ImageConstant
const SLOPE = 2.0f0
act(x::Float32) = SLOPE * x
end
