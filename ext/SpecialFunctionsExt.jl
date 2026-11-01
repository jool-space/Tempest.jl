# SpecialFunctions' error functions branch on their argument, among
# polynomials by its size, so their bodies do not trace: each is a primitive,
# as `tanh` is, one map that runs SpecialFunctions' own method on every element
# (on a GPU too: Tile IR has no error function, and cuTile compiles the
# polynomials), with the Gaussian for its derivative. `gelu_erf` and other
# functions written with them trace through.
module SpecialFunctionsExt

using SpecialFunctions: SpecialFunctions, erf, erfc
import Tempest: scalar_partials
import Tempest.Compiler: Traced, record_map

SpecialFunctions.erf(x::Traced) = record_map(erf, x)
SpecialFunctions.erfc(x::Traced) = record_map(erfc, x)

# The derivative of `erf`, 2/√π e^(-x²), as `erfc`'s is its negation.
gaussian(y::Traced{T}, x) where {T} = T(2 / sqrt(π)) * exp(-(x * x))
scalar_partials(::typeof(erf), y, x) = (gaussian(y, x),)
scalar_partials(::typeof(erfc), y, x) = (-gaussian(y, x),)

end
