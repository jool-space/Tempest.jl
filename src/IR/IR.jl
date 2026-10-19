"""
The vocabulary the compiler's levels share and implementations read: index
expressions over a domain, the algebra of folds, and the terms of a kernel's
body.
"""
module IR

using BFloat16s: BFloat16

include("expressions.jl")
include("folds.jl")
include("terms.jl")

end
