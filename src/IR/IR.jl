"""
The vocabulary the compiler's levels share and implementations read: index
expressions over a domain, how values are laid out in storage, the algebra of
folds, and the terms of a kernel's body.
"""
module IR

using BFloat16s: BFloat16

include("expressions.jl")
include("layout.jl")
include("folds.jl")
include("terms.jl")

end
