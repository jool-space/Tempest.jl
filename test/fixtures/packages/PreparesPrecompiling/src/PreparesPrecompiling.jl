# A package that prepares a program of its own as it precompiles, twice, as a
# package using Tempest may in its precompile workload, outside a store of its
# own: runtime/cache.jl precompiles it with a store named, which keeps nothing.
module PreparesPrecompiling
using Tempest
squares(x) = sum(abs2, x .+ 1)
let x = Float32[1, 2, 3], program = Tempest.Compiler.trace(squares, x)
    Tempest.Compiler.prepare(program, x; build = false)
    Tempest.Compiler.prepare(program, x; build = false)
end
end
