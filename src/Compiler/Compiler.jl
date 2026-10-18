"""
How a program becomes kernels: the program, index and plan levels, code
generation, the kernel cache, prepared programs, and the interface kernel
packages implement. Its exports are the levels beneath `compile`: `trace`,
`differentiate` and `interpret` for programs, and `prepare`, `bind!`,
`execute!`, `capture!`, `replay!` and `outputs` for prepared ones.
"""
module Compiler

using LinearAlgebra
using BFloat16s: BFloat16

export trace, differentiate, interpret, prepare, bind!, execute!, capture!, replay!, outputs

# The program level.
include("program/program.jl")
include("program/folds.jl")
include("program/tracing.jl")
include("program/updates.jl")
include("program/interpret.jl")
include("program/calls.jl")
include("program/differentiate.jl")

# The index level.
include("index/expressions.jl")
include("index/graph.jl")
include("index/lower.jl")
include("index/interpret.jl")

# Plans: what is stored, and the kernels.
include("plan/terms.jl")
include("plan/planner.jl")
include("plan/rewrites.jl")
include("plan/products.jl")
include("plan/cpu.jl")
include("plan/explain.jl")
include("plan/storage.jl")

# Code generation and prepared programs.
include("codegen/scalar.jl")
include("runtime/runtime.jl")
include("runtime/decide.jl")
include("runtime/cache.jl")
include("runtime/compile.jl")

end
