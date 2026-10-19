"""
How a program becomes kernels: the program, index and plan levels, code
generation, the kernel cache, and prepared programs. Its exports are the levels
beneath `compile`: `trace`, `differentiate` and `interpret` for programs, and
`prepare`, `bind!`, `execute!`, `capture!`, `replay!` and `outputs` for prepared
ones. Its other public names are the language programs are written in, which
`Tempest` makes its own, and what implementations read of a plan beyond
`Tempest.IR`.
"""
module Compiler

using LinearAlgebra
using BFloat16s: BFloat16
using Republic: @republic

# The vocabulary, and the arithmetic that builds expressions in it.
@republic republic=false inherit=:public using ..IR
using ..IR: constant, var, add, scale, normalize, fdiv, fmod, substitute, evaluate_index, affine_key,
    fold_array, fold_name, stats, stored_reads
import ..IR: mentions

# The protocol of implementations, which the cover calls and the generator implements.
@republic republic=false inherit=:public import ..Implementations
using ..Implementations: REGISTERED

export trace, differentiate, interpret, prepare, bind!, execute!, capture!, replay!, outputs

# What proposals read, until a view of the graph replaces it: the plan, and the
# view of each of its groups.
public Plan, group_view

# The program level.
include("program/program.jl")
include("program/tracing.jl")
include("program/updates.jl")
include("program/interpret.jl")
include("program/calls.jl")
include("program/differentiate.jl")

# The index level.
include("index/graph.jl")
include("index/lower.jl")
include("index/interpret.jl")

# Plans: what is stored, and the kernels.
include("plan/planner.jl")
include("plan/rewrites.jl")
include("plan/cpu.jl")
include("plan/explain.jl")
include("plan/storage.jl")

# Code generation and prepared programs.
include("codegen/scalar.jl")
include("runtime/runtime.jl")
include("runtime/view.jl")
include("runtime/decide.jl")
include("runtime/cache.jl")
include("runtime/compile.jl")

end
