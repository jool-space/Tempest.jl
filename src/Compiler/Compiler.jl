"""
How a program becomes kernels: the program, index and plan levels, the
cover, code generation, the kernel cache, and prepared programs. Its public
names are what packages outside it need: beneath `compile`, a program traced
(`trace`) and prepared (`prepare`, `bind!`, `execute!`, `capture!`, `replay!`
and `outputs`), for a step run in place; the language programs are written
in, which `Tempest` makes its own, and what an extension writes traced methods
with (`TracedArray`, `Traced`, `contract`); what implementations read of a
plan beyond `Tempest.IR`, and what they build kernels with (the kernel cache,
`generated` and `concurrently`); and the protocol a package that adds a
target defines (see `Target`).
"""
module Compiler

using LinearAlgebra
using BFloat16s: BFloat16
using Republic: @republic

# The vocabulary, and the arithmetic that builds expressions in it.
@republic republic=false inherit=:public using ..IR
using ..IR: constant, var, add, scale, mul, normalize, fdiv, fmod, evaluate_index, affine_key, isconstant,
    wide_axes, fold_array, fold_name, fold_eltype, law_function, statistic_name, stats, stored_reads
import ..IR: mentions

# The protocol of implementations, which the cover calls and the generator implements.
@republic republic=false inherit=:public import ..Implement
using ..Implement: REGISTERED

# Beneath `compile`: a program traced and prepared, for a step run in place.
public trace, prepare, bind!, execute!, capture!, replay!, outputs

# What proposals read, until a view of the graph replaces it: the plan, and the
# views of its groups; and how kernels are built side by side.
public Plan, group_views, concurrently

# The program level.
include("program/program.jl")
include("program/tracing.jl")
include("program/updates.jl")
include("program/interpret.jl")
include("program/differentiate.jl")

# The index level.
include("index/graph.jl")
include("index/lower.jl")
include("index/interpret.jl")

# Plans: what is stored, and the kernels.
include("plan/planner.jl")
include("plan/rewrites.jl")
include("plan/partition.jl")
include("plan/masks.jl")
include("plan/cpu.jl")
include("plan/explain.jl")
include("plan/storage.jl")

# Code generation and prepared programs.
include("codegen/scalar.jl")
include("runtime/target.jl")
include("runtime/runtime.jl")

# The cover: which groups there are, and how each runs.
include("cover/view.jl")
include("cover/cost.jl")
include("cover/choose.jl")
include("cover/cover.jl")
include("cover/calibrate.jl")
include("cover/record.jl")

# The kernel cache, and the front door.
include("runtime/cache.jl")
include("runtime/compile.jl")

end
