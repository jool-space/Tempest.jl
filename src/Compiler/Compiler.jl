"""
How a program becomes kernels: the program, index and plan levels, code
generation, the kernel cache, and prepared programs. Its exports are the levels
beneath `compile`: `trace`, `differentiate` and `interpret` for programs, and
`prepare`, `bind!`, `execute!`, `capture!`, `replay!` and `outputs` for prepared
ones. Its other public names are what implementations read of a plan (see
`Tempest.Implementations`).
"""
module Compiler

using LinearAlgebra
using BFloat16s: BFloat16

# The protocol of implementations, which the cover calls.
using ..Implementations: Implementations, REGISTERED
import ..Implementations: Implementation, Work, register!, claims, estimate, build_claim, candidates,
    record_name, recomputes_folds, in_place, claims_call, epilogue_results, build_call

export trace, differentiate, interpret, prepare, bind!, execute!, capture!, replay!, outputs

# What implementations read: a plan, its groups and their terms, and the
# addresses of reads over a group's domain.
public Plan, Group, Domain, Load, Apply, Recomputed, Cast, SumFold, members, subterms, group_bytes,
    address, is_affine, coefficient, group_stride, placement, product_axes

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
