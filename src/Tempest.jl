module Tempest

using LinearAlgebra
using Base.ScopedValues: ScopedValue, with

# Model authors use the exported names. Provider and verb authors extend the
# public names below through qualification or `using Tempest: support, ...`.
export trace,
    differentiate,
    with_update,
    bind_argument!,
    value_and_gradient,
    prepare,
    prepare_forward,
    execute!,
    bind!,
    outputs,
    capture!,
    replay!,
    explain,
    scalar_partials,
    recomputable,
    CPUTarget,
    CUDATarget,
    CPUKernels,
    LinearAlgebraGemm,
    CUDAKernels,
    CuTileKernels,
    CuDNNKernels,
    CuBLASGemm

public Implementation,
    RegionRequest,
    GraphRegion,
    GraphRequest,
    RegionProposal,
    ProposalRequest,
    propose_regions,
    ProposedRegion,
    ProposedRequest,
    standard_numbering,
    InPlaceRequest,
    GroupedRequest,
    CallRequest,
    Supported,
    Unsupported,
    support,
    is_supported,
    build_implementation,
    BuiltImplementation,
    ImplementationPlan,
    WorkspaceSpec,
    default_implementations,
    register!,
    activate,
    deactivate,
    implementation_name,
    implementation_notes,
    TensorSpec,
    output_spec,
    reference,
    verb_pullback,
    record_call,
    ReverseRule,
    reverse_rule,
    batched_mul,
    logsumexp,
    scatter_columns,
    take_rows,
    put_rows,
    dropout,
    attention,
    attention_backward,
    causal_fill,
    profile,
    Fold,
    fold,
    fold_init,
    fold_accumulator,
    fold_lift,
    fold_combine,
    fold_observe,
    fold_pullback,
    MaxFold,
    MinFold,
    ProdFold,
    LogSumExpFold,
    GenericFold,
    fold_resident,
    has_resident_lowering

# Graph IR: index mappings, operations and nodes, the schedule a plan is made
# of, and the reference interpreter.
include("ir/layout.jl")
include("ir/graph.jl")
include("ir/schedule.jl")
include("ir/interpret.jl")

# Array frontend: tracing Julia array code into the graph.
include("frontend/tensor.jl")
include("frontend/broadcast.jl")
include("frontend/operations.jl")
include("frontend/folds.jl")

# Semantic verbs: the call contract and the built-in verbs.
include("verbs/contract.jl")
include("verbs/builtin.jl")

# Program transformations.
include("transforms/reverse.jl")

# Implementations: targets, the provider protocol, and the CPU provider.
include("implementations/targets.jl")
include("implementations/protocol.jl")
include("implementations/registry.jl")
include("implementations/reverse_rules.jl")
include("implementations/cpu.jl")

# Compiler: semantic lowering, graph regions, planning, selection.
include("compiler/lowering.jl")
include("compiler/graph_regions.jl")
include("compiler/proposals.jl")
include("compiler/planning.jl")
include("compiler/selection.jl")

# Preparing and running executables.
include("runtime/hooks.jl")
include("runtime/state.jl")
include("runtime/execution.jl")
include("runtime/preparation.jl")

end
