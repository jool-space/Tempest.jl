module Tempest

using LinearAlgebra

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
    CuBLASGemm

public Implementation,
    RegionRequest,
    GraphRegion,
    GraphRequest,
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
    implementation_name,
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
    fold_lift,
    fold_combine,
    fold_observe,
    fold_pullback,
    MaxFold,
    MinFold,
    LogSumExpFold,
    GenericFold,
    fold_resident,
    has_resident_lowering

include("layout.jl")
include("graph_ir.jl")
include("tensor.jl")
include("broadcast.jl")
include("operations.jl")
include("verbs.jl")
include("primitives.jl")
include("reverse.jl")
include("reference.jl")
include("folds.jl")
include("planning.jl")
include("runtime.jl")
include("implementations.jl")
include("graph_regions.jl")
include("reverse_rules.jl")
include("semantic_lowering.jl")
include("cpu.jl")
include("state.jl")
include("execution.jl")
include("selection.jl")
include("preparation.jl")

end
