module cuDNNExt
using Tempest
using Tempest:
    CallRequest,
    ProposalRequest,
    ProposedRequest,
    RegionProposal,
    Supported,
    Unsupported,
    BuiltImplementation,
    ImplementationPlan,
    WorkspaceSpec,
    TensorSpec,
    ReverseRule,
    TracedArray,
    record_call,
    standard_numbering,
    MatMul,
    Pointwise,
    Constant,
    FoldReduction,
    SumFold
import Tempest: support, build_implementation, reverse_rule, output_spec, reference
using CUDACore
using LinearAlgebra: Transpose
import cuDNN
using cuDNN: Graph, tensor!, scalar!, sdpa_fwd!, sdpa_bwd!

include("launch.jl")
include("attention.jl")
include("matmul.jl")

end
