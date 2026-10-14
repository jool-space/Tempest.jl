module cuDNNExt
using Tempest
using Tempest:
    CallRequest,
    Supported,
    Unsupported,
    BuiltImplementation,
    ImplementationPlan,
    WorkspaceSpec,
    TensorSpec,
    ReverseRule,
    TracedArray,
    record_call
import Tempest: support, build_implementation, reverse_rule, output_spec, reference
using CUDACore
import cuDNN
using cuDNN: Graph, tensor!, scalar!, sdpa_fwd!, sdpa_bwd!

include("launch.jl")
include("attention.jl")

end
