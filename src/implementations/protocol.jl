# An implementation realizes a scheduled region. This is deliberately separate
# from a semantic operation and from the runtime's allocation policy.
"""
    Implementation

Supertype for implementations of semantic calls or scheduled regions. Define
`support` and `build_implementation`; candidates are tried in preparation order.
"""
abstract type Implementation end

abstract type Support end

"""A metadata check accepted a request. Construction may still return `Unsupported`."""
struct Supported <: Support end
"""A deliberate rejection with a printable reason; permits trying the next candidate."""
struct Unsupported <: Support
    reason::String
end

abstract type ImplementationRequest end

"""
Metadata for a scheduled region of type `R`; contains no example array contents.
Inspect `request.step.region` or specialize on `RegionRequest{<:Elementwise}`.
Reduction layouts and scatter selections belong to the region itself.
"""
struct RegionRequest{R <: Region} <: ImplementationRequest
    target::ExecutionTarget
    step::Step{R}
    dtype::DataType
    dims::Tuple
    input_types::Tuple
    input_dims::Tuple
end

"""
An elementwise request whose output aliases the specified input positions.
The planner proves same-lane reads and that all other old-state reads have ended.
Implementations must explicitly accept this request to permit in-place execution.
"""
struct InPlaceRequest{R <: RegionRequest} <: ImplementationRequest
    region::R
    aliased_inputs::Tuple{Vararg{Int}}
end

function build_implementation(
    impl::Implementation,
    request::InPlaceRequest,
    runtime,
    out,
    arrays,
)
    build_implementation(impl, request.region, runtime, out, arrays)
end

"""A semantic call's target and logical input/output contract, before storage planning."""
struct CallRequest <: ImplementationRequest
    target::ExecutionTarget
    call::SemanticCall
    inputs::Tuple
end

"""
    support(implementation, request) -> Supported() | Unsupported(reason)

Check a region's operations, types, shapes and layouts without launching it.
Supported means eligible for construction, not that a vendor engine must exist.
Construction can still return Unsupported. Unexpected errors must propagate.
"""
function support(::Implementation, request::ImplementationRequest)
    Unsupported("no support method for $(nameof(typeof(request)))")
end

"""
    is_supported(implementation, request) -> Bool

Check metadata eligibility through `support`. This does not build a kernel or
prove vendor engine availability. Invalid support results and exceptions propagate.
"""
function is_supported(impl::Implementation, request::ImplementationRequest)
    checked_support(impl, request) isa Supported
end

"""Return an implementation's diagnostic name, defaulting to its type name as a Symbol."""
implementation_name(impl::Implementation) = Symbol(nameof(typeof(impl)))

"""
A prepared, zero-argument launch with retained resource owners. workspace_bytes
reports owned device workspace (not vendor descriptor/driver metadata).
"""
struct BuiltImplementation{F, O}
    launch::F
    owners::O
    workspace_bytes::Int
end

function (b::BuiltImplementation)()
    GC.@preserve b begin
        b.launch()
    end
end

"""
    build_implementation(impl, request, runtime, out, inputs)

Return BuiltImplementation, ImplementationPlan, or Unsupported(reason). Called outside capture with
final bound arrays. For CallRequest, out has the complete declared result tree;
for RegionRequest it is one output buffer. Compile resources here, never during launch.
Declare shareable temporary buffers with ImplementationPlan; private persistent
resources remain owned by BuiltImplementation.
The implementation must write out, read only declared inputs, and retain owners
of every pointer used by launch/capture. Input contents are not specialization.
"""
function build_implementation end

"""
    WorkspaceSpec(dtype, count)

Request a temporary, flat buffer of at least `count` elements. Workspaces may be
reused by later launches; initialize what the launch reads and never retain data
in them across launches. Requests within one implementation do not alias.
"""
struct WorkspaceSpec
    dtype::DataType
    count::Int

    function WorkspaceSpec(dtype::Type, count::Integer)
        if !isconcretetype(dtype) || !isbitstype(dtype)
            throw(ArgumentError("workspace elements must have a concrete bits type"))
        end
        if count < 0
            throw(ArgumentError("workspace count must be nonnegative"))
        end
        new(dtype, Int(count))
    end
end

"""
    ImplementationPlan(bind, workspace::Tuple)

A successful construction that needs shared workspace before its launch can be
bound. `workspace` contains WorkspaceSpecs. After selection, Tempest assigns
stable buffers and calls `bind(buffers::Tuple)` once, outside capture. It must
return a BuiltImplementation; capability rejection belongs in support/build,
not in binding. Buffers may be larger than requested.

Return BuiltImplementation directly when no shared workspace is needed.
"""
struct ImplementationPlan{F}
    bind::F
    workspace::Tuple{Vararg{WorkspaceSpec}}
end

"""Generated scalar loops on CPU; matrix products use LinearAlgebraGemm."""
struct CPUKernels <: Implementation end

"""CPU matrix multiplication through LinearAlgebra.mul!."""
struct LinearAlgebraGemm <: Implementation end

"""CUDA scalar, indexing, scatter, and reduction kernels; load CUDACore and cuBLAS."""
struct CUDAKernels <: Implementation end

"""Tiled scalar and eligible reduction kernels; load CUDACore and cuTile."""
struct CuTileKernels <: Implementation end

"""CUDA matrix multiplication through an owned cuBLAS handle and workspace."""
struct CuBLASGemm <: Implementation end

"""
    CuDNNKernels(; float32_cast = nothing)

Semantic verbs realized as cuDNN graphs; load CUDACore and cuDNN. Currently
scaled dot-product attention, forward and backward. cuDNN's attention engines
take Float16 or BFloat16 operands. Float32 attention is rejected unless
`float32_cast` names one of those types. Then q, k, v and the output cotangent
are rounded to it, scores, softmax statistics and accumulation stay Float32, and
results return as Float32. The rounding is opt-in, like TF32 through
`CUDATarget(matmul_precision = :tf32)`, and `explain` reports it.
"""
struct CuDNNKernels <: Implementation
    float32_cast::Union{Nothing, DataType}

    function CuDNNKernels(; float32_cast = nothing)
        if !(float32_cast === nothing || float32_cast in (Float16, Core.BFloat16))
            throw(ArgumentError("float32_cast must be nothing, Float16 or BFloat16"))
        end
        new(float32_cast)
    end
end

implementation_name(::CPUKernels) = :cpu
implementation_name(::LinearAlgebraGemm) = :LinearAlgebra
implementation_name(::CUDAKernels) = :cuda
implementation_name(::CuTileKernels) = :cutile
implementation_name(::CuBLASGemm) = :cuBLAS
implementation_name(::CuDNNKernels) = :cuDNN

"""
    default_implementations(target) -> Tuple

Ordered candidates appended after the user's implementations during preparation.
Resolve this list once per executable. Unknown targets have no defaults.
"""
default_implementations(::ExecutionTarget) = ()
default_implementations(::CPUTarget) = (CPUKernels(), LinearAlgebraGemm())

# Apply the same contract checks during semantic eligibility and construction.
function checked_support(implementation, request)
    verdict = support(implementation, request)
    if !(verdict isa Union{Supported, Unsupported})
        error("support must return Supported or Unsupported")
    end
    verdict
end

function choose_implementation(candidates, request, runtime, out, arrays, notes)
    for candidate in candidates
        name = implementation_name(candidate)
        verdict = checked_support(candidate, request)
        if verdict isa Unsupported
            push!(notes, "$name: $(verdict.reason)")
            continue
        end
        built = build_implementation(candidate, request, runtime, out, arrays)
        if built isa Unsupported
            push!(notes, "$name: $(built.reason)")
            continue
        end
        if built isa BuiltImplementation
            check_built(built)
        elseif !(built isa ImplementationPlan)
            error(
                "build_implementation must return BuiltImplementation, ImplementationPlan, or Unsupported",
            )
        end
        return (name, built)
    end
    nothing
end

function check_built(built)
    if !(built isa BuiltImplementation)
        error("workspace binding must return BuiltImplementation")
    end
    if built.workspace_bytes < 0
        error("workspace_bytes must be nonnegative")
    end
    built
end

# Temporary workspace lives for one scheduled launch. Greedily reuse a slot of
# the same element type across launches, growing its capacity before allocation.
# Two requests made by one launch must receive distinct slots.
function plan_workspaces(chosen)
    specifications = WorkspaceSpec[]
    last_step = Int[]
    bindings = Vector{Int}[]

    for (index, (_, construction)) in enumerate(chosen)
        slots = Int[]
        requests = construction isa ImplementationPlan ? construction.workspace : ()
        for request in requests
            slot = findfirst(eachindex(specifications)) do candidate
                specifications[candidate].dtype === request.dtype &&
                    last_step[candidate] < index
            end
            if slot === nothing
                push!(specifications, request)
                push!(last_step, index)
                slot = length(specifications)
            else
                capacity = max(specifications[slot].count, request.count)
                specifications[slot] = WorkspaceSpec(request.dtype, capacity)
                last_step[slot] = index
            end
            push!(slots, slot)
        end
        push!(bindings, slots)
    end
    (; specifications, bindings)
end

function bind_implementations(runtime, chosen)
    plan = plan_workspaces(chosen)
    buffers = Any[allocate(runtime, spec.dtype, (spec.count,)) for spec in plan.specifications]
    calls = Any[]
    names = Symbol[]
    for (index, (name, construction)) in enumerate(chosen)
        if construction isa ImplementationPlan
            workspace = Tuple(buffers[slot] for slot in plan.bindings[index])
            built = check_built(construction.bind(workspace))
            # Retain workspace even when a binder forgets to include it in owners.
            # The binder must still retain its own descriptors and other resources.
            built = BuiltImplementation(
                built.launch,
                (built.owners, workspace),
                built.workspace_bytes,
            )
        else
            built = construction
        end
        push!(calls, built)
        push!(names, name)
    end
    (; calls, names, buffers, workspace_slots = plan.bindings)
end
