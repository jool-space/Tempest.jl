# Test implementations shared by implementations/protocol.jl, verbs/contract.jl
# and runtime/donation.jl.
struct TestImplementation <: Implementation
    mode::Symbol
end

Tempest.implementation_name(::TestImplementation) = :test_implementation

function Tempest.support(c::TestImplementation, r::RegionRequest)
    if c.mode === :reject
        return Unsupported("test metadata rejection")
    end
    if !(r.step.region isa Tempest.Elementwise)
        return Unsupported("pointwise only")
    end
    Supported()
end

function Tempest.build_implementation(
    c::TestImplementation,
    r::RegionRequest,
    runtime,
    out,
    arrays,
)
    if c.mode === :unavailable
        return Unsupported("test construction rejection")
    end
    if c.mode === :broken
        error("test unexpected compiler failure")
    end
    if c.mode === :invalid
        return nothing
    end
    build_implementation(CPUKernels(), r, runtime, out, arrays)
end

# A target may have no native implementation of an otherwise valid region.
# An accepted custom implementation must not consult that fallback at all.
struct ImplementationOnlyTarget <: Tempest.ExecutionTarget end
struct ImplementationOnlyRuntime <: Tempest.AbstractRuntime end
Tempest.make_runtime(::ImplementationOnlyTarget, storage, args) = ImplementationOnlyRuntime()
Tempest.allocate(::ImplementationOnlyRuntime, T, dims) = Array{T}(undef, dims)
Tempest.check_runtime(::ImplementationOnlyRuntime) = nothing


struct InvalidSupport <: Implementation end
Tempest.support(::InvalidSupport, ::RegionRequest) = true
Tempest.support(::InvalidSupport, ::CallRequest) = true
