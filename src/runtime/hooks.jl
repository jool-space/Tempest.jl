# A runtime owns a prepared executable's device resources. Target extensions
# implement these hooks; the CPU runtime needs none of them beyond allocation.
abstract type AbstractRuntime end

struct CPURuntime <: AbstractRuntime end

function make_runtime(::ExecutionTarget, storage, args)
    throw(ArgumentError("target is unavailable; for CUDATarget load CUDACore and cuBLAS"))
end

make_runtime(::CPUTarget, storage, args) = CPURuntime()
allocate(::CPURuntime, T, dims) = Array{T}(undef, dims)
check_runtime(::CPURuntime) = nothing

function capture_runtime!(runtime, ex)
    throw(ArgumentError("capture is unsupported by this execution target"))
end

function replay_runtime!(runtime)
    throw(ArgumentError("call capture! on a capture-capable executable first"))
end

explain_runtime(io, ::CPURuntime) = println(io, "Target: CPU")
