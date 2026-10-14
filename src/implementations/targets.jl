# Execution targets describe placement and policy, without importing device packages.
abstract type ExecutionTarget end

"""Execute on owned CPU arrays using LinearAlgebra and the reference scalar loops."""
struct CPUTarget <: ExecutionTarget end

"""
    CUDATarget(; device=nothing, kernels=:auto, blas_workspace_bytes=4<<20,
               matmul_precision=:full)

Prepare on the current CUDA device (or require `device` to be current).
Load CUDACore and cuBLAS to enable this target. `kernels=:auto` uses eligible
cuTile implementations when cuTile is loaded; `:cuda` uses only CUDA kernels;
`:cutile` requires cuTile but still uses CUDA for mapped/general axis operations.
Provider choices are fixed at preparation. Compilation errors are propagated.
`matmul_precision=:full` keeps Float32 products in full precision;
`:tf32` lets cuBLAS round Float32 inputs to TensorFloat-32 on tensor cores,
the default of several frameworks, which trades about three decimal digits of
input precision for speed.
"""
struct CUDATarget{D} <: ExecutionTarget
    device::D
    kernels::Symbol
    blas_workspace_bytes::Int
    matmul_precision::Symbol

    function CUDATarget(;
        device = nothing,
        kernels = :auto,
        blas_workspace_bytes = 4<<20,
        matmul_precision = :full,
    )
        if !(kernels in (:auto, :cuda, :cutile))
            throw(ArgumentError("kernels must be :auto, :cuda, or :cutile"))
        end
        if !(blas_workspace_bytes >= 0)
            throw(ArgumentError("blas_workspace_bytes must be nonnegative"))
        end
        if !(matmul_precision in (:full, :tf32))
            throw(ArgumentError("matmul_precision must be :full or :tf32"))
        end
        new{typeof(device)}(device, kernels, Int(blas_workspace_bytes), matmul_precision)
    end
end

# Wrappers inherit their parent's placement. Unknown owning arrays need an
# explicit target or an array_target method; never guess that device data is CPU.
array_target(::Array) = CPUTarget()
array_target(::AbstractRange) = CPUTarget()

function array_target(a::AbstractArray)
    p = parent(a)
    if p === a
        throw(ArgumentError("cannot infer placement of $(typeof(a)); pass an explicit target"))
    end
    array_target(p)
end

same_placement(::ExecutionTarget, ::ExecutionTarget) = false
same_placement(::CPUTarget, ::CPUTarget) = true
same_placement(a::CUDATarget, b::CUDATarget) = a.device == b.device

function infer_target(args)
    target = nothing
    for a in args
        if !(a isa AbstractArray)
            continue
        end
        t = array_target(a)
        if !(target === nothing || same_placement(target, t))
            throw(
                ArgumentError(
                    "array inputs have mixed placement; pass an explicit target to copy them",
                ),
            )
        end
        target = t
    end
    target === nothing ? CPUTarget() : target
end
