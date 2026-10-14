# One prepared program owns one handle and its coefficients. Workspace comes
# from the common schedule pool and is installed explicitly before each launch,
# avoiding ambient cuBLAS policy and capture-time pool allocations.
mutable struct BlasResources
    handle::cuBLAS.cublasHandle_t
    context::CUDACore.CuContext
    workspace_bytes::Int
    coefficients::Dict{DataType, Tuple{Any, Any}}
end

function BlasResources(workspace_bytes, precision = :full)
    handle = cuBLAS.cublasCreate()
    resource = BlasResources(
        handle,
        CUDACore.context(),
        workspace_bytes,
        Dict{DataType, Tuple{Any, Any}}(),
    )
    finalizer(resource) do r
        if CUDACore.isvalid(r.context)
            CUDACore.context!(r.context) do
                cuBLAS.cublasDestroy_v2(r.handle)
            end
        end
    end
    cuBLAS.cublasSetPointerMode_v2(handle, cuBLAS.CUBLAS_POINTER_MODE_DEVICE)
    # Explicit precision policy from the target; no ambient FAST_MATH override.
    mode = precision === :tf32 ? cuBLAS.CUBLAS_TF32_TENSOR_OP_MATH : cuBLAS.CUBLAS_DEFAULT_MATH
    cuBLAS.cublasSetMathMode(handle, mode)
    resource
end

# Half-precision products accumulate in Float32, as in cuDNN and the usual
# framework default, and round once to the output type.
const HALF_GEMM_TYPES = (Float16, Core.BFloat16)

function gemm_call(resources::BlasResources, y, a, b, workspace)
    T = eltype(y)
    if !(eltype(a) == eltype(b) == T)
        throw(ArgumentError("CUDA GEMM operands must have matching dtypes"))
    end
    C = T in HALF_GEMM_TYPES ? Float32 : T
    alpha, beta = get!(resources.coefficients, C) do
        (CUDACore.CuRef(one(C)), CUDACore.CuRef(zero(C)))
    end
    ta, aa = a isa Transpose ? ('T', parent(a)) : ('N', a)
    tb, bb = b isa Transpose ? ('T', parent(b)) : ('N', b)
    m, n, k = size(y, 1), size(y, 2), size(a, 2)
    if !(max(m, n, k) <= typemax(Int32))
        throw(ArgumentError("prototype GEMM dimensions exceed Int32"))
    end
    lda, ldb, ldc = max(1, stride(aa, 2)), max(1, stride(bb, 2)), max(1, stride(y, 2))
    fn = if T === Float32
        cuBLAS.cublasSgemm_v2
    elseif T === Float64
        cuBLAS.cublasDgemm_v2
    elseif T in HALF_GEMM_TYPES
        (handle, ta, tb, m, n, k, alpha, aa, lda, bb, ldb, beta, y, ldc) -> cuBLAS.cublasGemmEx(
            handle, ta, tb, m, n, k, alpha, aa, T, lda, bb, T, ldb, beta, y, T, ldc,
            cuBLAS.CUBLAS_COMPUTE_32F, cuBLAS.CUBLAS_GEMM_DEFAULT,
        )
    else
        throw(ArgumentError("unsupported GEMM dtype $T"))
    end
    return function ()
        # SetStream resets workspace, so install workspace afterwards. All launches
        # use the caller's current stream; concurrent use is deliberately unsupported.
        cuBLAS.cublasSetStream_v2(resources.handle, CUDACore.stream())
        # A shared slot may be larger because another implementation needs it.
        # Keep cuBLAS's configured workspace budget independent of that capacity.
        cuBLAS.cublasSetWorkspace_v2(resources.handle, workspace, resources.workspace_bytes)
        fn(resources.handle, ta, tb, m, n, k, alpha, aa, lda, bb, ldb, beta, y, ldc)
        nothing
    end
end

function batched_gemm_call(resources::BlasResources, y, a, b, transpose_a, transpose_b, workspace)
    T = eltype(y)
    alpha, beta = get!(resources.coefficients, T) do
        (CUDACore.CuRef(one(T)), CUDACore.CuRef(zero(T)))
    end
    ta, tb = transpose_a ? 'T' : 'N', transpose_b ? 'T' : 'N'
    m, n, batch = size(y)
    k = transpose_a ? size(a, 1) : size(a, 2)
    if !(max(m, n, k, batch) <= typemax(Int32))
        throw(ArgumentError("prototype GEMM dimensions exceed Int32"))
    end
    lda, ldb, ldc = max(1, size(a, 1)), max(1, size(b, 1)), max(1, m)
    stride_a, stride_b, stride_c = size(a, 1) * size(a, 2), size(b, 1) * size(b, 2), m * n
    fn = if T === Float32
        cuBLAS.cublasSgemmStridedBatched
    elseif T === Float64
        cuBLAS.cublasDgemmStridedBatched
    else
        throw(ArgumentError("unsupported GEMM dtype $T"))
    end
    return function ()
        cuBLAS.cublasSetStream_v2(resources.handle, CUDACore.stream())
        cuBLAS.cublasSetWorkspace_v2(resources.handle, workspace, resources.workspace_bytes)
        fn(resources.handle, ta, tb, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
           beta, y, ldc, stride_c, batch)
        nothing
    end
end

function support(::CuBLASGemm, request::RegionRequest)
    if !(request.target isa CUDATarget)
        return Unsupported("requires a CUDA target")
    end
    if !(request.step.region isa GemmRegion)
        return Unsupported("requires a matrix product")
    end
    if !(request.dtype in (Float32, Float64, HALF_GEMM_TYPES...)) ||
       !all(==(request.dtype), request.input_types)
        return Unsupported(
            "CUDA GEMM operands must have matching Float16, BFloat16, Float32 or Float64 dtypes",
        )
    end
    if max(request.dims..., request.input_dims[1][2]) > typemax(Int32)
        return Unsupported("GEMM dimensions exceed Int32")
    end
    Supported()
end

function build_implementation(::CuBLASGemm, request::RegionRequest, runtime, out, arrays)
    workspace = WorkspaceSpec(UInt8, runtime.target.blas_workspace_bytes)
    ImplementationPlan((workspace,)) do buffers
        if runtime.blas === nothing
            runtime.blas = BlasResources(
                runtime.target.blas_workspace_bytes,
                runtime.target.matmul_precision,
            )
        end
        work = only(buffers)
        launch = gemm_call(runtime.blas, out, arrays..., work)
        BuiltImplementation(launch, (runtime.blas, out, arrays, work), 0)
    end
end
