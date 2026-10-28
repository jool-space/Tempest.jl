# cuBLAS: a product as one library call. A library to Tempest: its templates
# are a batch of products, and the same of half-precision factors summed in
# Float32, and it supports those whose factors and result are matrices cuBLAS
# reads, each with a unit stride and a leading dimension; and instances of
# them alike but for where their operands start, as one call over arrays of
# their matrices.
module cuBLASExt

using CUDACore
using cuBLAS
using BFloat16s: BFloat16
import Tempest
using Tempest: CUDATarget
using Tempest.Compiler: contract
using Tempest.Compiler.Implement: Library, Template, Binding, cuBLASProducts
import Tempest.Compiler.Implement: templates, supports, build, loaded

# The library, `Tempest.cuBLASProducts`, can be made once this extension loads.
loaded(::Type{cuBLASProducts}) = true

product(A, B) = contract(A, B, (1, 3, 4), (3, 2, 4), (1, 2, 4))
widened(A, B) = product(Float32.(A), Float32.(B))

const PRODUCTS = (Template(product; inputs = (A = (:m, :k, :b), B = (:k, :n, :b)), outputs = (:C,)),
    Template(widened; inputs = (A = (:m, :k, :b), B = (:k, :n, :b)), outputs = (:C,)))

templates(::cuBLASProducts, ::CUDATarget) = PRODUCTS

const FACTOR_TYPES = (Float32, Float16, BFloat16)

"A factor or the result as cuBLAS reads it: column-major with a leading dimension."
struct Operand
    name::Symbol
    leading::Int
    batch_stride::Int
end

struct Gemm
    transposes::Tuple{Char, Char}
    m::Int
    n::Int
    k::Int
    batch::Int
    a::Operand
    b::Operand
    c::Operand
    factor::DataType
    result::DataType
end

# The compute type of a product of `T` factors under the target's numerics:
# Float32 in full or as TF32, half factors accumulating in Float32. The one place
# this extension reads precision; `:tf32x3` is met by full Float32.
compute_type(target::CUDATarget, T) =
    T === Float32 && target.precision === :tf32 ? cuBLAS.CUBLAS_COMPUTE_32F_FAST_TF32 : cuBLAS.CUBLAS_COMPUTE_32F

# A matrix of `rows` by `columns` with strides `(along rows, along columns)` as
# cuBLAS takes it: untransposed when rows are unit-stride, transposed when
# columns are, with a leading dimension cuBLAS accepts.
function orient(rows, columns, (sr, sc))
    if rows == 1 || sr == 1
        leading = columns == 1 ? max(rows, 1) : sc
        leading >= max(rows, 1) && return ('N', leading)
    end
    if columns == 1 || sc == 1
        leading = rows == 1 ? max(columns, 1) : sr
        leading >= max(columns, 1) && return ('T', leading)
    end
    nothing
end

# The call that computes a binding, or nothing when cuBLAS does not take it:
# products of matrices of more than one row and column, the widened
# template's of half-precision factors only.
function gemm(b::Binding)
    A, B, C = b.arrays.A, b.arrays.B, b.arrays.C
    T, R = A.eltype, C.eltype
    T in FACTOR_TYPES && B.eltype === T && (R === T || R === Float32) && (b.template == 1 || T !== Float32) || return nothing
    m, n, k, batch = b.extents.m, b.extents.n, b.extents.k, b.extents.b
    m > 1 && n > 1 && max(m, n, k, batch) < typemax(Int32) || return nothing
    left, right, out = (:A, A.strides[1:2], A.strides[3]), (:B, B.strides[1:2], B.strides[3]), (:C, C.strides[1:2], C.strides[3])
    # cuBLAS writes a column-major result: when the result's columns are unit
    # stride instead, compute its transpose, the factors swapped and transposed.
    if C.strides[1] != 1
        left, right = (right[1], reverse(right[2]), right[3]), (left[1], reverse(left[2]), left[3])
        out = (:C, reverse(out[2]), out[3])
        m, n = n, m
    end
    oc = orient(m, n, out[2])
    oc === nothing || oc[1] == 'N' || return nothing
    oa, ob = orient(m, k, left[2]), orient(k, n, right[2])
    (oc === nothing || oa === nothing || ob === nothing) && return nothing
    maximum((oa[2], ob[2], oc[2])) < typemax(Int32) || return nothing
    operand((name, _, bs), o) = Operand(name, o[2], bs)
    Gemm((oa[1], ob[1]), m, n, k, batch, operand(left, oa), operand(right, ob), operand(out, oc), T, R)
end

supports(::cuBLASProducts, ::CUDATarget, b::Binding) = gemm(b) !== nothing

# What products must share to be one call: all but where their operands start.
alike(c::Gemm) = (c.transposes, c.m, c.n, c.k, c.batch, c.a, c.b, c.c, c.factor, c.result)

function supports(::cuBLASProducts, ::CUDATarget, bs::Vector{Binding})
    cs = map(gemm, bs)
    !any(isnothing, cs) && allequal(map(alike, cs))
end

# One handle per context, its workspace and the device constants one and
# zero, set up once: pointers to device constants, Float32 accumulation that
# never reduces precision, and the precision of each call in its compute type.
struct Handle
    handle::cuBLAS.cublasHandle_t
    workspace::CuArray{UInt8, 1}
    one::CuArray{Float32, 1}
    zero::CuArray{Float32, 1}
end

const WORKSPACE_BYTES = 32 << 20
const HANDLES = Dict{CUDACore.CuContext, Handle}()

function handle()
    get!(HANDLES, CUDACore.context()) do
        handle = cuBLAS.cublasCreate()
        cuBLAS.cublasSetPointerMode_v2(handle, cuBLAS.CUBLAS_POINTER_MODE_DEVICE)
        cuBLAS.cublasSetMathMode(handle, cuBLAS.cublasMath_t(
            cuBLAS.CUBLAS_DEFAULT_MATH | cuBLAS.CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION))
        Handle(handle, CuArray{UInt8}(undef, WORKSPACE_BYTES), CuArray(Float32[1]), CuArray(Float32[0]))
    end
end

struct BuiltProduct
    gemm::Gemm
    compute::cuBLAS.cublasComputeType_t
end

struct BuiltProducts
    products::Vector{Gemm}
    compute::cuBLAS.cublasComputeType_t
end

Base.show(io::IO, b::BuiltProducts) = print(io, length(b.products), " ", BuiltProduct(first(b.products), b.compute), ", one call")

function Base.show(io::IO, b::BuiltProduct)
    c = b.gemm
    print(io, "cuBLAS product ", c.m, "×", c.n, " summing ", c.k)
    c.batch > 1 && print(io, ", batch ", c.batch)
    print(io, ", ", c.transposes[1], c.transposes[2], ", ",
        b.compute == cuBLAS.CUBLAS_COMPUTE_32F_FAST_TF32 ? "TF32" : c.factor === Float32 ? "full precision" : "$(c.factor) accumulating in Float32")
end

function build(::cuBLASProducts, target::CUDATarget, b::Binding)
    c = gemm(b)
    lib = handle()
    a, x, y = pointer(b.arrays[c.a.name]), pointer(b.arrays[c.b.name]), pointer(b.arrays[c.c.name])
    compute = compute_type(target, c.factor)
    (ta, tb), (m, n, k, batch) = c.transposes, (c.m, c.n, c.k, c.batch)
    keep = (lib, b)
    function launch()
        cuBLAS.cublasSetStream_v2(lib.handle, CUDACore.stream())
        # Setting the stream resets the workspace: install it after.
        cuBLAS.cublasSetWorkspace_v2(lib.handle, lib.workspace, sizeof(lib.workspace))
        cuBLAS.cublasGemmStridedBatchedEx(lib.handle, ta, tb, m, n, k, lib.one, a, c.factor, c.a.leading, c.a.batch_stride,
            x, c.factor, c.b.leading, c.b.batch_stride, lib.zero, y, c.result, c.c.leading, c.c.batch_stride, batch,
            compute, cuBLAS.CUBLAS_GEMM_DEFAULT)
        keep
        nothing
    end
    (launch, BuiltProduct(c, compute))
end

# The matrices of products alike, by arrays of their addresses on the device,
# fixed when it is built: the buffers of a prepared program do not move.
function build(::cuBLASProducts, target::CUDATarget, bs::Vector{Binding})
    cs = map(gemm, bs)
    c = first(cs)
    lib = handle()
    at(b, o, T, i) = Ptr{Cvoid}(reinterpret(UInt, pointer(b.arrays[o.name])) + i * o.batch_stride * sizeof(T))
    addresses(o, T) = CuArray(Ptr{Cvoid}[at(b, o, T, i) for b in bs for i in 0:(c.batch - 1)])
    a, x, y = addresses(c.a, c.factor), addresses(c.b, c.factor), addresses(c.c, c.result)
    compute = compute_type(target, c.factor)
    (ta, tb), (m, n, k), count = c.transposes, (c.m, c.n, c.k), length(bs) * c.batch
    keep = (lib, bs, a, x, y)
    function launch()
        cuBLAS.cublasSetStream_v2(lib.handle, CUDACore.stream())
        cuBLAS.cublasSetWorkspace_v2(lib.handle, lib.workspace, sizeof(lib.workspace))
        cuBLAS.cublasGemmBatchedEx(lib.handle, ta, tb, m, n, k, lib.one, a, c.factor, c.a.leading, x, c.factor, c.b.leading,
            lib.zero, y, c.result, c.c.leading, count, compute, cuBLAS.CUBLAS_GEMM_DEFAULT)
        keep
        nothing
    end
    (launch, BuiltProducts(cs, compute))
end

# Work the cuBLAS products do while they precompile, without a GPU: preparing
# small programs with products for CUDA without building them, which claims,
# estimates, proposes candidates and decides. Building needs a library
# handle, so it compiles in the session.

using PrecompileTools: PrecompileTools, @setup_workload, @compile_workload

# An extension has no preferences of its own, since no project names it: its
# workload follows Tempest's, so `[Tempest] precompile_workload = false` turns
# every workload off.
if !isdefined(PrecompileTools, :workload_enabled) || PrecompileTools.workload_enabled(Tempest)
    @setup_workload begin
        sample(T, dims...) = T.(reshape(sin.(1:prod(dims)), dims))
        gelu(x) = x / (1 + exp(-1.702f0 * x))
        losses = Tuple{Any, Any}[((w, x, b) -> sum(abs2, gelu.(w * x .+ b)), (sample(T, 32, 64), sample(T, 64, 48), sample(T, 32)))
            for T in FACTOR_TYPES]
        # A batch of products over (features, keys, heads, batch), as attention makes.
        q, k = sample(Float32, 16, 32, 2, 2), sample(Float32, 16, 32, 2, 2)
        push!(losses, ((q, k) -> sum(abs2, Tempest.Compiler.contract(k, q, (1, 2, 3, 4), (1, 5, 3, 4), (2, 5, 3, 4))), (q, k)))
        @compile_workload begin
            for (f, args) in losses, target in (CUDATarget(cuBLASProducts()), CUDATarget(cuBLASProducts(); precision = :tf32))
                program = Tempest.Compiler.trace((xs...) -> Tempest.gradient(f, xs...), args...)
                ex = Tempest.Compiler.prepare(program, args...; target, build = false)
                Tempest.explain(devnull, ex)
            end
        end
    end
end

end
