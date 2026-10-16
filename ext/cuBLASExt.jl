# cuBLAS: a product as one library call. It claims a group that is one sum of a
# product of two loads, each a strided matrix (or a batch of them) over the
# product's axes, written whole to its own buffer: no prologue, no epilogue.
module cuBLASExt

using Tempest
using CUDACore
using cuBLAS
import Tempest: CUDATarget, Plan, Group, Load, Apply, Domain

struct cuBLASProducts <: Tempest.Implementation end

"A factor or the result as cuBLAS reads it: column-major with a leading dimension."
struct Operand
    value::Int
    offset::Int
    leading::Int
    batch_stride::Int
end

struct GemmClaim
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

const FACTOR_TYPES = (Float32, Float16, Core.BFloat16)

# The compute type of a product of `T` factors under the target's numerics:
# Float32 in full or as TF32, half factors accumulating in Float32. The one place
# this extension reads precision; `:tf32x3` is met by full Float32.
compute_type(target::CUDATarget, T) =
    T === Float32 && target.precision === :tf32 ? cuBLAS.CUBLAS_COMPUTE_32F_FAST_TF32 : cuBLAS.CUBLAS_COMPUTE_32F

# Strides of value `v` read at `index` along each group of axes, and its offset;
# nothing when the read is not a strided view over them.
function strides(graph, v, index, g::Group, groups...)
    e = Tempest.address(graph, v, index, Domain(g.domain))
    Tempest.is_affine(e) || return nothing
    covered = reduce(vcat, groups; init = Int[])
    any(k -> g.domain[k] > 1 && Tempest.coefficient(e, k) != 0 && !(k in covered), eachindex(g.domain)) && return nothing
    s = map(grp -> Tempest.group_stride(e, grp, g.domain), groups)
    any(x -> x === nothing || x < 0, s) && return nothing
    (e.constant, s...)
end

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

# A factor's load: the load itself, or a half-precision load converted to
# Float32, whose products cuBLAS accumulates in Float32 exactly.
factor_load(t::Load, graph) = t
function factor_load(t::Apply, graph)
    t.f isa Tempest.Cast{Float32} && only(t.args) isa Load || return nothing
    l = only(t.args)
    graph.values[l.value].eltype <: Tempest.HalfFloat ? l : nothing
end
factor_load(t, graph) = nothing

function Tempest.claims(::cuBLASProducts, target::CUDATarget, p::Plan, g::Group)
    g.kind === :reduction && length(g.reductions) == 1 && isempty(g.outputs) || return GemmClaim[]
    graph = p.graph
    r = only(g.reductions)
    p.stored[r] && graph.values[r].definition.fold isa Tempest.SumFold || return GemmClaim[]
    found = Tempest.product_axes(graph, g)
    found === nothing && return GemmClaim[]
    M, N, K, B = found
    x, y = map(f -> factor_load(f, graph), g.terms[r].args)
    (x === nothing || y === nothing) && return GemmClaim[]
    T = graph.values[x.value].eltype
    R = graph.values[r].eltype
    T in FACTOR_TYPES && graph.values[y.value].eltype === T && (R === T || R === Float32) || return GemmClaim[]
    extent(axes) = prod(k -> g.domain[k], axes; init = 1)
    m, n, k, batch = extent(M), extent(N), extent(K), extent(B)
    max(m, n, k, batch) < typemax(Int32) || return GemmClaim[]
    sx = strides(graph, x.value, x.index, g, M, K, B)
    sy = strides(graph, y.value, y.index, g, K, N, B)
    sc = strides(graph, r, Tempest.placement(graph, g, r), g, M, N, B)
    (sx === nothing || sy === nothing || sc === nothing) && return GemmClaim[]
    left, right = (x.value, sx[1], (sx[2], sx[3]), sx[4]), (y.value, sy[1], (sy[2], sy[3]), sy[4])
    out = (r, sc[1], (sc[2], sc[3]), sc[4])
    # cuBLAS writes a column-major result: when the result's columns are unit
    # stride instead, compute its transpose, the factors swapped and transposed.
    if !(m == 1 || sc[2] == 1)
        left, right = (right[1], right[2], reverse(right[3]), right[4]), (left[1], left[2], reverse(left[3]), left[4])
        out = (r, sc[1], reverse(out[3]), sc[4])
        m, n = n, m
    end
    oc = orient(m, n, out[3])
    oc === nothing || oc[1] == 'N' || return GemmClaim[]
    oa, ob = orient(m, k, left[3]), orient(k, n, right[3])
    (oc === nothing || oa === nothing || ob === nothing) && return GemmClaim[]
    maximum((oa[2], ob[2], oc[2])) < typemax(Int32) || return GemmClaim[]
    operand((v, offset, _, bs), o) = Operand(v, offset, o[2], bs)
    [GemmClaim((oa[1], ob[1]), m, n, k, batch, operand(left, oa), operand(right, ob), operand(out, oc), T, R)]
end

# One handle per context, its workspace and the device constants one and
# zero, set up once: pointers to device constants, Float32 accumulation that
# never reduces precision, and the precision of each call in its compute type.
struct Library
    handle::cuBLAS.cublasHandle_t
    workspace::CuArray{UInt8, 1}
    one::CuArray{Float32, 1}
    zero::CuArray{Float32, 1}
end

const WORKSPACE_BYTES = 32 << 20
const LIBRARIES = Dict{CUDACore.CuContext, Library}()

function library()
    get!(LIBRARIES, CUDACore.context()) do
        handle = cuBLAS.cublasCreate()
        cuBLAS.cublasSetPointerMode_v2(handle, cuBLAS.CUBLAS_POINTER_MODE_DEVICE)
        cuBLAS.cublasSetMathMode(handle, cuBLAS.cublasMath_t(
            cuBLAS.CUBLAS_DEFAULT_MATH | cuBLAS.CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION))
        Library(handle, CuArray{UInt8}(undef, WORKSPACE_BYTES), CuArray(Float32[1]), CuArray(Float32[0]))
    end
end

struct BuiltProduct
    claim::GemmClaim
    compute::cuBLAS.cublasComputeType_t
end

function Base.show(io::IO, b::BuiltProduct)
    c = b.claim
    print(io, "cuBLAS product ", c.m, "×", c.n, " summing ", c.k)
    c.batch > 1 && print(io, ", batch ", c.batch)
    print(io, ", ", c.transposes[1], c.transposes[2], ", ",
        b.compute == cuBLAS.CUBLAS_COMPUTE_32F_FAST_TF32 ? "TF32" : c.factor === Float32 ? "full precision" : "$(c.factor) accumulating in Float32")
end

function Tempest.build_claim(::cuBLASProducts, c::GemmClaim, target, p::Plan, g::Group, buffers)
    lib = library()
    address(o::Operand, T) = pointer(buffers[o.value]) + o.offset * sizeof(T)
    a, b, y = address(c.a, c.factor), address(c.b, c.factor), address(c.c, c.result)
    compute = compute_type(target, c.factor)
    (ta, tb), (m, n, k, batch) = c.transposes, (c.m, c.n, c.k, c.batch)
    keep = (lib, buffers)
    function launch()
        cuBLAS.cublasSetStream_v2(lib.handle, CUDACore.stream())
        # Setting the stream resets the workspace: install it after.
        cuBLAS.cublasSetWorkspace_v2(lib.handle, lib.workspace, sizeof(lib.workspace))
        cuBLAS.cublasGemmStridedBatchedEx(lib.handle, ta, tb, m, n, k, lib.one, a, c.factor, c.a.leading, c.a.batch_stride,
            b, c.factor, c.b.leading, c.b.batch_stride, lib.zero, y, c.result, c.c.leading, c.c.batch_stride, batch,
            compute, cuBLAS.CUBLAS_GEMM_DEFAULT)
        keep
        nothing
    end
    (launch, BuiltProduct(c, compute))
end

__init__() = Tempest.register!(cuBLAS, cuBLASProducts())

end
