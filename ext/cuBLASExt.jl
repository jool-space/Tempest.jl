# cuBLAS: a product as one library call. It claims a group that is one sum of a
# product of two loads, each a strided matrix (or a batch of them) over the
# product's axes, written whole to its own buffer: no prologue, no epilogue.
module cuBLASExt

using CUDACore
using cuBLAS
using BFloat16s: BFloat16
using Republic: @republic
using Tempest: CUDATarget
@republic republic=false inherit=:public using Tempest.IR
@republic republic=false inherit=:public import Tempest.Implementations
using Tempest.Compiler: Plan, Group, members, group_bytes, placement, product_axes

struct cuBLASProducts <: Implementation end

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

const FACTOR_TYPES = (Float32, Float16, BFloat16)

# The compute type of a product of `T` factors under the target's numerics:
# Float32 in full or as TF32, half factors accumulating in Float32. The one place
# this extension reads precision; `:tf32x3` is met by full Float32.
compute_type(target::CUDATarget, T) =
    T === Float32 && target.precision === :tf32 ? cuBLAS.CUBLAS_COMPUTE_32F_FAST_TF32 : cuBLAS.CUBLAS_COMPUTE_32F

# Strides of value `v` read at `index` along each group of axes, and its offset;
# nothing when the read is not a strided view over them.
function strides(graph, v, index, g::Group, groups...)
    e = address(graph.values[v], index, Domain(g.domain))
    is_affine(e) || return nothing
    covered = reduce(vcat, groups; init = Int[])
    any(k -> g.domain[k] > 1 && coefficient(e, k) != 0 && !(k in covered), eachindex(g.domain)) && return nothing
    s = map(grp -> group_stride(e, grp, g.domain), groups)
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
    t.f isa Cast{Float32} && only(t.args) isa Load || return nothing
    l = only(t.args)
    graph.values[l.value].eltype <: HalfFloat ? l : nothing
end
factor_load(t, graph) = nothing

function claims(::cuBLASProducts, target::CUDATarget, p::Plan, g::Group)
    g.kind === :reduction && length(g.reductions) == 1 && isempty(g.outputs) || return GemmClaim[]
    graph = p.graph
    r = only(g.reductions)
    p.stored[r] && graph.values[r].definition.fold isa SumFold || return GemmClaim[]
    found = product_axes(graph, g)
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
    sc = strides(graph, r, placement(graph, g, r), g, M, N, B)
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

# Product rates cuBLAS reaches, measured on known devices (the scratchpad's
# rates.jl): Float32 in full, TF32, and half factors. Other devices scale
# GB10's by SM count.
const RATES = Dict("NVIDIA GB10" => (full = 17.0e12, tf32 = 38.0e12, half = 88.0e12))

function product_rate(target::CUDATarget, T)
    device = CUDACore.device()
    r = get(RATES, CUDACore.name(device)) do
        scale = CUDACore.attribute(device, CUDACore.DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) / 48
        map(x -> scale * x, RATES["NVIDIA GB10"])
    end
    T === Float32 ? (compute_type(target, T) == cuBLAS.CUBLAS_COMPUTE_32F_FAST_TF32 ? r.tf32 : r.full) : r.half
end

estimate(::cuBLASProducts, c::GemmClaim, target, p::Plan, g::Group) =
    [Work(group_bytes(p, g); flops = 2.0 * c.m * c.n * c.k * c.batch, rate = product_rate(target, c.factor))]

record_name(::cuBLASProducts, c::GemmClaim) = string("cuBLAS ", c.transposes..., " ", c.m, "×", c.n, "×", c.k,
    c.batch > 1 ? " batch $(c.batch)" : "", " leading ", (c.a.leading, c.b.leading, c.c.leading), " ", c.factor, " to ", c.result)

# Candidates: each product fold the generator groups with other work, alone,
# so that its group is this claim's and the work after it a kernel of its own,
# and each product fold it recomputes inside another's, stored by this claim.
function candidates(::cuBLASProducts, target::CUDATarget, p::Plan)
    graph = p.graph
    product(r, t) = graph.values[r].definition.fold isa SumFold && t isa Apply && t.f === (*) && length(t.args) == 2 &&
        all(f -> factor_load(f, graph) !== nothing, t.args)
    found = Vector{Int}[]
    for g in p.groups
        g.kind === :reduction || continue
        r = first(g.reductions)
        length(members(g)) > 1 && product(r, g.terms[r]) && product_axes(graph, g) !== nothing && push!(found, [r])
        roots = [g.terms[m] for m in members(g) if haskey(g.terms, m)]
        for t in subterms(roots; nested = true)
            t isa Recomputed && product(t.value, t.body) && !([t.value] in found) && push!(found, [t.value])
        end
    end
    found
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

function build_claim(::cuBLASProducts, c::GemmClaim, target, p::Plan, g::Group, buffers)
    lib = library()
    operand_pointer(o::Operand, T) = pointer(buffers[o.value]) + o.offset * sizeof(T)
    a, b, y = operand_pointer(c.a, c.factor), operand_pointer(c.b, c.factor), operand_pointer(c.c, c.result)
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

__init__() = register!(cuBLAS, cuBLASProducts())

end
