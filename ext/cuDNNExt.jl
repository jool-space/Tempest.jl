# cuDNN: attention as one library call. A library to Tempest: its templates
# are attention of half-precision queries, keys and values as cuDNN's SDPA
# engines compute it, the scores and their softmax in Float32, the weights
# meeting the values on the half-precision tensor cores, as a product of a
# half and a Float32 factor does in Tempest's own kernels, and the output
# rounded to the operands' type; with and without a causal mask, at the scale
# of one over the root of the features; and the same with the log-sum-exp of
# the scores beside the output, in Float32, as the engines write their
# statistics. A program that computes attention so is realized by cuDNN where
# it is faster; one that rounds the scores, keeps the output in Float32 or
# scales otherwise computes something else, and is not. It supports the
# bindings whose arrays the engines read: features contiguous, the query heads
# that share a key head next to one another, and every array and row 16 bytes
# aligned.
module cuDNNExt

using CUDACore
using cuDNN
using BFloat16s: BFloat16
import Tempest
using Tempest: CUDATarget
using Tempest.Compiler: contract
using Tempest.Compiler.Implement: Library, Template, Binding, Strided, Work, cuDNNAttention
import Tempest.Compiler.Implement: templates, supports, build, estimate, loaded

# The library, `Tempest.cuDNNAttention`, can be made once this extension loads.
loaded(::Type{cuDNNAttention}) = true

# The labels of `contract`: features, query heads per key head, key heads,
# queries, batch, keys.
const D, G, H, M, B, N = 1, 2, 3, 4, 5, 6

# Queries `q` are (features, query heads per key head, key heads, queries,
# batch): query head `g + G·h` reads key head `h`, as cuDNN groups them. With
# `statistics`, the log-sum-exp of each query's scores too, (1, query heads
# per key head, key heads, queries, batch), as a training forward keeps it for
# the backward.
function attention(q, k, v; causal, statistics = false)
    s = contract(Float32.(k), Float32.(q), (D, H, N, B), (D, G, H, M, B), (N, G, H, M, B)) .* Float32(inv(sqrt(size(q, 1))))
    causal && (s = ifelse.(reshape(1:size(s, 1), :, 1, 1, 1) .<= reshape(1:size(s, 4), 1, 1, 1, :), s, -Inf32))
    lse = Tempest.logsumexp(s; dims = 1)
    o = contract(v, exp.(s .- lse), (D, H, N, B), (N, G, H, M, B), (D, G, H, M, B))
    statistics ? (eltype(q).(o), lse) : eltype(q).(o)
end

# Without and with the statistics, each without and with the causal mask: the
# `k`th is causal when `k` is even, and writes the statistics past the second.
const ATTENTION = Tuple(Template((q, k, v) -> attention(q, k, v; causal, statistics);
    inputs = (q = (:d, :g, :h, :m, :b), k = (:d, :h, :n, :b), v = (:e, :h, :n, :b)), outputs = statistics ? (:o, :lse) : (:o,))
    for statistics in (false, true) for causal in (false, true))
causal(b::Binding) = iseven(b.template)

# Online: the engines rescale the softmax's sums as its maximum grows, which
# a target that keeps the two passes does not permit.
templates(::cuDNNAttention, target::CUDATarget) = target.online ? ATTENTION : ()

"An array as the SDPA engines declare it: (features, heads, positions, batch) and its strides."
struct Operand
    dims::NTuple{4, Int}
    strides::NTuple{4, Int}
end

# Queries or the output, their query heads one axis: the key heads step over
# the heads of a group. The statistics are one feature wide.
function heads(s::Strided)
    d, g, h, m, b = length(s.size) == 5 ? s.size : (1, s.size...)
    sd, sg, sh, sm, sb = length(s.strides) == 5 ? s.strides : (0, s.strides...)
    stride = g == 1 ? sh : h == 1 || sh == g * sg ? sg : nothing
    stride === nothing ? nothing : Operand((d, g * h, m, b), (d == 1 ? 1 : sd, stride, sm, sb))
end

struct Sdpa
    eltype::DataType
    causal::Bool
    q::Operand
    k::Operand
    v::Operand
    o::Operand
    stats::Union{Nothing, Operand}
end

const HALF = (Float16, BFloat16)

# The call that computes a binding, or nothing when the engines do not take
# its arrays: half-precision, as many features in the keys as in the values,
# a multiple of 8 up to 256 of them, contiguous, and every other stride and
# where each array starts a multiple of 16 bytes. The statistics are Float32,
# laid out as Tempest stores them, a query's heads next to one another, or
# any other way: the engines write them through their strides.
function sdpa(b::Binding)
    q, k, v, o = b.arrays.q, b.arrays.k, b.arrays.v, b.arrays.o
    T = q.eltype
    T in HALF && k.eltype === v.eltype === o.eltype === T || return nothing
    e = b.extents
    e.d == e.e && e.d % 8 == 0 && e.d <= 256 || return nothing
    Q, O = heads(q), heads(o)
    (Q === nothing || O === nothing) && return nothing
    K, V = Operand(k.size, k.strides), Operand(v.size, v.strides)
    all(x -> x.strides[1] == 1 && all(s -> s % 8 == 0, x.strides[2:4]), (Q, K, V, O)) || return nothing
    all(s -> s.offset % 8 == 0, (q, k, v, o)) || return nothing
    stats = nothing
    if haskey(b.arrays, :lse)
        lse = b.arrays.lse
        lse.eltype === Float32 && lse.offset % 4 == 0 || return nothing
        stats = heads(lse)
        stats === nothing && return nothing
    end
    Sdpa(T, causal(b), Q, K, V, O, stats)
end

# A graph built for each call alike, on each device, or nothing when cuDNN
# has no engine for it.
const GRAPHS = Dict{Tuple{CUDACore.CuContext, Sdpa}, Union{Nothing, cuDNN.Graph}}()
const GRAPHS_LOCK = ReentrantLock()

function graph(c::Sdpa)
    @lock GRAPHS_LOCK get!(GRAPHS, (CUDACore.context(), c)) do
        g = cuDNN.Graph(; io_dtype = c.eltype, intermediate_dtype = Float32, compute_dtype = Float32)
        operand(name, x; output = false) = cuDNN.tensor!(g; dims = x.dims, strides = x.strides, dtype = c.eltype, name, output)
        q, k, v, o = operand("Q", c.q), operand("K", c.k), operand("V", c.v), operand("O", c.o; output = true)
        stats = c.stats === nothing ? nothing :
            cuDNN.tensor!(g; dims = c.stats.dims, strides = c.stats.strides, dtype = Float32, name = "Stats", output = true)
        cuDNN.sdpa_fwd!(g, q, k, v; o, stats, scale = cuDNN.scalar!(g, Float32; rank = 4, name = "Scale"), causal = c.causal)
        try
            # Engines that keep the operands' types: none that rounds them further.
            cuDNN.build!(g; math_mode = CUDACore.DEFAULT_MATH, deterministic = true)
        catch err
            cuDNN.graph_unsupported(err) || rethrow()
            nothing
        end
    end
end

# Whether an engine runs a call is known only on a device, in the session:
# while precompiling, no graph is built.
builds_graphs() = ccall(:jl_generating_output, Cint, ()) == 0 && cuDNN.functional()

supports(::cuDNNAttention, ::CUDATarget, b::Binding) = (c = sdpa(b)) !== nothing && (!builds_graphs() || graph(c) !== nothing)

# Both products on the half-precision tensor cores, a causal mask skipping
# half of each: what the group reads and writes, and the products' work.
function estimate(::cuDNNAttention, b::Binding, target, g)
    e = b.extents
    flops = 2.0 * (e.d + e.e) * e.g * e.h * e.m * e.b * e.n
    [Work(g.bytes; flops = causal(b) ? flops / 2 : flops, factor = b.arrays.q.eltype)]
end

# One handle per context, which carries the stream: execution plans do not
# belong to a handle.
const HANDLES = Dict{CUDACore.CuContext, cuDNN.cudnnHandle_t}()

handle() = @lock GRAPHS_LOCK get!(cuDNN.cudnnCreate, HANDLES, CUDACore.context())

struct BuiltAttention
    call::Sdpa
    workspace::Int
end

function Base.show(io::IO, a::BuiltAttention)
    c = a.call
    d, heads, queries, batch = c.q.dims
    print(io, "cuDNN attention", c.causal ? ", causal" : "", ", ", heads, " heads", c.k.dims[2] == heads ? "" : " over $(c.k.dims[2])",
        " of ", d, " features, ", queries, " queries by ", c.k.dims[3], " keys", batch > 1 ? ", batch $batch" : "", ", ", c.eltype,
        c.stats === nothing ? "" : ", with its log-sum-exp")
end

# Every pointer the plan reads is fixed when it is built, in a variant pack:
# the buffers of a prepared program do not move. A launch sets the stream and
# executes, which a graph captures.
function build(::cuDNNAttention, target::CUDATarget, b::Binding)
    c = sdpa(b)
    g = graph(c)
    h = handle()
    workspace = CuArray{UInt8}(undef, g.workspace_size)
    scale, mask = Ref(Float32(inv(sqrt(c.q.dims[1])))), Ref(-Inf32)
    arrays = Dict("Q" => b.arrays.q, "K" => b.arrays.k, "V" => b.arrays.v, "O" => b.arrays.o)
    c.stats === nothing || (arrays["Stats"] = b.arrays.lse)
    pointers = map(g.variant_tensors) do t
        t.name == "Scale" && return Base.unsafe_convert(Ptr{Float32}, scale)
        t.name == "MaskValue" && return Base.unsafe_convert(Ptr{Float32}, mask)
        pointer(arrays[t.name])
    end
    pack = cuDNN.variant_pack(; uids = [t.uid for t in g.variant_tensors], pointers,
        workspace = g.workspace_size == 0 ? C_NULL : pointer(workspace))
    keep = (g, pack, scale, mask, workspace, b)
    function launch()
        cuDNN.cudnnSetStream(h, CUDACore.stream())
        cuDNN.cudnnBackendExecute(h, g.plan.ptr, pack.ptr)
        keep
        nothing
    end
    (launch, BuiltAttention(c, g.workspace_size))
end

# Work the library does while it precompiles, without a GPU: preparing
# attention programs for CUDA without building them, which traces and plans
# the templates, matches them, estimates and decides.

using PrecompileTools: PrecompileTools, @setup_workload, @compile_workload

# An extension has no preferences of its own (see cuBLASExt): its workload
# follows Tempest's.
if !isdefined(PrecompileTools, :workload_enabled) || PrecompileTools.workload_enabled(Tempest)
    @setup_workload begin
        sample(T, dims...) = T.(reshape(sin.(1:prod(dims)), dims))
        q, k, v = sample(BFloat16, 64, 2, 2, 32, 1), sample(BFloat16, 64, 2, 32, 1), sample(BFloat16, 64, 2, 32, 1)
        @compile_workload begin
            for causal in (false, true), statistics in (false, true)
                program = Tempest.Compiler.trace((q, k, v) -> attention(q, k, v; causal, statistics), q, k, v)
                ex = Tempest.Compiler.prepare(program, q, k, v; target = CUDATarget(cuDNNAttention()), build = false)
                Tempest.explain(devnull, ex)
            end
        end
    end
end

end
