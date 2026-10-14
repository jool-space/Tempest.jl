# Scaled dot-product attention on cuDNN's SDPA engines.
#
# Tempest's arrays are (features, heads, positions, batch). The forward graph
# declares them in that order and cuDNN.jl's sdpa_fwd! presents them to the
# engine; the backward graph declares the same memory as (features, positions,
# heads, batch), the order cuDNN.jl's composite sdpa_bwd! expects.
#
# cuDNN's softmax statistic is the natural-log log-sum-exp of the scaled
# scores, Tempest's `lse`, in Float32. The forward writes it in Tempest's
# (1, heads, positions, batch) order for the public verb. The backward reads it
# densely packed as (1, positions, heads, batch); the training rule saves it in
# that order instead, so the backward reads the forward's statistic directly.
#
# On cuDNN 9.27 with GB10, the SDPA engines take Float16 or BFloat16 operands
# only: Float32 has no engine, nor does Float32 I/O around in-graph casts or half
# inputs with a Float32 output. Float32 attention therefore needs an explicit
# CuDNNKernels(float32_cast = T).

const ATTENTION_FORWARD = Tempest.attention_forward
const ATTENTION_BACKWARD = Tempest.attention_backward
const HALF_TYPES = (Float16, Core.BFloat16)

# The type cuDNN computes with for `T` operands, or a reason.
function operand_type(implementation::CuDNNKernels, T)
    T in HALF_TYPES && return T
    if T === Float32
        cast = implementation.float32_cast
        cast === nothing || return cast
        return Unsupported(
            "cuDNN has no Float32 attention engine; " *
            "CuDNNKernels(float32_cast = BFloat16) or Float16 rounds operands explicitly",
        )
    end
    Unsupported("cuDNN attention takes Float16 or BFloat16 operands, not $T")
end

function attention_shape_support(q::TensorSpec, k::TensorSpec, v::TensorSpec)
    all(s -> all(>(0), s.dims), (q, k, v)) ||
        return Unsupported("cuDNN attention needs nonempty arrays")
    D = q.dims[1]
    v.dims[1] == D || return Unsupported("cuDNN attention needs equal key and value features")
    # Rows of 8 half-precision features keep each head 16-byte aligned.
    D % 8 == 0 || return Unsupported("cuDNN attention needs a multiple of 8 features")
    Supported()
end

function public_attention_support(implementation, request)
    request.target isa CUDATarget || return Unsupported("requires a CUDA target")
    q, k, v = request.inputs
    all(s -> s.dtype === q.dtype, (k, v)) ||
        return Unsupported("cuDNN attention needs one operand dtype")
    H = operand_type(implementation, q.dtype)
    H isa Unsupported && return H
    attention_shape_support(q, k, v)
end

# The training rule's private calls receive operands already in a half type.
function private_attention_support(request)
    request.target isa CUDATarget || return Unsupported("requires a CUDA target")
    q, k, v = request.inputs[1:3]
    q.dtype in HALF_TYPES ||
        return Unsupported("private cuDNN attention calls take Float16 or BFloat16")
    # The backward's fifth input is the Float32 statistic.
    operands = request.call.f === sdpa_backward ? request.inputs[[1, 2, 3, 4, 6]] : request.inputs
    all(s -> s.dtype === q.dtype, operands) ||
        return Unsupported("cuDNN attention needs one operand dtype")
    attention_shape_support(q, k, v)
end

function support(implementation::CuDNNKernels, request::CallRequest)
    f = request.call.f
    f === ATTENTION_FORWARD && return public_attention_support(implementation, request)
    if f === ATTENTION_BACKWARD
        return Unsupported(
            "cuDNN's attention backward has no log-sum-exp cotangent input; " *
            "the cuDNN reverse rule uses it when that cotangent is zero",
        )
    end
    f in (sdpa_forward, sdpa_backward) && return private_attention_support(request)
    Unsupported("no cuDNN implementation of $f")
end

# cuDNN is offered whole semantic verbs only. Scalar regions are not cuDNN
# patterns: with cuDNN 9.26 on GB10 a lone sum had no engine, and pointwise
# regions belong to the scalar-fusion providers. Without a GraphRequest method,
# cuDNN does not take part in graph-region selection.
function support(::CuDNNKernels, ::Tempest.RegionRequest)
    Unsupported("cuDNN implements semantic verbs, not scalar regions")
end

## Private verbs of the training rule

"""
    sdpa_forward(q, k, v; causal, scale) -> (; o, stats)

Attention on half-precision operands, as cuDNN computes it: Float32 scores and
softmax, `o` rounded to the operand type, and the Float32 log-sum-exp `stats`
laid out `(1, queries, heads, batch)`, the order cuDNN's backward reads.
"""
sdpa_forward(q::TracedArray, k::TracedArray, v::TracedArray; causal, scale) =
    record_call(sdpa_forward, q, k, v; causal, scale)

function output_spec(::typeof(sdpa_forward), q, k, v; causal, scale)
    D, heads, queries, batch = size(q)
    (o = TensorSpec(eltype(q), size(q)), stats = TensorSpec(Float32, (1, queries, heads, batch)))
end

to_float32(x) = Float32.(x)

function reference(::typeof(sdpa_forward), q, k, v; causal, scale)
    y = ATTENTION_FORWARD(to_float32(q), to_float32(k), to_float32(v); causal, scale)
    (o = eltype(q).(y.o), stats = permutedims(y.lse, (1, 3, 2, 4)))
end

"""
    sdpa_backward(q, k, v, o, stats, dout; causal, scale) -> (; dq, dk, dv)

The attention pullback for a zero log-sum-exp cotangent, from `sdpa_forward`'s
half-precision output and Float32 statistics.
"""
sdpa_backward(q::TracedArray, k, v, o, stats, dout; causal, scale) =
    record_call(sdpa_backward, q, k, v, o, stats, dout; causal, scale)

output_spec(::typeof(sdpa_backward), q, k, v, o, stats, dout; causal, scale) =
    (dq = TensorSpec(q), dk = TensorSpec(k), dv = TensorSpec(v))

function reference(::typeof(sdpa_backward), q, k, v, o, stats, dout; causal, scale)
    lse = permutedims(stats, (1, 3, 2, 4))
    g = ATTENTION_BACKWARD(
        to_float32(q), to_float32(k), to_float32(v), to_float32(o), lse, to_float32(dout), zero(lse); causal, scale,
    )
    T = eltype(q)
    (dq = T.(g.dq), dk = T.(g.dk), dv = T.(g.dv))
end

## Training rule

convert_values(x, T) = eltype(x) === T ? x : T.(x)

# Tempest represents the cotangent of an unused result by the symbolic zero
# `zero(x)`, a constant node.
function symbolic_zero(x::TracedArray)
    op = x.nodes[x.id].op
    op isa Tempest.Constant && iszero(op.value)
end

# Explicit casts are ordinary graph operations, so they can fuse with the
# operations that produce q, k and v or consume the gradients.
function attention_rule(H, q, k, v; causal, scale)
    T = eltype(q)
    qh, kh, vh = convert_values(q, H), convert_values(k, H), convert_values(v, H)
    saved = sdpa_forward(qh, kh, vh; causal, scale)
    o = convert_values(saved.o, T)
    lse = convert_values(permutedims(saved.stats, (1, 3, 2, 4)), T)
    function pullback(dy)
        if !symbolic_zero(dy.lse)
            g = ATTENTION_BACKWARD(q, k, v, o, lse, dy.o, dy.lse; causal, scale)
            return (g.dq, g.dk, g.dv)
        end
        dout = convert_values(dy.o, H)
        g = sdpa_backward(qh, kh, vh, saved.o, saved.stats, dout; causal, scale)
        (convert_values(g.dq, T), convert_values(g.dk, T), convert_values(g.dv, T))
    end
    (; o, lse), pullback
end

function reverse_rule(implementation::CuDNNKernels, request::CallRequest)
    request.call.f === ATTENTION_FORWARD ||
        return Unsupported("cuDNN has a reverse rule for attention only")
    verdict = public_attention_support(implementation, request)
    verdict isa Unsupported && return verdict
    H = operand_type(implementation, request.inputs[1].dtype)
    ReverseRule((q, k, v; causal, scale) -> attention_rule(H, q, k, v; causal, scale))
end

## Graphs

dense_strides(dims) = cumprod((1, dims[1:(end - 1)]...))

# q and o are (D, heads, queries, batch), k and v (D, kv_heads, keys, batch).
# The statistic has dims (1, heads, queries, batch) and the given strides.
function forward_graph(H, qdims, kdims, stats_strides, causal)
    graph = Graph(; io_dtype = H, intermediate_dtype = Float32, compute_dtype = Float32)
    q = tensor!(graph; dims = qdims, dtype = H, name = "Q")
    k = tensor!(graph; dims = kdims, dtype = H, name = "K")
    v = tensor!(graph; dims = kdims, dtype = H, name = "V")
    o = tensor!(graph; dims = qdims, dtype = H, name = "O", output = true)
    _, heads, queries, batch = qdims
    stats = tensor!(
        graph;
        dims = (1, heads, queries, batch),
        strides = stats_strides,
        dtype = Float32,
        name = "Stats",
        output = true,
    )
    scale = scalar!(graph, Float32; rank = 4, name = "Scale")
    sdpa_fwd!(graph, q, k, v; o, stats, scale, causal)
    graph
end

# The backward's operands are the forward's arrays, declared (D, positions,
# heads, batch) over the same memory.
# Layout from Stiletto's SdpaBwd emission (Stiletto src/compiler/emission.jl).
function backward_graph(H, qdims, kdims, causal)
    graph = Graph(; io_dtype = H, intermediate_dtype = Float32, compute_dtype = Float32)
    function operand(name, (D, heads, positions, batch); output = false)
        tensor!(
            graph;
            dims = (D, positions, heads, batch),
            strides = (1, D * heads, D, D * heads * positions),
            dtype = H,
            name,
            output,
        )
    end
    q, o, dout = operand("Q", qdims), operand("O", qdims), operand("dO", qdims)
    k, v = operand("K", kdims), operand("V", kdims)
    dq = operand("dQ", qdims; output = true)
    dk = operand("dK", kdims; output = true)
    dv = operand("dV", kdims; output = true)
    _, heads, queries, batch = qdims
    stats = tensor!(graph; dims = (1, queries, heads, batch), dtype = Float32, name = "Stats")
    scale = scalar!(graph, Float32; rank = 4, name = "Scale")
    sdpa_bwd!(graph, q, k, v, o, dout, stats; dQ = dq, dK = dk, dV = dv, scale, causal)
    graph
end

# By-value inputs shared by both directions. The masks' fill value and the
# composite backward's dropout-scale slot are graph inputs, not attributes.
function scalar_values(graph, scale)
    values = Dict{String, Any}("Scale" => scale)
    names = Set(t.name for t in graph.variant_tensors)
    "MaskValue" in names && (values["MaskValue"] = -Inf32)
    "One" in names && (values["One"] = 1.0f0)
    values
end

function launch_summary(direction, T, H, graph)
    precision = T === H ? "$(nameof(H)) operands" : "$(nameof(T)) operands rounded to $(nameof(H))"
    "cuDNN attention $direction: $precision, Float32 softmax and accumulation; " *
    "engine workspace $(graph.workspace_size) bytes"
end

function bound_launch(runtime, graph, values, engine, before, after, summary)
    pack, scalars = bind_graph(graph, values, engine)
    GraphLaunch(runtime_handle(runtime), graph, pack, scalars, before, after, summary)
end

## Construction

function build_implementation(
    implementation::CuDNNKernels,
    request::CallRequest,
    runtime,
    out,
    inputs,
)
    f = request.call.f
    f === ATTENTION_FORWARD && return build_public_forward(implementation, request, runtime, out, inputs)
    f === sdpa_forward && return build_private_forward(request, runtime, out, inputs)
    f === sdpa_backward && return build_private_backward(request, runtime, out, inputs)
    throw(ArgumentError("cuDNN cannot build $f"))
end

# The public verb returns lse in the operand dtype and Tempest's order. Half
# operands write the statistic to a Float32 workspace and round it into lse;
# Float32 operands are rounded into workspaces, and the half output widened.
function build_public_forward(implementation, request, runtime, out, inputs)
    (; causal, scale) = request.call.kwargs
    q, k, v = inputs
    T = eltype(q)
    H = operand_type(implementation, T)
    graph = forward_graph(H, size(q), size(k), dense_strides(size(out.lse)), causal)
    built = build_graph(graph, "attention forward with $H operands")
    built isa Unsupported && return built
    engine = WorkspaceSpec(UInt8, graph.workspace_size)
    if H === T
        specs = (engine, WorkspaceSpec(Float32, length(out.lse)))
    else
        specs = (engine, map(x -> WorkspaceSpec(H, length(x)), (q, k, v, out.o))...)
    end
    summary = launch_summary("forward", T, H, graph)
    ImplementationPlan(specs) do buffers
        values = scalar_values(graph, scale)
        if H === T
            stats = workspace_array(buffers[2], size(out.lse))
            merge!(values, Dict("Q" => q, "K" => k, "V" => v, "O" => out.o, "Stats" => stats))
            before, after = (), (conversion(out.lse, stats),)
        else
            qh, kh, vh, oh = map(workspace_array, buffers[2:5], size.((q, k, v, out.o)))
            merge!(values, Dict("Q" => qh, "K" => kh, "V" => vh, "O" => oh, "Stats" => out.lse))
            before = (conversion(qh, q), conversion(kh, k), conversion(vh, v))
            after = (conversion(out.o, oh),)
        end
        launch = bound_launch(runtime, graph, values, buffers[1], before, after, summary)
        BuiltImplementation(launch, (out, inputs, buffers), 0)
    end
end

function build_private_forward(request, runtime, out, inputs)
    (; causal, scale) = request.call.kwargs
    q, k, v = inputs
    H = eltype(q)
    _, heads, queries, batch = size(q)
    # Packing from Stiletto's Sdpa emission (Stiletto src/compiler/emission.jl).
    stats_strides = (1, queries, 1, queries * heads)
    graph = forward_graph(H, size(q), size(k), stats_strides, causal)
    built = build_graph(graph, "attention forward with $H operands")
    built isa Unsupported && return built
    summary = launch_summary("forward", H, H, graph)
    ImplementationPlan((WorkspaceSpec(UInt8, graph.workspace_size),)) do buffers
        values = scalar_values(graph, scale)
        merge!(values, Dict("Q" => q, "K" => k, "V" => v, "O" => out.o, "Stats" => out.stats))
        launch = bound_launch(runtime, graph, values, only(buffers), (), (), summary)
        BuiltImplementation(launch, (out, inputs, buffers), 0)
    end
end

# Besides its engine workspace, the composite backward writes Float32 row sums
# and a Float32 dQ accumulator and, with grouped heads, per-query-head dK and dV
# before folding them onto the shared heads. The engine initializes all of them,
# so shared workspace slots suffice.
function build_private_backward(request, runtime, out, inputs)
    (; causal, scale) = request.call.kwargs
    q, k, v, o, stats, dout = inputs
    H = eltype(q)
    D, heads, queries, batch = size(q)
    keys = size(k, 3)
    graph = backward_graph(H, size(q), size(k), causal)
    built = build_graph(graph, "attention backward with $H operands and $D features")
    built isa Unsupported && return built
    grouped = size(k, 2) != heads
    full_head = (D, keys, heads, batch)
    specs = (
        WorkspaceSpec(UInt8, graph.workspace_size),
        WorkspaceSpec(Float32, length(stats)),
        WorkspaceSpec(Float32, length(q)),
        (grouped ? (WorkspaceSpec(H, prod(full_head)), WorkspaceSpec(H, prod(full_head))) : ())...,
    )
    summary = launch_summary("backward", H, H, graph)
    ImplementationPlan(specs) do buffers
        values = scalar_values(graph, scale)
        merge!(
            values,
            Dict(
                "Q" => q,
                "K" => k,
                "V" => v,
                "O" => o,
                "Stats" => stats,
                "dO" => dout,
                "dQ" => out.dq,
                "dK" => out.dk,
                "dV" => out.dv,
                "SoftmaxSum" => workspace_array(buffers[2], (1, queries, heads, batch)),
                "dQAccum" => workspace_array(buffers[3], (D, queries, heads, batch)),
            ),
        )
        if grouped
            values["dKFullhead"] = workspace_array(buffers[4], full_head)
            values["dVFullhead"] = workspace_array(buffers[5], full_head)
        end
        launch = bound_launch(runtime, graph, values, buffers[1], (), (), summary)
        BuiltImplementation(launch, (out, inputs, buffers), 0)
    end
end
