# Fused scaled dot-product attention (FlashAttention-2 style) for the
# `attention_forward` and `attention_backward` verbs. Scores are never
# materialized: the forward pass keeps an online softmax per query tile and
# stores the log-sum-exp; the backward pass recomputes the probabilities from it.
# dK and dV accumulate per key block and dQ per query block, so no atomics are
# needed and the result is deterministic. Products run on tensor cores in
# TensorFloat-32, so the kernels are offered only under
# `CUDATarget(matmul_precision = :tf32)`; full precision keeps the composed form.
using Tempest: CallRequest

@inline tensor_float(a) = convert(ct.Tile{ct.TFloat32}, a)
@inline mm(a, b, c) = muladd(tensor_float(a), tensor_float(b), c)

# Arrays are (features, heads, positions, batch). The second grid axis covers
# heads × batch; query head h reads key/value head cld(h, G).
@inline function head_tile(x, head, block, batch, D, M)
    reshape(ct.load(x; index = (Int32(1), head, block, batch), shape = (D, 1, M, 1)), (D, M))
end

@inline function row_tile(x, head, block, batch, M)
    reshape(ct.load(x; index = (Int32(1), head, block, batch), shape = (1, 1, M, 1)), (1, M))
end

@inline store_head!(x, tile, head, block, batch, D, M) =
    ct.store(x; index = (Int32(1), head, block, batch), tile = reshape(tile, (D, 1, M, 1)))

function attention_forward_kernel(out, lse, q, k, v, scale::Float32, D::Int, M::Int, N::Int, causal::Bool, H::Int, G::Int)
    block = ct.bid(1)
    head = mod1(ct.bid(2), Int32(H))
    batch = cld(ct.bid(2), Int32(H))
    kv = cld(head, Int32(G))
    keys = size(k, 3)
    qt = head_tile(q, head, block, batch, D, M)
    offs_m = reshape((block - Int32(1)) * Int32(M) .+ ct.arange(M; dtype = Int32), (1, M))
    m_i = fill(-Inf32, (1, M))
    l_i = zeros(Float32, (1, M))
    acc = zeros(Float32, (D, M))
    last = causal ? cld(block * Int32(M), Int32(N)) : cld(keys, Int32(N))
    for j in Int32(1):last
        kt = head_tile(k, kv, j, batch, D, N)
        s = mm(transpose(kt), qt, zeros(Float32, (N, M))) .* scale
        if causal
            offs_n = reshape((j - Int32(1)) * Int32(N) .+ ct.arange(N; dtype = Int32), (N, 1))
            s = s .+ ifelse.(offs_m .>= offs_n, 0.0f0, -Inf32)
        end
        m_new = max.(m_i, maximum(s; dims = 1))
        p = exp.(s .- m_new)
        alpha = exp.(m_i .- m_new)
        l_i = l_i .* alpha .+ sum(p; dims = 1)
        acc = acc .* alpha
        acc = mm(head_tile(v, kv, j, batch, D, N), p, acc)
        m_i = m_new
    end
    store_head!(out, acc ./ l_i, head, block, batch, D, M)
    store_head!(lse, m_i .+ log.(l_i), head, block, batch, 1, M)
    return
end

# delta[m] = sum(dout[:, m] .* out[:, m]) - dlse[m], the softmax backward's row
# term; a log-sum-exp cotangent enters the scores exactly like this.
function attention_delta_kernel(delta, o, dout, dlse, D::Int, M::Int, H::Int)
    block = ct.bid(1)
    head = mod1(ct.bid(2), Int32(H))
    batch = cld(ct.bid(2), Int32(H))
    dt = head_tile(dout, head, block, batch, D, M)
    ot = head_tile(o, head, block, batch, D, M)
    g = row_tile(dlse, head, block, batch, M)
    store_head!(delta, sum(dt .* ot; dims = 1) .- g, head, block, batch, 1, M)
    return
end

# dK, dV for one key block of one key/value head: loops over the query heads
# of its group and the query blocks that can see the keys.
function attention_dkdv_kernel(dk, dv, q, k, v, lse, delta, dout, scale::Float32, D::Int, M::Int, N::Int, causal::Bool, H::Int, G::Int)
    block = ct.bid(1)
    kv_heads = Int32(H ÷ G)
    kv = mod1(ct.bid(2), kv_heads)
    batch = cld(ct.bid(2), kv_heads)
    queries = size(q, 3)
    kT = transpose(head_tile(k, kv, block, batch, D, N))
    vT = transpose(head_tile(v, kv, block, batch, D, N))
    offs_n = reshape((block - Int32(1)) * Int32(N) .+ ct.arange(N; dtype = Int32), (N, 1))
    acc_k = zeros(Float32, (D, N))
    acc_v = zeros(Float32, (D, N))
    first = causal ? fld((block - Int32(1)) * Int32(N), Int32(M)) + Int32(1) : Int32(1)
    for g in Int32(1):Int32(G)
        head = (kv - Int32(1)) * Int32(G) + g
        for i in first:cld(queries, Int32(M))
            qt = head_tile(q, head, i, batch, D, M)
            dt = head_tile(dout, head, i, batch, D, M)
            l = row_tile(lse, head, i, batch, M)
            r = row_tile(delta, head, i, batch, M)
            s = mm(kT, qt, zeros(Float32, (N, M))) .* scale
            p = exp.(s .- l)
            if causal
                offs_m = reshape((i - Int32(1)) * Int32(M) .+ ct.arange(M; dtype = Int32), (1, M))
                p = ifelse.(offs_m .>= offs_n, p, 0.0f0)
            end
            acc_v = mm(dt, transpose(p), acc_v)
            dp = mm(vT, dt, zeros(Float32, (N, M)))
            ds = p .* (dp .- r) .* scale
            acc_k = mm(qt, transpose(ds), acc_k)
        end
    end
    store_head!(dk, acc_k, kv, block, batch, D, N)
    store_head!(dv, acc_v, kv, block, batch, D, N)
    return
end

# dQ for one query block of one query head; loops over the visible key blocks.
function attention_dq_kernel(dq, q, k, v, lse, delta, dout, scale::Float32, D::Int, M::Int, N::Int, causal::Bool, H::Int, G::Int)
    block = ct.bid(1)
    head = mod1(ct.bid(2), Int32(H))
    batch = cld(ct.bid(2), Int32(H))
    kv = cld(head, Int32(G))
    keys = size(k, 3)
    qt = head_tile(q, head, block, batch, D, M)
    dt = head_tile(dout, head, block, batch, D, M)
    l = row_tile(lse, head, block, batch, M)
    r = row_tile(delta, head, block, batch, M)
    offs_m = reshape((block - Int32(1)) * Int32(M) .+ ct.arange(M; dtype = Int32), (1, M))
    acc = zeros(Float32, (D, M))
    last = causal ? cld(block * Int32(M), Int32(N)) : cld(keys, Int32(N))
    for j in Int32(1):last
        kt = head_tile(k, kv, j, batch, D, N)
        vt = head_tile(v, kv, j, batch, D, N)
        s = mm(transpose(kt), qt, zeros(Float32, (N, M))) .* scale
        p = exp.(s .- l)
        if causal
            offs_n = reshape((j - Int32(1)) * Int32(N) .+ ct.arange(N; dtype = Int32), (N, 1))
            p = ifelse.(offs_m .>= offs_n, p, 0.0f0)
        end
        dp = mm(transpose(vt), dt, zeros(Float32, (N, M)))
        ds = p .* (dp .- r) .* scale
        acc = mm(kt, ds, acc)
    end
    store_head!(dq, acc, head, block, batch, D, M)
    return
end

const ATTENTION_TILE = 64

function attention_shape_support(q, k, v)
    D = q.dims[1]
    D == v.dims[1] || return Unsupported("fused attention needs equal key and value features")
    D in (16, 32, 64, 128) || return Unsupported("fused attention needs 16, 32, 64 or 128 features")
    all(s -> s.dtype === Float32, (q, k, v)) || return Unsupported("fused attention is Float32")
    q.dims[3] % ATTENTION_TILE == 0 && k.dims[3] % ATTENTION_TILE == 0 ||
        return Unsupported("fused attention needs positions divisible by $ATTENTION_TILE")
    Supported()
end

function support(::CuTileKernels, request::CallRequest)
    f = request.call.f
    f in (Tempest.attention_forward, Tempest.attention_backward) ||
        return Unsupported("no cuTile kernel for $f")
    request.target isa CUDATarget || return Unsupported("requires a CUDA target")
    request.target.matmul_precision === :tf32 ||
        return Unsupported("fused attention computes products in TF32; the target asks for full precision")
    q, k, v = request.inputs
    attention_shape_support(q, k, v)
end

function build_implementation(::CuTileKernels, request::CallRequest, runtime, out, inputs)
    (; causal, scale) = request.call.kwargs
    q, k = inputs[1], inputs[2]
    D, H, T, B = size(q)
    KV, keys = size(k, 2), size(k, 3)
    tile = ct.Constant(ATTENTION_TILE)
    heads = (ct.Constant(H), ct.Constant(H ÷ KV))
    shape = (ct.Constant(D), tile, tile, ct.Constant(causal), heads...)
    grid = (cld(T, ATTENTION_TILE), H * B)
    if request.call.f === Tempest.attention_forward
        q, k, v = inputs
        o, lse = out.o, out.lse
        kernel = CUDACore.@cuda backend=ct launch=false attention_forward_kernel(o, lse, q, k, v, scale, shape...)
        launch = () -> begin
            args = ct.cuTileconvert((o, lse, q, k, v))
            kernel(args..., scale, shape...; blocks = grid)
            nothing
        end
        return BuiltImplementation(launch, (out, inputs, kernel), 0)
    end
    q, k, v, o, lse, dout, dlse = inputs
    dq, dk, dv = out.dq, out.dk, out.dv
    specification = (WorkspaceSpec(Float32, length(lse)),)
    ImplementationPlan(specification) do buffers
        delta = reshape(view(only(buffers), 1:length(lse)), size(lse))
        delta_kernel = CUDACore.@cuda backend=ct launch=false attention_delta_kernel(
            delta, o, dout, dlse, ct.Constant(D), tile, ct.Constant(H),
        )
        dkdv = CUDACore.@cuda backend=ct launch=false attention_dkdv_kernel(
            dk, dv, q, k, v, lse, delta, dout, scale, shape...,
        )
        dqk = CUDACore.@cuda backend=ct launch=false attention_dq_kernel(
            dq, q, k, v, lse, delta, dout, scale, shape...,
        )
        launch = () -> begin
            d = ct.cuTileconvert((delta, o, dout, dlse))
            delta_kernel(d..., ct.Constant(D), tile, ct.Constant(H); blocks = grid)
            a = ct.cuTileconvert((dk, dv, q, k, v, lse, delta, dout))
            dkdv(a..., scale, shape...; blocks = (cld(keys, ATTENTION_TILE), KV * B))
            b = ct.cuTileconvert((dq, q, k, v, lse, delta, dout))
            dqk(b..., scale, shape...; blocks = grid)
            nothing
        end
        BuiltImplementation(launch, (out, inputs, buffers, delta_kernel, dkdv, dqk), 0)
    end
end
