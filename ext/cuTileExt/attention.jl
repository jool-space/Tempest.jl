# Hand-written attention kernels (FlashAttention-2 style): an escape hatch at
# the edge. They claim the attention verbs; the planner sees one opaque group
# per call, whose boundary is storage. Scores are never materialized: the
# forward pass keeps an online softmax per query tile and stores the
# log-sum-exp; the backward pass recomputes the probabilities from it. dK and dV
# accumulate per key block, dQ per query block, so no atomics are needed.
#
# The forward kernel applies an epilogue to its output tile before storing it:
# the pointwise terms the planner absorbed after the attention output, such as
# a residual add, compiled into a callable that carries the arrays it reads.
# Limits: pointwise work only; reads aligned with the output tile (the same
# index, or extent one along an axis); positions are not available; each
# distinct epilogue compiles its own kernel variant.

struct cuTileAttention <: Tempest.Implementation end

const ATTENTION_TILE = 64

@inline tensor_float(a) = convert(ct.Tile{ct.TFloat32}, a)
@inline mm(a, b, c, tf32::Bool) = tf32 ? muladd(tensor_float(a), tensor_float(b), c) : muladd(a, b, c)

# Arrays are (features, heads, positions, batch). The second grid axis covers
# heads × batch; query head h reads key/value head cld(h, G).
@inline head_tile(x, head, block, batch, D, M) =
    reshape(ct.load(x; index = (Int32(1), head, block, batch), shape = (D, 1, M, 1)), (D, M))
@inline row_tile(x, head, block, batch, M) =
    reshape(ct.load(x; index = (Int32(1), head, block, batch), shape = (1, 1, M, 1)), (1, M))
@inline store_head!(x, tile, head, block, batch, D, M) =
    ct.store(x; index = (Int32(1), head, block, batch), tile = reshape(tile, (D, 1, M, 1)))
# A result nobody reads is passed as `nothing` and not stored.
@inline store_head!(::Nothing, tile, head, block, batch, D, M) = nothing

"An epilogue: a generated function of the output tile, its index, and the arrays it reads."
struct Epilogue{F, A}
    f::F
    arrays::A
end
(e::Epilogue)(y, index) = e.f(y, e.arrays, index)

struct NoEpilogue end
(::NoEpilogue)(y, index) = y

function attention_forward_kernel(out, lse, q, k, v, epilogue, scale::Float32, D::Int, M::Int, N::Int,
        causal::Bool, H::Int, G::Int, tf32::Bool)
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
        s = mm(transpose(kt), qt, zeros(Float32, (N, M)), tf32) .* scale
        if causal
            offs_n = reshape((j - Int32(1)) * Int32(N) .+ ct.arange(N; dtype = Int32), (N, 1))
            s = s .+ ifelse.(offs_m .>= offs_n, 0.0f0, -Inf32)
        end
        m_new = max.(m_i, maximum(s; dims = 1))
        p = exp.(s .- m_new)
        alpha = exp.(m_i .- m_new)
        l_i = l_i .* alpha .+ sum(p; dims = 1)
        acc = acc .* alpha
        acc = mm(head_tile(v, kv, j, batch, D, N), p, acc, tf32)
        m_i = m_new
    end
    index = (Int32(1), head, block, batch)
    ct.store(out; index, tile = epilogue(reshape(acc ./ l_i, (D, 1, M, 1)), index))
    store_head!(lse, m_i .+ log.(l_i), head, block, batch, 1, M)
    return
end

# delta[m] = Σ_d dout·o − dlse, the softmax backward's row term.
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
function attention_dkdv_kernel(dk, dv, q, k, v, lse, delta, dout, scale::Float32, D::Int, M::Int, N::Int,
        causal::Bool, H::Int, G::Int, tf32::Bool)
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
            s = mm(kT, qt, zeros(Float32, (N, M)), tf32) .* scale
            p = exp.(s .- l)
            if causal
                offs_m = reshape((i - Int32(1)) * Int32(M) .+ ct.arange(M; dtype = Int32), (1, M))
                p = ifelse.(offs_m .>= offs_n, p, 0.0f0)
            end
            acc_v = mm(dt, transpose(p), acc_v, tf32)
            dp = mm(vT, dt, zeros(Float32, (N, M)), tf32)
            ds = p .* (dp .- r) .* scale
            acc_k = mm(qt, transpose(ds), acc_k, tf32)
        end
    end
    store_head!(dk, acc_k, kv, block, batch, D, N)
    store_head!(dv, acc_v, kv, block, batch, D, N)
    return
end

# dQ for one query block of one query head; loops over the visible key blocks.
function attention_dq_kernel(dq, q, k, v, lse, delta, dout, scale::Float32, D::Int, M::Int, N::Int,
        causal::Bool, H::Int, G::Int, tf32::Bool)
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
        s = mm(transpose(kt), qt, zeros(Float32, (N, M)), tf32) .* scale
        p = exp.(s .- l)
        if causal
            offs_n = reshape((j - Int32(1)) * Int32(N) .+ ct.arange(N; dtype = Int32), (N, 1))
            p = ifelse.(offs_m .>= offs_n, p, 0.0f0)
        end
        dp = mm(transpose(vt), dt, zeros(Float32, (N, M)), tf32)
        ds = p .* (dp .- r) .* scale
        acc = mm(kt, ds, acc, tf32)
    end
    store_head!(dq, acc, head, block, batch, D, M)
    return
end

# Claimed only under TF32: with full Float32 tile products the forward kernel
# took 8.65 ms on GB10 at d=64, 12 heads, T=256, B=16, causal, against 2.13 ms
# for today's composed form on cuBLAS. The kernels keep the full-precision
# variant behind their `tf32` flag. A claim competes by cost with the
# attention generated from the reference.
function Tempest.claims_call(::cuTileAttention, target::CUDATarget, f, kwargs, inputs)
    target.precision === :tf32 || return false
    f in (Tempest.attention_forward, Tempest.attention_backward) || return false
    (qd, qt), (kd, kt), (vd, vt) = inputs[1], inputs[2], inputs[3]
    qt === kt === vt === Float32 || return false
    qd[1] in (16, 32, 64, 128) && vd[1] == qd[1] || return false
    qd[3] % ATTENTION_TILE == 0 && kd[3] % ATTENTION_TILE == 0 || return false
    all(i -> i[2] === Float32, inputs)
end

Tempest.epilogue_results(::cuTileAttention, ::typeof(Tempest.attention_forward), kwargs) = (1,)

# The launches' work: the forward kernel computes two products per query and
# key, the backward a row term, then four products per key block and three
# per query block; a causal mask halves them. Products run at the generator's
# rate.
function Tempest.estimate(::cuTileAttention, claim, target, p::Plan, g::Group)
    _, d = call_parts(p, g)
    op = d.node.op
    (D, H, T, B), (_, KV, keys, _) = first(d.shapes[1]), first(d.shapes[2])
    product = 2.0 * D * T * keys * H * B / (op.kwargs.causal ? 2 : 1)
    rate, bytes = product_rate(target, Float32), Tempest.group_bytes(p, g)
    blocks = cld(T, ATTENTION_TILE) * H * B
    op.f === Tempest.attention_forward && return [Tempest.Work(bytes; flops = 2product, rate, blocks)]
    [Tempest.Work(0.0; blocks), Tempest.Work(bytes; flops = 4product, rate, blocks = cld(keys, ATTENTION_TILE) * KV * B),
     Tempest.Work(0.0; flops = 3product, rate, blocks)]
end

# The opaque results of the group, by result number, and the operands.
function call_parts(p, g)
    outs = sort([v for v in g.outputs if p.graph.values[v].definition isa Tempest.Opaque];
        by = v -> p.graph.values[v].definition.node.op.output)
    d = p.graph.values[first(outs)].definition
    outs, d
end

tile_array(buffer, dims) = ct.TileArray(reshape(buffer, dims))

# The epilogue of result `o`: its terms as a function of the output tile `y`,
# the tile's index, and the arrays it loads, aligned with the tile.
function epilogue_for(p, g, o, buffers, shape)
    haskey(g.epilogues, o) || return NoEpilogue(), o
    e = g.epilogues[o]
    t = g.terms[e]
    graph = p.graph
    rank = length(shape)
    # The group's domain is the result's axes wider than one; a load spans
    # those it reads along.
    result = graph.values[o].definition.node.dims
    wide = Tempest.wide_axes(result)
    function spanned(s)
        axes = Tempest.aligned_axes(graph, s, g.domain)
        ntuple(k -> k in wide && findfirst(==(k), wide) in axes ? result[k] : 1, rank)
    end
    arrays = Any[]
    slot = Dict{Int, Int}()
    names = Dict{Term, Any}()
    body = Any[]
    for s in Tempest.subterms((t,))
        if s isa Lit
            names[s] = s.value
            continue
        end
        name = gensym(:e)
        expr = if s isa Stat
            :y
        elseif s isa Load
            dims = spanned(s)
            j = get!(slot, s.value) do
                push!(arrays, tile_array(buffers[s.value], dims))
                length(arrays)
            end
            index = ntuple(k -> dims[k] == 1 ? :(Int32(1)) : :(index[$k]), rank)
            tile = ntuple(k -> dims[k] == 1 ? 1 : shape[k], rank)
            :($(ct.load)(arrays[$j]; index = ($(index...),), shape = $tile))
        elseif s isa Apply
            args = map(a -> names[a], s.args)
            s.f isa Tempest.Cast ? :(convert($(ct.Tile){$(s.eltype)}, $(only(args)))) : :($(s.f).($(args...)))
        else
            error("no epilogue lowering for $(typeof(s))")
        end
        push!(body, :($name = $expr))
        names[s] = name
    end
    f = Tempest.generated(:(function (y, arrays, index)
        $(body...)
        return $(names[t])
    end))
    Epilogue(f, Tuple(arrays)), e
end


struct BuiltAttention
    f::Any
    epilogue::Bool
    tf32::Bool
end

Base.show(io::IO, b::BuiltAttention) = print(io, "cuTile attention kernels for ", b.f,
    b.tf32 ? " (TF32)" : " (Float32)", b.epilogue ? ", with an epilogue on the output" : "")

function Tempest.build_call(::cuTileAttention, p::Plan, g::Group, buffers, target::CUDATarget)
    outs, d = call_parts(p, g)
    op = d.node.op
    (; causal, scale) = op.kwargs
    tf32 = target.precision === :tf32
    shapes = map(first, d.shapes)
    arrays = map((o, s) -> tile_array(buffers[o.value], s), d.operands, shapes)
    q, k = shapes[1], shapes[2]
    D, H, T, B = q
    KV, keys = k[2], k[3]
    G = H ÷ KV
    tile = ct.Constant(ATTENTION_TILE)
    constants = (ct.Constant(D), tile, tile, ct.Constant(causal), ct.Constant(H), ct.Constant(G), ct.Constant(tf32))
    grid = (cld(T, ATTENTION_TILE), H * B)
    s = Float32(scale)
    if op.f === Tempest.attention_forward
        o, lse = outs
        epilogue, target_o = epilogue_for(p, g, o, buffers, (D, 1, ATTENTION_TILE, 1))
        lse_array = buffers[lse] === nothing ? nothing : tile_array(buffers[lse], (1, H, T, B))
        args = (tile_array(buffers[target_o], (D, H, T, B)), lse_array, arrays..., epilogue, s, constants...)
        return (tile_launch(attention_forward_kernel, args, grid), BuiltAttention(op.f, !(epilogue isa NoEpilogue), tf32))
    end
    qa, ka, va, oa, lsea, douta, dlsea = arrays
    # The kernels write all three gradients; one nobody reads goes to scratch.
    places = map(r -> something(buffers[r], CuArray{Float32}(undef, prod(p.graph.values[r].dims))), outs)
    dq, dk, dv = map((r, b) -> tile_array(b, p.graph.values[r].definition.node.dims), outs, places)
    workspace = CuArray{Float32}(undef, H * T * B)
    delta = tile_array(workspace, (1, H, T, B))
    delta_args = (delta, oa, douta, dlsea, ct.Constant(D), tile, ct.Constant(H))
    dkdv_args = (dk, dv, qa, ka, va, lsea, delta, douta, s, constants...)
    dq_args = (dq, qa, ka, va, lsea, delta, douta, s, constants...)
    kv_grid = (cld(keys, ATTENTION_TILE), KV * B)
    kernels = ((attention_delta_kernel, delta_args, grid), (attention_dkdv_kernel, dkdv_args, kv_grid),
        (attention_dq_kernel, dq_args, grid))
    launches = Tempest.concurrently(k -> tile_launch(k...), kernels, target)
    launch = () -> (foreach(l -> l(), launches); workspace; places; nothing)
    (launch, BuiltAttention(op.f, false, tf32))
end
