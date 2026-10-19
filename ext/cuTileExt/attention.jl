# Hand-written attention kernels (FlashAttention-2 style): an escape hatch at
# the edge. They are cuTile's kernels for the attention operations; the planner
# sees one opaque group per call, whose boundary is storage. Scores are never
# materialized: the forward pass keeps an online softmax per query tile and
# stores the log-sum-exp; the backward pass recomputes the probabilities from
# it. dK and dV accumulate per key block, dQ per query block, so no atomics are
# needed.
#
# The forward kernel applies the term after its output to the output tile
# before storing it: the pointwise work the planner absorbed after the
# attention output, such as a residual add, compiled into a tile function that
# carries the arrays it reads (see `epilogue_term`). Limits: pointwise work
# only; reads aligned with the output tile (the same index, or extent one along
# an axis); positions are not available; each distinct term compiles its own
# kernel variant.

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
# variant behind their `tf32` flag. Queries and keys come in whole tiles, and
# the head dimension is one a tile product takes.
tiled(target, q, k, v) = target.precision === :tf32 && size(q, 1) in (16, 32, 64, 128) && size(v, 1) == size(q, 1) &&
    size(q, 3) % ATTENTION_TILE == 0 && size(k, 3) % ATTENTION_TILE == 0

# The constants of the tile programs: the head dimension, the query and key
# tiles, the mask, the query heads, the query heads per key/value head, and
# TF32 products.
constants(q, k, causal) =
    map(ct.Constant, (size(q, 1), ATTENTION_TILE, ATTENTION_TILE, causal, size(q, 2), size(q, 2) ÷ size(k, 2), true))

# The flops of a product over every query and key, which a causal mask halves.
products(q, k, causal) = 2.0 * size(q, 1) * size(q, 3) * size(k, 3) * size(q, 2) * size(q, 4) / (causal ? 2 : 1)

# The forward kernel computes two products per query and key, reads the
# arguments and writes the output and the log-sum-exp.
function forward(target::CUDATarget, q::Spec{Float32, 4}, k::Spec{Float32, 4}, v::Spec{Float32, 4}; causal, scale)
    tiled(target, q, k, v) || return nothing
    _, H, T, B = size(q)
    grid = (cld(T, ATTENTION_TILE), H * B)
    cost = Work(2bytes(q) + bytes(k) + bytes(v) + 4H * T * B; flops = 2products(q, k, causal), blocks = prod(grid))
    fixed = constants(q, k, causal)
    Kernel(; cost, epilogue = (o = pointwise,)) do (o, lse), (q, k, v), after
        launch(attention_forward_kernel, grid, o, lse, q, k, v, after.o, Float32(scale), fixed...)
    end
end

# The backward kernels: a row term, then four products per key block, which
# reads the arguments and writes dK and dV, and three per query block, which
# writes dQ. A gradient nobody reads is computed and not stored.
function backward(target::CUDATarget, q::Spec{Float32, 4}, k::Spec{Float32, 4}, v::Spec{Float32, 4}, o::Spec{Float32, 4},
        lse::Spec{Float32, 4}, dout::Spec{Float32, 4}, dlse::Spec{Float32, 4}; causal, scale)
    tiled(target, q, k, v) || return nothing
    D, H, T, B = size(q)
    grid, kv_grid = (cld(T, ATTENTION_TILE), H * B), (cld(size(k, 3), ATTENTION_TILE), size(k, 2) * B)
    product = products(q, k, causal)
    moved = sum(bytes, (q, k, v, o, lse, dout, dlse)) + bytes(q) + bytes(k) + bytes(v)
    cost = [Work(0.0; blocks = prod(grid)), Work(moved; flops = 4product, blocks = prod(kv_grid)),
        Work(0.0; flops = 3product, blocks = prod(grid))]
    fixed = constants(q, k, causal)
    Kernel(; cost, scratch = (Spec(Float32, 1, H, T, B),)) do (dq, dk, dv), (q, k, v, o, lse, dout, dlse), (delta,)
        launch(attention_delta_kernel, grid, delta, o, dout, dlse, ct.Constant(D), ct.Constant(ATTENTION_TILE), ct.Constant(H))
        launch(attention_dkdv_kernel, kv_grid, dk, dv, q, k, v, lse, delta, dout, Float32(scale), fixed...)
        launch(attention_dq_kernel, grid, dq, q, k, v, lse, delta, dout, Float32(scale), fixed...)
    end
end

tile_array(buffer, dims) = ct.TileArray(reshape(buffer, dims))

# The term absorbed after result `r`, as a tile function of the output tile `y`
# and its index that gives the tile stored in its place; one that leaves the
# tile as it is where the group absorbed nothing after `r`. A load of the term
# spans the tile along the axes it reads aligned with the result, and is one
# wide along the others. It takes its extents from the tile, so the term holds
# nothing of the kernel's tiling, and any tile program that stores tiles of the
# result in its program shape can apply it.
function Tempest.Compiler.epilogue_term(::CUDATarget, p, g, r, buffers)
    haskey(g.epilogues, r) || return NoEpilogue()
    graph = p.graph
    t = g.terms[g.epilogues[r]]
    # The group's domain is the result's axes wider than one.
    result = graph.values[r].definition.node.dims
    rank = length(result)
    wide = wide_axes(result)
    function spanned(s)
        axes = aligned_axes(graph.values[s.value].dims, s.index, g.domain)
        ntuple(k -> k in wide && findfirst(==(k), wide) in axes ? result[k] : 1, rank)
    end
    arrays = Any[]
    slot = Dict{Int, Int}()
    names = Dict{Term, Any}()
    body = Any[]
    for s in subterms((t,))
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
            tile = ntuple(k -> dims[k] == 1 ? 1 : :(size(y, $k)), rank)
            :($(ct.load)(arrays[$j]; index = ($(index...),), shape = ($(tile...),)))
        elseif s isa Apply
            args = map(a -> names[a], s.args)
            s.f isa Cast ? :(convert($(ct.Tile){$(s.eltype)}, $(only(args)))) : :($(s.f).($(args...)))
        else
            error("no epilogue lowering for $(typeof(s))")
        end
        push!(body, :($name = $expr))
        names[s] = name
    end
    f = Tempest.Compiler.generated(:(function (y, arrays, index)
        $(body...)
        return $(names[t])
    end))
    Epilogue(f, Tuple(arrays))
end
