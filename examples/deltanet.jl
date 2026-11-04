# Gated DeltaNet's forward, chunked
#
# The gated delta rule, S = exp(a) (I - β k kᵀ) S + β k vᵀ and o = Sᵀ q at each
# position, in flash-linear-attention's chunked form: within a chunk of 64
# positions everything is products, and across chunks a matrix state is
# carried, written as Base's `accumulate` over the chunks. Each chunk needs
# (I + A)⁻¹ for a strictly lower triangular A. By doubling, (I - A)(I + A²)
# (I + A⁴)…, that is ten products of whole 64 × 64 matrices; by blocks, as
# flash-linear-attention's `solve_tril` does, the 16 × 16 blocks on the
# diagonal are inverted and merged pairwise, multiplying a fourteenth of the
# flops. Both are plain Julia; the reference is the rule token by token in
# Float64 on the CPU.
#
#   julia --project=test examples/deltanet.jl

using Tempest, CUDACore, cuTile, cuBLAS, NNlib, Printf

tr(x) = permutedims(x, (2, 1, 3))

# (I + A)⁻¹ for A strictly lower triangular, n × n: A^n = 0, so the product
# of (I + (-A)^(2^j)) for 2^j < n is the whole series.
function doubling(A)
    n = size(A, 1)
    I = eltype(A).(reshape(1:n, :, 1) .== reshape(1:n, 1, :))
    L = -A
    inverse = I .+ L
    for _ in 2:round(Int, log2(n))
        L = L ⊠ L
        inverse = inverse ⊠ (I .+ L)
    end
    inverse
end

# The inverse of [P 0; Q R] from P⁻¹ and R⁻¹.
merged(Pi, Ri, Q) = vcat(hcat(Pi, zero(Pi)), hcat(-(Ri ⊠ (Q ⊠ Pi)), Ri))

# The same by blocks: the b × b blocks on the diagonal, side by side along the
# batch, inverted by doubling; then neighbours merged, b → 2b → … → n.
function blocks(A; b = 16)
    n, m = size(A, 1), size(A, 3)
    r(i, w) = ((i - 1) * w + 1):(i * w)
    part(x, k) = x[:, :, r(k, m)]
    inverse = doubling(cat([A[r(i, b), r(i, b), :] for i in 1:(n ÷ b)]...; dims = 3))
    w = b
    while w < n
        pairs = 1:(n ÷ 2w)
        Q = cat([A[r(2j, w), r(2j - 1, w), :] for j in pairs]...; dims = 3)
        Pi = cat([part(inverse, 2j - 1) for j in pairs]...; dims = 3)
        Ri = cat([part(inverse, 2j) for j in pairs]...; dims = 3)
        inverse = merged(Pi, Ri, Q)
        w *= 2
    end
    inverse
end

# q and k are (dk, positions, heads), v (dv, positions, heads), a and β
# (positions, heads); the result is (positions, dv, heads).
function deltanet(q, k, v, a, β; C = 64, inverse = blocks)
    dk, T, H = size(q)
    N = T ÷ C
    E = eltype(v)
    cols(x) = reshape(x, size(x, 1), C, N * H)
    K, Q = cols(k), cols(q)
    Kr, Qr, Vr = tr(K), tr(Q), tr(cols(v))
    ca = cumsum(reshape(a, C, 1, N * H); dims = 1)
    b = reshape(β, C, 1, N * H)
    t, s = reshape(1:C, :, 1), reshape(1:C, 1, :)
    decay(keep) = exp.(ifelse.(keep, ca .- tr(ca), typemin(E)))
    # Within each chunk: the WY form's W and Ũ, and the queries' scores.
    T⁻¹ = inverse(b .* decay(t .> s) .* (Kr ⊠ K))
    W = T⁻¹ ⊠ (b .* exp.(ca) .* Kr)
    Ũ = T⁻¹ ⊠ (b .* Vr)
    P = decay(t .>= s) .* (Qr ⊠ K)
    Qγ = exp.(ca) .* Qr
    Kd = exp.(ca[C:C, :, :] .- ca) .* Kr
    g = exp.(ca[C:C, :, :])
    # Across chunks: the state carried, each chunk's output beside it.
    chunks(x) = eachslice(reshape(x, size(x, 1), size(x, 2), N, H); dims = 3)
    S0 = fill!(similar(v, dk, size(v, 1), H), 0)
    steps = accumulate(zip(chunks(Ũ), chunks(W), chunks(Kd), chunks(g), chunks(Qγ), chunks(P)); init = (S0, nothing)) do (S, _), (ũ, w, kd, g, qγ, p)
        U = ũ .- w ⊠ S
        o = qγ ⊠ S .+ p ⊠ U
        (reshape(g, 1, 1, H) .* S .+ tr(kd) ⊠ U, o)
    end
    vcat(last.(steps)...)
end

# The reference: the rule token by token.
function sequential(q, k, v, a, β)
    dk, T, H = size(q)
    o = zeros(T, size(v, 1), H)
    for h in 1:H
        S = zeros(dk, size(v, 1))
        for t in 1:T
            kt = k[:, t, h]
            S = exp(a[t, h]) .* (S .- β[t, h] .* kt * (kt' * S)) .+ β[t, h] .* kt * v[:, t, h]'
            o[t, :, h] = S' * q[:, t, h]
        end
    end
    o
end

# 1024 positions, 32 heads, Qwen3.5's 128 features; unit keys, gates in (0.88, 0.98).
T, H, d = 1024, 32, 128
w(n, s) = sin.(s .* reshape(1:n, :))
unit(x) = x ./ sqrt.(sum(abs2, x; dims = 1))
q = unit(reshape(w(d * T * H, 0.37), d, T, H)) ./ sqrt(d)
k = unit(reshape(w(d * T * H, 0.53), d, T, H))
v = reshape(w(d * T * H, 0.71), d, T, H)
a = -0.05 .* (1.5 .+ reshape(w(T * H, 0.13), T, H))
β = 1 ./ (1 .+ exp.(-reshape(w(T * H, 0.29), T, H)))
reference = sequential(q, k, v, a, β)
args = map(x -> CuArray(Float32.(x)), (q, k, v, a, β))

function run(name, inverse)
    c = compile((x...) -> deltanet(x...; inverse), args...)
    o = c(args...)
    ms = minimum(1:20) do _
        t = time_ns()
        c(args...)
        CUDACore.synchronize()
        (time_ns() - t) / 1.0e6
    end
    err = maximum(abs, Array(o) .- reference) / maximum(abs, reference)
    @printf("%-9s relative difference from Float64 %.1e, %d kernels, %.2f ms\n", name, err, length(c.executable.plan.groups), ms)
    c
end

run("doubling", doubling)
c = run("blocks", blocks)
println()
explain(c)
