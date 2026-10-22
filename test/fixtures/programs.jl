# Programs several tests run: each a function, and its arguments in Float64
# so that finite differences are accurate.

swish(x) = x / (1 + exp(-x))
leaky(x) = x > 0 ? x : oftype(x, 0.01) * x
T.scalar_partials(::typeof(leaky), g, y, x) = (ifelse(x > 0, g, oftype(g, 0.01) * g),)
rmsnorm(x, w) = x ./ sqrt.(sum(abs2, x; dims = 1) ./ size(x, 1) .+ 1.0e-5) .* w
softmax_columns(x) = exp.(x .- T.logsumexp(x; dims = 1))
softmax_rows(x) = exp.(x .- T.logsumexp(x; dims = 2))
batched(a, b) = C.contract(a, b, (1, 2, 4), (2, 3, 4), (1, 3, 4))
function layernorm(x, w)
    m = sum(x; dims = 1) ./ size(x, 1)
    v = sum(abs2, x .- m; dims = 1) ./ size(x, 1)
    (x .- m) ./ sqrt.(v .+ 1.0e-5) .* w
end
mlp(x, w1, b1, w2, b2) = w2 * tanh.(w1 * x .+ b1) .+ b2
gelu(x) = x / 2 * (1 + tanh(oftype(x, 0.7978846) * (x + oftype(x, 0.044715) * x * x * x)))
# A product spelled as a broadcast and a sum over the shared axis.
spelled_product(a, b) = dropdims(sum(reshape(a, size(a, 1), size(a, 2), 1) .* reshape(b, 1, size(b, 1), size(b, 2)); dims = 2); dims = 2)

# Writes into an array the program allocates, as NNlib pads.
function padded(x)
    y = fill!(similar(x, size(x) .+ 2), 0)
    y[2:(end - 1), 2:(end - 1)] = x
    y
end

# Julia arrays a program closes over, as factors of products: constants, in
# the precision of the program's inputs.
const WEIGHTS, BATCH = data(5, 4; scale = 0.2), data(4, 3, 2; scale = 0.3)

const PROGRAMS = [
    ("dot product", (a, b) -> sum(a .* b), (data(32), data(32; scale = 0.3))),
    ("sum abs2 tanh", x -> sum(abs2, tanh.(x)), (data(8, 6),)),
    ("full logsumexp", x -> T.logsumexp(x), (data(8, 6),)),
    ("maximum and shifted sum", x -> sum(exp.(x .- maximum(x))), (data(8, 6),)),
    ("x .- mean", x -> sum(abs2, x .- sum(x) / length(x)), (data(8, 6),)),
    ("column maxima", x -> sum(maximum(x; dims = 1) .* data(1, 6)), (data(8, 6),)),
    ("row minima", x -> sum(minimum(x; dims = 2)), (data(8, 6),)),
    ("products", x -> sum(prod(x; dims = 1)), (data(5, 3; scale = 0.4),)),
    ("softmax columns", (x, t) -> sum(softmax_columns(x) .* t), (data(8, 6), data(8, 6; scale = 0.7))),
    ("softmax rows", (x, t) -> sum(softmax_rows(x) .* t), (data(8, 6), data(8, 6; scale = 0.7))),
    ("RMSNorm", (x, w) -> sum(abs2, rmsnorm(x, w)), (data(8, 6), data(8; scale = 0.1))),
    ("layer norm", (x, w) -> sum(abs2, layernorm(x, w)), (data(8, 6), data(8; scale = 0.1))),
    ("rank 3 softmax", (x, t) -> sum(exp.(x .- T.logsumexp(x; dims = 3)) .* t), (data(4, 3, 5), data(4, 3, 5; scale = 0.2))),
    ("MLP", (x, w1, b1, w2, b2) -> sum(abs2, mlp(x, w1, b1, w2, b2)),
     (data(6, 10), data(7, 6; scale = 0.1), data(7), data(6, 7; scale = 0.2), data(6))),
    ("adjoint product", (a, b) -> sum(tanh.(a' * b)), (data(6, 4), data(6, 5; scale = 0.3))),
    ("product spelled as a broadcast sum", (a, b) -> sum(abs2, spelled_product(a, b)), (data(5, 7; scale = 0.3), data(7, 6))),
    ("linear, bias, GELU", (w, x, b) -> sum(abs2, gelu.(w * x .+ b)), (data(9, 7; scale = 0.2), data(7, 10), data(9))),
    ("residual epilogue", (w, x) -> sum(abs2, w * x .+ x), (data(7, 7; scale = 0.2), data(7, 10))),
    ("scaling prologue", (w, x) -> sum(abs2, w * (x .* 0.5) ./ 3), (data(9, 7; scale = 0.2), data(7, 10))),
    ("products of transposes", (a, b) -> sum(abs2, a' * b' .+ 1), (data(7, 5; scale = 0.2), data(6, 7))),
    ("batched product", (a, b) -> sum(abs2, batched(a, b)), (data(3, 4, 2), data(4, 5, 2; scale = 0.3))),
    ("matrix vector", (a, v) -> sum(abs2, a * v), (data(5, 4), data(4; scale = 0.5))),
    ("a constant times x", x -> sum(abs2, eltype(x).(WEIGHTS) * x), (data(4, 3),)),
    ("x' times a constant", x -> sum(abs2, tanh.(x' * eltype(x).(WEIGHTS))), (data(5, 3),)),
    ("a constant times a vector", v -> sum(abs2, eltype(v).(WEIGHTS) * v), (data(4; scale = 0.5),)),
    ("batched product with a constant", a -> sum(abs2, batched(a, eltype(a).(BATCH))), (data(3, 4, 2),)),
    ("constant batched product", b -> sum(abs2, batched(eltype(b).(BATCH), b)), (data(3, 5, 2),)),
    ("swish", x -> sum(swish.(x)), (data(8, 6),)),
    ("opaque leaky with a rule", x -> sum(abs2, leaky.(x)), (data(8, 6),)),
    ("powers", x -> sum(x .^ 2 .+ x .^ 3 .+ sqrt.(abs.(x) .+ 1)), (data(8, 6),)),
    ("clamp and max", x -> sum(clamp.(x, -0.5, 0.5) .+ max.(x, 0.1)), (data(8, 6),)),
    ("permutation and reshape", x -> sum(abs2, reshape(permutedims(x, (2, 1, 3)), 3, 20) .* data(3, 20)), (data(4, 3, 5),)),
    ("strided view and index", x -> sum(abs2, view(x, 2:2:8, :) .+ x[3, 2]), (data(8, 6),)),
    ("gather by a constant list", x -> sum(abs2, x[:, [3, 1, 3, 6]]), (data(8, 6),)),
    ("rotation by a constant permutation", (x, c, s) -> sum(abs2, x .* c .+ x[[3, 4, 1, 2], :] .* s), (data(4, 6), data(4, 1), data(4, 1; scale = 0.3))),
    ("vec and dropdims", x -> sum(abs2, vec(dropdims(sum(x; dims = 1); dims = 1))), (data(8, 6),)),
    ("Cartesian indices, the diagonal and dot", x -> sum(abs2, diag(x)) + x[CartesianIndex(2, 3)] + dot(x, tanh.(x)), (data(6, 6),)),
    ("map over two arrays", (x, y) -> sum(abs2, map((a, b) -> a * b + a, x, y)), (data(8, 6), data(8, 6; scale = 0.4))),
    ("scalar input", (x, s) -> sum(abs2, x .* s .+ s^2), (data(8, 6), 0.7)),
    ("attention, grouped heads", (q, k, v) -> sum(abs2, Ops.attention(q, k, v)),
     (data(4, 4, 6, 2; scale = 0.3), data(4, 2, 6, 2; scale = 0.2), data(4, 2, 6, 2; scale = 0.1))),
    ("causal attention with a residual", (q, k, v, x) -> sum(abs2, Ops.attention(q, k, v; causal = true) .+ x),
     (data(4, 4, 6, 2; scale = 0.3), data(4, 2, 6, 2; scale = 0.2), data(4, 2, 6, 2; scale = 0.1), data(4, 4, 6, 2; scale = 0.5))),
    ("dropout", (x, s) -> sum(abs2, Ops.dropout(x, s; rate = 0.25) .* 2), (data(8, 6), Int32[7])),
    ("gather by a traced vector", (x, i) -> sum(abs2, x[:, i] .* 2), (data(8, 6), [3, 1, 3, 6])),
    ("gather by a traced Int32 matrix", (x, i) -> sum(abs2, x[:, i]), (data(8, 6), Int32[2 5; 6 1; 1 1])),
    ("gather rows by a traced Int64 matrix", (x, i) -> sum(abs2, tanh.(x[i, :])), (data(8, 6), [8 1; 2 8])),
    ("gather by linear and scalar indices", (x, i, t) -> sum(abs2, x[i]) + x[2, t], (data(8, 6), [40, 3, 17], 5)),
    ("a write of a slice", (x, v) -> (y = copy(x); y[2:4, :] = v; sum(abs2, tanh.(y))), (data(8, 6), data(3, 6; scale = 0.3))),
    ("padding by a write", x -> sum(abs2, padded(x) .* data(10, 8)), (data(8, 6),)),
    ("writes through views", x -> (y = zero(x); view(y, :, 2:3) .= x[:, 1:2] .* 2; y[1, :] .= 1; sum(abs2, y .+ x)), (data(8, 6),)),
    ("a mean divided in place", x -> (r = sum(x; dims = 1); r ./= size(x, 1); sum(abs2, x .- r)), (data(8, 6),)),
    ("concatenations", (x, v) -> sum(abs2, vcat(x, v) .* data(11, 6)) + sum(hcat(x, x) .* 2), (data(8, 6), data(3, 6; scale = 0.3))),
    ("strided and reversed writes", (x, v) -> (y = copy(x); y[1:2:8, end:-2:1] = v; sum(abs2, sin.(y))), (data(8, 6), data(4, 3; scale = 0.3))),
    ("scalar writes in a loop", x -> (y = similar(x, 6); for i in 1:6; y[i] = sum(x[:, i]); end; sum(abs2, y)), (data(8, 6),)),
    ("a scalar written into place", (x, s) -> (y = copy(x); y[3, 2] = s * 2; sum(abs2, y)), (data(8, 6), 0.7)),
    ("permutedims! and copyto!", x -> (y = similar(x, 6, 8); permutedims!(y, x, (2, 1)); z = similar(x, 48); copyto!(z, y); sum(abs2, z .* data(48))),
     (data(8, 6),)),
    ("embedding lookup and layer norm", (E, P, t, w) -> sum(abs2, layernorm(E[:, t] .+ P, w)),
     (data(8, 10; scale = 0.3), data(8, 3; scale = 0.2), Int32[3 1; 10 3; 7 7], data(8; scale = 0.1))),
]

