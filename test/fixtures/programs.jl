# Programs several tests run: each a function, and its arguments in Float64
# so that finite differences are accurate.

swish(x) = x / (1 + exp(-x))
leaky(x) = x > 0 ? x : oftype(x, 0.01) * x
T.scalar_partials(::typeof(leaky), g, y, x) = (ifelse(x > 0, g, oftype(g, 0.01) * g),)
rmsnorm(x, w) = x ./ sqrt.(sum(abs2, x; dims = 1) ./ size(x, 1) .+ 1.0e-5) .* w
softmax_columns(x) = exp.(x .- T.logsumexp(x; dims = 1))
softmax_rows(x) = exp.(x .- T.logsumexp(x; dims = 2))
# The log-sum-exp written out, with the maximum that keeps it finite.
written_logsumexp(x; dims) = (m = maximum(x; dims); m .+ log.(sum(exp.(x .- m); dims)))
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

# A fold of the user's with a tuple state, Welford's count, mean and sum of
# squared deviations: named with `GenericFold`, and as Base spells it, the
# same lift and merge given to `mapreduce` and the observe a broadcast after.
function welford(a, b)
    n = a[1] + b[1]
    w = ifelse(n == 0, zero(n), b[1] / n)
    d = b[2] - a[2]
    (n, a[2] + d * w, a[3] + b[3] + d * d * a[1] * w)
end
welford_lift(x) = (one(x), x, zero(x))
const VARIANCE = T.GenericFold(; init = T -> (zero(T), zero(T), zero(T)), lift = welford_lift, combine = welford, observe = s -> s[3] / s[1])
base_variance(x; dims) = (s -> s[3] / s[1]).(mapreduce(welford_lift, welford, x; dims, init = ntuple(_ -> zero(eltype(x)), 3)))

# A linear recurrence `h = a h′ + b`, as Base spells its scan: pairs of a
# factor and an offset, merged by a combine that associates and does not
# commute, from `h = 0`.
chain((a1, b1), (a2, b2)) = (a2 * a1, a2 * b1 + b2)
recurrence(a, b) = last.(accumulate(chain, tuple.(a, b); init = (one(eltype(a)), zero(eltype(b)))))
# Base combines a function given to `accumulate` in sequence; a `GenericFold`
# of the same combine claims that it associates.
const RECURRENCE = T.GenericFold(; init = E -> (one(fieldtype(E, 1)), zero(fieldtype(E, 2))), lift = identity, combine = chain, observe = last)
# An average running from a start that is no identity, in sequence.
running_average(x; dims) = accumulate((s, e) -> s / 2 + e, x; dims, init = one(eltype(x)))
# A zero, through which a product's derivative passes.
with_zero(x) = (y = copy(x); y[2, 3] = 0; y)

const PROGRAMS = [
    ("dot product", (a, b) -> sum(a .* b), (data(32), data(32; scale = 0.3))),
    ("sum abs2 tanh", x -> sum(abs2, tanh.(x)), (data(8, 6),)),
    ("full logsumexp", x -> T.logsumexp(x), (data(8, 6),)),
    ("maximum and shifted sum", x -> sum(exp.(x .- maximum(x))), (data(8, 6),)),
    ("log-sum-exp written out", x -> sum(written_logsumexp(x; dims = 1) .* data(1, 6)), (data(8, 6),)),
    ("softmax by the maximum, the maximum read", (x, t) -> (m = maximum(x; dims = 1); e = exp.(x .- m); sum(e ./ sum(e; dims = 1) .* t) + sum(m)),
     (data(8, 6), data(8, 6; scale = 0.7))),
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
    ("a write at index arrays", (x, v) -> (y = copy(x); y[:, [5, 2, 6]] = v; sum(abs2, tanh.(y))), (data(8, 6), data(8, 3; scale = 0.3))),
    ("a write at traced positions", (x, v, i) -> (y = copy(x); y[:, i] .= v; y[i[1], 1] = 2 * v[3]; sum(abs2, tanh.(y))),
     (data(8, 6), data(8, 2; scale = 0.3), Int32[4, 1])),
    ("scalar writes in a loop", x -> (y = similar(x, 6); for i in 1:6; y[i] = sum(x[:, i]); end; sum(abs2, y)), (data(8, 6),)),
    ("a scalar written into place", (x, s) -> (y = copy(x); y[3, 2] = s * 2; sum(abs2, y)), (data(8, 6), 0.7)),
    ("permutedims! and copyto!", x -> (y = similar(x, 6, 8); permutedims!(y, x, (2, 1)); z = similar(x, 48); copyto!(z, y); sum(abs2, z .* data(48))),
     (data(8, 6),)),
    ("embedding lookup and layer norm", (E, P, t, w) -> sum(abs2, layernorm(E[:, t] .+ P, w)),
     (data(8, 10; scale = 0.3), data(8, 3; scale = 0.2), Int32[3 1; 10 3; 7 7], data(8; scale = 0.1))),
    ("cumulative sums", x -> sum(cumsum(x; dims = 1) .* data(8, 6; scale = 0.3)) + sum(abs2, cumsum(x[:, 2])), (data(8, 6),)),
    ("cumulative products through a zero", x -> sum(cumprod(x; dims = 2) .* data(5, 4)), (with_zero(data(5, 4; scale = 0.4)),)),
    ("running maxima and minima", x -> sum(accumulate(max, x; dims = 1) .* data(8, 6; scale = 0.3)) + sum(accumulate(min, x)),
     (data(8, 6; scale = 0.7),)),
    ("a linear recurrence", (a, b) -> sum(abs2, recurrence(a, b)), (0.9 .+ 0.1 .* data(30; scale = 0.3), data(30; scale = 0.7))),
    ("a linear recurrence named with GenericFold", (a, b) -> sum(abs2, accumulate(RECURRENCE, tuple.(a, b))),
     (0.9 .+ 0.1 .* data(30; scale = 0.3), data(30; scale = 0.7))),
    ("a running average", x -> sum(abs2, running_average(x; dims = 2)), (data(8, 6),)),
    ("Welford's variance as Base spells it", x -> sum(base_variance(x; dims = 1) .* data(1, 6)), (data(8, 6),)),
    ("Welford's variance named with GenericFold", x -> sum(reduce(VARIANCE, x; dims = 1) .* data(1, 6)), (data(8, 6),)),
    ("extremes and where they are", x -> sum(findmax(x; dims = 1)[1] .* data(1, 6)) + sum(getindex.(argmin(x; dims = 2), 2)) + x[argmax(x)] +
        argmin(abs, x) + sum(last.(extrema(x; dims = 2)) .* data(8)), (data(8, 6),)),
    ("the extremes of a vector", v -> v[argmax(v)] * findmin(v)[2] + sum(extrema(v)) + findmax(abs2, v)[1], (data(30),)),
]

# Writes where several elements land at one position: the interpreters and
# the CPU's floor take them in order, the last landing, and a device lands
# any one of them, so these are checked on the CPU.
const REPEATED_WRITES = [
    ("a write at repeated positions", (x, v) -> (y = copy(x); y[[2, 7, 2], :] .= v; sum(abs2, sin.(y))), (data(8, 6), data(3, 6; scale = 0.3))),
    ("a write at repeated traced positions", (x, v, i) -> (y = copy(x); y[:, i] .= v; sum(abs2, tanh.(y))),
     (data(8, 6), data(8, 3; scale = 0.3), Int32[4, 1, 4])),
]

# Extremes and where they are, as Base finds them, exactly: the first of
# equals, a NaN found by `findmax` and `findmin` alike, 0.0 above -0.0, an
# index into a matrix a `CartesianIndex`.
# A tuple, which keeps each array's element type.
const EXTREMES = (
    Float32[1 5 2; 3 -5 7; 0 4 -1; 2 2 9], Float32[3 5 1; 3 5 1; 1 5 1; 3 0 1], Float32[1 NaN 2; NaN 4 NaN; 3 NaN 1; 0 1 NaN],
    Float32[0.0 -0.0 1; -0.0 0.0 -1; 0.0 -0.0 -1; -0.0 -0.0 1], Float32[-Inf -Inf Inf; -Inf 2 Inf; -Inf -Inf 1; NaN -Inf Inf],
    Int32[4 -2 9; 4 7 -9; 1 7 0; -3 7 9], Bool[0 1 0; 1 1 0; 0 0 0; 1 0 0],
)
const FOUND = [
    findmax, findmin, x -> findmax(x; dims = 1), x -> findmin(x; dims = 2), x -> findmax(x; dims = (1, 2)), x -> findmin(x; dims = 3),
    argmax, x -> argmin(x; dims = 1), x -> argmax(x[:, 2]), x -> findmin(x[:, 3]; dims = 1), x -> getindex.(argmax(x; dims = 1), 1),
    x -> Tuple.(argmin(x; dims = 2)), x -> x[argmax(x; dims = 1)], x -> x[argmin(x)], extrema, x -> extrema(x; dims = 1),
    x -> findmax(abs2, x; dims = 1), x -> argmin(abs, x), x -> extrema(abs, x; dims = 2),
]
# A result's parts, a number standing for an array of no dimensions.
found(x::Tuple) = reduce(vcat, map(found, x); init = Any[])
found(x) = Any[on_host(x)]
