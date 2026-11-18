# Programs over arrays of no elements, each a name, a function and its
# arguments: maps and layouts of none, folds over no elements whose values
# have some, a product through an axis of none, pieces and writes of none,
# and a loop one of whose states has none.
nothing32(dims...) = zeros(Float32, dims...)
counted32(dims...) = Float32.(reshape(1:prod(dims), dims...))

function write_none(x, y)
    w = copy(y)
    w[1:0] .= x
    w .* 2.0f0
end

function loop_beside_none(x, y)
    a, b = x, y
    for _ in 1:4
        a = a .* 2.0f0
        b = b .+ sum(a) .+ 1.0f0
    end
    (a, b)
end

const OF_NONE = [
    ("a map", x -> x .+ 1.0f0, (nothing32(0),)),
    ("a masked map", x -> ifelse.(Int32.(1:length(x)) .< Int32(3), x, 0.0f0), (nothing32(0),)),
    ("a sum over an axis of none", x -> sum(x; dims = 1), (nothing32(0, 3),)),
    ("a sum of none", x -> sum(x), (nothing32(0, 3),)),
    ("a sum of none, read by a map", (x, y) -> y .* sum(abs2, x), (nothing32(0), counted32(4))),
    ("a product over an axis of none", x -> prod(x; dims = 1), (nothing32(0, 3),)),
    ("a largest from `init`", x -> maximum(x; dims = 1, init = 1.0f0), (nothing32(0, 3),)),
    ("a largest of elements from `init`", x -> maximum(x; dims = 1, init = 4.0f0), (counted32(2, 3),)),
    ("a smallest of none from `init`", x -> minimum(x; init = 7.0f0), (nothing32(0),)),
    ("a largest magnitude of none", x -> maximum(abs, x; dims = 1), (nothing32(0, 3),)),
    ("any and all of none", x -> (any(x .> 0; dims = 1), all(x .> 0; dims = 1)), (nothing32(0, 3),)),
    ("a count of none", x -> count(>(0), x; dims = 1), (nothing32(0, 3),)),
    ("a log-sum-exp of none", x -> T.logsumexp(x; dims = 1), (nothing32(0, 3),)),
    ("a mean of none", x -> sum(x) / length(x), (nothing32(0),)),
    ("a matrix product through an axis of none", (w, x) -> w * x, (nothing32(3, 0), nothing32(0, 4))),
    ("a matrix product through an axis of none, read by a map", (w, x, y) -> tanh.(w * x .+ y),
        (nothing32(3, 0), nothing32(0, 4), counted32(3, 4))),
    ("a matrix product of no rows", (w, x) -> w * x, (nothing32(0, 2), counted32(2, 4))),
    ("a running sum along an axis of none", x -> cumsum(x; dims = 1), (nothing32(0, 3),)),
    ("a piece of none joined to one of some", (x, y) -> vcat(x, y) .* 2.0f0, (nothing32(0, 3), counted32(2, 3))),
    ("pieces of none joined", x -> hcat(x, x), (nothing32(3, 0),)),
    ("a result of none beside one of some", (x, y) -> (x .* 2.0f0, y .+ 1.0f0), (nothing32(0, 2), counted32(4))),
    ("a gather of no indices", (x, i) -> x[i] .+ 1.0f0, (counted32(4), Int[])),
    ("a slice of none", x -> x[:, 1:0] .* 2.0f0, (counted32(3, 4),)),
    ("a reshape of none", x -> reshape(x, 3, 0), (nothing32(0, 3),)),
    ("a write of none", write_none, (nothing32(0), counted32(4))),
    ("a loop with a state of none beside one of some", loop_beside_none, (nothing32(0), counted32(3))),
]

# The results of a program as Julia's are: of its shapes and element types,
# and its values up to rounding, what is no number where Julia's is none.
as_julia(got, expected) = length(got) == length(expected) &&
    all(map((a, b) -> size(a) == size(b) && eltype(a) === eltype(b) && isapprox(a, b; rtol = 1.0e-6, nans = true), got, expected))
listed(r) = r isa Tuple ? collect(Any, r) : Any[r]
