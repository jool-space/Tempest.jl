module StatisticsExt

# `mean`, `var` and `std` of traced arrays, spelled with Tempest's sums and
# pointwise operations: the same nodes as `sum(x; dims) ./ n`, so they fuse,
# plan and differentiate as that spelling does. Statistics' generic methods
# would index the array element by element, which tracing refuses.

using Tempest: TracedArray, TracedScalar
import Statistics

# The number of elements each slice of a reduction over `dims` holds.
slice_length(x, ::Colon) = length(x)
slice_length(x, dims) = prod(size(x, d) for d in unique(dims isa Integer ? (dims,) : dims); init = 1)

function Statistics.mean(x::TracedArray; dims = :)
    total = sum(x; dims)
    total ./ slice_length(x, dims)
end

Statistics.mean(f, x::TracedArray; dims = :) = Statistics.mean(map(f, x); dims)

# The sum of squared deviations from `mean` over `dims`, divided by `n - 1` when
# `corrected` and by `n` otherwise. A given `mean` is used as it is: pass the
# value of `mean(x; dims)` to share one reduction between the two statistics.
function Statistics.var(x::TracedArray; dims = :, corrected::Bool = true, mean = nothing)
    center = mean === nothing ? Statistics.mean(x; dims) : mean
    squares = sum(abs2, x .- center; dims)
    squares ./ (slice_length(x, dims) - Int(corrected))
end

Statistics.std(x::TracedArray; dims = :, corrected::Bool = true, mean = nothing) =
    sqrt.(Statistics.var(x; dims, corrected, mean))

end
