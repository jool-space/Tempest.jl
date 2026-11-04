# Statistics' variance loops over the elements, which traces as scalars; on a
# traced array it is the sum of squares about the mean, which the plan makes
# one fold relative to the mean (see `rewrite`). `mean` needs nothing: it is
# a sum divided by a count.
module StatisticsExt

using Statistics: Statistics
using Tempest.Compiler: TracedArray

count_over(x, dims) = dims isa Colon ? length(x) : prod(d -> size(x, d), dims; init = 1)

# Both of Statistics' methods, for a mean that is an array or a number. The
# variance along `dims` is an array, of no dimensions too, as Statistics'
# is; over all of `x` it is a number.
for M in (AbstractArray, Number)
    @eval function Statistics.varm(x::TracedArray, m::$M; dims = :, corrected::Bool = true)
        n = count_over(x, dims)
        squares = sum(abs2, Broadcast.broadcast_preserving_zero_d(-, x, m); dims)
        dims isa Colon ? squares / (n - corrected) : Broadcast.broadcast_preserving_zero_d(/, squares, n - corrected)
    end
end

Statistics.var(x::TracedArray; dims = :, corrected::Bool = true, mean = nothing) =
    Statistics.varm(x, mean === nothing ? Statistics.mean(x; dims) : mean; dims, corrected)

function Statistics.std(x::TracedArray; kws...)
    v = Statistics.var(x; kws...)
    v isa TracedArray ? map(sqrt, v) : sqrt(v)
end

end
