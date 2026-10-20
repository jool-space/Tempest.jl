# Statistics' variance loops over the elements, which traces as scalars; on a
# traced array it is the sum of squares about the mean, which the plan makes
# one fold relative to the mean (see `rewrite`). `mean` needs nothing: it is
# a sum divided by a count.
module StatisticsExt

using Statistics: Statistics
using Tempest.Compiler: TracedArray

count_over(x, dims) = dims isa Colon ? length(x) : prod(d -> size(x, d), dims; init = 1)

# Both of Statistics' methods, for a mean that is an array or a number.
for M in (AbstractArray, Number)
    @eval function Statistics.varm(x::TracedArray, m::$M; dims = :, corrected::Bool = true)
        n = count_over(x, dims)
        sum(abs2, x .- m; dims) ./ (n - corrected)
    end
end

Statistics.var(x::TracedArray; dims = :, corrected::Bool = true, mean = nothing) =
    Statistics.varm(x, mean === nothing ? Statistics.mean(x; dims) : mean; dims, corrected)

Statistics.std(x::TracedArray; kws...) = sqrt.(Statistics.var(x; kws...))

end
