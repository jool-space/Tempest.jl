# Statistics' mean, var and std on traced arrays (ext/StatisticsExt.jl): the
# values of Statistics' own methods, the nodes of the sums and pointwise work
# they are spelled with, and their gradients.
import Statistics
using Statistics: mean, var, std

# The prepared value: a number for a scalar result, else a copy of the array.
evaluated(f, x) = (y = execute!(prepare_forward(f, x)); ndims(y) == 0 ? y[] : copy(y))

@testset "mean, var and std of traced arrays" begin
    x = reshape(sin.(1.0:60.0) .+ 0.3 .* cos.(0.7 .* (1:60)), 5, 4, 3)
    for dims in (:, 1, 2, 3, (1, 2), (2, 3), (1, 3))
        @test evaluated(x -> mean(x; dims), x) ≈ mean(x; dims)
        @test evaluated(x -> mean(abs2, x; dims), x) ≈ mean(abs2, x; dims)
        for corrected in (true, false)
            @test evaluated(x -> var(x; dims, corrected), x) ≈
                  var(x; dims, corrected)
            @test evaluated(x -> std(x; dims, corrected), x) ≈
                  std(x; dims, corrected)
            # A given mean is used as it is.
            given(x) = var(x; dims, corrected, mean = mean(x; dims))
            @test evaluated(given, x) ≈ var(x; dims, corrected)
        end
    end
    # A reduction over all axes gives a number; over every axis by name, an array.
    @test trace(x -> mean(x), x).nodes[end].scalar
    @test trace(x -> var(x), x).nodes[end].scalar
    @test trace(x -> std(x; dims = (1, 2, 3)), x).nodes[end].dims == (1, 1, 1)
    @test evaluated(x -> var(x; mean = 0.5), x) ≈ var(x; mean = 0.5)
    # Element types follow Statistics: Float32 in, Float32 out.
    y = Float32.(x)
    @test trace(x -> var(x; dims = 1), y).nodes[end].dtype === Float32
    @test evaluated(x -> std(x; dims = 2), y) ≈ std(y; dims = 2) rtol = 1e-5
end

# The statistics are their sums: a layer norm written with mean and var traces
# to the same operations as the spelling with sums, so it plans and fuses
# alike (test/cutile/spellings.jl checks the GPU plan).
function layernorm_statistics(x)
    m = mean(x; dims = 1)
    v = var(x; dims = 1, mean = m, corrected = false)
    (x .- m) ./ sqrt.(v .+ 1.0e-5)
end

function layernorm_sums(x)
    m = sum(x; dims = 1) ./ size(x, 1)
    v = sum(abs2, x .- m; dims = 1) ./ size(x, 1)
    (x .- m) ./ sqrt.(v .+ 1.0e-5)
end

@testset "Statistics trace as their sums" begin
    x = reshape(sin.(1.0:48.0), 8, 6)
    operations(f) = [(typeof(node.op), node.inputs, node.dims) for node in trace(f, x).nodes]
    @test operations(layernorm_statistics) == operations(layernorm_sums)
    @test operations(x -> std(x; dims = 2)) ==
          operations(x -> sqrt.(sum(abs2, x .- sum(x; dims = 2) ./ 6; dims = 2) ./ 5))
    @test layernorm_statistics(x) ≈ layernorm_sums(x)
    @test evaluated(layernorm_statistics, x) ≈ layernorm_sums(x)
end

@testset "Gradients of statistics" begin
    x = reshape(sin.(1.0:24.0) .+ 0.1 .* (1:24), 6, 4)
    w = reshape(cos.(1.0:24.0), 6, 4)
    for loss in (
        (x, w) -> mean(x .* w),
        (x, w) -> var(x .+ w),
        (x, w) -> std(x),
        (x, w) -> sum(abs2, layernorm_statistics(x) .* w),
        (x, w) -> sum(std(x; dims = 1, corrected = false) .* sum(w; dims = 1)),
        (x, w) -> sum(var(x; dims = 2) .+ mean(exp, x .* w; dims = 2)),
    )
        compare_prepared(loss, x, w; atol = 1e-6, rtol = 1e-6)
        @test value_and_gradient(trace(loss, x, w), x, w)[2][1] ≈
              finite_difference(loss, (x, w), 1) rtol = 1e-5 atol = 1e-7
    end
end
