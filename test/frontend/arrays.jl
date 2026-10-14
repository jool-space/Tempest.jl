include("../fixtures/frontend.jl")

@testset "Symbolic array and scalar interface" begin
    @test isempty(Test.detect_ambiguities(Tempest; recursive = true))
    a = reshape(collect(range(-0.5, 0.7; length = 12)), 3, 4)
    trace(a) do x
        @test x isa AbstractMatrix{Float64}
        @test x isa TA{Float64, 2}
        @test size(x) == (3, 4)
        @test size(x, 3) == 1
        @test axes(x) == (Base.OneTo(3), Base.OneTo(4))
        @test axes(x, 3) == Base.OneTo(1)
        @test ndims(x) == 2
        @test length(x) == 12
        @test eltype(x) === Float64
        @test occursin("TracedArray", sprint(show, x))
        s = sum(x)
        @test s isa TS{Float64}
        @test s isa Real
        @test !(s isa AbstractArray)
        @test one(s) == 1.0
        @test zero(s) == 0.0
        @test occursin("TracedScalar", sprint(show, s))
        @test sum(x; dims = 1) isa AbstractMatrix{Float64}
        @test size(sum(x; dims = 1)) == (1, 4)
        @test size(sum(x; dims = (1, 2))) == (1, 1)
        @test size(sum(x; dims = 3)) == size(x)
        @test eltype(sum(x .> 0)) === Int
        @test eltype(sum(x .> 0; dims = ())) === Int
        # Broadcasting is lazy until materialization: one arbitrary-arity tree.
        n = length(x.nodes)
        bc = Base.Broadcast.broadcasted(+, x, 2.0, 3.0)
        @test length(x.nodes) == n
        @test copy(bc) isa TA{Float64, 2}
        @test length(x.nodes) > n
        s
    end
    @test_throws ArgumentError trace(a) do x
        x[1]
        sum(x)
    end
    @test_throws ArgumentError trace(a) do x
        x[CartesianIndex(1, 1)]
        sum(x)
    end
    @test_throws ArgumentError trace(a) do x
        x[1] = 0
        sum(x)
    end
    @test_throws ArgumentError trace(a) do x
        x .= 0
        sum(x)
    end
    @test_throws ArgumentError trace(x -> sum(similar(x)), a)
    @test_throws ArgumentError trace(x -> sum(x) > 0 ? sum(x) : -sum(x), a)
    @test_throws ArgumentError trace(x -> sum(x .+ ones(size(x))), a)
    @test_throws DimensionMismatch trace((x, b) -> sum(x + b), a, ones(3, 1))
    @test trace(x -> sum(reshape(x', :)), a) isa Tempest.TensorProgram
    @test_throws ArgumentError trace(x -> sum(x; dims = 0), a)
end

@testset "Scalars versus zero-dimensional arrays" begin
    a = reshape(collect(range(-0.3, 0.4; length = 12)), 3, 4)
    scalar_loss(x, scale::Real) = sum(abs2, x * scale) / scale
    p = trace(scalar_loss, a, 1.2)
    @test p.nodes[p.arguments[2]].scalar
    ex = compare_prepared(scalar_loss, a, 1.2)
    @test outputs(ex)[3][] ≈ finite_difference(scalar_loss, (a, 1.2), 2) atol=1e-8
    bind!(ex, a, 2.3)
    @test execute!(ex)[1][] ≈ scalar_loss(a, 2.3)
    @test_throws ArgumentError bind!(ex, a, fill(2.3))
    z = fill(1.2)
    zero_dim_loss(x, z::AbstractArray{T, 0}) where {T} = sum(abs2, x .* z) / z[]
    p0 = trace(zero_dim_loss, a, z)
    @test !p0.nodes[p0.arguments[2]].scalar
    ex0 = compare_prepared(zero_dim_loss, a, z)
    @test size(outputs(ex0)[3]) == ()
    @test outputs(ex0)[3][] ≈ outputs(compare_prepared(scalar_loss, a, 1.2))[3][]
    @test_throws ArgumentError bind!(ex0, a, 1.2)
    @test_throws ArgumentError differentiate(trace(identity, z))
    trace(z) do x
        @test x isa TA{Float64, 0}
        @test x[] isa TS{Float64}
        @test (x .+ 1) isa TA{Float64, 0}
        @test (x[] .+ 1) isa TS{Float64}
        sum(x)
    end
end

@testset "Broadcast axes, reshape aliases, and reduction gradients" begin
    x = reshape(collect(range(-0.6, 0.7; length = 24)), 3, 4, 2)
    array(m, n, s) = reshape(s .* sin.(collect(1:(m * n))), m, n)
    cases = (
        ((x, b) -> sum(abs2, x .+ b), (x, reshape([0.2, -0.1, 0.3], 3, 1, 1))),
        ((x, b) -> sum(abs2, x .* b), (x, reshape([0.3, 0.5, 0.7, 0.9], 1, 4))),
        ((x, b) -> sum(abs2, x .+ b'), (array(3, 4, 0.5), [0.2, 0.1, -0.1, 0.3])),
        (x -> sum(abs2, sum(x; dims = 1)), (x,)),
        (x -> sum(abs2, sum(x; dims = 2)), (x,)),
        (x -> sum(abs2, sum(x; dims = (1, 3))), (x,)),
        (x -> sum(abs2, sum(x; dims = (1, 2, 3))), (x,)),
        (x -> sum(abs2, sum(x; dims = ())), (x,)),
        (x -> sum(abs2, reshape(x, 6, :)), (x,)),
        (x -> sum(abs2, reshape(x, 6, 4)), (x,)),
        (x -> sum(abs2, reshape(x, (6, 4))), (x,)),
        (x -> sum(abs2, reshape(x, 6, 4)'), (x,)),
        (x -> sum(sin.(reshape(reshape(x, 6, 4)', 3, 8))), (x,)),
        (x -> sum(abs2, dropdims(sum(x; dims = (1, 3)); dims = (1, 3))), (x,)),
        ((x, b, c) -> sum(@. tanh(x + b + c) ^ 2), (array(3, 4, 0.4), ones(3), ones(1, 4))),
        ((x, b, c) -> sum(@. x * b * c), (array(3, 4, 0.4), ones(3), ones(1, 4))),
        ((x, b, c) -> sum(triple.(x, b, c)), (array(3, 4, 0.4), ones(3), ones(1, 4))),
        (x -> sum(map(sin, x)), (x,)),
        (x -> sum(mapreduce(abs2, +, x; dims = 2)), (x,)),
        (x -> sum(x .^ 3), (x,)),
        (let power=3
            x -> sum(x .^ power)
        end, (x,)),
        (x -> sum(x) ^ 3, (x,)),
        (x -> sum(abs2, x) ^ (1 // 2), (x,)),
        (x -> sum(reshape(x, :)), ([0.2, -0.5, 0.8],)),
        (norm_loss, (array(3, 4, 0.5), [0.9, 1.1, 0.8], [0.1, -0.2, 0.1])),
    )
    for (f, args) in cases
        for reuse in (true, false)
            ex = compare_prepared(f, args...; reuse)
            for i in eachindex(args)
                @test Array(outputs(ex)[i + 1]) ≈ finite_difference(f, args, i) atol=2e-7 rtol=2e-6
            end
        end
    end
end
