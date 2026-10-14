# Provider and verb APIs are public but not exported.
using Tempest:
    Implementation,
    RegionRequest,
    GraphRegion,
    GroupedRequest,
    CallRequest,
    Supported,
    Unsupported,
    support,
    is_supported,
    build_implementation,
    BuiltImplementation,
    ImplementationPlan,
    WorkspaceSpec,
    default_implementations,
    TensorSpec,
    output_spec,
    reference,
    verb_pullback,
    record_call,
    ReverseRule

function finite_difference(f, arrays, which; h = 1e-6)
    a = arrays[which]
    if a isa Number
        return (
            f(Base.setindex(arrays, a + h, which)...) -
            f(Base.setindex(arrays, a - h, which)...)
        ) / (2h)
    end
    g = similar(a)
    for i in eachindex(a)
        old = a[i]
        a[i] = old + h
        hi = f(arrays...)
        a[i] = old - h
        lo = f(arrays...)
        a[i] = old
        g[i] = (hi - lo) / (2h)
    end
    g
end

function compare_prepared(
    f,
    args...;
    target = CPUTarget(),
    reuse = true,
    atol = 1e-8,
    rtol = 1e-7,
)
    p = trace(f, args...)
    before = length(p.nodes)
    d = differentiate(p)
    @test length(p.nodes) == before
    ref = value_and_gradient(d, args...)
    ex = prepare(d, args...; target, reuse)
    got = execute!(ex)
    @test Array(got[1])[] ≈ ref[1] atol=atol rtol=rtol
    for (g, r) in zip(got[2:end], ref[2])
        observed = r isa Number ? only(Array(g)) : Array(g)
        @test observed ≈ r atol=atol rtol=rtol
    end
    ex
end

# Residual normalization fixture shared by the graph-region and training tests.
fusion_activation(x) = x * x * x
Tempest.scalar_partials(::typeof(fusion_activation)) = (x -> 3 * x * x,)

function residual_normalization(x, residual, weight, bias, epsilon)
    h = x .+ residual
    rstd = inv.(sqrt.(sum(abs2, h; dims = 1) ./ eltype(x)(size(x, 1)) .+ epsilon))
    y = fusion_activation.(h .* rstd .* weight .+ bias)
    (; y, h, rstd)
end

function fusion_inputs(T, rows = 5, columns = 4)
    x = reshape(T.(sin.(1:(rows * columns))), rows, columns)
    residual = T(0.2) .* reverse(x; dims = 1)
    weight = T.(range(0.3, 0.8; length = rows))
    bias = reshape(T.(range(-0.2, 0.1; length = columns)), 1, columns)
    (x, residual, weight, bias, T(0.03))
end
