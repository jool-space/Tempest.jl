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

# Examples included by tests: rmsnorm kernels and the GPT training step.
const EXAMPLES = joinpath(dirname(@__DIR__), "examples")
