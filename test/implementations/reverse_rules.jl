include(joinpath(EXAMPLES, "kernel_composition", "rmsnorm.jl"))
using .RMSNormExample

struct NormRuleCPU <: Implementation
    mode::Symbol
end

function Tempest.reverse_rule(impl::NormRuleCPU, request::CallRequest)
    request.call.f === rmsnorm || return Unsupported("not RMSNorm")
    request.target isa CPUTarget || return Unsupported("CPU only")
    ReverseRule(rms_rule)
end

function Tempest.support(impl::NormRuleCPU, request::CallRequest)
    request.call.f in (rms_forward, rms_backward) ||
        return Unsupported("private norm calls only")
    if impl.mode === :no_backward && request.call.f === rms_backward
        return Unsupported("backward unavailable")
    end
    Supported()
end

function Tempest.build_implementation(
    impl::NormRuleCPU,
    request::CallRequest,
    runtime,
    out,
    inputs,
)
    if request.call.f === rms_backward
        impl.mode === :reject_build && return Unsupported("backward construction rejected")
        impl.mode === :broken && error("backward compiler bug")
    end
    launch =
        () -> begin
            result = reference(request.call.f, inputs...; request.call.kwargs...)
            for (destination, source) in
                zip(Tempest.tree_leaves(out), Tempest.tree_leaves(result))
                copyto!(destination, source)
            end
            nothing
        end
    BuiltImplementation(launch, (out, inputs), 0)
end

struct RecomputeNorm <: Implementation end

function Tempest.reverse_rule(::RecomputeNorm, request::CallRequest)
    request.call.f === rmsnorm || return Unsupported("not RMSNorm")
    function recompute(x, weight; eps)
        y = reference(rmsnorm, x, weight; eps)
        y, dy -> verb_pullback(rmsnorm, (x, weight), y, dy; eps)
    end
    ReverseRule(recompute)
end

@testset "Paired reverse rules and private saved values" begin
    x = reshape(collect(range(-0.8, 0.7; length = 35)), 7, 5)
    w = collect(range(0.6, 1.1; length = 7))
    loss(x, w) = sum(abs2, rmsnorm(x, w; eps = 0.03) .+ x)
    p = trace(loss, x, w)
    derivative = differentiate(p)
    value, (dx, dw) = value_and_gradient(derivative, x, w)
    @test dx ≈ finite_difference(loss, (x, w), 1) atol=1e-7
    @test dw ≈ finite_difference(loss, (x, w), 2) atol=1e-7

    for mode in (:accept, :no_backward, :reject_build), reuse in (false, true)
        ex = prepare(derivative, x, w; reuse, implementations = (NormRuleCPU(mode),))
        actual = execute!(ex)
        @test actual[1][] ≈ value
        @test actual[2] ≈ dx
        @test actual[3] ≈ dw
        opaque = filter(s -> s.region isa Tempest.OpaqueCall, ex.storage.steps)
        if mode === :accept
            @test length(opaque) == 2
            forward, backward = opaque
            statistic = forward.region.call.results.rstd
            @test statistic in backward.dependencies
            @test statistic ∉ ex.storage.program.results
            @test ex.storage.lastuse[statistic] >= findfirst(==(backward), ex.storage.steps)
            @test occursin("reverse rule NormRuleCPU", sprint(explain, ex))
        else
            @test isempty(opaque)
            @test occursin("rejected as a pair", sprint(explain, ex))
        end
        bind!(ex, 0.7x, w)
        @test execute!(ex)[1][] ≈ loss(0.7x, w)
    end

    # A construction failure retries a different complete rule, including its
    # choice to recompute instead of saving statistics.
    ex = prepare(
        derivative,
        x,
        w;
        implementations = (NormRuleCPU(:reject_build), RecomputeNorm()),
    )
    @test execute!(ex)[2] ≈ dx
    @test occursin("reverse rule RecomputeNorm", sprint(explain, ex))
    @test !any(s -> s.region isa Tempest.OpaqueCall, ex.storage.steps)
    @test_throws "backward compiler bug" prepare(
        derivative,
        x,
        w;
        implementations = (NormRuleCPU(:broken),),
    )

    # Forward-only execution does not run a training rule or save its statistics.
    ex = prepare_forward(rmsnorm, x, w; implementations = (NormRuleCPU(:broken),))
    @test execute!(ex) ≈ rmsnorm(x, w)
    @test !occursin("reverse rule", sprint(explain, ex))

    # Independent invocations must not share statistics with overlapping lives.
    two(x, w) = sum(abs2, rmsnorm(rmsnorm(x, w), w) .+ x)
    ex = prepare(two, x, w; implementations = (NormRuleCPU(:accept),))
    @test execute!(ex)[2] ≈ finite_difference(two, (x, w), 1) atol=1e-6
    forwards = filter(
        s -> s.region isa Tempest.OpaqueCall && s.region.call.f === rms_forward,
        ex.storage.steps,
    )
    @test length(forwards) == 2
    @test ex.storage.slots[forwards[1].region.call.results.rstd] !=
          ex.storage.slots[forwards[2].region.call.results.rstd]
end

struct InvalidNormRule <: Implementation
    mode::Symbol
end

function Tempest.reverse_rule(impl::InvalidNormRule, request::CallRequest)
    request.call.f === rmsnorm || return Unsupported("not RMSNorm")
    impl.mode === :invalid_return && return true
    function invalid(x, weight; eps)
        if impl.mode === :wrong_output
            return weight, dy -> (x, weight)
        end
        y = reference(rmsnorm, x, weight; eps)
        y, dy -> (weight, x)
    end
    ReverseRule(invalid)
end

@testset "Reverse rules validate their symbolic contract" begin
    x, w = ones(Float32, 3, 2), ones(Float32, 3)
    loss(x, w) = sum(rmsnorm(x, w))
    @test_throws "reverse_rule must return" prepare(
        loss,
        x,
        w;
        implementations = (InvalidNormRule(:invalid_return),),
    )
    @test_throws "metadata differs" prepare(
        loss,
        x,
        w;
        implementations = (InvalidNormRule(:wrong_output),),
    )
    @test_throws "cotangent metadata" prepare(
        loss,
        x,
        w;
        implementations = (InvalidNormRule(:wrong_gradient),),
    )

    # Selecting only the weights still gives the specialized backward its full
    # input contract, while exposing just the requested gradient to the caller.
    p = differentiate(trace(loss, x, w); wrt = (2,))
    ex = prepare(p, x, w; implementations = (NormRuleCPU(:accept),))
    @test length(execute!(ex)) == 2
    @test execute!(ex)[2] ≈ value_and_gradient(p, x, w)[2][1]
end

rule_pair(x::Tempest.TracedArray) = record_call(rule_pair, x)
Tempest.output_spec(::typeof(rule_pair), x) = (a = TensorSpec(x), nested = (TensorSpec(x),))
Tempest.reference(::typeof(rule_pair), x) = (a = x .* x, nested = (x .* x .* x,))
Tempest.verb_pullback(::typeof(rule_pair), xs, y, g) =
    (2 .* only(xs) .* g.a .+ 3 .* only(xs) .* only(xs) .* only(g.nested),)

struct PairRule <: Implementation end
function Tempest.reverse_rule(::PairRule, request::CallRequest)
    request.call.f === rule_pair || return Unsupported("not the pair operation")
    function rule(x)
        y = reference(rule_pair, x)
        y, g -> (2 .* x .* g.a .+ 3 .* y.a .* only(g.nested),)
    end
    ReverseRule(rule)
end

@testset "Reverse rules preserve nested outputs and unused cotangents" begin
    x = [0.2, -0.4, 0.6]
    loss(x) = sum(only(rule_pair(x).nested))
    ex = prepare(loss, x; implementations = (PairRule(),))
    @test execute!(ex)[1][] ≈ sum(x .^ 3)
    @test execute!(ex)[2] ≈ 3 .* x .* x
end


@testset "RMSNorm epsilon follows input precision" begin
    x, w = ones(Float32, 3, 2), ones(Float32, 3)
    @test eltype(rmsnorm(x, w; eps = 0.03)) === Float32
    loss(x, w) = sum(rmsnorm(x, w; eps = 0.03))
    ex = prepare(loss, x, w; implementations = (NormRuleCPU(:accept),))
    @test execute!(ex)[1][] ≈ loss(x, w)
end

rule_pair(x::Tempest.TracedScalar) = record_call(rule_pair, x)

@testset "Reverse rules preserve scalar and zero-dimensional identity" begin
    function scalar_loss(x)
        y = rule_pair(x)
        y.a + only(y.nested)
    end
    scalar = prepare(scalar_loss, 0.2; implementations = (PairRule(),))
    @test execute!(scalar)[1][] ≈ 0.048
    @test execute!(scalar)[2][] ≈ 0.52

    function array_loss(x)
        y = rule_pair(x)
        sum(y.a .+ only(y.nested))
    end
    array = prepare(array_loss, fill(0.2); implementations = (PairRule(),))
    @test execute!(array)[1][] ≈ 0.048
    @test execute!(array)[2] ≈ fill(0.52)
end
