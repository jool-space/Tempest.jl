@test Base.get_extension(Tempest, :ChainRulesCoreExt) === nothing
using ChainRules
import ChainRulesCore
using ChainRulesCore: ProjectTo, @thunk, NoTangent, ZeroTangent, unthunk

# Exercise projection and deferred cotangents without modifying package rules.
projected_square(x::Real) = x * x

function ChainRulesCore.rrule(::typeof(projected_square), x::Real)
    project = ProjectTo(x)
    x * x, delta -> (NoTangent(), @thunk(project(2 * x * unthunk(delta))))
end

ignore_second(x::Real, y::Real) = x
ChainRulesCore.rrule(::typeof(ignore_second), x::Real, y::Real) =
    (x, delta -> (NoTangent(), delta, ZeroTangent()))

nondifferentiable_second(x::Real, y::Real) = x
ChainRulesCore.rrule(::typeof(nondifferentiable_second), x::Real, y::Real) =
    (x, delta -> (NoTangent(), delta, NoTangent()))

broken_rule(x::Real) = x

function ChainRulesCore.rrule(::typeof(broken_rule), x::Real)
    x, delta -> error("bug inside pullback")
end

function check_rule(f, inputs)
    loss(xs...) = sum(f.(xs...))
    program = differentiate(trace(loss, inputs...))
    expected = map(i -> finite_difference(loss, inputs, i), eachindex(inputs))
    for reuse in (false, true)
        executable = prepare(program, inputs...; reuse)
        actual = execute!(executable)
        @test only(first(actual)) ≈ loss(inputs...)
        for (gradient, baseline) in zip(Base.tail(actual), expected)
            @test gradient ≈ baseline atol = 1e-7 rtol = 1e-6
        end
        changed = map(copy, inputs)
        for input in changed
            input .+= 0.1
        end
        bind!(executable, changed...)
        actual = execute!(executable)
        @test only(first(actual)) ≈ loss(changed...)
        for (i, gradient) in enumerate(Base.tail(actual))
            @test gradient ≈ finite_difference(loss, changed, i) atol = 1e-7 rtol = 1e-6
        end
    end
    program
end

@testset "ChainRules scalar graphs lifted over arrays" begin
    x = reshape(collect(range(0.2, 0.8; length = 6)), 2, 3)
    for f in (exp, log, sqrt, inv, tanh, abs2, projected_square)
        check_rule(f, (x,))
    end
    for f in (+, -, *, /, ignore_second)
        check_rule(f, (x, reshape([0.4, 0.6], 2, 1)))
    end
    check_rule(+, (x, reshape([0.4, 0.6], 2, 1), fill(0.3)))
    @test_throws "NoTangent for a data input" prepare(
        (x, y) -> sum(nondifferentiable_second.(x, y)),
        x,
        x,
    )
    @test_throws "bug inside pullback" prepare(x -> sum(broken_rule.(x)), x)
end


@testset "Shared rules extend scalar coverage" begin
    x = reshape(collect(range(0.2, 0.7; length = 6)), 2, 3)
    for f in (sin, cos, expm1, log1p, sinh, cosh, tan, asin, acos, atan, abs)
        check_rule(f, (x,))
    end
    check_rule(abs, (-x,))
    check_rule(atan, (x, reshape([0.4, 0.6], 2, 1)))
    # min/max/hypot still need symbolic predicates in their upstream pullbacks.
    for f in (hypot, min, max)
        @test_throws "data-dependent scalar comparison" prepare((x, y) -> sum(f.(x, y)), x, x)
    end
    @test only(execute!(prepare(x -> sum(abs.(x)), [0.0]))[2]) == 0.0
    @test isempty(Test.detect_ambiguities(Tempest; recursive = true))
end

explicit_square(x::Real) = x * x
Tempest.scalar_partials(::typeof(explicit_square)) = (x -> 2x,)
ChainRulesCore.rrule(::typeof(explicit_square), x::Real) = error("explicit rule must win")

no_shared_rule(x::Real) = x * x

@testset "Explicit rules retain priority" begin
    check_rule(explicit_square, ([0.2, 0.5],))
    @test_throws "no scalar derivative rule" prepare(x -> sum(no_shared_rule.(x)), [0.3])
end

constant_rule(x::Real) = 2.0
ChainRulesCore.rrule(::typeof(constant_rule), x::Real) =
    (2.0, delta -> (NoTangent(), ZeroTangent()))

@testset "Constant scalar rule outputs" begin
    check_rule(constant_rule, ([0.2, 0.5],))
end

shared_exp(x::Real) = exp(x)

function ChainRulesCore.rrule(::typeof(shared_exp), x::Real)
    output = shared_exp(x)
    output, delta -> (NoTangent(), delta * output)
end

@testset "Shared rules reuse the forward output" begin
    program = check_rule(shared_exp, ([0.2, 0.5],))
    functions = [node.op.f for node in program.nodes if node.op isa Tempest.Pointwise]
    @test count(==(shared_exp), functions) == 1
    @test exp ∉ functions
end

delegated_sin(x::Real) = sin(x)
ChainRulesCore.rrule(::typeof(delegated_sin), x::Real) = ChainRulesCore.rrule(sin, x)

@testset "Shared rules may use sincos" begin
    check_rule(delegated_sin, ([0.2, 0.5],))
end
