include("../fixtures/planning.jl")

@testset "Explicit derivatives and prepared CPU schedule" begin
    x = reshape(collect(range(-0.6, 0.7; length = 35)), 5, 7)
    w = reshape(collect(range(-0.4, 0.5; length = 15)), 5, 3)
    f(x, w) = sum(abs2, ripple.(w' * x)) / length(x)
    ex = compare_prepared(f, x, w)
    @test any(n -> n.op isa Tempest.Pointwise && n.op.f === ripple, ex.storage.program.nodes)
    @test length(ex.calls) < length(ex.storage.program.nodes)
    @test occursin(
        "explicit forward + backward",
        sprint(show, MIME"text/plain"(), ex.storage.program),
    )
    @test occursin("slot", sprint(explain, ex))
    ref = value_and_gradient(trace(f, x, w), x, w)
    @test ref[2][1] ≈ finite_difference(f, (x, w), 1) atol=1e-8
    @test ref[2][2] ≈ finite_difference(f, (x, w), 2) atol=1e-8
    @test_throws ArgumentError replay!(ex)
    @test_throws ArgumentError capture!(ex)
    @test_throws ArgumentError bind!(ex, x[:, 1:2], w)
    # New input contents; output pointers remain fixed.
    old = map(pointer, outputs(ex))
    bind!(ex, 2x, w)
    execute!(ex)
    @test outputs(ex)[1][] ≈ f(2x, w)
    @test map(pointer, outputs(ex)) == old

    compare_prepared(x -> sum(abs2, x' * x), x)
    compare_prepared(x -> sum(ripple.(x')), x)
    compare_prepared((x, u) -> sum(abs2, x), x, w)
    compare_prepared(x -> sum(x .* sum(abs2, x)), x) # scalar cotangent must reduce
    compare_prepared(x -> sum(exp.(0 * x)), x) # padded f(0) != 0
    compare_prepared(x -> sum(log.(exp.(x))), x)
    compare_prepared(x -> sum(-x), x)
    scalar_loss(x) = sum(x) * sum(x) / sum(abs2, x)
    compare_prepared(scalar_loss, x)
    @test value_and_gradient(trace(scalar_loss, x), x)[2][1] ≈
          finite_difference(scalar_loss, (x,), 1) atol=1e-8
    compare_prepared(x -> sum(Float32.(x)), x; atol = 1e-6, rtol = 1e-6)

    unsupported(x) = x * x * x
    @test_throws ArgumentError differentiate(trace(x -> sum(unsupported.(x)), x))
    @test_throws DimensionMismatch trace((x, b) -> sum(x .+ b), x, ones(4, 1))
    @test_throws ArgumentError prepare(f, x, w; target = :unknown)
end

@testset "Residual gated MLP finite differences" begin
    array(m, n, s) = reshape(s .* sin.(collect(1:(m * n))), m, n)
    args = (
        array(3, 5, 0.2),
        array(3, 2, 0.3),
        array(3, 2, 0.4),
        array(2, 3, 0.3),
        array(3, 5, 0.1),
    )
    ex = compare_prepared(gated_loss, args...)
    for i in eachindex(args)
        @test Array(outputs(ex)[i + 1]) ≈ finite_difference(gated_loss, args, i) atol=1e-8
    end
end

@testset "Lifetimes and exact-size reuse" begin
    function chain(x, w)
        for _ = 1:5
            x = tanh.(w' * x)
        end
        sum(abs2, x)
    end
    x = reshape(collect(range(-0.2, 0.3; length = 64)), 8, 8)
    w = reshape(collect(range(-0.1, 0.2; length = 64)), 8, 8)
    yes = compare_prepared(chain, x, w; reuse = true)
    no = compare_prepared(chain, x, w; reuse = false)
    @test length(yes.buffers) < length(no.buffers)
    @test sum(sizeof, yes.buffers) < sum(sizeof, no.buffers)
    # Every aliased pair has disjoint scheduled lifetimes, and input dependencies
    # are never reclaimed at their consuming step.
    births = Dict(s.output => i for (i, s) in enumerate(yes.storage.steps))
    ids = collect(keys(yes.storage.slots))
    for a in ids, b in ids
        if a >= b
            continue
        end
        if !(yes.storage.slots[a] == yes.storage.slots[b])
            continue
        end
        @test yes.storage.lastuse[a] < births[b] || yes.storage.lastuse[b] < births[a]
    end
    # Dead forward work does not occupy a step or slot.
    function dead(x)
        unused = exp.(x)
        sum(abs2, x)
    end
    ex = compare_prepared(dead, x)
    deadid =
        findfirst(n -> n.op isa Tempest.Pointwise && n.op.f === exp, ex.storage.program.nodes)
    @test !haskey(ex.storage.slots, deadid)
end

# A rule can exist and still contain a bug. Its MethodError must reach the caller.
broken_derivative(x) = x * x

function missing_rule_helper end

Tempest.scalar_partials(::typeof(broken_derivative)) = missing_rule_helper()

@testset "Derivative rules remain inspectable" begin
    x = collect(range(0.2, 0.8; length = 11))
    for activation in (tanh, sin, cos, exp, log, sqrt, inv)
        loss(x) = sum(activation.(x))
        program = differentiate(trace(loss, x))
        _, (gradient,) = value_and_gradient(program, x)
        @test gradient ≈ finite_difference(loss, (copy(x),), 1) atol=1e-8

        # Built-ins must remain visible to region backends as ordinary operations.
        functions = [node.op.f for node in program.nodes if node.op isa Tempest.Pointwise]
        @test all(
            f -> f in (activation, +, -, *, /, sin, cos, log, inv) || f isa Tempest.Cast,
            functions,
        )
    end
    program = differentiate(trace(x -> sum(tanh.(x)), x))
    @test count(node -> node.op isa Tempest.Pointwise && node.op.f === tanh, program.nodes) ==
          1

    error = try
        differentiate(trace(x -> sum(broken_derivative.(x)), x))
        nothing
    catch exception
        exception
    end
    @test error isa MethodError
    @test error.f === missing_rule_helper
end
