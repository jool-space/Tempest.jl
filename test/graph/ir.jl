# An external operation supplies mathematical and scheduling methods. It needs
# no entry in the interpreter, reverse traversal, or planner's branch tables.
struct ScaleOperation{T <: Real} <: Tempest.Operation
    factor::T
end

function scaled(x::Tempest.TracedArray, factor::Real)
    Tempest.emit(x.nodes, ScaleOperation(factor), (x.id,), size(x), eltype(x); scalar = false)
end

scaled(x::AbstractArray, factor::Real) = factor .* x

Tempest.evaluate(op::ScaleOperation, node, xs) = op.factor .* only(xs)
Tempest.pullback(op::ScaleOperation, xs, y, g) = (op.factor .* g,)

function Tempest.schedule_region(op::ScaleOperation, nodes, id, dependencies, expression)
    node = nodes[id]
    input = expression(only(node.inputs), dependencies, node.dims)
    product = Tempest.ScalarCall(*, (Tempest.Literal(op.factor), input))
    Tempest.Elementwise(product, prod(node.dims))
end

struct MissingLowering <: Tempest.Operation end

@testset "Operation methods extend the compiler" begin
    x = reshape(Float32[-0.4, 0.1, 0.6, 0.2, -0.3, 0.5], 2, 3)
    loss(x) = sum(abs2, scaled(x, 3.0f0))
    program = trace(loss, x)
    @test any(node -> node.op isa ScaleOperation, program.nodes)
    reference_loss, (reference_gradient,) = value_and_gradient(program, x)
    @test reference_loss ≈ loss(x)
    @test reference_gradient ≈ 18x
    @test occursin("ScaleOperation", sprint(show, MIME"text/plain"(), program))

    for reuse in (false, true)
        executable = prepare(program, x; reuse)
        value, gradient = execute!(executable)
        @test only(value) ≈ reference_loss
        @test gradient ≈ reference_gradient
        bind!(executable, 2x)
        value, gradient = execute!(executable)
        @test only(value) ≈ loss(2x)
        @test gradient ≈ 36x
    end

    # An unknown operation must reach its missing lowering, never be treated as
    # pointwise work by default. This failure identifies the extension method.
    unsupported(x) =
        Tempest.emit(x.nodes, MissingLowering(), (x.id,), size(x), eltype(x); scalar = false)
    error = try
        prepare_forward(unsupported, x)
        nothing
    catch exception
        exception
    end
    @test error isa MethodError
    @test error.f === Tempest.schedule_region
end

struct UnknownScalarRegion <: Tempest.ScalarRegion end

@testset "Native support rejects unknown regions" begin
    step = Tempest.Step(UnknownScalarRegion(), 2, [1])
    request = RegionRequest(CPUTarget(), step, Float32, (3,), (Float32,), ((3,),))
    verdict = support(CPUKernels(), request)
    @test verdict isa Unsupported
    @test occursin("UnknownScalarRegion", verdict.reason)
end
