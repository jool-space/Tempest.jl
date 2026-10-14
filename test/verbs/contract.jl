include("../fixtures/implementations.jl")
include("../fixtures/verbs.jl")

@testset "General forward results" begin
    x = reshape(Float32.(1:12), 3, 4)
    f(x) = (; a = x, nested = (view(x, 1:2, :), (; s = sum(x), z = view(x, 1, 1))), again = x)
    p = trace(f, x)
    @test_throws ArgumentError differentiate(p)
    for reuse in (false, true)
        ex = prepare_forward(p, x; reuse)
        y = execute!(ex)
        @test y.a == x
        @test y.a === y.again
        @test y.nested[1] == view(x, 1:2, :)
        @test y.nested[2].s[] == sum(x)
        @test size(y.nested[2].z) == ()
        @test occursin("(forward)", sprint(explain, ex))
        bind!(ex, 2x)
        @test execute!(ex).nested[2].s[] == sum(2x)
    end
    @test execute!(prepare_forward(x -> x * x, 2.0f0))[] == 4.0f0
    @test execute!(prepare_forward(identity, x)) == x
    @test_throws ArgumentError trace(x -> (), x)
    @test_throws ArgumentError trace(x -> (x, 1), x)
end

@testset "Semantic calls, references, and pullbacks" begin
    x = collect(range(-0.8, 0.6; length = 17))
    function loss(x)
        y = pair_wave(x)
        sum(abs2, y.s .+ 2 .* y.c)
    end
    expected = loss(x)
    dx = finite_difference(loss, (copy(x),), 1)
    p = trace(loss, x)
    @test count(n -> n.op isa Tempest.CallResult, p.nodes) == 2
    @test value_and_gradient(p, x)[1] ≈ expected
    @test value_and_gradient(p, x)[2][1] ≈ dx atol=1e-8
    for mode in (:accept, :reject, :unavailable), reuse in (false, true)
        ex = prepare(p, x; reuse, implementations = (WaveImplementation(mode),))
        y, g = execute!(ex)
        @test y[] ≈ expected
        @test g ≈ dx atol=1e-8
        @test (:wave_kernel in ex.providers) == (mode === :accept)
        if mode === :accept
            @test count(s -> s.region isa Tempest.OpaqueCall, ex.storage.steps) == 2
            @test all(
                length(s.outputs) == 2 for s in ex.storage.steps if
                s.region isa Tempest.OpaqueCall && s.region.call.f === pair_wave
            )
        else
            @test occursin("expanded reference", sprint(explain, ex))
            @test !any(s -> s.region isa Tempest.OpaqueCall, ex.storage.steps)
        end
        bind!(ex, 2x)
        @test execute!(ex)[1][] ≈ loss(2x)
    end
    @test_throws ErrorException prepare(p, x; implementations = (WaveImplementation(:broken),))
    ex = prepare_forward(pair_wave, x; implementations = (WaveImplementation(:accept),))
    @test execute!(ex).s ≈ sin.(x)
    @test execute!(ex).c ≈ cos.(x)
    @test length(ex.storage.steps) == 1
    # Only one output contributes; the other cotangent is an explicit zero.
    ex = prepare(x -> sum(pair_wave(x).s), x; implementations = (WaveImplementation(:accept),))
    @test execute!(ex)[2] ≈ cos.(x)
    # Mapped inputs materialize before an opaque kernel that expects native arrays.
    ex = prepare_forward(
        x -> pair_wave(view(x, 1:2:17)),
        x;
        implementations = (WaveImplementation(:accept),),
    )
    @test execute!(ex).s ≈ sin.(x[1:2:17])
    @test length(ex.storage.steps) == 2
end

no_pullback(x::Tempest.TracedArray) = record_call(no_pullback, x)
Tempest.output_spec(::typeof(no_pullback), x) = TensorSpec(x)
Tempest.reference(::typeof(no_pullback), x) = x .* x
bad_reference(x::Tempest.TracedArray) = record_call(bad_reference, x)
Tempest.output_spec(::typeof(bad_reference), x) = (; a = TensorSpec(x))
Tempest.reference(::typeof(bad_reference), x) = (; b = x)
recursive_reference(x::Tempest.TracedArray) = record_call(recursive_reference, x)
Tempest.output_spec(::typeof(recursive_reference), x) = TensorSpec(x)
Tempest.reference(::typeof(recursive_reference), x) = recursive_reference(x)
integer_result(x::Tempest.TracedArray) = record_call(integer_result, x)
Tempest.output_spec(::typeof(integer_result), x) = TensorSpec(Int32, size(x))
@testset "Semantic contract errors" begin
    x = Float32[1, 2]
    @test execute!(prepare_forward(no_pullback, x)) == x .* x
    @test_throws ArgumentError prepare(x -> sum(no_pullback(x)), x)
    @test_throws ArgumentError prepare_forward(bad_reference, x)
    @test_throws ArgumentError prepare_forward(recursive_reference, x)
    @test_throws ArgumentError TensorSpec(Float32, (2,); scalar = true)
    # Integer specs describe index inputs; Bool is not an index type, and verb
    # outputs must still be floating.
    @test TensorSpec(Int32, (2,)).dtype === Int32
    @test_throws ArgumentError TensorSpec(Bool, (2,))
    @test_throws ArgumentError trace(x -> record_call(integer_result, x), x)
    @test_throws ArgumentError trace(x -> record_call(pair_wave, x; configuration = [1, 2]), x)
end

wrapped_wave(x::Tempest.TracedArray) = record_call(wrapped_wave, x)
Tempest.output_spec(::typeof(wrapped_wave), x) = output_spec(pair_wave, x)
Tempest.reference(::typeof(wrapped_wave), x) = pair_wave(x)
scalar_and_array(x::Tempest.TracedArray) = record_call(scalar_and_array, x)

function Tempest.output_spec(::typeof(scalar_and_array), x)
    (; total = TensorSpec(eltype(x), (); scalar = true), extra = (TensorSpec(x),))
end

Tempest.reference(::typeof(scalar_and_array), x) = (; total = sum(x), extra = (2 .* x,))
opaque_copy(x::Tempest.TracedArray) = record_call(opaque_copy, x)
Tempest.output_spec(::typeof(opaque_copy), x) = TensorSpec(x)
struct CopyImplementation <: Implementation end

function Tempest.support(::CopyImplementation, r::CallRequest)
    r.call.f === opaque_copy && r.target isa CPUTarget ? Supported() : Unsupported("copy only")
end

function Tempest.build_implementation(
    ::CopyImplementation,
    r::CallRequest,
    runtime,
    out,
    inputs,
)
    BuiltImplementation(() -> copyto!(out, only(inputs)), (out, inputs), 0)
end

@testset "Nested references and opaque-only forward execution" begin
    x = Float32[0.1, 0.3, 0.7]
    for mode in (:accept, :unavailable)
        ex = prepare_forward(wrapped_wave, x; implementations = (WaveImplementation(mode),))
        @test execute!(ex).s ≈ sin.(x)
        @test (:wave_kernel in ex.providers) == (mode === :accept)
    end
    ex = prepare_forward(
        pair_wave,
        x;
        implementations = (WaveImplementation(:unavailable), WaveImplementation(:accept)),
    )
    @test :wave_kernel in ex.providers
    @test execute!(ex).c ≈ cos.(x)
    @test !occursin("expanded reference", sprint(explain, ex))
    ex = prepare(x -> sum(pair_wave(x).c), x; implementations = (WaveImplementation(:accept),))
    @test execute!(ex)[2] ≈ -sin.(x)
    result = execute!(prepare_forward(scalar_and_array, x))
    @test result.total[] ≈ sum(x)
    @test result.extra[1] == 2x
    @test execute!(
        prepare_forward(opaque_copy, x; implementations = (CopyImplementation(),)),
    ) == x
    @test_throws ArgumentError prepare_forward(opaque_copy, x)
end

@testset "Semantic support follows the same contract as regions" begin
    @test_throws "support must return" prepare_forward(
        pair_wave,
        Float32[0.1, 0.3];
        implementations = (InvalidSupport(),),
    )
end
