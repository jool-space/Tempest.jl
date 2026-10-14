struct TestImplementation <: Implementation
    mode::Symbol
end

Tempest.implementation_name(::TestImplementation) = :test_implementation

function Tempest.support(c::TestImplementation, r::RegionRequest)
    if c.mode === :reject
        return Unsupported("test metadata rejection")
    end
    if !(r.step.region isa Tempest.Elementwise)
        return Unsupported("pointwise only")
    end
    Supported()
end

function Tempest.build_implementation(
    c::TestImplementation,
    r::RegionRequest,
    runtime,
    out,
    arrays,
)
    if c.mode === :unavailable
        return Unsupported("test construction rejection")
    end
    if c.mode === :broken
        error("test unexpected compiler failure")
    end
    if c.mode === :invalid
        return nothing
    end
    build_implementation(CPUKernels(), r, runtime, out, arrays)
end

@testset "Implementation support and construction" begin
    x = Float32[0.2, 0.4, 0.7]
    f(x) = sum(abs2, x)
    ordinary = prepare(f, x)
    expected = deepcopy(execute!(ordinary))
    ex = prepare(
        f,
        x;
        implementations = (TestImplementation(:reject), TestImplementation(:accept)),
    )
    @test :test_implementation in ex.providers
    @test execute!(ex) == expected
    @test occursin("test metadata rejection", sprint(explain, ex))
    @test occursin("Owned implementation workspace: 0 bytes", sprint(explain, ex))
    bind!(ex, 2x)
    @test outputs(ex)[2] == expected[2] # bind alone does not run the schedule
    @test execute!(ex)[2] == 4x
    fallback = prepare(f, x; implementations = (TestImplementation(:unavailable),))
    @test execute!(fallback) == expected
    @test occursin("test construction rejection", sprint(explain, fallback))
    @test_throws ErrorException prepare(f, x; implementations = (TestImplementation(:broken),))
    @test_throws ErrorException prepare(f, x; implementations = (TestImplementation(:invalid),))
    @test_throws ArgumentError prepare(f, x; implementations = (:invalid,))
    s = first(ordinary.storage.steps)
    r = RegionRequest(CPUTarget(), s, Float32, (), (Float32,), ((3,),))
    @test !is_supported(TestImplementation(:reject), r)
    @test support(TestImplementation(:reject), r).reason == "test metadata rejection"
end

# A target may have no native implementation of an otherwise valid region.
# An accepted custom implementation must not consult that fallback at all.
struct ImplementationOnlyTarget <: Tempest.ExecutionTarget end
struct ImplementationOnlyRuntime <: Tempest.AbstractRuntime end
Tempest.make_runtime(::ImplementationOnlyTarget, storage, args) = ImplementationOnlyRuntime()
Tempest.allocate(::ImplementationOnlyRuntime, T, dims) = Array{T}(undef, dims)
Tempest.check_runtime(::ImplementationOnlyRuntime) = nothing


struct InvalidSupport <: Implementation end
Tempest.support(::InvalidSupport, ::RegionRequest) = true
Tempest.support(::InvalidSupport, ::CallRequest) = true

@testset "Custom selection precedes native restrictions" begin
    x = Float32[0.1, 0.3, 0.8]
    ex = prepare_forward(
        x -> sin.(x),
        x;
        target = ImplementationOnlyTarget(),
        implementations = (TestImplementation(:accept),),
    )
    @test execute!(ex) ≈ sin.(x)
    @test ex.providers == [:test_implementation]
    @test isempty(ex.scratch)
    @test_throws "no supported implementation" prepare_forward(
        x -> sin.(x),
        x;
        target = ImplementationOnlyTarget(),
    )
    @test_throws "support must return" prepare_forward(
        x -> sin.(x),
        x;
        implementations = (InvalidSupport(),),
    )

    request = RegionRequest(
        CPUTarget(),
        first(ex.storage.steps),
        Float32,
        size(x),
        (Float32,),
        (size(x),),
    )
    @test_throws "support must return" is_supported(InvalidSupport(), request)
end

struct TemporaryPair <: Implementation
    bindings::Vector{Any}
end

Tempest.implementation_name(::TemporaryPair) = :temporary_pair
Tempest.support(::TemporaryPair, request::RegionRequest) = support(CPUKernels(), request)

function Tempest.build_implementation(
    impl::TemporaryPair,
    request::RegionRequest,
    runtime,
    out,
    inputs,
)
    built = build_implementation(CPUKernels(), request, runtime, out, inputs)
    count = length(out)
    specs = (WorkspaceSpec(eltype(out), count), WorkspaceSpec(eltype(out), count + 2))
    ImplementationPlan(specs) do workspace
        push!(impl.bindings, workspace)
        first, second = workspace
        launch = () -> begin
            built()
            copyto!(first, 1, out, 1, count)
            fill!(second, zero(eltype(out)))
            for index in eachindex(out)
                out[index] = first[index] + second[index]
            end
        end
        BuiltImplementation(launch, (built, workspace), 0)
    end
end

struct InvalidWorkspaceBinding <: Implementation end
Tempest.support(::InvalidWorkspaceBinding, request::RegionRequest) =
    support(CPUKernels(), request)
Tempest.build_implementation(
    ::InvalidWorkspaceBinding,
    request::RegionRequest,
    runtime,
    out,
    inputs,
) = ImplementationPlan(_ -> Unsupported("too late to reject"), ())

@testset "Shared workspace is bound after selection" begin
    x = Float32.(range(0.1, 0.7; length = 7))
    bindings = Any[]
    impl = TemporaryPair(bindings)
    f(x) = (sin.(view(x, 1:3)), cos.(x))
    ex = prepare_forward(f, x; implementations = (impl,))
    @test execute!(ex) == f(x)
    @test length(bindings) == 2
    @test length(ex.scratch) == 2
    @test bindings[1][1] !== bindings[1][2]
    @test bindings[1][1] === bindings[2][1]
    @test bindings[1][2] === bindings[2][2]
    @test length(bindings[1][1]) >= length(x)
    @test length(bindings[1][2]) >= length(x) + 2
    @test ex.candidates == [:temporary_pair, :cpu, :LinearAlgebra]
    @test occursin("uses workspace slots", sprint(explain, ex))
    pointers = map(pointer, ex.scratch)
    bind!(ex, 2x)
    GC.gc()
    @test execute!(ex) == f(2x)
    @test map(pointer, ex.scratch) == pointers

    rejected =
        prepare_forward(x -> sin.(x), x; implementations = (TestImplementation(:unavailable),))
    @test rejected.providers == [:cpu]
    @test isempty(rejected.scratch)
    @test_throws "workspace binding must return" prepare_forward(
        x -> sin.(x),
        x;
        implementations = (InvalidWorkspaceBinding(),),
    )
    @test_throws ArgumentError WorkspaceSpec(Float32, -1)
    @test_throws ArgumentError WorkspaceSpec(Real, 1)
end

@testset "Default implementations use the public protocol" begin
    x = reshape(Float32.(1:6), 2, 3)
    ex = prepare_forward(x -> x' * x, x)
    @test execute!(ex) == x' * x
    @test ex.providers == [:LinearAlgebra]
    @test occursin("CPU scalar kernels do not handle matmul", sprint(explain, ex))
    @test default_implementations(CPUTarget()) == (CPUKernels(), LinearAlgebraGemm())
end
