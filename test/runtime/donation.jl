include("../fixtures/implementations.jl")

@testset "Direct donation proof and staged fallback" begin
    a, b = [1.0, 2.0], [3.0, 4.0]
    # Equal element counts alone do not prove that a broadcast reads its lane.
    read = Tempest.Read{1}(Tempest.dense_map((2, 3)), (2, 3), (3, 2))
    @test !Tempest.same_lane_read(read, 1, 6)
    ex = prepare_forward(a -> (a .+ 1,), a; donate = (1,))
    @test length(ex.storage.donations) == 1
    @test isempty(ex.buffers)
    @test isempty(ex.state.commits)
    @test only(execute!(ex)) == a .+ 1
    @test only(execute!(ex)) == a .+ 2
    for f in (a -> (a[[2, 1]] .+ 1,), a -> (a .+ sum(a),), a -> (a .+ 1, a))
        direct = prepare_forward(f, a; donate = (1,))
        staged = prepare_forward(f, a; donate = (1,), inplace = false)
        for _ = 1:3
            @test execute!(direct) == execute!(staged)
        end
    end
    # The output aliases an old-state observation; this must stay staged.
    observed = prepare_forward(a -> (a .+ 1, a), a; donate = (1,))
    @test isempty(observed.storage.donations)
    mapped = prepare_forward(a -> (a[[2, 1]] .+ 1,), a; donate = (1,))
    @test isempty(mapped.storage.donations)
    repeated = prepare_forward((a, b) -> begin
        x = a .+ b
        (x, x)
    end, a, b; donate = (1, 2))
    @test isempty(repeated.storage.donations)
    @test execute!(repeated) == (a + b, a + b)
    # This target has only an implementation that rejects aliasing and grouping.
    fallback = prepare_forward(
        a -> (a .+ 1,),
        a;
        donate = (1,),
        target = ImplementationOnlyTarget(),
        implementations = (TestImplementation(:accept),),
    )
    @test isempty(fallback.storage.donations)
    @test length(fallback.state.commits) == 1
    @test only(execute!(fallback)) == a .+ 1
    @test any(note -> occursin("staged donation", note), fallback.semantic_notes)
end

@testset "Independent array grouping" begin
    arrays = ntuple(i -> fill(Float32(i), 3i + 1), 20)
    update(xs) = (map(x -> x .* 0.9f0 .+ 1.0f0, xs),)
    for inplace in (false, true), reuse in (false, true)
        grouped = prepare_forward(update, arrays; donate = (1,), inplace, reuse)
        separate =
            prepare_forward(update, arrays; donate = (1,), inplace, reuse, grouped = false)
        @test length(grouped.calls) == 1
        @test length(separate.calls) == 20
        for _ = 1:3
            @test execute!(grouped) == execute!(separate)
        end
        @test occursin("groups regions", sprint(explain, grouped))
    end
    chain(xs) = begin
        a = xs[1] .+ 1.0f0
        b = a .+ 1.0f0
        (a, b)
    end
    ex = prepare_forward(chain, (ones(Float32, 4),))
    @test length(ex.calls) == 2
    @test execute!(ex) == (fill(2.0f0, 4), fill(3.0f0, 4))
    storage = ones(4)
    @test !Tempest.independent_bindings(
        (output = storage, inputs = (zeros(4),)),
        (output = zeros(4), inputs = (view(storage, :),)),
    )
    fallback = prepare_forward(
        update,
        arrays;
        target = ImplementationOnlyTarget(),
        implementations = (TestImplementation(:accept),),
    )
    @test length(fallback.calls) == 20
    @test execute!(fallback) == update(arrays)
end

@testset "Wide input tree traversal" begin
    args = ntuple(i -> fill(Float32(i), 2), 1000)
    leaves = Tempest.tree_leaves(args)
    @test length(leaves) == 1000
    mapped = Tempest.tree_map(identity, args)
    @test all(i -> mapped[i] === args[i], eachindex(args))
    names = Tuple(Symbol("parameter_", i) for i = 1:100)
    named = NamedTuple{names}(args[1:100])
    @test keys(Tempest.tree_map(identity, named)) == names
    @test Tempest.same_structure(named, Tempest.tree_map(_ -> nothing, named))
end

struct RejectGroupedBuild <: Implementation end
Tempest.support(::RejectGroupedBuild, ::GroupedRequest) = Supported()
Tempest.support(::RejectGroupedBuild, ::RegionRequest) = Supported()
Tempest.build_implementation(
    ::RejectGroupedBuild,
    ::GroupedRequest,
    runtime,
    outputs,
    inputs,
) = Unsupported("group engine unavailable")
Tempest.build_implementation(
    ::RejectGroupedBuild,
    request::RegionRequest,
    runtime,
    out,
    inputs,
) = build_implementation(CPUKernels(), request, runtime, out, inputs)

@testset "Grouped construction can decline" begin
    arrays = (ones(3), ones(4), ones(5))
    ex = prepare_forward(
        xs -> map(x -> x .+ 1, xs),
        arrays;
        target = ImplementationOnlyTarget(),
        implementations = (RejectGroupedBuild(),),
    )
    @test length(ex.calls) == 3
    @test execute!(ex) == (fill(2.0, 3), fill(2.0, 4), fill(2.0, 5))
    @test any(
        notes -> any(note -> occursin("group engine unavailable", note), notes),
        ex.selection_notes,
    )
end
