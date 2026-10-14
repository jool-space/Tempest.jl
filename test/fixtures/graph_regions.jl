# A deliberately allocating CPU graph provider, shared by compiler/graph_regions.jl
# and compiler/proposals.jl. Its interpreter is independent of scheduling and
# storage, so tests using it exercise region boundaries and planning.
mutable struct GraphTestImplementation <: Implementation
    mode::Symbol
    member_limit::Int
    requests::Vector{Any}
    attempts::Vector{Any}
    built::Vector{Any}
end

GraphTestImplementation(; mode = :accept, member_limit = typemax(Int)) =
    GraphTestImplementation(mode, member_limit, Any[], Any[], Any[])
Tempest.implementation_name(::GraphTestImplementation) = :graph_test

function Tempest.support(impl::GraphTestImplementation, request::Tempest.GraphRequest)
    push!(impl.requests, request)
    request.target isa CPUTarget || return Unsupported("graph test provider requires CPU")
    impl.mode === :reject && return Unsupported("graph test metadata rejection")
    if impl.mode === :two_sums
        count(
            id -> request.region.program.nodes[id].op isa Tempest.SumReduction,
            request.region.members,
        ) >= 2 || return Unsupported("requires two sums together")
    elseif impl.mode === :row_sums
        for id in request.region.members
            operation = request.region.program.nodes[id].op
            if operation isa Tempest.Pointwise && !Tempest.recomputable(operation.f)
                return Unsupported("scalar function has not granted recomputation")
            elseif operation isa Tempest.SumReduction && operation.axes != (1,)
                return Unsupported("only leading-axis sums are supported")
            end
        end
        all(spec -> length(spec.dims) == 2, request.outputs) ||
            return Unsupported("outputs must retain the matrix axes")
    end
    Supported()
end

function graph_test_values(region, inputs)
    nodes = region.program.nodes
    values = Dict{Int, Any}()
    for (id, input) in zip(region.inputs, inputs)
        values[id] = nodes[id].scalar ? only(input) : input
    end
    for id in region.members
        node = nodes[id]
        operands = map(i -> values[i], node.inputs)
        values[id] = Tempest.evaluate(node.op, node, operands)
    end
    values
end

function Tempest.build_implementation(
    impl::GraphTestImplementation,
    request::Tempest.GraphRequest,
    runtime,
    outputs,
    inputs,
)
    push!(impl.attempts, request)
    length(impl.attempts) <= 256 || error("graph selection retried without terminating")
    impl.mode === :broken && error("graph test unexpected compiler failure")
    if impl.mode === :late_reject || length(request.region.members) > impl.member_limit
        return Unsupported("graph test construction rejection")
    end

    region = request.region
    @test inputs isa Tuple
    @test outputs isa Tuple
    @test length(inputs) == length(region.inputs) == length(request.inputs)
    @test length(outputs) == length(region.outputs) == length(request.outputs)
    for (arrays, specs, ids) in
        ((inputs, request.inputs, region.inputs), (outputs, request.outputs, region.outputs))
        for (array, spec, id) in zip(arrays, specs, ids)
            node = region.program.nodes[id]
            @test spec.dims == node.dims == size(array)
            @test spec.dtype === node.dtype === eltype(array)
            @test spec.scalar == node.scalar
        end
    end
    # No in-place contract is supplied by this provider. A graph region may
    # read any input multiple times before producing its final results.
    @test all(!Base.mightalias(out, input) for out in outputs for input in inputs)
    push!(impl.built, request)
    launch = () -> begin
        values = graph_test_values(region, inputs)
        for (output, id) in zip(outputs, region.outputs)
            if region.program.nodes[id].scalar
                output[] = values[id]
            else
                copyto!(output, values[id])
            end
        end
        nothing
    end
    BuiltImplementation(launch, (region, inputs, outputs), 0)
end

function check_graph_test_outputs(actual, expected)
    @test Tempest.same_structure(actual, expected)
    a = Tempest.tree_leaves(actual)
    b = Tempest.tree_leaves(expected)
    @test length(a) == length(b)
    for (observed, reference) in zip(a, b)
        @test size(observed) == size(reference)
        value = reference isa Number ? only(observed) : observed
        @test value ≈ reference
    end
end
