# julia --project=. examples/research/graph_regions_tests.jl
include("graph_regions.jl")
include("reduction_scopes.jl")

using .GraphRegions
using .ReductionScopes
using Tempest
using Tempest: TensorSpec, record_call
using Random
using Test
const D = Tempest
const R = ReductionScopes

function computation_ids(program)
    execution = GraphRegions.execution_program(program)
    live = D.live_nodes(execution.nodes, execution.results)
    findall(eachindex(execution.nodes)) do id
        live[id] && !(execution.nodes[id].op isa D.Argument)
    end
end

function check_result(actual, expected)
    @test D.same_structure(actual, expected)
    observed_leaves, expected_leaves = D.tree_leaves(actual), D.tree_leaves(expected)
    @test length(observed_leaves) == length(expected_leaves)
    for (observed, reference) in zip(observed_leaves, expected_leaves)
        @test (observed isa Number) == (reference isa Number)
        @test size(observed) == size(reference)
        @test eltype(observed) === eltype(reference)
        @test observed ≈ reference rtol=2e-5 atol=2e-6
    end
end

custom_scalar(x) = x / (1 + abs(x))
custom_composition(x) = custom_scalar.(x) .+ sum(x; dims = 1)

const pair_evaluations = Ref(0)
paired(x) = (square = x .* x, wave = sin.(x))
paired(x::D.TracedArray) = record_call(paired, x)
D.output_spec(::typeof(paired), x) = (square = TensorSpec(x), wave = TensorSpec(x))
function D.reference(::typeof(paired), x)
    pair_evaluations[] += 1
    paired(x)
end

function run_tests()
    rng = MersenneTwister(20260921)
    x = randn(rng, Float32, 5, 3)
    residual = randn(rng, Float32, size(x))
    weight = randn(rng, Float32, size(x, 1))

    @testset "Semantic regions preserve unfamiliar computations" begin
        cases = ((R.rmsnorm, (x, residual, weight)),
                 (R.layernorm, (x, weight)),
                 (R.composed, (x,)),
                 (R.softmax, (x,)),
                 (custom_composition, (x,)))
        for (f, inputs) in cases
            program = trace(f, inputs...)
            boundary = region_boundary(program, computation_ids(program))
            @test length(boundary.order) == 1
            @test boundary.outputs == sort(unique(D.tree_leaves(program.output)))
            check_result(execute_partition(boundary, inputs...), f(inputs...))
        end
        # An orthogonal reduction is legal mathematics even though the earlier
        # one-axis scope scheduler and the current TensorIR adapter are bounded.
        orthogonal(x) = (rows = sum(x; dims = 2), columns = sum(x; dims = 1))
        program = trace(orthogonal, x)
        boundary = region_boundary(program, computation_ids(program))
        check_result(execute_partition(boundary, x), orthogonal(x))
    end

    @testset "Escapes and graph contraction" begin
        function branching(x, residual)
            h = x .+ residual
            y = tanh.(h)
            (y = y, observation = sum(h))
        end
        program = trace(branching, x, residual)
        y = program.output.y
        h = only(program.nodes[y].inputs)
        boundary = region_boundary(program, [h, y])
        @test boundary.inputs == program.arguments
        @test boundary.outputs == [h, y]
        @test length(boundary.order) == 2
        check_result(execute_partition(boundary, x, residual), branching(x, residual))

        function independent(x)
            a = sin.(x)
            b = exp.(x)
            c = cos.(a)
            (c = c, b = b)
        end
        program = trace(independent, x)
        c = program.output.c
        a = only(program.nodes[c].inputs)
        boundary = region_boundary(program, [a, c])
        @test c > a + 1 # an unrelated operation lies between the selected nodes
        @test boundary.outputs == [c]
        check_result(execute_partition(boundary, x), independent(x))

        function cyclic(x)
            a = sin.(x)
            b = cos.(a)
            a .* b
        end
        program = trace(cyclic, x)
        last = program.output
        a, b = program.nodes[last].inputs
        @test_throws r"dependency cycle" region_boundary(program, [a, last])
        @test_throws r"arguments are region inputs" region_boundary(program, [1])
        @test_throws r"at least one" region_boundary(program, Int[])
        function dead(x)
            sin.(x)
            cos.(x)
        end
        program = trace(dead, x)
        @test_throws r"dead value" region_boundary(program, [2])
    end

    @testset "Logical maps and escaping aliases" begin
        function aliases(x)
            h = sin.(x)
            (h = h, window = view(h, 1:2, :), y = tanh.(h))
        end
        program = trace(aliases, x)
        boundary = region_boundary(program, [program.output.h, program.output.y])
        actual = execute_partition(boundary, x)
        check_result(actual, aliases(x))
        @test Base.mightalias(actual.h, actual.window)
        @test boundary.mappings[program.output.window][1] == program.output.h

        # The alias can also be exported from inside the region. It carries a
        # mapping to h rather than pretending it is an unrelated dense value.
        boundary = region_boundary(program, computation_ids(program))
        actual = execute_partition(boundary, x)
        @test Base.mightalias(actual.h, actual.window)
        @test size(actual.window) == (2, 3)

        # Here only the view escapes; its full storage root remains internal.
        # The CPU view retains that root. A GPU provider would need to declare
        # root retention or choose to materialize the escaping presentation.
        function hidden_root(x)
            h = sin.(x)
            (window = view(h, 1:2, :), y = tanh.(h))
        end
        program = trace(hidden_root, x)
        boundary = region_boundary(program, computation_ids(program))
        root, mapping = boundary.mappings[program.output.window]
        @test root in boundary.members
        @test root ∉ boundary.outputs
        check_result(execute_partition(boundary, x), hidden_root(x))

        square = randn(rng, Float32, 4, 4)
        permuted(x) = (x = x, transposed = transpose(x))
        program = trace(permuted, square)
        boundary = region_boundary(program, computation_ids(program))
        root, mapping = boundary.mappings[program.output.transposed]
        @test root == only(program.arguments)
        @test !D.linear_identity(mapping)
        check_result(execute_partition(boundary, square), permuted(square))
    end

    @testset "Semantic calls remain atomic" begin
        function f(x)
            pair = paired(x)
            pair.square .+ pair.wave
        end
        program = trace(f, x)
        siblings = collect(program.nodes[program.output].inputs)
        @test_throws r"result leaves must stay together" region_boundary(program, [first(siblings)])
        boundary = region_boundary(program, siblings)
        pair_evaluations[] = 0
        check_result(execute_partition(boundary, x), f(x))
        @test pair_evaluations[] == 1
        @test boundary.outputs == siblings

        # The call may instead be an outside producer. Its two result leaves
        # still arrive together, from exactly one reference invocation.
        boundary = region_boundary(program, [program.output])
        @test boundary.inputs == siblings
        pair_evaluations[] = 0
        check_result(execute_partition(boundary, x), f(x))
        @test pair_evaluations[] == 1

        # Even a result unused by the caller belongs to the call's atomic unit.
        single(x) = paired(x).square
        program = trace(single, x)
        @test_throws r"result leaves must stay together" region_boundary(program, [program.output])
        boundary = region_boundary(program, computation_ids(program))
        @test boundary.outputs == [program.output]
        pair_evaluations[] = 0
        check_result(execute_partition(boundary, x), single(x))
        @test pair_evaluations[] == 1
    end

    @testset "Scalars, casts, and functional state" begin
        function scalar_and_array(x, scale)
            rounded = Float32.(Float64.(x) .* scale)
            (value = rounded, total = sum(rounded), repeated = rounded)
        end
        program = trace(scalar_and_array, x, 0.7)
        boundary = region_boundary(program, computation_ids(program))
        result = execute_partition(boundary, x, 0.7)
        check_result(result, scalar_and_array(x, 0.7))
        @test result.value === result.repeated
        @test result.total isa Float32
        check_result(execute_partition(boundary, x, 1.1), scalar_and_array(x, 1.1))

        # The same graph vocabulary contains backward and a functional update.
        # Buffer donation remains a later physical decision; this probe mutates
        # neither the input parameters nor the old moments.
        loss(p, moments) = sum(abs2, p)
        moments = zero(x)
        gradient = differentiate(trace(loss, x, moments); wrt = (1,))
        step = with_update(gradient) do loss, gradients, p, old_moments
            next_moments = 0.9f0 .* old_moments .+ only(gradients)
            (params = p .- 0.01f0 .* next_moments, moments = next_moments, loss = loss)
        end
        boundary = region_boundary(step, computation_ids(step))
        original = copy(x)
        expected_moments = 0.9f0 .* moments .+ 2f0 .* x
        expected = (params = x .- 0.01f0 .* expected_moments,
                    moments = expected_moments, loss = sum(abs2, x))
        check_result(execute_partition(boundary, x, moments), expected)
        @test x == original
        @test iszero(moments)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_tests()
end
