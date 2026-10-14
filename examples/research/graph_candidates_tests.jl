# julia --project=. examples/research/graph_candidates_tests.jl
include("graph_candidates.jl")
include("reduction_scopes.jl")

using .GraphCandidates
using .GraphRegions
using .ReductionScopes
using Tempest
using Tempest: TensorSpec, record_call
using Random
using Test
const D = Tempest
const G = GraphCandidates
const R = ReductionScopes

# The instance itself must survive outlining; this is not a built-in opcode.
struct ShiftedSquare{T}
    offset::T
end
(f::ShiftedSquare)(x) = (x + f.offset) * (x + f.offset)

barrier(x) = sin.(x)
barrier(x::D.TracedArray) = record_call(barrier, x)
D.output_spec(::typeof(barrier), x) = TensorSpec(x)
D.reference(::typeof(barrier), x) = barrier(x)

function all_values(program, args...)
    execution = GraphRegions.execution_program(program)
    inputs = D.validate(execution.primal, args)
    values = Dict{Int, Any}(zip(execution.primal.arguments, inputs))
    live = D.live_nodes(execution.nodes, execution.results)
    operations = findall(eachindex(execution.nodes)) do id
        live[id] && !(execution.nodes[id].op isa D.Argument)
    end
    merge!(values, GraphRegions.evaluate_unit(execution.nodes, operations, values))
end

function check_candidates(program, args...)
    boundaries = reduction_candidates(program)
    values = all_values(program, args...)
    members = [id for boundary in boundaries for id in boundary.members]
    @test length(members) == length(unique(members))
    for boundary in boundaries
        outlined = standalone_program(boundary)
        inputs = Tuple(values[id] for id in boundary.inputs)
        expected = Tuple(values[id] for id in boundary.outputs)
        actual = Tuple(D.interpret(D.ForwardProgram(outlined.program), inputs...))
        @test D.same_structure(actual, expected)
        @test all(a ≈ b for (a, b) in zip(actual, expected))
        @test all((a isa Number) == (b isa Number) for (a, b) in zip(actual, expected))
        for id in boundary.members
            local_id = outlined.original_to_local[id]
            @test outlined.local_to_original[local_id] == id
            @test outlined.program.nodes[local_id].op === boundary.program.nodes[id].op
        end
    end
    boundaries
end

function run_tests()
    rng = MersenneTwister(71)
    x = randn(rng, Float32, 9, 5)
    residual = randn(rng, Float32, size(x))
    weight = randn(rng, Float32, size(x, 1))

    @testset "Automatic connected reduction candidates" begin
        cases = ((R.rmsnorm, (x, residual, weight)),
                 (R.layernorm, (x, weight)),
                 (R.composed, (x,)),
                 (R.rms_input_backward, (x, weight, residual)))
        for (f, inputs) in cases
            program = trace(f, inputs...)
            boundary = only(check_candidates(program, inputs...))
            @test boundary.outputs == sort!(unique!(Int[D.tree_leaves(program.output)...]))
            @test length(boundary.order) == 1
        end

        callable = ShiftedSquare(0.7f0)
        custom(x) = callable.(x) ./ (sum(abs2, x; dims = 1) .+ 1f0)
        program = trace(custom, x)
        boundary = only(check_candidates(program, x))
        outlined = standalone_program(boundary)
        custom_nodes = filter(outlined.program.nodes) do node
            node.op isa D.Pointwise && node.op.f === callable
        end
        custom_node = only(custom_nodes)
        @test custom_node.op.f === callable

        scalar(x, scale) = scale * sum(abs2, x)
        program = trace(scalar, x, 0.7f0)
        boundary = only(check_candidates(program, x, 0.7f0))
        @test only(program.nodes[id].scalar for id in boundary.outputs)
        @test isempty(reduction_candidates(trace(x -> sin.(x), x)))
    end

    @testset "Matmul and layouts stay outside" begin
        function mixed(a, b, weight)
            product = a * b
            h = product .+ weight
            y = h ./ sqrt.(sum(abs2, h; dims = 1) .+ 1f0)
            (y = y, saved = h, product = product)
        end
        a = randn(rng, Float32, 9, 4)
        b = randn(rng, Float32, 4, 5)
        program = trace(mixed, a, b, weight)
        boundary = only(check_candidates(program, a, b, weight))
        product = program.output.product
        @test product in boundary.inputs
        @test product ∉ boundary.members
        @test boundary.outputs == sort([program.output.y, program.output.saved])
        @test any(program.nodes[id].op isa D.MatMul for id in boundary.inputs)
        @test execute_partition(boundary, a, b, weight).y ≈ mixed(a, b, weight).y

        function layout_boundary(x)
            a = sin.(x)
            b = transpose(a)
            sum(a .+ b; dims = 1)
        end
        square = x[1:5, :]
        program = trace(layout_boundary, square)
        boundary = only(check_candidates(program, square))
        @test any(program.nodes[id].op isa D.TransposeOp for id in boundary.inputs)
        @test all(!(program.nodes[id].op isa D.LayoutOperation) for id in boundary.members)
        @test length(boundary.inputs) == 2
    end

    @testset "Leaving and re-entering partitions safely" begin
        function across_barrier(x)
            a = sin.(x)
            s = sum(a; dims = 1)
            b = barrier(a)
            sum(a .+ b .+ s; dims = 1)
        end
        program = trace(across_barrier, x)
        boundaries = check_candidates(program, x)
        @test length(boundaries) == 2
        all_members = [id for boundary in boundaries for id in boundary.members]
        @test_throws r"dependency cycle" region_boundary(program, all_members)
        @test all(!(program.nodes[id].op isa D.CallResult) for id in all_members)
        live = D.live_nodes(program.nodes, Int[D.tree_leaves(program.output)...])
        groups = [boundary.members for boundary in boundaries]
        @test G.can_contract(D.ForwardProgram(program), groups, live)
        for boundary in boundaries
            @test execute_partition(boundary, x) ≈ across_barrier(x)
        end

        function intervening(x)
            a = x .* x
            outside = transpose(x)
            y = a ./ (sum(a; dims = 1) .+ 1f0)
            (y = y, outside = outside)
        end
        program = trace(intervening, x)
        boundary = only(check_candidates(program, x))
        @test program.output.outside ∉ boundary.members
        @test last(boundary.members) - first(boundary.members) + 1 > length(boundary.members)
    end

    @testset "Several contractions remain jointly acyclic" begin
        function crossed(x)
            a = sin.(x)
            b = cos.(x)
            outside_a = barrier(a)
            outside_b = barrier(b)
            left = sum(a .+ outside_b; dims = 1)
            right = sum(b .+ outside_a; dims = 1)
            (; left, right)
        end
        program = trace(crossed, x)
        left = program.output.left
        right = program.output.right
        left_input = only(program.nodes[left].inputs)
        right_input = only(program.nodes[right].inputs)
        a = first(program.nodes[left_input].inputs)
        b = first(program.nodes[right_input].inputs)
        proposed = [[a, left_input, left], [b, right_input, right]]
        # Each alone is convex; together they induce A -> opaque -> B -> opaque -> A.
        @test all(region_boundary(program, group) isa RegionBoundary for group in proposed)
        live = D.live_nodes(program.nodes, Int[D.tree_leaves(program.output)...])
        @test !G.can_contract(D.ForwardProgram(program), proposed, live)
        boundaries = check_candidates(program, x)
        @test length(boundaries) == 2
        @test G.can_contract(
            D.ForwardProgram(program), [boundary.members for boundary in boundaries], live,
        )
    end

    @testset "Standalone inputs retain aliases and scalar metadata" begin
        function viewed(x)
            window = view(x, 2:8, :)
            window ./ (sum(abs2, window; dims = 1) .+ 1f0)
        end
        program = trace(viewed, x)
        boundary = only(check_candidates(program, x))
        @test program.nodes[only(boundary.inputs)].op isa D.ViewOp
        values = all_values(program, x)
        @test values[only(boundary.inputs)] isa SubArray
        outlined = standalone_program(boundary)
        @test outlined.program.nodes[1].op isa D.Argument
        @test outlined.program.nodes[1].dims == (7, 5)

        called = trace(barrier, x)
        selected = region_boundary(called, [called.output])
        @test_throws r"semantic calls" standalone_program(selected)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_tests()
end
