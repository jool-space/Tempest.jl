# Run: julia --project=examples examples/research/cutile_sweeps_tests.jl
include("cutile_sweeps.jl")
include("graph_candidates.jl")

using .GraphCandidates

using .CuTileSweeps
using Tempest
using CUDACore
using Random
using Test
import cuBLAS
const D = Tempest
const S = CuTileSweeps

# Neither tracing nor the scheduler knows this function's scalar formula. Its
# explicit contract permits repeated evaluation on the same input. Nested
# scalar captures exercise cuTile's specialization without hiding tensor reads.
struct WarpParameters
    slope::Float32
    offset::Float32
end

struct ResearchWarp
    parameters::WarpParameters
end

function (f::ResearchWarp)(x)
    p = f.parameters
    exp(ifelse(x > 0f0, p.slope * x, p.offset * x))
end
S.ReductionScopes.pure_scalar(::ResearchWarp) = true

const warp = ResearchWarp(WarpParameters(0.25f0, 0.125f0))

function custom_normalization(x, residual, weight)
    h = warp.(x .+ residual)
    mean = sum(h; dims = 1) ./ size(h, 1)
    centered = h .- mean
    variance = sum(abs2, centered; dims = 1) ./ size(h, 1)
    y = centered ./ sqrt.(variance .+ 1f-5) .* weight
    (y = y, shared = h, mean = mean, variance = variance)
end

function three_reductions(x)
    a = sum(exp.(x); dims = 1) ./ size(x, 1)
    b = sum(abs2, x .- a; dims = 1) ./ size(x, 1)
    c = sum(sin.(x .+ b); dims = 1) ./ size(x, 1)
    (y = (x .+ c) ./ sqrt.(b .+ 1f-3), statistic = c)
end

function parallel_reductions(x)
    a = sum(exp.(x); dims = 1)
    b = sum(abs2, x; dims = 1)
    (y = x .+ a ./ (b .+ 1f0), a = a, b = b)
end

function padding_sensitive(x)
    h = log.(x)
    mean = sum(h; dims = 1) ./ size(x, 1)
    (y = h .- mean, mean = mean)
end

function check_outputs(actual, expected)
    @test D.same_structure(actual, expected)
    for (device, reference) in zip(D.tree_leaves(actual), D.tree_leaves(expected))
        @test size(device) == size(reference)
        @test eltype(device) === eltype(reference)
        observed = Array(device)
        @test all(isfinite, observed)
        @test all(isapprox.(observed, reference; rtol = 3f-4, atol = 3f-5))
    end
end

addresses(ex) = map(pointer, (ex.inputs..., ex.outputs...))

function check_capture(ex, f, inputs, replacements)
    expected = f(inputs...)
    check_outputs(launch!(ex), expected)
    fixed = addresses(ex)
    CUDACore.synchronize()
    graph = CUDACore.capture() do
        launch!(ex)
    end
    extension = Base.get_extension(Tempest, :CUDAExt)
    kinds = extension.graph_node_types(graph)
    @test count(==(CUDACore.CU_GRAPH_NODE_TYPE_KERNEL), kinds) == 1
    @test all(k -> k ∉ (CUDACore.CU_GRAPH_NODE_TYPE_MEM_ALLOC,
                        CUDACore.CU_GRAPH_NODE_TYPE_MEM_FREE), kinds)
    executable = CUDACore.instantiate(graph)
    for updated in replacements
        bind_inputs!(ex, updated...)
        foreach(a -> fill!(a, NaN32), ex.outputs)
        GC.gc()
        CUDACore.launch(executable)
        # launch! returns the structured borrowed outputs; do not launch another
        # kernel here, or a broken replay could be masked by ordinary execution.
        ids = typeof(ex.program).parameters[3]
        actual = D.tree_map(ex.analysis.program.output) do id
            ex.outputs[findfirst(==(id), ids)]
        end
        check_outputs(actual, f(updated...))
        @test addresses(ex) == fixed
    end
    # Normal launches on another stream must still observe buffer ownership.
    CUDACore.stream!(CUDACore.CuStream()) do
        check_outputs(launch!(ex), f(last(replacements)...))
    end
    graph, executable
end

function run_tests()
    if !CUDACore.functional()
        @info "Skipping streamed cuTile tests: CUDACore is not functional"
        return
    end
    rng = MersenneTwister(20260921)
    for rows in (259, 8193)
        @testset "Custom function, dependent reductions, rows=$rows" begin
            inputs = (randn(rng, Float32, rows, 7),
                      0.2f0 .* randn(rng, Float32, rows, 7),
                      randn(rng, Float32, rows))
            program = trace(custom_normalization, inputs...)
            boundary = only(reduction_candidates(program))
            @test boundary.inputs == program.arguments
            outlined = standalone_program(boundary)
            @test all(id -> outlined.local_to_original[outlined.original_to_local[id]] == id,
                      boundary.members)
            ex = prepare_sweeps(outlined.program, CuArray.(inputs)...)
            function ordered_reference(args...)
                values = Dict(zip(D.tree_leaves(program.output),
                                  D.tree_leaves(custom_normalization(args...))))
                Tuple(values[id] for id in boundary.outputs)
            end
            @test length(typeof(ex.program).parameters[4]) == 2
            @test ex.tile == 256
            changed = map(x -> 0.3f0 .* randn(rng, Float32, size(x)), inputs)
            zeroed = (zero(inputs[1]), zero(inputs[2]), ones(Float32, rows))
            check_capture(ex, ordered_reference, inputs, (changed, zeroed))
            before = map(Array, ex.inputs)
            @test_throws r"overlaps" bind_inputs!(ex, ex.inputs[2], ex.inputs[1], zeroed[3])
            @test map(Array, ex.inputs) == before
            explain_sweeps(stdout, ex)
        end
    end

    @testset "Unfamiliar compositions and padding" begin
        for (f, phase_count) in ((three_reductions, 3), (parallel_reductions, 1),
                                  (padding_sensitive, 1))
            input = 0.2f0 .* rand(rng, Float32, 259, 3) .+ 0.5f0
            ex = prepare_sweeps(trace(f, input), CuArray(input); tile = 128)
            @test length(typeof(ex.program).parameters[4]) == phase_count
            check_capture(ex, f, (input,), ((ones(Float32, size(input)),),))
        end
    end

    @testset "Column broadcasts and repeated statistic outputs" begin
        function statistics_only(x, bias)
            h = x .+ bias
            statistic = sum(abs2, h; dims = 1)
            (statistic, statistic)
        end
        input = randn(rng, Float32, 259, 3)
        bias = randn(rng, Float32, 1, 3)
        ex = prepare_sweeps(trace(statistics_only, input, bias), CuArray(input), CuArray(bias))
        @test length(ex.outputs) == 1
        actual = launch!(ex)
        @test actual[1] === actual[2]
        check_capture(ex, statistics_only, (input, bias), ((2f0 .* input, zero(bias)),))
    end

    @testset "Dead captures do not enter compiled schedules" begin
        scale = Float32[2]
        unused(x) = x * scale[1]
        function with_dead_code(x, unused_array)
            unused.(x)
            sum(x; dims = 1)
        end
        input = randn(rng, Float32, 17, 3)
        unused_array = ones(Float32, 2, 3, 4)
        graph = trace(with_dead_code, input, unused_array)
        ex = prepare_sweeps(graph, CuArray(input), CuArray(unused_array))
        @test isbits(ex.program)
        check_outputs(launch!(ex), with_dead_code(input, unused_array))
    end

    @testset "Unsupported schedules reject before compilation" begin
        input = randn(rng, Float32, 17, 3)
        unknown(x) = (t -> t * t).(x) .+ sum(x; dims = 1)
        @test_throws r"purity contract" prepare_sweeps(trace(unknown, input), CuArray(input))
        across(x) = x .- sum(x; dims = 2)
        @test_throws r"reduces rows" prepare_sweeps(trace(across, input), CuArray(input))
        f(x) = x .- sum(x; dims = 1)
        wide = Float64.(input)
        @test_throws r"Float32" prepare_sweeps(trace(f, wide), CuArray(wide))
        @test_throws r"power of two" prepare_sweeps(trace(f, input), CuArray(input); tile = 129)
        @test_throws r"dense CuArrays" prepare_sweeps(trace(f, input), input)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_tests()
end
