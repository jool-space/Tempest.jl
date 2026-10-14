# julia --project=examples examples/research/tensor_ir_training.jl LIBRARY
# Run with tileiras on PATH; see docs/src/tensor-ir-training.md.
include("tensor_ir_normalization.jl")

using .TensorIRBackend
using .TensorIRNormalization
using Tempest
using Tempest: TensorSpec
using CUDACore
using Random
using Test
import cuBLAS

const epsilon = 0.03f0
const N = TensorIRNormalization
const B = TensorIRBackend

loss(x, residual, weight) =
    sum(abs2, residual_rmsnorm(x, residual, weight; eps = epsilon)) / length(x)

function sample_inputs(rng, ::Type{T}, rows, columns) where {T}
    (randn(rng, T, rows, columns),
     T(0.3) .* randn(rng, T, rows, columns),
     randn(rng, T, rows))
end

function check_reference_rule(rng)
    @testset "Reference derivative against finite differences" begin
        inputs = sample_inputs(rng, Float64, 4, 3)
        value, gradients = value_and_gradient(trace(loss, inputs...), inputs...)
        @test value ≈ loss(inputs...)
        for index in eachindex(inputs)
            direction = randn(rng, size(inputs[index]))
            step = 1e-6
            plus = Base.setindex(inputs, inputs[index] .+ step .* direction, index)
            minus = Base.setindex(inputs, inputs[index] .- step .* direction, index)
            numerical = (loss(plus...) - loss(minus...)) / (2step)
            analytic = sum(gradients[index] .* direction)
            @test analytic ≈ numerical rtol=1e-7 atol=1e-9
        end
    end
end

function check_outputs(actual, inputs)
    value, gradients = value_and_gradient(trace(loss, inputs...), inputs...)
    @test length(actual) == 1 + length(gradients)
    @test only(Array(actual[1])) ≈ value rtol=3e-5 atol=3e-6
    for (device, expected) in zip(actual[2:end], gradients)
        @test size(device) == size(expected)
        observed = Array(device)
        @test all(isfinite, observed)
        @test all(isapprox.(observed, expected; rtol = 3e-4, atol = 3e-6))
    end
end

addresses(ex) = map(pointer, (ex.inputs..., ex.buffers...))

function check_saved_forward(compiler, rng)
    @testset "Forward exposes its saved values" begin
        inputs = sample_inputs(rng, Float32, 64, 8)
        forward = (x, residual, weight) -> N.forward_region(x, residual, weight; eps = epsilon)
        ex = prepare_forward(forward, CuArray.(inputs)...;
                             implementations = (TensorIRImplementation(compiler),))
        @test ex.providers == [:tensor_ir]
        expected = N.forward_saved(inputs...; eps = epsilon)
        foreach(array -> fill!(array, NaN32), values(outputs(ex)))
        actual = execute!(ex)
        for key in keys(expected)
            observed = Array(actual[key])
            @test all(isfinite, observed)
            @test all(isapprox.(observed, expected[key]; rtol = 3e-5, atol = 3e-6))
        end

        # Compilation does not recognize a normalization name or pattern.
        reference = x -> sin.(x) .+ sum(x .* x; dims = 1)
        generic = GraphCall(:unrelated_graph, reference, TensorSpec)
        other = prepare_forward(generic, CuArray(first(inputs));
                                implementations = (TensorIRImplementation(compiler),))
        @test other.providers == [:tensor_ir]
        @test Array(execute!(other)) ≈ reference(first(inputs)) rtol=3e-5 atol=3e-6
    end
end

function check_training(compiler, rng)
    for (rows, columns) in ((64, 8), (256, 32)), reuse in (false, true)
        @testset "Training $(rows)×$(columns), reuse=$reuse" begin
            inputs = sample_inputs(rng, Float32, rows, columns)
            before = compiler.compilations
            ex = prepare(loss, CuArray.(inputs)...; reuse,
                         implementations = (NormalizationImplementation(compiler),))
            @test count(==(:tensor_ir), ex.providers) == 3
            @test compiler.compilations == before + 3
            fixed_addresses = addresses(ex)
            check_outputs(execute!(ex), inputs)

            # Exercise a launch on a different stream before capture. CUDACore
            # must see pointer access again to transfer memory ownership.
            other_stream = CUDACore.CuStream()
            CUDACore.stream!(other_stream) do
                check_outputs(execute!(ex), inputs)
            end

            capture!(ex)
            compiled = compiler.compilations
            for case in (:changed, :zero)
                updated = sample_inputs(rng, Float32, rows, columns)
                if case === :zero
                    fill!(updated[1], 0)
                    fill!(updated[2], 0)
                end
                bind!(ex, updated...)
                GC.gc()
                check_outputs(replay!(ex), updated)
                @test addresses(ex) == fixed_addresses
                @test compiler.compilations == compiled
            end
            capture!(ex)
            check_outputs(replay!(ex),
                          (Array(ex.inputs[1]), Array(ex.inputs[2]), Array(ex.inputs[3])))
            @test compiler.compilations == compiled
            @test addresses(ex) == fixed_addresses
            println("$(rows)×$(columns), reuse=$reuse: ", ex.providers,
                    "; TensorIR compilations = ", compiled - before)
        end
    end
end

function check_fallback(compiler, rng)
    @testset "Unsupported signatures retain the reference" begin
        # H=259 rejects the forward. N=7 rejects only weight backward: the
        # selected forward/backward pair must still roll back as a whole.
        for (T, rows, columns, reason) in ((Float32, 259, 8, "power-of-two"),
                                          (Float32, 64, 7, "power-of-two"),
                                          (Float64, 64, 8, "Float32"))
            inputs = sample_inputs(rng, T, rows, columns)
            before = compiler.compilations
            ex = prepare(loss, CuArray.(inputs)...;
                         implementations = (NormalizationImplementation(compiler),))
            @test :tensor_ir ∉ ex.providers
            @test compiler.compilations == before
            @test occursin(reason, sprint(explain, ex))
            check_outputs(execute!(ex), inputs)
            capture!(ex)
            check_outputs(replay!(ex), inputs)
        end
    end
end

function check_runtime_contract(compiler)
    @testset "Runtime rejects invalid bindings and preserves compiler errors" begin
        error = try
            B.compile_program(compiler, "this is not MLIR")
            nothing
        catch caught
            caught
        end
        @test error isa B.TensorIRFailure
        @test occursin("parse", sprint(showerror, error))

        input = zeros(Float32, 8, 4)
        source = TensorIRExport.export_mlir(trace(x -> x .+ 1f0, input))
        program = B.compile_program(compiler, source)
        specs = (TensorSpec(input), TensorSpec(input))
        arrays = (CuArray(input), CuArray(input))
        try
            @test_throws ArgumentError B.bind_arguments(program, (arrays[1],), specs)
            @test_throws DimensionMismatch B.bind_arguments(
                program, (CuArray(zeros(Float32, 4, 8)), arrays[2]), specs)
            @test_throws ArgumentError B.bind_arguments(
                program, (CuArray(Float64.(input)), arrays[2]), specs)
        finally
            B.release!(program)
        end
    end
end

function main(library_path)
    rng = MersenneTwister(20260921)
    check_reference_rule(rng)
    if !CUDACore.functional()
        @info "Skipping TensorIR GPU checks: CUDACore is not functional"
        return
    end
    compiler = Compiler(library_path)
    check_saved_forward(compiler, rng)
    check_training(compiler, rng)
    check_fallback(compiler, rng)
    check_runtime_contract(compiler)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("Pass libtempest_tensor_ir_runtime.so")
    main(only(ARGS))
end
