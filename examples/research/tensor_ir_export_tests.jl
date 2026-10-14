# Loaded on demand by TensorIRExport.examples/check_exporter, or run directly.
if !isdefined(@__MODULE__, :TensorIRExport)
    include("tensor_ir_export.jl")
end
if !isdefined(@__MODULE__, :ReductionScopes)
    include("reduction_scopes.jl")
end

module TensorIRExportTests

using Tempest
using Test
using ..TensorIRExport
using ..ReductionScopes

const D = Tempest
const R = ReductionScopes
const E = TensorIRExport
const export_mlir = E.export_mlir
E.reduction_description(op::R.ResearchMaximum) = (mode = "max", axes = op.axes)

function examples(; rows = 259, columns = 7)
    x = zeros(Float32, rows, columns)
    w = ones(Float32, rows)
    (
        rmsnorm = D.trace(R.rmsnorm, x, x, w),
        layernorm = D.trace(R.layernorm, x, w),
        softmax = D.trace(R.softmax, x),
        composed = D.trace(R.composed, x),
    )
end

unsupported_verb(x::D.TracedArray) = D.record_call(unsupported_verb, x)
D.output_spec(::typeof(unsupported_verb), x) = D.TensorSpec(x)

function rmsnorm_backward_input(dy, h, w, rstd)
    scaled = Float32.(dy) .* Float32.(w)
    projection = sum(scaled .* h; dims = 1) ./ Float32(size(h, 1))
    scaled .* rstd .- h .* rstd .^ 3 .* projection
end

function rmsnorm_backward_weight(dy, h, rstd)
    reshape(sum(dy .* h .* rstd; dims = 2), size(h, 1))
end

function backward_examples(; rows = 64, columns = 8)
    h = zeros(Float32, rows, columns)
    w = ones(Float32, rows)
    rstd = ones(Float32, 1, columns)
    (
        rmsnorm_dx = D.trace(rmsnorm_backward_input, h, h, w, rstd),
        rmsnorm_dw = D.trace(rmsnorm_backward_weight, h, h, rstd),
    )
end

function check_exporter()
    x = zeros(Float32, 5, 3)
    @testset "Bounded TensorIR research export" begin
        for (name, program) in pairs(examples())
            @test occursin("@$name", export_mlir(program; name = string(name)))
        end
        vector = export_mlir(D.trace((x, w) -> x .* w, x, ones(Float32, 5)))
        @test occursin("tensor<5xf32> -> tensor<5x1xf32>", vector)
        @test occursin("stride = \"(1,5)\"", vector)
        column = export_mlir(D.trace((x, w) -> x .* w,
                                     zeros(Float32, 5, 1), ones(Float32, 5)))
        @test occursin("tensor<5xf32> -> tensor<5x1xf32>", column)
        @test !occursin(" = broadcast ", column)
        shared = export_mlir(D.trace(x -> (a = exp.(x); (a, a .+ 1f0)), x))
        @test length(findall(" = exp ", shared)) == 1
        @test_throws r"scalar function" export_mlir(D.trace(x -> (t -> t * t).(x), x))
        @test_throws r"layout mapping" export_mlir(D.trace(transpose, x))
        @test_throws r"non-Float32" export_mlir(D.trace(identity, Float64.(x)))
        @test_throws r"exactly representable" export_mlir(D.trace(x -> x .+ 16777217, x))
        @test_throws r"Float32 literals" export_mlir(D.trace(x -> x .+ 0.1, x))
        @test_throws r"scalar" export_mlir(D.trace(sum, x))
        @test_throws r"layout mapping" export_mlir(D.trace(unsupported_verb, x))
        @test_throws r"runtime scalars" export_mlir(D.trace((x, a) -> x .* a, x, 2f0))
        @test_throws r"nonfinite" export_mlir(D.trace(x -> x .+ Inf32, x))
    end

    @testset "Backward reductions and bounded layout semantics" begin
        programs = backward_examples()
        dx = export_mlir(programs.rmsnorm_dx; name = "rmsnorm_dx")
        dw = export_mlir(programs.rmsnorm_dw; name = "rmsnorm_dw")
        @test occursin("dimensions = [0]", dx)
        @test occursin("dimensions = [1]", dw)
        @test occursin("tensor<64x1xf32> -> tensor<64xf32>", dw)
        @test occursin("_square = mul", dx)
        @test occursin("stride = \"(1)\"", dw)
        @test occursin(" = neg ", export_mlir(D.trace(x -> .-x, x)))

        # n-ary arithmetic keeps Julia's left-to-right operand order.
        for (f, operation) in ((+, "add"), (*, "mul"))
            source = export_mlir(D.trace((a, b, c, d) -> f.(a, b, c, d), x, x, x, x))
            @test length(findall(" = $operation ", source)) == 3
            @test occursin("_fold2 = $operation %v1, %v2", source)
            @test occursin("_fold3 = $operation %v5_fold2, %v3", source)
        end

        cast = export_mlir(D.trace(x -> Float32.(exp.(x)), x))
        @test length(findall(" = exp ", cast)) == 1
        @test_throws E.UnsupportedExport export_mlir(D.trace(x -> reshape(x, 15), x))
        @test_throws E.UnsupportedExport export_mlir(D.trace(x -> (x, x), x))
        @test_throws E.UnsupportedExport export_mlir(D.trace(x -> x .^ 4, x))
        @test_throws E.UnsupportedExport export_mlir(D.trace(x -> Float32.(x), Float64.(x)))
        @test_throws E.UnsupportedExport export_mlir(D.trace(x -> x .+ 16777217, x))
    end
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    TensorIRExportTests.check_exporter()
end
