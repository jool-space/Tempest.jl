# Independently check the exact 256×7 RMSNorm artifact exported by this experiment.
# julia --project=examples examples/research/tensor_ir_rmsnorm_check.jl rmsnorm.h256.cubin
# This is a fixed artifact ABI check, not a generic TensorIR loader or backend.
using CUDACore
using Random
using Test

if !isdefined(@__MODULE__, :ReductionScopes)
    include("reduction_scopes.jl")
end

function check_rmsnorm_artifact(path)
    CUDACore.functional() || error("This research check requires functional CUDA")
    module_handle = CUDACore.CuModule(read(path))
    kernel = CUDACore.CuFunction(module_handle, "rmsnorm")
    inputs = (CuArray{Float32}(undef, 256, 7), CuArray{Float32}(undef, 256, 7),
              CuArray{Float32}(undef, 256))
    outputs = (similar(inputs[1]), similar(inputs[1]), CuArray{Float32}(undef, 1, 7))
    arrays = (inputs..., outputs...)
    rng = MersenneTwister(1234)

    @testset "TensorIR RMSNorm against eager Julia" begin
        for case in (:random, :changed_inputs, :zero)
            x = case === :zero ? zeros(Float32, 256, 7) : randn(rng, Float32, 256, 7)
            residual = case === :zero ? zero(x) : randn(rng, Float32, size(x))
            weight = randn(rng, Float32, 256)
            expected = ReductionScopes.rmsnorm(x, residual, weight)
            for (destination, source) in zip(inputs, (x, residual, weight))
                copyto!(destination, source)
            end
            for output in outputs
                fill!(output, NaN32) # Incomplete writes cannot pass the finite check.
            end

            # TensorIR v0.1.4 CudaTileRuntimeKernel.cpp uses blockDim=(1,1,1),
            # sharedMemBytes=0. cuTile.jl uses threads=1 for the same convention:
            # the TileIR cubin carries the actual worker-thread metadata.
            # This artifact's grid maps blockId_x to one of seven columns.
            # The runtime's optional cluster SPREAD preference is omitted here;
            # the artifact uses one CTA per cluster. The six args are pointers,
            # with no Julia CuDeviceArray descriptors or implicit seed argument.
            GC.@preserve arrays begin
                CUDACore.cudacall(kernel, NTuple{6, CUDACore.CuPtr{Float32}},
                                 pointer.(arrays)...; blocks = 7, threads = 1)
                CUDACore.synchronize()
            end

            for (name, device_result, reference) in zip(keys(expected), outputs, values(expected))
                actual = Array(device_result)
                @test all(isfinite, actual)
                @test all(isfinite, reference)
                @test all(isapprox.(actual, reference; rtol = 2f-5, atol = 2f-6))
                println(case, " / ", name, ": max absolute error = ",
                        maximum(abs.(actual .- reference)))
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("Pass the exported 256×7 RMSNorm cubin path")
    check_rmsnorm_artifact(only(ARGS))
end
