# Run with: julia --project=examples examples/semantic_kernels.jl
# The public function describes results. A separate implementation supplies
# destination-writing CUDA launches, including an explicit backward operation.
module SemanticKernelExample
using Tempest
using Tempest:
    Implementation, CallRequest, Supported, Unsupported, BuiltImplementation, TensorSpec, reference, record_call
using CUDACore
import cuBLAS

# A public mathematical function, plus its symbolic boundary. No allocation
# wrapper or implementation keyword is required at the model call site.
pair_wave(x) = reference(pair_wave, x)
pair_wave(x::Tempest.TracedArray) = record_call(pair_wave, x)
Tempest.output_spec(::typeof(pair_wave), x) = (; s = TensorSpec(x), c = TensorSpec(x))
Tempest.reference(::typeof(pair_wave), x) = (; s = sin.(x), c = cos.(x))
wave_back(gs, gc, s, c) = reference(wave_back, gs, gc, s, c)
wave_back(gs::Tempest.TracedArray, gc, s, c) = record_call(wave_back, gs, gc, s, c)
Tempest.output_spec(::typeof(wave_back), gs, gc, s, c) = TensorSpec(gs)
Tempest.reference(::typeof(wave_back), gs, gc, s, c) = gs .* c .- gc .* s
Tempest.verb_pullback(::typeof(pair_wave), xs, y, g) = (wave_back(g.s, g.c, y.s, y.c),)

struct WaveCUDA <: Implementation end
Tempest.implementation_name(::WaveCUDA) = :wave_cuda

function Tempest.support(::WaveCUDA, r::CallRequest)
    if !(r.call.f in (pair_wave, wave_back))
        return Unsupported("not a wave operation")
    end
    if !(r.target isa CUDATarget)
        return Unsupported("CUDA only")
    end
    if !(all(s -> s.dtype === Float32, r.inputs))
        return Unsupported("Float32 only")
    end
    Supported()
end

function wave_forward_kernel(s, c, x, n)
    i = (CUDACore.blockIdx().x - 1) * CUDACore.blockDim().x + CUDACore.threadIdx().x
    if i <= n
        s[i], c[i] = sincos(x[i])
    end
    nothing
end

function wave_backward_kernel(dx, gs, gc, s, c, n)
    i = (CUDACore.blockIdx().x - 1) * CUDACore.blockDim().x + CUDACore.threadIdx().x
    if i <= n
        dx[i] = gs[i] * c[i] - gc[i] * s[i]
    end
    nothing
end

# These are ordinary destination-writing launch wrappers. Preparing them compiles
# the kernel and binds actual device arrays, before capture.
function wave_forward_call(out, x::CuArray)
    n = length(x)
    kernel = CUDACore.@cuda launch=false wave_forward_kernel(out.s, out.c, x, n)
    () -> begin
        if n != 0
            kernel(out.s, out.c, x, n; threads = 256, blocks = cld(n, 256))
        end
        nothing
    end
end

function wave_backward_call(out::CuArray, gs, gc, s, c)
    n = length(out)
    kernel = CUDACore.@cuda launch=false wave_backward_kernel(out, gs, gc, s, c, n)
    () -> begin
        if n != 0
            kernel(out, gs, gc, s, c, n; threads = 256, blocks = cld(n, 256))
        end
        nothing
    end
end

function Tempest.build_implementation(::WaveCUDA, r::CallRequest, runtime, out, inputs)
    launch = if r.call.f === pair_wave
        wave_forward_call(out, only(inputs))
    else
        wave_backward_call(out, inputs...)
    end
    BuiltImplementation(launch, (out, inputs), 0)
end


function demo()
    x = Float32.(range(-0.6, 0.7; length = 513))
    function loss(x)
        y = pair_wave(x)
        sum(abs2, y.s .+ 2.0f0 .* y.c)
    end
    if CUDACore.functional()
        ex = prepare(loss, CuArray(x); implementations = (WaveCUDA(),))
        expected, (dx,) = value_and_gradient(trace(loss, x), x)
        y, g = execute!(ex)
        @assert isapprox(Array(y)[], expected; rtol = 2e-5)
        @assert isapprox(Array(g), dx; rtol = 2e-5, atol = 2e-5)
        capture!(ex)
        bind!(ex, 2x)
        @assert isapprox(Array(replay!(ex)[1])[], loss(2x); rtol = 2e-5)
        explain(ex)
    else
        ex = prepare(loss, x) # Both semantic calls expand into their references.
        execute!(ex)
        explain(ex)
    end
end

end

if abspath(PROGRAM_FILE) == (@__FILE__)
    SemanticKernelExample.demo()
end
