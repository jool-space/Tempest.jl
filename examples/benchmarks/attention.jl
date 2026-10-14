# Causal attention on GPT-like shapes: cuTile's fused TF32 kernels against
# cuDNN's SDPA engines, forward and forward + backward.
#
# cuDNN takes half-precision operands only. `cuDNN bf16 cast` and `cuDNN fp16
# cast` run on Float32 data with CuDNNKernels(float32_cast = ...), so they
# include the casts as graph operations; `cuDNN bf16` runs on BFloat16 data, the
# engines alone. Every executable is checked against cuTile's result before
# timing. Device times are per step from one CUDA graph holding `repetitions`
# back-to-back executions of one input set, so host submission is excluded and
# the L2 may hold the inputs. Candidates are measured in alternating order;
# each figure is the median of `samples` samples. Compilation is excluded.
# TFLOP/s counts 4·b·h·t²·d for a forward and 3.5 times that for forward +
# backward, halved for the causal mask.
# Run: julia --project=examples examples/benchmarks/attention.jl
using Tempest, CUDACore, Statistics
import cuBLAS, cuTile, cuDNN
using Tempest: attention

function operands(D, heads, positions, batch, T)
    shaped(s) = T.(reshape(Float32.(sin.((1:(D * heads * positions * batch)) .* s)), D, heads, positions, batch))
    map(CuArray, (shaped(0.3), shaped(0.7), shaped(1.1)))
end

forward(q, k, v) = attention(q, k, v; causal = true)
loss(q, k, v) = sum(abs2, attention(q, k, v; causal = true))

const CANDIDATES = (
    (name = "cuTile tf32", provider = :cutile, T = Float32, flags = (; target = CUDATarget(matmul_precision = :tf32))),
    (name = "cuDNN bf16 cast", provider = :cuDNN, T = Float32, flags = (; implementations = (CuDNNKernels(float32_cast = Core.BFloat16),))),
    (name = "cuDNN fp16 cast", provider = :cuDNN, T = Float32, flags = (; implementations = (CuDNNKernels(float32_cast = Float16),))),
    (name = "cuDNN bf16", provider = :cuDNN, T = Core.BFloat16, flags = (; implementations = (CuDNNKernels(),))),
)

function prepared(candidate, training, shape)
    args = operands(shape..., candidate.T)
    ex = training ? prepare(loss, args...; candidate.flags...) : prepare_forward(forward, args...; candidate.flags...)
    candidate.provider in ex.providers || error("$(candidate.name) did not select $(candidate.provider)")
    ex
end

result(ex, training) = Float32.(Array(training ? execute!(ex)[2] : execute!(ex)))

function check(executables, training)
    reference = result(first(executables), training)
    for ex in executables[2:end]
        got = result(ex, training)
        error = sqrt(sum(abs2, got .- reference) / sum(abs2, reference))
        error < 5e-2 || @warn "relative difference $error from cuTile"
    end
end

function batch(ex, repetitions)
    CUDACore.synchronize()
    graph = CUDACore.capture() do
        for _ = 1:repetitions
            execute!(ex)
        end
    end
    CUDACore.instantiate(graph)
end

device_sample(executable, repetitions) =
    1e6 * CUDACore.@elapsed(CUDACore.launch(executable)) / repetitions

function measure(executables; samples = 7, repetitions = 20)
    graphs = map(ex -> batch(ex, repetitions), executables)
    foreach(CUDACore.launch, graphs)
    CUDACore.synchronize()
    times = [Float64[] for _ in graphs]
    for sample = 1:samples
        order = isodd(sample) ? eachindex(graphs) : reverse(eachindex(graphs))
        for i in order
            push!(times[i], device_sample(graphs[i], repetitions))
        end
    end
    map(median, times)
end

flops(D, heads, positions, batch, training) =
    (training ? 3.5 : 1.0) * 4 * batch * heads * positions^2 * D / 2

function run(shapes)
    CUDACore.functional() || error("requires CUDA")
    println(CUDACore.name(CUDACore.device()), "; cuDNN ", cuDNN.version())
    println("pass | d×heads×seq×batch | ", join((c.name for c in CANDIDATES), " | "), " (μs, TFLOP/s)")
    for training in (false, true), shape in shapes
        executables = map(c -> prepared(c, training, shape), CANDIDATES)
        check(executables, training)
        times = measure(executables)
        work = flops(shape..., training)
        cells = map(t -> string(round(t; digits = 1), " (", round(work / t / 1e6; digits = 1), ")"), times)
        println(training ? "fwd+bwd" : "fwd", " | ", join(shape, "×"), " | ", join(cells, " | "))
        executables = nothing
        GC.gc()
    end
end

const SHAPES = get(ENV, "SHAPES", "64x12x1024x8,64x12x2048x4,128x12x1024x8,128x12x2048x4")
run(Tuple(Tuple(parse.(Int, split(s, "x"))) for s in split(SHAPES, ",")))
