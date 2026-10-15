# MLP blocks with half-precision operands: the default plan against regions
# that CuDNNKernels proposes, a matrix product fused with its epilogue.
#
# The block is y = W2 * tanh.(W1 * x .+ b1) .+ b2 with a hidden width four
# times the model width, as in a transformer's feed-forward layer, and the
# training loss is sum(abs2, y). `default` is Tempest's plan without cuDNN:
# cuBLAS products (Float32 accumulation) and generated epilogue kernels. `cuDNN`
# adds CuDNNKernels(), whose proposals replace a product and its epilogue with
# one cuDNN graph: forward bias and activation, and in training the
# activation's derivative and the bias gradient after the backward product.
#
# Every executable is checked against the default result first. Device times
# are per step from one CUDA graph holding `repetitions` back-to-back
# executions of one input set, so host submission is excluded and the L2 may
# hold the inputs. Candidates are measured in alternating order; each figure is
# the median of `samples` samples. Preparation is excluded. TFLOP/s counts the
# products only: 2·n·(d·4d + 4d·d) for a forward, three times that for
# forward + backward.
#
# With LAYERS=m×k×n,... it instead times single layers tanh.(w * x .+ b) in
# Float16: the default plan, cuDNN with its limits switched off (one launch,
# never declined), and cuDNN with them (column chunks within L2, the cost
# estimate), the measurements behind those limits.
# Run: julia --project=examples examples/benchmarks/matmul_epilogue.jl
using Tempest, CUDACore, Statistics
import cuBLAS, cuTile, cuDNN

block(w1, b1, w2, b2, x) = w2 * tanh.(w1 * x .+ b1) .+ b2
loss(w1, b1, w2, b2, x) = sum(abs2, block(w1, b1, w2, b2, x))

function operands(d, n, T)
    shaped(s, dims...) = T.(reshape(Float32.(sin.((1:prod(dims)) .* s)), dims...) ./ sqrt(Float32(dims[end])))
    map(CuArray, (shaped(0.3, 4d, d), shaped(0.5, 4d), shaped(0.7, d, 4d), shaped(1.1, d), shaped(1.3, d, n)))
end

const CANDIDATES = (
    (name = "default", flags = (;)),
    (name = "cuDNN", flags = (; implementations = (CuDNNKernels(),))),
)

function prepared(candidate, training, d, n, T)
    args = operands(d, n, T)
    ex = training ? prepare(loss, args...; candidate.flags...) : prepare_forward(block, args...; candidate.flags...)
    fused = :cuDNN in ex.providers
    fused == (candidate.name == "cuDNN") || error("$(candidate.name): unexpected providers $(unique(ex.providers))")
    ex
end

# The forward output, or in training the first weight's gradient.
result(ex, training) = Float32.(Array(training ? execute!(ex)[2] : execute!(ex)))

function check(executables, training)
    reference = result(first(executables), training)
    for ex in executables[2:end]
        got = result(ex, training)
        error = sqrt(sum(abs2, got .- reference) / sum(abs2, reference))
        error < 2e-2 || @warn "relative difference $error from the default plan"
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

function measure(executables; samples = 9, repetitions = 20)
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

flops(d, n, training) = (training ? 3 : 1) * 2 * n * 8 * d^2

function run(shapes, types)
    CUDACore.functional() || error("requires CUDA")
    println(CUDACore.name(CUDACore.device()), "; cuDNN ", cuDNN.version())
    println("pass | type | d×n | ", join((c.name for c in CANDIDATES), " | "), " (μs, TFLOP/s) | speedup")
    for training in (false, true), T in types, (d, n) in shapes
        executables = map(c -> prepared(c, training, d, n, T), CANDIDATES)
        check(executables, training)
        times = measure(executables)
        work = flops(d, n, training)
        cells = map(t -> string(round(t; digits = 1), " (", round(work / t / 1e6; digits = 1), ")"), times)
        speedup = round(times[1] / times[2]; digits = 2)
        println(training ? "fwd+bwd" : "fwd", " | ", nameof(T), " | ", d, "×", n, " | ", join(cells, " | "), " | ", speedup)
        executables = nothing
        GC.gc()
    end
end

layer(w, b, x) = tanh.(w * x .+ b)

# cuDNN with its limits off (one launch, any inner dimension), and with them.
const LAYER_CANDIDATES = (
    CANDIDATES[1],
    (name = "cuDNN unlimited", flags = (; implementations = (CuDNNKernels(l2_fraction = Inf, fused_efficiency = Inf),))),
    CANDIDATES[2],
)

function run_layers(shapes)
    println(CUDACore.name(CUDACore.device()), "; cuDNN ", cuDNN.version())
    println("m×k×n | right operand | ", join((c.name for c in LAYER_CANDIDATES), " | "), " (μs) | cuDNN plan")
    for (m, k, n) in shapes
        shaped(s, dims...) = CuArray(Float16.(reshape(sin.((1:prod(dims)) .* s), dims...) ./ sqrt(dims[end])))
        args = (shaped(0.3, m, k), shaped(0.5, m), shaped(0.7, k, n))
        executables = map(c -> prepare_forward(layer, args...; c.flags...), LAYER_CANDIDATES)
        foreach(execute!, executables)
        times = measure(executables)
        summary = something(Tempest.implementation_summary(last(executables).calls[1]), "default plan")
        plan = occursin("chunks", summary) ? match(r"in (\d+) column chunks", summary)[1] * " chunks" :
               :cuDNN in last(executables).providers ? "one launch" : "declined"
        println(m, "×", k, "×", n, " | ", (2k * n) >> 20, " MiB | ", join(round.(times; digits = 1), " | "), " | ", plan)
    end
end

parse_shapes(text) = Tuple(Tuple(parse.(Int, split(s, "x"))) for s in split(text, ","))

if haskey(ENV, "LAYERS")
    run_layers(parse_shapes(ENV["LAYERS"]))
else
    run(
        parse_shapes(get(ENV, "SHAPES", "256x4096,512x4096,768x8192,1024x8192")),
        Tuple(t == "BFloat16" ? Core.BFloat16 : getfield(Base, Symbol(t)) for t in split(get(ENV, "TYPES", "Float16,BFloat16"), ",")),
    )
end
