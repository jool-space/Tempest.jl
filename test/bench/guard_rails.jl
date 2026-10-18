# Device time of the default plan for the guard-rail cases, measured as the
# earlier Tempest measured them: the minimum over SAMPLES launches of a
# captured CUDA graph of a few milliseconds of executions.
#   julia --project=test test/bench/guard_rails.jl [output.tsv]
using Tempest, Tempest.Compiler, CUDACore, cuBLAS, cuTile, Printf
import Tempest as T
import Tempest.Compiler as C

const SAMPLES = 15

function device_microseconds(ex; samples = SAMPLES, milliseconds = 4.0)
    execute!(ex)
    replay!(capture!(ex))
    once = minimum(_ -> CUDACore.@elapsed(replay!(ex)), 1:3)
    repetitions = clamp(round(Int, 1.0e-3 * milliseconds / once), 2, 200)
    replay!(capture!(ex; repetitions))
    minimum(_ -> 1.0e6 * CUDACore.@elapsed(replay!(ex)) / repetitions, 1:samples)
end

guard_data(dims...; scale = 1.0f0) = 2.0f0 .* sin.(scale .* reshape(Float32.(1:prod(dims)), dims...))
guard_weight(n) = Float32.(range(0.5, 1.5; length = n))

softmax_rows(x) = exp.(x .- T.logsumexp(x; dims = 2))
softmax_columns(x) = exp.(x .- T.logsumexp(x; dims = 1))
softmax_naive(x) = exp.(x) ./ sum(exp.(x); dims = 1)
rmsnorm(x, w) = x ./ sqrt.(sum(abs2, x; dims = 1) ./ size(x, 1) .+ 1.0f-5) .* w
function layernorm(x, w)
    n = size(x, 1)
    m = sum(x; dims = 1) ./ n
    (x .- m) ./ sqrt.(sum(abs2, x .- m; dims = 1) ./ n .+ 1.0f-5) .* w
end

x = guard_data(1024, 4096)
const CASES = [
    ("linear + bias + GELU, 2304×768 · 768×4096", (w, x, b) -> gelu.(w * x .+ b),
     (guard_data(2304, 768; scale = 0.01f0), guard_data(768, 4096), guard_weight(2304))),
    ("linear + bias + GELU, 3072×768 · 768×4096", (w, x, b) -> gelu.(w * x .+ b),
     (guard_data(3072, 768; scale = 0.01f0), guard_data(768, 4096), guard_weight(3072))),
    ("linear + bias + GELU, 768×3072 · 3072×4096", (w, x, b) -> gelu.(w * x .+ b),
     (guard_data(768, 3072; scale = 0.01f0), guard_data(3072, 4096), guard_weight(768))),
    ("dot product, 2^20", (a, b) -> sum(a .* b), (guard_data(2^20), guard_data(2^20; scale = 0.3f0))),
    ("sum(abs2, tanh.(x)), 1024×4096", x -> sum(abs2, tanh.(x)), (x,)),
    ("sum(abs2, x), 4096×4096", x -> sum(abs2, x), (guard_data(4096, 4096),)),
    ("full logsumexp, 2048×2048", x -> T.logsumexp(x), (guard_data(2048, 2048),)),
    ("full maximum and shifted sum, 1024×4096", x -> sum(exp.(x .- maximum(x))), (x,)),
    ("x .- mean(x), 1024×4096", x -> x .- sum(x) ./ length(x), (x,)),
    ("sum of 4096 values", x -> sum(abs2, x), (guard_data(4096),)),
    ("Float16 sum(abs2, x), 2048×4096", x -> sum(abs2, x), (Float16.(guard_data(2048, 4096)),)),
    ("Float16 column sums, 2048×4096", x -> sum(abs2, x; dims = 1), (Float16.(guard_data(2048, 4096)),)),
    ("column sums, 1024×4096", x -> sum(abs2, x; dims = 1), (x,)),
    ("row sums, 1024×4096", x -> sum(abs2, x; dims = 2), (x,)),
    ("row maxima, 4096×4096", x -> maximum(x; dims = 2), (guard_data(4096, 4096),)),
    ("split row sums, 128×16384", x -> sum(x; dims = 2), (guard_data(128, 16384),)),
    ("split column sums, 65536×16", x -> sum(x; dims = 1), (guard_data(65536, 16),)),
    ("packed column sums, 32×8192", x -> sum(x; dims = 1), (guard_data(32, 8192),)),
    ("bias gradient, 2048×1024", (g, h) -> sum(g .* (1 .- h .^ 2); dims = 2),
     (guard_data(2048, 1024), guard_data(2048, 1024; scale = 0.3f0))),
    ("softmax down columns, 1024×4096", softmax_columns, (x,)),
    ("naive softmax down columns, 1024×4096", softmax_naive, (x,)),
    ("softmax down columns, 65536×128", softmax_columns, (guard_data(65536, 128),)),
    ("softmax along rows, 1024×4096", softmax_rows, (x,)),
    ("softmax along rows, 4096×4096", softmax_rows, (guard_data(4096, 4096),)),
    ("softmax along rows, 65536×64", softmax_rows, (guard_data(65536, 64),)),
    ("softmax, 64×64", softmax_columns, (guard_data(64, 64),)),
    ("logsumexp down columns, 65536×128", x -> T.logsumexp(x; dims = 1), (guard_data(65536, 128),)),
    ("RMSNorm, 1024×4096", rmsnorm, (x, guard_weight(1024))),
    ("layer norm by mean and var, 1024×4096", layernorm, (x, guard_weight(1024))),
    ("softmax over axis 1, 1024×256×16", softmax_columns, (guard_data(1024, 256, 16),)),
    ("softmax over axis 3, 1024×256×16", x -> exp.(x .- T.logsumexp(x; dims = 3)), (guard_data(1024, 256, 16),)),
    ("layer norm, 1024×256×16", layernorm, (guard_data(1024, 256, 16), guard_weight(1024))),
]

mlp(x, w1, b1, w2, b2) = w2 * tanh.(w1 * x .+ b1) .+ b2
gelu(x) = 0.5f0 * x * (1.0f0 + tanh(0.7978846f0 * (x + 0.044715f0 * x * x * x)))
# As in the earlier guard rails.
mlp_arguments = (guard_data(512, 1024; scale = 0.5f0), 0.05f0 .* guard_data(2048, 512), guard_data(2048; scale = 0.1f0),
                 0.05f0 .* guard_data(512, 2048; scale = 0.3f0), guard_data(512; scale = 0.2f0))

# Products with few output tiles and a long sum: split-K's case.
push!(CASES, ("product with a small output, 256×8192 · 8192×256", (a, b) -> a * b,
    (guard_data(256, 8192; scale = 0.01f0), guard_data(8192, 256))))

const TRAINING = [
    ("MLP block training, 512→2048→512 × 1024", (x, w1, b1, w2, b2) -> sum(abs2, mlp(x, w1, b1, w2, b2)), mlp_arguments),
    ("RMSNorm training, 1024×4096", (x, w) -> sum(abs2, rmsnorm(x, w)), (x, guard_weight(1024))),
    ("softmax training, 1024×4096", (x, t) -> sum(softmax_columns(x) .* t), (x, guard_data(1024, 4096; scale = 0.7f0))),
]

host(x) = x isa AbstractArray ? (a = Array(x); ndims(a) == 0 ? a[] : a) : x

function main(path = nothing)
    rows = String[]
    for (name, f, args, training) in [map(c -> (c..., false), CASES); map(c -> (c..., true), TRAINING)]
        p = training ? differentiate(trace(f, args...)) : trace(f, args...)
        ex = prepare(p, map(a -> CuArray(a), args)...; target = TARGET)
        got = map(host, C.leaves(outputs(execute!(ex))))
        ref = C.leaves(interpret(p, args...))
        ok = all(map((g, r) -> isapprox(g, r; rtol = TARGET.precision === :tf32 ? 2.0e-2 : 2.0e-3, atol = 1.0e-3), got, ref))
        if TARGET.precision === :tf32x3
            println("    max relative error ", maximum(map((g, r) -> maximum(abs.(g .- r)) / maximum(abs.(r)), got, ref)))
        end
        t = device_microseconds(ex)
        kinds = join((d isa C.FloorKernels ? "floor" : hasproperty(d, :schedule) && hasproperty(d.schedule, :sketch) ? string(d.schedule.sketch) : string(d) for d in ex.kernels), "+")
        line = @sprintf("%-44s %9.1f  %-5s %s", name, t, ok ? "ok" : "WRONG", kinds)
        println(line)
        push!(rows, string(name, '\t', t, '\t', ok, '\t', kinds))
    end
    path === nothing || write(path, join(rows, '\n') * '\n')
end

# `--tf32` or `--tf32x3` anywhere in the arguments sets the products' precision.
const TARGET = T.CUDATarget(; precision = "--tf32" in ARGS ? :tf32 : "--tf32x3" in ARGS ? :tf32x3 : :full)
filter!(a -> !(a in ("--tf32", "--tf32x3")), ARGS)

if abspath(PROGRAM_FILE) != @__FILE__
    # Included for its cases.
elseif !isempty(ARGS) && ARGS[1] == "--cases"
    # Several cases in one process, each timed `repeats` times: for comparing
    # builds case by case.
    repeats = parse(Int, get(ARGS, 3, "5"))
    for name in split(ARGS[2], '|')
        (_, f, args) = only(filter(c -> c[1] == name, [CASES; TRAINING]))
        p = name in map(first, TRAINING) ? differentiate(trace(f, args...)) : trace(f, args...)
        ex = prepare(p, map(a -> CuArray(a), args)...; target = TARGET)
        times = [device_microseconds(ex) for _ in 1:repeats]
        println(rpad(name, 44), " min ", round(minimum(times); digits = 1), "  all ", join(round.(times; digits = 1), ", "))
    end
elseif isempty(ARGS) || ARGS[1] != "--case"
    main(get(ARGS, 1, nothing))
else
    name = ARGS[2]
    (_, f, args) = only(filter(c -> c[1] == name, [CASES; TRAINING]))
    p = name in map(first, TRAINING) ? differentiate(trace(f, args...)) : trace(f, args...)
    ex = prepare(p, map(a -> CuArray(a), args)...; target = TARGET)
    println(name, " (", TARGET.precision, "): ", join((round(device_microseconds(ex); digits = 1) for _ in 1:parse(Int, get(ARGS, 3, "5"))), ", "), " μs")
end
