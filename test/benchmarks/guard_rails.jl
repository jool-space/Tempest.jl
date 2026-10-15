# Speed guard rails: the default plan against the plan without cuTile, by time.
#
# Plan-shape tests pass while a plan is slow; these compare device time. For
# each case the default plan (cuTile active) must not take longer than
# RATIO_LIMIT times the CUDA kernels' plan (`Tempest.deactivate(cuTile)`), and
# a forward case larger than twice L2 must move its inputs and outputs at
# BANDWIDTH_FLOOR or more of the copy bandwidth measured in the same run.
#
# The same run checks the cost model's choices (see below).
#
# Times are the minimum, over SAMPLES launches, of a captured CUDA graph of
# enough executions to take a few milliseconds, with the two plans' launches
# alternating so clock changes reach both. Run-to-run variation on GB10 is up
# to 10%, so the limits leave room for it.
#
# Timing needs the GPU to itself, so these tests run only when named:
#   julia --project=test test/runtests.jl benchmarks
# Run them before a commit that changes a provider's schedules or the
# partitioner, and on a new device. The table they print is the record.
using Printf

const RATIO_LIMIT = 1.25
# Absolute slack for launch-bound cases, where one launch is a large ratio.
const SLACK_MICROSECONDS = 0.5
const BANDWIDTH_FLOOR = 0.5
const SAMPLES = 15

function replay_graph(ex; repetitions)
    execute!(ex)
    CUDACore.synchronize()
    graph = CUDACore.capture() do
        for _ = 1:repetitions
            foreach(call -> call(), ex.calls)
        end
    end
    CUDACore.instantiate(graph)
end

function device_microseconds(executables; samples = SAMPLES, milliseconds = 4.0)
    once = map(executables) do ex
        graph = replay_graph(ex; repetitions = 1)
        CUDACore.launch(graph)
        minimum(_ -> CUDACore.@elapsed(CUDACore.launch(graph)), 1:3)
    end
    repetitions = clamp(round(Int, 1.0e-3 * milliseconds / maximum(once)), 2, 200)
    graphs = map(ex -> replay_graph(ex; repetitions), executables)
    foreach(graph -> (CUDACore.launch(graph); CUDACore.launch(graph)), graphs)
    times = [Float64[] for _ in graphs]
    for sample = 1:samples
        order = isodd(sample) ? eachindex(graphs) : reverse(eachindex(graphs))
        for i in order
            seconds = CUDACore.@elapsed CUDACore.launch(graphs[i])
            push!(times[i], 1.0e6 * seconds / repetitions)
        end
    end
    map(minimum, times)
end

function copy_bandwidth(; bytes = 1 << 28)
    source = CuArray{UInt8}(undef, bytes)
    destination = similar(source)
    fill!(source, 0x01)
    copyto!(destination, source)
    2 * bytes / minimum(_ -> CUDACore.@elapsed(copyto!(destination, source)), 1:7)
end

# Every input read and every output written once: a single pass's traffic.
single_pass_bytes(ex) =
    sum(sizeof, ex.inputs; init = 0) +
    sum(a -> a isa AbstractArray ? sizeof(a) : 0, Tempest.tree_leaves(outputs(ex)); init = 0)

host(x) = x isa AbstractArray ? Array(x) : x
results_agree(a, b) = all(map((x, y) -> isapprox(host(x), host(y); rtol = 1.0e-3, atol = 1.0e-4),
                              Tempest.tree_leaves(a), Tempest.tree_leaves(b)))

guard_data(dims...; scale = 1.0f0) =
    CuArray(2.0f0 .* sin.(scale .* reshape(Float32.(1:prod(dims)), dims...)))
guard_weight(n) = CuArray(Float32.(range(0.5, 1.5; length = n)))

softmax_rows(x) = exp.(x .- Tempest.logsumexp(x; dims = 2))
softmax_columns(x) = exp.(x .- Tempest.logsumexp(x; dims = 1))
softmax_naive(x) = (e = exp.(x .- maximum(x; dims = 1)); e ./ sum(e; dims = 1))
rmsnorm(x, w) = x .* w .* inv.(sqrt.(sum(abs2, x; dims = 1) ./ size(x, 1) .+ 1.0f-5))
function layernorm(x, w)
    m = Statistics.mean(x; dims = 1)
    (x .- m) ./ sqrt.(Statistics.var(x; dims = 1, mean = m, corrected = false) .+ 1.0f-5) .* w
end
mlp(x, w1, b1, w2, b2) = w2 * tanh.(w1 * x .+ b1) .+ b2

# (name, function, arguments, training)
function guard_cases()
    d, h, n = 512, 2048, 1024
    w1, w2 = 0.05f0 .* guard_data(h, d), 0.05f0 .* guard_data(d, h; scale = 0.3f0)
    mlp_arguments = (guard_data(d, n; scale = 0.5f0), w1, guard_data(h; scale = 0.1f0), w2, guard_data(d; scale = 0.2f0))
    x = guard_data(1024, 4096)
    t = guard_data(1024, 4096; scale = 0.7f0)
    [
        # Full reductions: partial sums and their completion.
        ("dot product, 2^20", (a, b) -> sum(a .* b), (guard_data(2^20), guard_data(2^20; scale = 0.3f0)), false),
        ("sum(abs2, tanh.(x)), 1024×4096", x -> sum(abs2, tanh.(x)), (x,), false),
        ("sum(abs2, x), 4096×4096", x -> sum(abs2, x), (guard_data(4096, 4096),), false),
        ("full logsumexp, 2048×2048", x -> Tempest.logsumexp(x), (guard_data(2048, 2048),), false),
        ("full maximum and shifted sum, 1024×4096", x -> sum(exp.(x .- maximum(x))), (x,), false),
        ("x .- mean(x), 1024×4096", x -> x .- Statistics.mean(x), (x,), false),
        ("sum of 4096 values", x -> sum(abs2, x), (guard_data(4096),), false),
        # Float16 values leave graph regions; cuTile's scalar kernels take them.
        ("Float16 sum(abs2, x), 2048×4096", x -> sum(abs2, x), (Float16.(guard_data(2048, 4096)),), false),
        ("Float16 column sums, 2048×4096", x -> sum(abs2, x; dims = 1), (Float16.(guard_data(2048, 4096)),), false),
        # Axis reductions: resident, streamed, split and packed.
        ("column sums, 1024×4096", x -> sum(abs2, x; dims = 1), (x,), false),
        ("row sums, 1024×4096", x -> sum(abs2, x; dims = 2), (x,), false),
        ("row maxima, 4096×4096", x -> maximum(x; dims = 2), (guard_data(4096, 4096),), false),
        ("split row sums, 128×16384", x -> sum(x; dims = 2), (guard_data(128, 16384),), false),
        ("split column sums, 65536×16", x -> sum(x; dims = 1), (guard_data(65536, 16),), false),
        ("packed column sums, 32×8192", x -> sum(x; dims = 1), (guard_data(32, 8192),), false),
        ("bias gradient, 2048×1024", (g, h) -> sum(g .* (1 .- h .^ 2); dims = 2),
         (guard_data(2048, 1024), guard_data(2048, 1024; scale = 0.3f0)), false),
        # Softmax, logsumexp and norm spellings.
        ("softmax down columns, 1024×4096", softmax_columns, (x,), false),
        ("naive softmax down columns, 1024×4096", softmax_naive, (x,), false),
        ("softmax down columns, 65536×128", softmax_columns, (guard_data(65536, 128),), false),
        ("softmax along rows, 1024×4096", softmax_rows, (x,), false),
        ("softmax along rows, 4096×4096", softmax_rows, (guard_data(4096, 4096),), false),
        ("softmax along rows, 65536×64", softmax_rows, (guard_data(65536, 64),), false),
        ("softmax, 64×64", softmax_columns, (guard_data(64, 64),), false),
        ("logsumexp down columns, 65536×128", x -> Tempest.logsumexp(x; dims = 1), (guard_data(65536, 128),), false),
        ("RMSNorm, 1024×4096", rmsnorm, (x, guard_weight(1024)), false),
        ("layer norm by mean and var, 1024×4096", layernorm, (x, guard_weight(1024)), false),
        # Rank 3: leading and trailing axes collapse to a matrix domain.
        ("softmax over axis 1, 1024×256×16", softmax_columns, (guard_data(1024, 256, 16),), false),
        ("softmax over axis 3, 1024×256×16", x -> exp.(x .- Tempest.logsumexp(x; dims = 3)), (guard_data(1024, 256, 16),), false),
        ("layer norm, 1024×256×16", layernorm, (guard_data(1024, 256, 16), guard_weight(1024)), false),
        # Training steps.
        ("RMSNorm training, 1024×4096", (x, w) -> sum(abs2, rmsnorm(x, w)), (x, guard_weight(1024)), true),
        ("softmax training, 1024×4096", (x, t) -> sum(softmax_columns(x) .* t), (x, t), true),
        ("MLP block training, 512→2048→512 × 1024",
         (x, w1, b1, w2, b2) -> sum(abs2, mlp(x, w1, b1, w2, b2)), mlp_arguments, true),
        # The attention reference: products and a softmax over 1024 × 1024 × 48.
        ("causal attention reference training", (q, k, v, t) -> sum(Tempest.attention(q, k, v; causal = true) .* t),
         Tuple(0.5f0 .* guard_data(64, 12, 1024, 4; scale) for scale in (0.3f0, 0.7f0, 1.1f0, 1.3f0)), true),
    ]
end

prepare_case(f, arguments, training) =
    training ? prepare(f, arguments...) : prepare_forward(f, arguments...)

graph_regions(ex) = count(step -> step.region isa Tempest.GraphRegion, ex.storage.steps)
# The largest ratio of a kept region's estimate to the default kernels'.
function estimate_ratio(ex)
    ratios = map(filter(note -> occursin("kept by estimate", note), ex.semantic_notes)) do note
        m = match(r"estimate: ([0-9.]+) μs .* against ([0-9.]+) μs", note)
        parse(Float64, m[1]) / parse(Float64, m[2])
    end
    isempty(ratios) ? NaN : maximum(ratios)
end

same_plan(a, b) = [Tempest.region_name(s.region) for s in a.storage.steps] ==
    [Tempest.region_name(s.region) for s in b.storage.steps]

# The cost model (src/compiler/costs.jl) keeps a graph region only when it is
# estimated to beat the default kernels for its members. Its choice, the
# default plan, is checked against both alternatives: every supported region
# kept (estimates off) and none formed. A choice is wrong when the plan it
# declined is faster by more than run-to-run variation.
const CHOICE_LIMIT = 1.1

@testset "Default plans against the CUDA kernels and the cost model's alternatives" begin
    copy = copy_bandwidth()
    l2 = CUDACore.attribute(CUDACore.device(), CUDACore.DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
    @printf("copy bandwidth %.1f GB/s; ratio limit %.2f; bandwidth floor %.0f%% above %d MiB\n",
            copy / 1.0e9, RATIO_LIMIT, 100 * BANDWIDTH_FLOOR, 2l2 >> 20)
    @printf("%-44s %8s %8s %6s %6s %8s %8s %6s %6s\n", "case", "default", "CUDA", "ratio", "copy",
            "regions", "none", "kept", "est")
    wrong = String[]
    for (name, f, arguments, training) in guard_cases()
        default = prepare_case(f, arguments, training)
        baseline = without_cutile(() -> prepare_case(f, arguments, training))
        forced = Tempest.with(() -> prepare_case(f, arguments, training), Tempest.REGION_COSTS => false)
        unfused = training ? prepare(f, arguments...; fuse_regions = false) :
            prepare_forward(f, arguments...; fuse_regions = false)
        @test results_agree(execute!(default), execute!(baseline))
        fast, slow, all_regions, none = device_microseconds((default, baseline, forced, unfused))
        bytes = single_pass_bytes(default)
        share = bytes / (1.0e-6 * fast) / copy
        kept = "$(graph_regions(default))/$(graph_regions(forced))"
        @printf("%-44s %8.1f %8.1f %6.2f %5.0f%% %8.1f %8.1f %6s %6.2f\n", name, fast, slow, fast / slow,
                100 * share, all_regions, none, kept, estimate_ratio(default))
        @test fast <= RATIO_LIMIT * slow + SLACK_MICROSECONDS
        !training && bytes >= 2l2 && @test share >= BANDWIDTH_FLOOR
        # The default plan is one of the alternatives when the model kept every
        # region or none: both timings sample that plan.
        chosen, other = if same_plan(default, forced)
            min(fast, all_regions), none
        elseif same_plan(default, unfused)
            min(fast, none), all_regions
        else
            fast, min(all_regions, none)
        end
        chosen <= CHOICE_LIMIT * other + SLACK_MICROSECONDS || push!(wrong, name)
    end
    isempty(wrong) || println("the cost model chose the slower plan: ", join(wrong, "; "))
    @test length(wrong) <= 1
end
