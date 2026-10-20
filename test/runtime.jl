# Prepared programs compute what programs compute, through generated kernels,
# on the CPU (and on CUDA when it is loaded and functional).
include("fixtures/programs.jl")

# The proposals the cover took.
taken(ex) = [c.proposal for c in ex.considered if c.accepted]

function check_prepared(device, expected_target; rtol = 1.0e-10, precision = identity)
    @testset "$name" for (name, f, args) in PROGRAMS
        args = map(precision, args)
        for p in (trace(f, args...), differentiate(trace(f, args...)))
            reference = filter(!isnothing, C.leaves(interpret(p, args...)))
            ex = prepare(p, map(device, args)...)
            @test ex.target == expected_target
            got = filter(!isnothing, C.leaves(outputs(execute!(ex))))
            @test agree(got, reference; rtol)
        end
    end
end

@testset "prepared on the CPU" begin
    check_prepared(identity, T.CPUTarget())
end

# The results are a place: the same arrays every time, which each run fills
# again. `execute!` launches kernel by kernel, captured or not; `replay!` runs
# the capture.
@testset "running a prepared program" begin
    x = data(4)
    ex = prepare(trace(x -> (sum(x), 2 .* x), x), x)
    out = outputs(ex)
    @test execute!(ex) === ex && outputs(ex) === out
    @test out[1][] ≈ sum(x) && out[2] ≈ 2 .* x
    @test occursin("capture! records the program first", message(() -> replay!(ex)))
    bind!(ex, [1.0, 2.0, 3.0, 4.0])
    @test replay!(capture!(ex)) === ex && out[1][] == 10.0
    bind!(ex, [2.0, 2.0, 2.0, 2.0])
    execute!(ex)
    @test out[1][] == 8.0 && out[2] == [4.0, 4.0, 4.0, 4.0]
end

# Values other than inputs and results share slots where their lifetimes do
# not overlap, so training steps that keep values for their backward pass
# store less.
function slots_disjoint(s::C.Storage)
    slotted = [v for v in eachindex(s.slots) if s.slots[v] > 0]
    all(isempty(intersect(s.lifetimes[v], s.lifetimes[w])) for v in slotted for w in slotted if v < w && s.slots[v] == s.slots[w])
end

@testset "slots" begin
    for (name, f, args) in PROGRAMS
        ex = prepare(differentiate(trace(f, args...)), args...)
        @test slots_disjoint(ex.storage)
        bytes, apart = C.footprint(ex)
        @test name in ("MLP", "attention, grouped heads") ? bytes < apart : bytes <= apart
    end
end

# A training step as one program: an MLP's gradient, then Adam's update.
# Prepared with its parameters and Adam's state replacing themselves, it runs
# step after step with only the batch bound, writing every parameter and
# moment into its own storage, and matches the interpreter step by step.
function adam(params, state, grads; rate = 1.0f-2, β1 = 0.9f0, β2 = 0.999f0, ϵ = 1.0f-8)
    t = state.t + 1
    m = map((m, g) -> β1 .* m .+ (1 - β1) .* g, state.m, grads)
    v = map((v, g) -> β2 .* v .+ (1 - β2) .* g .^ 2, state.v, grads)
    new = map((p, m, v) -> p .- rate .* (m ./ (1 - β1^t)) ./ (sqrt.(v ./ (1 - β2^t)) .+ ϵ), params, m, v)
    new, (; m, v, t)
end
mlp_loss(p, batch) = sum(abs2, p.w2 * tanh.(p.w1 * batch.x .+ p.b1) .+ p.b2 .- batch.y) / size(batch.y, 2)
function training_step(params, state, batch)
    loss, grads, _ = T.gradient(mlp_loss, params, batch)
    new_params, new_state = adam(params, state, grads)
    (loss, new_params, new_state)
end

params = map(single, (w1 = data(16, 8; scale = 0.3) ./ 4, b1 = data(16; scale = 0.2) ./ 4, w2 = data(4, 16; scale = 0.5) ./ 4,
    b2 = data(4; scale = 0.7) ./ 4))
adam_state = (m = map(zero, params), v = map(zero, params), t = 0.0f0)
batches = [map(single, (x = data(8, 32; scale = 0.1k), y = data(4, 32; scale = 0.05k))) for k in 1:4]
step_program = trace(training_step, params, adam_state, batches[1])

# Captured, each step replays one graph, the batch bound between replays.
function check_steps(device; rtol, by_reference = (), captured = false)
    p = step_program
    on(x) = C.rebuild(a -> a isa AbstractArray ? device(a) : a, x)
    inputs = (on(params), on(adam_state), on(batches[1]))
    ex = prepare(p, inputs...; replaces = (2 => 1, 3 => 2), by_reference)
    captured && capture!(ex)
    @test length(ex.storage.donated) == 13 && isempty(ex.storage.staged)
    # The results that replace arguments are the arguments' storage.
    @test all(C.leaves(outputs(ex)[2:3]) .=== ex.results[2:end])
    @test all(pointer.(ex.results[2:end]) .== pointer.(ex.arguments[1:13]))
    expected = (params, adam_state)
    for batch in batches
        bind!(ex, 3 => on(batch))
        got = C.leaves(outputs(captured ? replay!(ex) : execute!(ex)))
        reference = interpret(p, expected..., batch)
        @test agree(got, C.leaves(reference); rtol)
        expected = reference[2:3]
    end
    # The same step runs eagerly.
    @test all(map((a, b) -> isapprox(a, b; rtol = 1.0e-5), C.leaves(training_step(params, adam_state, batches[1])),
        C.leaves(interpret(p, params, adam_state, batches[1]))))
    # Arrays bound by reference are the executable's storage, and the results
    # that replace them write into them.
    for j in by_reference
        @test agree(filter(x -> x isa AbstractArray, C.leaves(inputs[j])), filter(x -> x isa AbstractArray, C.leaves(expected[j])); rtol)
    end
    ex
end

# A kernel that writes a result in place of an argument runs after the
# others that read the argument, here a kernel over another domain the plan
# put after it; a result written where the kernel reads the argument at
# another address is staged, then copied.
function check_replacing(device)
    a, x = data(6), data(6, 5; scale = 0.3)
    p = trace((a, x) -> (a .+ vec(sum(x; dims = 2)), a[2:3] .* 3), a, x)
    ex = prepare(p, device(a), device(x); replaces = (1 => 1,))
    @test length(ex.storage.donated) == 1
    expected = a
    for _ in 1:2
        reference = interpret(p, expected, x)
        @test agree(outputs(execute!(ex)), reference; rtol = 1.0e-6)
        expected = reference[1]
    end
    w = data(5, 5; scale = 0.4)
    p = trace(w -> (permutedims(w) .* 2,), w)
    ex = prepare(p, device(w); replaces = (1 => 1,))
    @test isempty(ex.storage.donated) && length(ex.storage.staged) == 1
    @test occursin("staged, then copied into input 1", sprint(explain, ex))
    execute!(ex)
    @test Array(only(outputs(execute!(ex)))) ≈ 4 .* w
    @test_throws ArgumentError prepare(trace(w -> (w .* 2, w), w), device(w); replaces = (1 => 1,))
    @test_throws ArgumentError prepare(trace(w -> (sum(w),), w), device(w); replaces = (1 => 1,))
end

@testset "a training step as one program, on the CPU" begin
    check_steps(copy; rtol = 1.0e-5, by_reference = (1, 2))
    check_steps(identity; rtol = 1.0e-5, captured = true)
    check_replacing(identity)
end

# An implementation that proposes each product alone, grown from its fold, in
# place of the generator's product with the work after it; it claims a lone
# product as free, so the cover takes it. A candidate over several domains
# (the product and a sum over its result) does not form a group.
struct HostProducts <: C.Implementation end
lone_product(g) = g.kind === :reduction && all(in(g.reductions), C.members(g)) && C.product_axes(g) !== nothing
C.candidates(::HostProducts, ::T.CPUTarget, p) = [C.grow(p.graph, r, set -> length(set) <= 1)
    for g in map(g -> C.group_view(p, g), p.groups) if g.kind === :reduction && C.product_axes(g) !== nothing for r in g.reductions]
C.claims(::HostProducts, ::T.CPUTarget, g) = lone_product(g) ? [:host] : []
C.estimate(::HostProducts, claim, target, g) = [C.Work(0.0)]
C.build_claim(::HostProducts, claim, target, g, buffers) = g.reference(target, buffers)

@testset "candidates from implementations" begin
    a, b = data(6, 5; scale = 0.3), data(5, 4)
    p = trace((a, b) -> tanh.(a * b), a, b)
    @test length(C.plan(C.lower(p)).groups) == 1 # the generator fuses the tanh
    C.register!(@__MODULE__, HostProducts())
    try
        ex = prepare(p, a, b)
        @test length(only(taken(ex))) == 1
        @test length(ex.plan.groups) == 2
        @test isapprox(only(C.leaves(outputs(execute!(ex)))), interpret(p, a, b); rtol = 1.0e-12)
        @test occursin("from HostProducts replaces", sprint(explain, ex))
        # The product, the tanh and a sum over the result span two domains.
        graph = C.lower(trace((a, b) -> sum(tanh.(a * b); dims = 2), a, b))
        planned = C.plan(graph)
        r = only(g.reductions[1] for g in planned.groups if C.product_axes(C.group_view(planned, g)) !== nothing)
        @test_throws C.NotAGroup C.plan(graph; fixed = [C.grow(graph, r, set -> length(set) <= 3)])
    finally
        C.register!(@__MODULE__)
    end
end

# The floor evaluates recomputed folds too: an implementation that asks for
# them and leaves the kernels to the floor.
struct RecomputeOnTheFloor <: C.Implementation end
C.recomputes_folds(::RecomputeOnTheFloor, target) = true
C.claims(::RecomputeOnTheFloor, target, g) = isempty(C.recomputed_in(g)) ? () : (:floor,)
C.build_claim(::RecomputeOnTheFloor, claim, target, g, buffers) = error("left to the floor")

function with_recomputing_floor(f)
    C.register!(@__MODULE__, RecomputeOnTheFloor())
    try
        f()
    finally
        C.register!(@__MODULE__)
    end
end

attention_cases = [
    ("causal attention, grouped heads", (q, k, v) -> Ops.attention(q, k, v; causal = true),
     (data(4, 4, 9, 2; scale = 0.3), data(4, 2, 9, 2; scale = 0.2), data(4, 2, 9, 2; scale = 0.1))),
    ("attention", (q, k, v) -> Ops.attention(q, k, v), (data(8, 2, 7, 3; scale = 0.3), data(8, 2, 7, 3; scale = 0.2), data(8, 2, 7, 3; scale = 0.1))),
]

@testset "recomputed folds on the CPU floor" begin
    with_recomputing_floor() do
        @testset "$name" for (name, f, args) in attention_cases
            p = trace(f, args...)
            ex = prepare(p, args...)
            @test !isempty(C.recomputed_in(only(ex.plan.groups)))
            @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, args...)])
        end
    end
end

# A proposal the cover tries leaves other folds to recompute, and a fold
# recomputed where no implementation claims its reader stays stored, as in
# the generator's plan. This implementation claims groups over three axes, or
# that recompute nothing, finds recomputing slow, and proposes storing each
# fold it recomputes: storing `q` lets `s` be recomputed in the logsumexp,
# over two axes, which it does not claim.
struct RecomputesOverThree <: C.Implementation end
C.recomputes_folds(::RecomputesOverThree, target) = true
C.claims(::RecomputesOverThree, target, g) = isempty(C.recomputed_in(g)) || length(g.domain) >= 3 ? (:claim,) : ()
C.estimate(::RecomputesOverThree, claim, target, g) = [C.Work(0.0; flops = isempty(C.recomputed_in(g)) ? 1.0e-6 : 1.0)]
C.candidates(::RecomputesOverThree, target::T.CPUTarget, p) = [[v] for g in p.groups for v in C.recomputed_in(g)]

@testset "a fold recomputed where nothing claims its reader stays stored" begin
    C.register!(@__MODULE__, RecomputesOverThree())
    try
        x, w = data(8, 6), data(8, 8)
        function f(x, w)
            q = T.contract(w, x, (1, 2), (1, 3), (2, 3))
            s = T.contract(q, q, (1, 2), (1, 3), (2, 3))
            T.logsumexp(s; dims = 1)
        end
        d = C.Decider(T.CPUTarget(), C.lower(trace(f, x, w)); measure = false)
        p, considered = C.cover(d)
        @test any(c -> c.accepted, considered)
        @test all(g -> !isempty(C.claims(RecomputesOverThree(), d.target, C.group_view(p, g))), p.groups)
    finally
        C.register!(@__MODULE__)
    end
end

if CUDA_FUNCTIONAL
    @testset "recomputed folds on the CUDA floor" begin
        with_recomputing_floor() do
            @testset "$name" for (name, f, args) in attention_cases
                p = trace(f, args...)
                ex = prepare(p, map(CuArray, args)...; target = T.CUDATarget(@__MODULE__))
                @test all(d -> d isa C.FloorKernels, ex.kernels)
                @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, args...)]; rtol = 1.0e-8)
            end
        end
    end
    @testset "prepared on CUDA" begin
        check_prepared(x -> x isa AbstractArray ? CuArray(x) : x, T.CUDATarget())
    end
    # Claims are timed on buffers of their own, which a kernel reads as data:
    # an integer array is an index there, and zero is outside every axis. The
    # buffers hold ones, so that timing every option of a gather (the floor
    # too, with bounds checked as under `Pkg.test`) reads in bounds.
    @testset "timed on trial buffers, a gather reads in bounds" begin
        @testset "$name" for (name, f, args) in filter(((_, _, args),) -> any(a -> a isa AbstractArray{<:Integer}, args), PROGRAMS)
            for p in (trace(f, args...), differentiate(trace(f, args...)))
                graph = C.lower(p)
                planned = C.plan(graph)
                d = C.Decider(T.CUDATarget(), graph; measure = false)
                foreach(g -> C.buffers!(d, planned, g), planned.groups)
                indices = [b for (v, b) in zip(graph.values, d.buffers) if b !== nothing && v.eltype <: Integer && !(v.definition isa C.ConstantArray)]
                @test !isempty(indices) && all(b -> all(>=(1), Array(b)), indices)
                reference = filter(!isnothing, C.leaves(interpret(p, args...)))
                ex = prepare(p, map(x -> x isa AbstractArray ? CuArray(x) : x, args)...; target = T.CUDATarget(), measure = :all)
                @test agree(filter(!isnothing, C.leaves(outputs(execute!(ex)))), reference)
            end
        end
    end
    @testset "a training step as one program, on CUDA" begin
        check_steps(CuArray; rtol = 1.0e-3, by_reference = (1, 2))
        check_steps(CuArray; rtol = 1.0e-3)
        check_steps(CuArray; rtol = 1.0e-3, by_reference = (1, 2), captured = true)
        ex = check_steps(CuArray; rtol = 1.0e-3, captured = true)
        @test ex.captured !== nothing
        check_replacing(CuArray)
    end
    using cuTile
    @testset "prepared on CUDA with cuTile, Float32" begin
        check_prepared(x -> x isa AbstractArray ? CuArray(x) : x, T.CUDATarget(); rtol = 2.0e-4, precision = single)
    end
    @testset "products on cuTile" begin
        w, x, b = Float32.(data(96, 80; scale = 0.1)), Float32.(data(80, 112)), Float32.(data(96))
        for target in (T.CUDATarget(cuTile), T.CUDATarget(cuTile; precision = :tf32))
            p = trace((w, x, b) -> gelu.(w * x .+ b), w, x, b)
            ex = prepare(p, CuArray(w), CuArray(x), CuArray(b); target)
            @test any(d -> !(d isa C.FloorKernels) && d.schedule.sketch === :product, ex.kernels)
            @test isapprox(Array(only(C.leaves(outputs(execute!(ex))))), interpret(p, w, x, b); rtol = target.precision === :tf32 ? 1.0e-2 : 1.0e-4)
        end
        # A conversion to BFloat16 is part of the epilogue: one kernel, and
        # only the BFloat16 result is stored.
        p = trace((w, x, b) -> BFloat16.(gelu.(w * x .+ b)), w, x, b)
        ex = prepare(p, CuArray(w), CuArray(x), CuArray(b); target = T.CUDATarget(cuTile; precision = :tf32))
        @test length(ex.kernels) == 1 && only(ex.kernels).schedule.sketch === :product
        y = only(C.leaves(outputs(execute!(ex))))
        @test eltype(y) === BFloat16
        @test isapprox(Float32.(Array(y)), Float32.(interpret(p, w, x, b)); rtol = 1.0e-2)
        # Half operands accumulate in Float32 and round once.
        h, y = Float16.(data(64, 48; scale = 0.1)), Float16.(data(48, 32))
        p = trace((h, y) -> h * y, h, y)
        ex = prepare(p, CuArray(h), CuArray(y); target = T.CUDATarget(cuTile))
        @test any(d -> !(d isa C.FloorKernels) && d.schedule.sketch === :product, ex.kernels)
        @test isapprox(Array(only(C.leaves(outputs(execute!(ex))))), interpret(p, h, y); rtol = eps(Float16))
        # A product spelled as a broadcast and a sum is a product too.
        a, c = Float32.(data(64, 48; scale = 0.1)), Float32.(data(48, 80))
        ex = prepare(trace(spelled_product, a, c), CuArray(a), CuArray(c); target = T.CUDATarget(cuTile))
        @test any(d -> !(d isa C.FloorKernels) && d.schedule.sketch === :product, ex.kernels)
        @test isapprox(Array(only(C.leaves(outputs(execute!(ex))))), a * c; rtol = 1.0e-4)
        # A long sum over a small output splits among blocks (split-K) and merges.
        u, z = Float32.(data(64, 3000; scale = 0.01)), Float32.(data(3000, 48))
        for target in (T.CUDATarget(cuTile), T.CUDATarget(cuTile; precision = :tf32))
            p = trace((u, z) -> u * z, u, z)
            ex = prepare(p, CuArray(u), CuArray(z); target)
            @test any(d -> !(d isa C.FloorKernels) && d.schedule.sketch === :product && d.schedule.splits > 1, ex.kernels)
            @test isapprox(Array(only(C.leaves(outputs(execute!(ex))))), interpret(p, u, z); rtol = target.precision === :tf32 ? 1.0e-2 : 1.0e-4)
        end
    end
    using cuBLAS
    @testset "products on cuBLAS" begin
        @test Base.get_extension(Tempest, :cuBLASExt) !== nothing
        built(ex) = [d for d in ex.kernels if d isa Base.get_extension(Tempest, :cuBLASExt).BuiltProduct]
        a, b = Float32.(data(96, 80; scale = 0.1)), Float32.(data(80, 112))
        at, bt = permutedims(a), permutedims(b)
        A, B = Float32.(data(32, 24, 5; scale = 0.1)), Float32.(data(24, 40, 5; scale = 0.2))
        # Factors in either orientation, a result stored transposed, a batch,
        # and half factors accumulating in Float32.
        layouts = [((a, b) -> a * b, (a, b)), ((at, b) -> transpose(at) * b, (at, b)),
            ((a, bt) -> a * transpose(bt), (a, bt)), ((at, bt) -> transpose(at) * transpose(bt), (at, bt)),
            ((a, b) -> T.contract(a, b, (1, 2), (2, 3), (3, 1)), (a, b)), ((A, B) -> T.batched_mul(A, B), (A, B)),
            ((a, b) -> a * b, (Float16.(a), Float16.(b))), ((a, b) -> a * b, (BFloat16.(a), BFloat16.(b)))]
        for (f, args) in layouts, target in (T.CUDATarget(cuBLAS), T.CUDATarget(cuBLAS; precision = :tf32))
            p = trace(f, args...)
            ex = prepare(p, map(CuArray, args)...; target)
            @test length(built(ex)) == 1
            rtol = eltype(args[1]) === Float32 ? (target.precision === :tf32 ? 1.0e-2 : 1.0e-5) : 2 * eps(eltype(args[1]))
            @test isapprox(Float32.(Array(only(C.leaves(outputs(execute!(ex)))))), Float32.(interpret(p, args...)); rtol)
        end
        # With work after the product, cuBLAS's candidate is the product
        # alone, and the work after it a kernel of its own: at a size where
        # the product by cuBLAS is cheaper than the fused reduction.
        left, right = Float32.(data(256, 192; scale = 0.1)), Float32.(data(192, 224))
        w = Float32.(data(256; scale = 0.3))
        p = trace((a, b, w) -> tanh.(a * b .+ w), left, right, w)
        ex = prepare(p, CuArray(left), CuArray(right), CuArray(w); target = T.CUDATarget(cuBLAS))
        @test length(built(ex)) == 1 && length(taken(ex)) == 1 && length(ex.kernels) == 2
        @test isapprox(Array(only(C.leaves(outputs(execute!(ex))))), interpret(p, left, right, w); rtol = 1.0e-5)
        ex = prepare(trace((a, b) -> a * b, a, b), CuArray(a), CuArray(b); target = T.CUDATarget(cuTile))
        @test isempty(built(ex))
    end
    @testset "choosing between claims" begin
        # Half-precision products cuBLAS and cuTile both claim, every option
        # measured at preparation and kept in the tuning record.
        a, b = Float16.(data(1024, 256; scale = 0.1)), Float16.(data(256, 1024))
        p = trace((a, b) -> a * b, a, b)
        target = T.CUDATarget()
        prepared() = prepare(p, CuArray(a), CuArray(b); target, measure = :all)
        ex = prepared()
        x = only(ex.decisions)
        @test x.how === :measured && length(x.options) == 2
        @test occursin("chosen by measurement", sprint(explain, ex))
        path = C.record_path(target)
        lines = readlines(path)
        @test length(lines) >= 2
        # Given the record, preparing again measures nothing and chooses alike.
        again = prepared()
        @test readlines(path) == lines && only(again.decisions).choice.key == x.choice.key
        # A measurement overrides the estimate only beyond the noise margin.
        rewrite(f) = write(path, join((string(k, '\t', f(k, parse(Float64, t))) for (k, t) in split.(lines, '\t')), '\n') * '\n')
        other = only(o for o in x.options if o.key != x.choice.key)
        rewrite((k, t) -> k == other.key ? 1.0e-7 : 1.0e-5)
        @test only(prepared().decisions).choice.key == other.key
        rewrite((k, t) -> 1.0e-5)
        @test only(prepared().decisions).choice.key == first(x.options).key
        @test isapprox(Array(only(C.leaves(outputs(execute!(prepared()))))), interpret(p, a, b); rtol = 2 * eps(Float16))
    end
    @testset "implementation sets" begin
        w, x, b = Float32.(data(96, 80; scale = 0.1)), Float32.(data(80, 112)), Float32.(data(96))
        p = trace((w, x, b) -> gelu.(w * x .+ b), w, x, b)
        # Loading a package makes its implementations available; a target
        # names the ones it uses, and the floor is always there.
        @test C.used_packages(T.CUDATarget()) == [cuBLAS, cuTile]
        ex = prepare(p, CuArray(w), CuArray(x), CuArray(b); target = T.CUDATarget(cuTile))
        @test ex.packages == [cuTile]
        @test occursin("implementations from cuTile, and the floor", sprint(explain, ex))
        @test_throws ArgumentError prepare(p, CuArray(w), CuArray(x), CuArray(b); target = T.CUDATarget(Test))
    end
end

# Sizes that take several blocks: full reductions merge partial states, and
# folds over other than the swept axes (the loss, a weight gradient) merge
# theirs, under every sketch.
rmsnorm32(x, w) = x ./ sqrt.(sum(abs2, x; dims = 1) ./ size(x, 1) .+ 1.0f-5) .* w
function layernorm32(x, w)
    m = sum(x; dims = 1) ./ size(x, 1)
    (x .- m) ./ sqrt.(sum(abs2, x .- m; dims = 1) ./ size(x, 1) .+ 1.0f-5) .* w
end

const LARGE = [
    ("vector sum", x -> sum(abs2, x), (data(1 << 16),)),
    ("dot product", (a, b) -> sum(a .* b), (data(1 << 16), data(1 << 16; scale = 0.3))),
    ("matrix sum and shift", x -> sum(exp.(x .- maximum(x))), (data(512, 300),)),
    ("RMSNorm training, resident", (x, w) -> sum(abs2, rmsnorm32(x, w)), (data(256, 700), data(256; scale = 0.1))),
    ("softmax training, resident", (x, t) -> sum(softmax_columns(x) .* t), (data(256, 700), data(256, 700; scale = 0.7))),
    ("softmax training, streamed", (x, t) -> sum(softmax_columns(x) .* t), (data(9000, 40), data(9000, 40; scale = 0.7))),
    ("row softmax training, split", (x, t) -> sum(softmax_rows(x) .* t), (data(100, 3000), data(100, 3000; scale = 0.7))),
    ("layer norm training", (x, w) -> sum(abs2, layernorm32(x, w)), (data(256, 700), data(256; scale = 0.1))),
]

if CUDA_FUNCTIONAL
    @testset "several blocks with cuTile, Float32" begin
        @testset "$name" for (name, f, args) in LARGE
            args = map(single, args)
            for p in (trace(f, args...), differentiate(trace(f, args...)))
                reference = C.leaves(interpret(p, args...))
                ex = prepare(p, map(x -> x isa AbstractArray ? CuArray(x) : x, args)...)
                @test all(d -> !(d isa C.FloorKernels), ex.kernels)
                @test agree(C.leaves(outputs(execute!(ex))), reference; rtol = 2.0e-3)
            end
        end
    end
end

# A fold of the user's with a tuple state, Welford's count, mean and sum of
# squared deviations, generated like the built-in folds under every sketch.
function welford(a, b)
    n = a[1] + b[1]
    w = ifelse(n == 0, zero(n), b[1] / n)
    d = b[2] - a[2]
    (n, a[2] + d * w, a[3] + b[3] + d * d * a[1] * w)
end
const VARIANCE = T.GenericFold(; init = T -> (zero(T), zero(T), zero(T)), lift = x -> (one(x), x, zero(x)),
    combine = welford, observe = s -> s[3] / s[1])

if CUDA_FUNCTIONAL
    @testset "a fold with a tuple state on cuTile" begin
        for (sketch, x, dims) in ((:resident, data(300, 200), 1), (:streamed, data(9000, 40), 1), (:split, data(100, 3000), 2),
            (:full, data(512, 300), (1, 2)))
            x = single(x)
            p = trace(x -> T.fold(VARIANCE, x; dims), x)
            ex = prepare(p, CuArray(x); target = T.CUDATarget(cuTile))
            @test only(ex.kernels).schedule.sketch === sketch
            @test isapprox(Array(only(C.leaves(outputs(execute!(ex))))), interpret(p, x); rtol = 1.0e-4)
        end
    end
end

# Positions on tiles, and formulas of the axes the grid binds computed per
# block: a causal mask after a product, positions in pointwise work and in a
# fold, and key heads shared among query heads, spelled as a broadcast that
# reads key head h ÷ G.
causal_scores(a, b) = (s = a' * b; ifelse.((1:size(s, 1)) .<= (1:size(s, 2))', s, -Inf32))
function grouped_scores(q, k)
    D, H, S, B = size(q)
    G = H ÷ size(k, 2)
    shared = reshape(reshape(k, D, 1, size(k, 2), S, B) .* ones(Float32, 1, G, 1, 1, 1), D, H, S, B)
    T.contract(shared, q, (1, 2, 3, 4), (1, 2, 5, 4), (3, 5, 2, 4))
end

const POSITIONAL = [
    ("causal mask after a product", (a, b) -> exp.(causal_scores(a, b) .- T.logsumexp(causal_scores(a, b); dims = 1)),
     (data(24, 70; scale = 0.1), data(24, 70; scale = 0.2))),
    ("positions in pointwise work", x -> x .* reshape(1:size(x, 2), 1, :) .+ reshape(1:length(x), size(x)), (data(50, 30, 3),)),
    ("positions with arithmetic", x -> x .* fld.(reshape(1:size(x, 2), 1, :) .- 1, 4) .+ (2 .* reshape(Int32(1):Int32(size(x, 1)), :, 1)), (data(50, 30, 3),)),
    ("positions in a fold", x -> sum(ifelse.((1:size(x, 1)) .<= (1:size(x, 2))', x, 0.0f0); dims = 1), (data(300, 200),)),
    ("grouped heads by division", grouped_scores, (data(16, 6, 40, 2; scale = 0.1), data(16, 2, 40, 2; scale = 0.2))),
    ("shared columns by division in a fold",
     x -> sum(reshape(reshape(x, 64, 1, 3, 10) .* ones(Float32, 1, 2, 1, 1), 64, 6, 10); dims = 1), (data(64, 3, 10),)),
]

if CUDA_FUNCTIONAL
    @testset "positions and per-block formulas on cuTile" begin
        @testset "$name" for (name, f, args) in POSITIONAL
            args = map(single, args)
            p = trace(f, args...)
            ex = prepare(p, map(CuArray, args)...; target = T.CUDATarget(cuTile))
            @test all(d -> !(d isa C.FloorKernels), ex.kernels)
            @test agree(C.leaves(outputs(execute!(ex))), C.leaves(interpret(p, args...)); rtol = 2.0e-4)
        end
    end
end

# Gathers by runtime indices, element by element inside the kernels that read
# them and never stored: an embedding lookup with pointwise work and a sum; a
# lookup with a layer norm, where the embedding plus a bias, read three times,
# joins the kernel that reads it (a proposal of the planner's own); a long sum
# over gathered columns; and a product with a gathered factor.
tokens = Int32.(mod1.(37 .* (1:240), 1000))
const GATHERS = [
    ("embedding, pointwise work and a sum", (E, t) -> sum(abs2, tanh.(E[:, t]); dims = 1),
     (data(64, 1000; scale = 0.01), reshape(tokens, 80, 3)), 1),
    ("embedding and layer norm", (E, t, b, w) -> layernorm32(E[:, t] .+ b, w),
     (data(64, 1000; scale = 0.01), reshape(tokens, 80, 3), data(64; scale = 0.3), data(64; scale = 0.1)), 1),
    ("long sum over gathered columns", (E, t) -> sum(E[:, t]; dims = 2), (data(64, 1000; scale = 0.01), tokens), 1),
    ("product with a gathered factor", (W, E, t) -> W * E[:, t], (data(96, 64; scale = 0.02), data(64, 1000; scale = 0.01), tokens), 1),
]

if CUDA_FUNCTIONAL
    @testset "gathers on cuTile" begin
        @testset "$name" for (name, f, args, kernels) in GATHERS
            args = map(single, args)
            p = trace(f, args...)
            ex = prepare(p, map(CuArray, args)...; target = T.CUDATarget(cuTile))
            @test length(ex.kernels) == kernels && all(d -> !(d isa C.FloorKernels), ex.kernels)
            @test agree(C.leaves(outputs(execute!(ex))), C.leaves(interpret(p, args...)); rtol = 1.0e-3)
        end
    end
end

# Folds and broadcasts over part of the axes a schedule collapses into one
# tile axis: a sum over the middle axis of an array, spelled directly and
# through `permutedims`, and a long one that splits; a positional embedding
# broadcast over the batch; a table broadcast over heads and batch, as rotary
# positions read it. The gradient of the embedding and layer norm sums over
# the batch alone; only its scatter is on the floor.
const PARTIAL_COLLAPSE = [
    ("sum over the middle axis", x -> sum(x; dims = 2), (data(8, 64, 6),), false),
    ("sum over the middle axis, permuted", x -> sum(permutedims(x, (2, 1, 3)); dims = 1), (data(64, 8, 6),), false),
    ("long sum over the middle axis", x -> sum(abs2, x; dims = 2), (data(40, 3000, 7),), false),
    ("positional embedding", (E, t, P) -> E[:, t] .+ P, (data(64, 1000; scale = 0.01), reshape(tokens, 80, 3), data(64, 80)), false),
    ("a table broadcast over heads and batch", (x, c) -> x .* c, (data(32, 4, 64, 16), data(32, 1, 64)), false),
    ("positional embedding and layer norm, training", (E, t, P, w) -> sum(abs2, layernorm32(E[:, t] .+ P, w)),
     (data(64, 1000; scale = 0.01), reshape(tokens, 80, 3), data(64, 80; scale = 0.2), data(64; scale = 0.1)), true),
]

if CUDA_FUNCTIONAL
    @testset "folds over part of a collapsed axis on cuTile" begin
        @testset "$name" for (name, f, args, training) in PARTIAL_COLLAPSE
            args = map(single, args)
            p = training ? differentiate(trace(f, args...)) : trace(f, args...)
            ex = prepare(p, map(CuArray, args)...; target = T.CUDATarget(cuTile))
            @test all(k -> (ex.kernels[k] isa C.FloorKernels) == (ex.plan.groups[k].kind === :opaque), eachindex(ex.kernels))
            @test agree(filter(!isnothing, C.leaves(outputs(execute!(ex)))), filter(!isnothing, C.leaves(interpret(p, args...))); rtol = 1.0e-4)
        end
    end
end

# A gather's adjoint scatters on the device: one item per element scattered,
# adding atomically, so the sums at an index selected more than once come in
# no fixed order and differ from the interpreter's by rounding (Promise 3: up
# to summation order). With repeated indices, two index arrays, and ranges
# and integers beside an index array.
const SCATTERS = [
    ("embedding", (E, t, g) -> sum(E[:, t] .* g), (data(64, 109), Int32.(mod1.(37 .* (1:1024) .^ 2, 109)), data(64, 1024; scale = 0.37))),
    ("repeated indices in two arrays", (x, i, j) -> sum(abs2, x[i, j]), (data(8, 6), [3, 1, 3, 8, 3], Int32[6, 6, 1])),
    ("a strided range and an integer beside index arrays", x -> sum(abs2, x[2:2:8, [1, 4, 4]]) + sum(abs2, x[3, [2, 2]]), (data(8, 6),)),
    ("a reversed range", x -> sum(abs2, x[8:-3:1, [2, 2, 5]]), (data(8, 6),)),
]

if CUDA_FUNCTIONAL
    @testset "scatters on the device" begin
        @testset "$name" for (name, f, args) in SCATTERS
            args = map(single, args)
            p = differentiate(trace(f, args...))
            ex = prepare(p, map(CuArray, args)...)
            scatters = [k for (k, g) in enumerate(ex.plan.groups) if g.kind === :opaque]
            @test !isempty(scatters) && all(k -> ex.kernels[k] isa C.FloorKernels, scatters)
            reference = C.leaves(interpret(p, args...))[2:2]
            @test agree(C.leaves(outputs(execute!(ex)))[2:2], reference; rtol = 1.0e-5)
            # A scatter runs on the device, so the program can be captured.
            capture!(ex)
            @test agree(C.leaves(outputs(replay!(ex)))[2:2], reference; rtol = 1.0e-5)
        end
    end
end

# Attention generated from its reference: one kernel, a fold over keys whose
# scores and logsumexp are recomputed in it, online or in two passes.
const GENERATED_ATTENTION = [
    ("causal, grouped heads", (16, 4, 2, 200, 2), true),
    ("grouped heads", (16, 4, 2, 200, 2), false),
    ("causal, odd sizes", (32, 3, 1, 70, 3), true),
    ("one head", (8, 1, 1, 133, 1), false),
]

if CUDA_FUNCTIONAL
    @testset "attention generated on cuTile" begin
        @testset "$name" for (name, (D, H, KV, S, B), causal) in GENERATED_ATTENTION
            q, k, v = single(data(D, H, S, B; scale = 0.1)), single(data(D, KV, S, B; scale = 0.2)), single(data(D, KV, S, B; scale = 0.3))
            p = trace((q, k, v) -> Ops.attention(q, k, v; causal), q, k, v)
            for online in (true, false)
                ex = prepare(p, map(CuArray, (q, k, v))...; target = T.CUDATarget(cuTile; online))
                kernel = only(ex.kernels)
                @test !(kernel isa C.FloorKernels)
                @test occursin("online softmax", string(kernel)) == online
                @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, q, k, v)]; rtol = 1.0e-4)
            end
        end
        # Softmax spelled with a maximum and a sum: rewritten into the same one
        # kernel, at full precision and in TF32.
        q, k, v = single(data(16, 2, 70, 2; scale = 0.1)), single(data(16, 2, 70, 2; scale = 0.2)), single(data(16, 2, 70, 2; scale = 0.3))
        p = trace(q, k, v) do q, k, v
            s = T.contract(k, q, (1, 2, 3, 4), (1, 2, 5, 4), (3, 2, 5, 4)) .* 0.25f0
            s = ifelse.(reshape(1:size(s, 1), :, 1, 1) .<= reshape(1:size(s, 3), 1, 1, :), s, -Inf32)
            e = exp.(s .- maximum(s; dims = 1))
            T.contract(v, e ./ sum(e; dims = 1), (1, 2, 3, 4), (3, 2, 5, 4), (1, 2, 5, 4))
        end
        for precision in (:full, :tf32)
            ex = prepare(p, map(CuArray, (q, k, v))...; target = T.CUDATarget(cuTile; precision))
            @test occursin("online softmax", string(only(ex.kernels)))
            @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, q, k, v)]; rtol = precision === :tf32 ? 1.0e-2 : 1.0e-4)
        end
        # In half precision the rewrite reads the exponential through the
        # product's conversion to Float32: one kernel, online, the weights
        # rounded to BFloat16 for the tensor cores.
        qh, kh, vh = map(x -> BFloat16.(single(x)), (data(16, 2, 70, 2; scale = 0.1), data(16, 2, 70, 2; scale = 0.2), data(16, 2, 70, 2; scale = 0.3)))
        p = trace((q, k, v) -> Ops.attention(q, k, v; causal = true), qh, kh, vh)
        ex = prepare(p, map(CuArray, (qh, kh, vh))...; target = T.CUDATarget(cuTile))
        @test occursin("online softmax", string(only(ex.kernels)))
        @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, qh, kh, vh)]; rtol = 1.0e-2)
        # The reference's gradient: every product generated, also those that
        # sum over key heads shared by one query head each.
        q, k, v = single(data(16, 2, 70, 2; scale = 0.1)), single(data(16, 2, 70, 2; scale = 0.2)), single(data(16, 2, 70, 2; scale = 0.3))
        d = differentiate(trace((q, k, v) -> sum(abs2, Ops.attention(q, k, v; causal = true)), q, k, v))
        ex = prepare(d, map(CuArray, (q, k, v))...; target = T.CUDATarget(cuTile))
        @test all(d -> !(d isa C.FloorKernels), ex.kernels)
        @test agree(filter(!isnothing, C.leaves(outputs(execute!(ex)))), filter(!isnothing, C.leaves(interpret(d, q, k, v))); rtol = 1.0e-4)
        # With grouped heads, the products into the key and value gradients sum
        # over the query heads of a group and the queries, which no strided
        # axis spans: the queries are tiled and the heads swept in turn.
        g = single(data(16, 4, 70, 2; scale = 0.1))
        d = differentiate(trace((q, k, v) -> sum(abs2, Ops.attention(q, k, v; causal = true)), g, k, v))
        ex = prepare(d, map(CuArray, (g, k, v))...; target = T.CUDATarget(cuTile))
        @test all(d -> !(d isa C.FloorKernels), ex.kernels)
        @test any(d -> occursin("in turn", string(d)), ex.kernels)
        @test agree(filter(!isnothing, C.leaves(outputs(execute!(ex)))), filter(!isnothing, C.leaves(interpret(d, g, k, v))); rtol = 1.0e-4)
        # A residual is part of the one kernel, also with a key head per query
        # head, where the reference groups heads along an axis of extent one.
        x = single(data(16, 2, 70, 2; scale = 0.5))
        p = trace((q, k, v, x) -> Ops.attention(q, k, v; causal = true) .+ x, q, k, v, x)
        ex = prepare(p, map(CuArray, (q, k, v, x))...; target = T.CUDATarget(cuTile))
        @test length(ex.kernels) == 1 && !(only(ex.kernels) isa C.FloorKernels)
        @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, q, k, v, x)]; rtol = 1.0e-4)
    end
end
