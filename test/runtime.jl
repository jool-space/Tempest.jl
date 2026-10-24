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

# Writes where elements land at one position, on the CPU's floor: in order,
# the last landing, as the interpreter places them.
@testset "repeated writes on the CPU floor" begin
    @testset "$name" for (name, f, args) in REPEATED_WRITES
        for p in (trace(f, args...), differentiate(trace(f, args...)))
            @test agree(filter(!isnothing, C.leaves(outputs(execute!(prepare(p, args...))))),
                filter(!isnothing, C.leaves(interpret(p, args...))); rtol = 1.0e-10)
        end
    end
end

# Extremes prepared on the CPU find what Base finds, exactly (`EXTREMES` and
# `FOUND` in fixtures/programs.jl).
@testset "extremes on the CPU" begin
    @testset "$(eltype(x)) $i" for (i, x) in enumerate(EXTREMES), f in FOUND
        @test isequal(found(outputs(execute!(prepare(trace(f, x), x)))), found(f(x)))
    end
end

# A BFloat16 scatter on the floor sums in Float32 and rounds once, as the
# reference does: 41 from 4096 gradients of 0.01 at one position, not 4.
@testset "a BFloat16 scatter on the floor" begin
    E, t, g = BFloat16s.BFloat16.(fill(0.5f0, 4, 3)), Int32.(fill(2, 4096)), BFloat16s.BFloat16.(fill(0.01f0, 4, 4096))
    p = differentiate(trace((E, t, g) -> sum(E[:, t] .* g), E, t, g))
    @test outputs(execute!(prepare(p, E, t, g)))[2] == interpret(p, E, t, g)[2]
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
# not overlap, or where a fold by key writes over the value it starts from, so
# training steps that keep values for their backward pass store less.
function slots_disjoint(s::C.Storage)
    slotted = [v for v in eachindex(s.slots) if s.slots[v] > 0]
    over(v, w) = (v => w) in s.overwrites || (w => v) in s.overwrites
    all(isempty(intersect(s.lifetimes[v], s.lifetimes[w])) || over(v, w) for v in slotted for w in slotted if v < w && s.slots[v] == s.slots[w])
end

@testset "slots" begin
    for (name, f, args) in PROGRAMS
        ex = prepare(differentiate(trace(f, args...)), args...)
        @test slots_disjoint(ex.storage)
        bytes, apart = C.footprint(ex)
        @test name in ("MLP", "attention, grouped heads") ? bytes < apart : bytes <= apart
    end
end

# A write takes the storage of the contents it starts from where nothing reads
# them after it, and writes only its positions: a cache the step replaces,
# appended per sequence, three writes each over the last, is where the step
# leaves it, and no kernel copies it.
function appended(K, k, p)
    K = copy(K)
    for b in axes(K, 4)
        K[:, p[b], :, b] .= k[:, :, b]
    end
    K
end

function check_appended_in_place(device, target)
    K, k, p = data(4, 16, 2, 3), data(4, 2, 3), Int32[3, 16, 1]
    program = trace(appended, K, k, p)
    ex = prepare(program, device(K), device(k), device(p); target, replaces = (1 => 1,))
    @test length(ex.storage.donated) == 1 && isempty(ex.storage.staged) && length(ex.storage.overwrites) == 3
    @test pointer(ex.results[1]) == pointer(ex.arguments[1])
    @test count("written over", sprint(explain, ex)) == 3
    # Each kernel writes the elements of one sequence's token.
    @test all(k -> !(k isa C.FloorKernels) || all(s -> s.items == 8, k.kernels), ex.kernels)
    @test !any(k -> occursin("initial states", string(k)), ex.kernels)
    # Each write is priced by the elements it writes: into a longer cache it
    # moves no more bytes and takes no longer.
    longer = data(4, 256, 2, 3)
    ey = prepare(trace(appended, longer, k, p), device(longer), device(k), device(p); target, replaces = (1 => 1,))
    estimates(e) = [d.choice.estimate for d in e.decisions if d !== nothing]
    @test map(v -> v.bytes, C.group_views(ey)) == map(v -> v.bytes, C.group_views(ex))
    @test estimates(ey) == estimates(ex)
    expected = K
    for positions in (p, Int32[5, 2, 9])
        bind!(ex, 3 => device(positions))
        reference = interpret(program, expected, k, positions)
        @test Array(outputs(execute!(ex))) == reference
        expected = reference
    end
end

# A value the program computes is written over where the write is its last
# reader; the contents are copied where the elements written read them, and
# from an input nothing replaces, which the program never writes.
@testset "writes in place" begin
    check_appended_in_place(identity, T.CPUTarget())
    f = (x, v, i) -> (y = x .* 2; y[:, i] .= v; sum(abs2, tanh.(y)))
    args = (data(4, 9), data(4, 3), Int32[7, 2, 5])
    for p in (trace(f, args...), differentiate(trace(f, args...)))
        ex = prepare(p, args...)
        @test !isempty(ex.storage.overwrites) && slots_disjoint(ex.storage)
        @test agree(filter(!isnothing, C.leaves(outputs(execute!(ex)))), filter(!isnothing, C.leaves(interpret(p, args...))); rtol = 1.0e-10)
    end
    g = (x, i, j) -> (y = copy(x); y[:, i] .= x[:, j]; y)
    args = (data(4, 9), Int32[7, 2], Int32[1, 7])
    ex = prepare(trace(g, args...), args...; replaces = (1 => 1,))
    @test isempty(ex.storage.overwrites) && length(ex.storage.staged) == 1
    @test outputs(execute!(ex)) == interpret(trace(g, args...), args...)
    x, v, i = data(4, 9), data(4, 3), Int32[7, 2, 5]
    unwritten = copy(x)
    ex = prepare(trace((x, v, i) -> (y = copy(x); y[:, i] .= v; y), x, v, i), x, v, i; by_reference = (1,))
    @test isempty(ex.storage.overwrites)
    @test outputs(execute!(ex))[:, [7, 2, 5]] == v && x == unwritten
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

@testset "prepared without building" begin
    x = single(data(8, 6))
    p = trace(x -> sum(abs2, tanh.(x); dims = 1), x)
    ex = prepare(p, x; build = false)
    @test !ex.built && isempty(ex.launches) && length(ex.kernels) == length(ex.plan.groups)
    @test occursin("not built", sprint(explain, ex))
    for f in (() -> execute!(ex), () -> bind!(ex, x), () -> capture!(ex))
        @test occursin("build = false", message(f))
    end
    # The plan and the decisions of a built preparation on the same nominal rates.
    built = prepare(p, x; measure = :none)
    @test [g.terms for g in ex.plan.groups] == [g.terms for g in built.plan.groups]
    how(ds) = [d === nothing ? nothing : (d.choice.implementation, d.how) for d in ds]
    @test how(ex.decisions) == how(built.decisions)
    if CUDA_FUNCTIONAL
        # Inputs on the host: only their shapes and types are read.
        ex = prepare(p, x; target = T.CUDATarget(), build = false)
        @test !ex.built && occursin("cuTile", sprint(explain, ex))
    end
end

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

# Recomputing a fold where it is read is a proposal: priced only where an
# implementation that recomputes folds claims every kernel that recomputes
# it, and taken by cost. These implementations claim groups over three axes,
# or that recompute nothing; one finds recomputing slow, the other cheap.
# Recomputing `q` in the product of `s`, over three axes, is proposed and
# priced; recomputing `s` in the logsumexp, over two axes, is never priced,
# since nothing claims that kernel, and `s` reads `q` inside its sweep.
struct RecomputesOverThree{Cheap} <: C.Implementation end
C.recomputes_folds(::RecomputesOverThree, target) = true
C.claims(::RecomputesOverThree, target, g) = isempty(C.recomputed_in(g)) || length(g.domain) >= 3 ? (:claim,) : ()
C.estimate(::RecomputesOverThree{Cheap}, claim, target, g) where {Cheap} =
    [C.Work(0.0; flops = isempty(C.recomputed_in(g)) == Cheap ? 1.0 : 1.0e-6)]

@testset "recomputing a fold is a proposal, priced where its kernels are claimed" begin
    x, w = data(8, 6), data(8, 8)
    function f(x, w)
        q = C.contract(w, x, (1, 2), (1, 3), (2, 3))
        s = C.contract(q, q, (1, 2), (1, 3), (2, 3))
        T.logsumexp(s; dims = 1)
    end
    graph = C.lower(trace(f, x, w))
    q = only(C.recomputable_folds(graph))
    @test graph.values[q].definition isa C.Reduction && graph.values[q].dims == (8, 6)
    for cheap in (false, true)
        C.register!(@__MODULE__, RecomputesOverThree{cheap}())
        try
            d = C.Decider(T.CPUTarget(), graph; measure = false)
            p, considered = C.cover(d)
            proposed = filter(c -> c.proposal isa C.Recompute, considered)
            @test map(c -> c.proposal.value, proposed) == [q]
            @test only(proposed).accepted == cheap
            @test (q in reduce(union, map(C.recomputed_in, p.groups); init = Set{Int}())) == cheap
            @test all(g -> !isempty(C.claims(RecomputesOverThree{cheap}(), d.target, C.group_view(p, g))), p.groups)
        finally
            C.register!(@__MODULE__)
        end
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
    # buffers hold integers spread over the axes they index, so that timing
    # every option of a gather (the floor too, with bounds checked as under
    # `Pkg.test`) reads in bounds, and a gather or a scatter reads or writes
    # as many places as it has indices: the arguments here are indices, and
    # no two of an argument's entries are alike.
    @testset "timed on trial buffers, a gather reads in bounds" begin
        @testset "$name" for (name, f, args) in filter(((_, _, args),) -> any(a -> a isa AbstractArray{<:Integer}, args), PROGRAMS)
            for p in (trace(f, args...), differentiate(trace(f, args...)))
                graph = C.lower(p)
                planned = C.plan(graph)
                d = C.Decider(T.CUDATarget(), graph; measure = false)
                foreach(g -> C.buffers!(d, planned, g), planned.groups)
                indices = [b for (v, b) in zip(graph.values, d.buffers) if b !== nothing && v.eltype <: Integer && !(v.definition isa C.ConstantArray)]
                @test !isempty(indices) && all(b -> all(>=(1), Array(b)), indices)
                held = [d.buffers[a] for a in graph.arguments if graph.values[a].eltype <: Integer && d.buffers[a] !== nothing]
                @test !isempty(held) && all(b -> allunique(Array(b)), held)
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
            ((a, b) -> C.contract(a, b, (1, 2), (2, 3), (3, 1)), (a, b)), ((A, B) -> C.contract(A, B, (1, 2, 4), (2, 3, 4), (1, 3, 4)), (A, B)),
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
        # Half-precision products cuBLAS claims and cuTile claims under each
        # of its tiles, every option measured at preparation and kept in the
        # tuning record.
        a, b = Float16.(data(1024, 256; scale = 0.1)), Float16.(data(256, 1024))
        p = trace((a, b) -> a * b, a, b)
        target = T.CUDATarget()
        prepared() = prepare(p, CuArray(a), CuArray(b); target, measure = :all)
        ex = prepared()
        x = only(ex.decisions)
        @test x.how === :measured && length(x.options) >= 2
        @test length(unique(C.package_of(o.implementation) for o in x.options)) == 2
        @test occursin("chosen by measurement", sprint(explain, ex))
        path = C.record_path(target)
        lines = readlines(path)
        @test length(lines) >= 2
        # Given the record, preparing again measures nothing and chooses alike.
        again = prepared()
        @test readlines(path) == lines && only(again.decisions).choice.key == x.choice.key
        # A measurement overrides the estimate only beyond the noise margin.
        rewrite(f) = write(path, join((string(k, '\t', f(k, parse(Float64, t))) for (k, t) in split.(lines, '\t')), '\n') * '\n')
        other = first(o for o in x.options if o.key != x.choice.key)
        rewrite((k, t) -> k == other.key ? 1.0e-7 : 1.0e-5)
        @test only(prepared().decisions).choice.key == other.key
        rewrite((k, t) -> 1.0e-5)
        @test only(prepared().decisions).choice.key == first(x.options).key
        @test isapprox(Array(only(C.leaves(outputs(execute!(prepared()))))), interpret(p, a, b); rtol = 2 * eps(Float16))
    end
    @testset "the search is staged" begin
        # The generator claims a product under every output tile at the
        # default depth, and the cover tries the variants of the one that
        # measures best: its other depths, and its sum split the other way. A
        # close-call search tries fewer tiles than measuring every option
        # does, and chooses alike each time from the record. A product with
        # an epilogue, on a target of cuTile alone, at full precision:
        # depths 16 and 32, and a grid of few output tiles, so that the sum
        # splits.
        a, b, c = single(data(256, 4096; scale = 0.1)), single(data(4096, 256)), single(data(256))
        p = trace((a, b, c) -> gelu.(a * b .+ c), a, b, c)
        target = T.CUDATarget(cuTile)
        dir = mktempdir()
        rates = C.rates_path(target)
        isfile(rates) && (mkpath(joinpath(dir, "tuning")); cp(rates, joinpath(dir, "tuning", basename(rates))))
        ext = Base.get_extension(Tempest, :cuTileExt)
        tile(o) = (s = o.claim.schedule; (s.tiling[1].tile, s.tiling[3].tile, s.tiling[2].tile, s.splits))
        withenv("TEMPEST_CACHE" => dir) do
            prepared(measure) = only(x for x in prepare(p, CuArray(a), CuArray(b), CuArray(c); target, measure).decisions
                if x !== nothing && x.choice.implementation isa ext.cuTileGenerator && x.choice.claim.schedule.sketch === :product)
            staged = prepared(:close)
            tiles = map(tile, staged.options)
            @test staged.how === :measured
            # Every output tile at depth 16, the default, and the variants of
            # the best: at least its other depth and its split, fewer than all.
            @test length(unique(t[1:2] for t in tiles if t[3] == 16)) == 4
            @test length(tiles) >= 6
            again = prepared(:close)
            @test again.choice.key == staged.choice.key && length(again.options) == length(staged.options)
            # Every option: the four output tiles at both depths, split and
            # not, but the tile of 32 output blocks, which does not split.
            every = prepared(:all)
            @test length(staged.options) < length(every.options) == 14
            @test all(t -> t[1:3] in [(tm, tn, tk) for (tm, tn) in ((64, 64), (128, 64), (64, 128), (32, 64)) for tk in (16, 32)],
                map(tile, every.options))
        end
    end
    @testset "a matrix-vector product searches its step along the sum" begin
        # A matrix stored along its sum times one column is the streamed
        # reduction of the sum, and the cover tries its longer steps. Rows
        # enough for 64 blocks, so that the sum streams rather than splits.
        w, x = single(data(512, 4096; scale = 0.1)), single(data(512, 1))
        p = trace((w, x) -> Einops.@einsum(w, x, "k m, k n -> m n"), w, x)
        target = T.CUDATarget(cuTile)
        dir = mktempdir()
        rates = C.rates_path(target)
        isfile(rates) && (mkpath(joinpath(dir, "tuning")); cp(rates, joinpath(dir, "tuning", basename(rates))))
        ext = Base.get_extension(Tempest, :cuTileExt)
        step(o) = (s = o.claim.schedule; s.tiling[ext.sweep_axis(s)].tile)
        withenv("TEMPEST_CACHE" => dir) do
            ex = prepare(p, CuArray(w), CuArray(x); target)
            d = only(d for d in ex.decisions if d !== nothing && d.choice.implementation isa ext.cuTileGenerator)
            @test sort([step(o) for o in d.options if o.implementation isa ext.cuTileGenerator]) == [32, 64, 128]
            @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, w, x)]; rtol = 1.0e-4)
        end
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

# A fold of the user's with a tuple state (`VARIANCE`, in
# fixtures/programs.jl), generated like the built-in folds under every sketch,
# named and as Base spells it, its observe composed from the broadcast after.

if CUDA_FUNCTIONAL
    @testset "a fold with a tuple state on cuTile" begin
        for (sketch, x, dims) in ((:resident, data(300, 200), 1), (:streamed, data(9000, 40), 1), (:split, data(100, 3000), 2),
            (:full, data(512, 300), (1, 2))), f in (x -> reduce(VARIANCE, x; dims), x -> base_variance(x; dims))
            x = single(x)
            p = trace(f, x)
            ex = prepare(p, CuArray(x); target = T.CUDATarget(cuTile))
            @test only(ex.kernels).schedule.sketch === sketch
            @test isapprox(Array(only(C.leaves(outputs(execute!(ex))))), interpret(p, x); rtol = 1.0e-4)
        end
    end
end

# Extremes on cuTile find what Base finds: the first of equals, a NaN, 0.0
# above -0.0, within a block's tile and among the blocks a long column splits
# into, in Float32, BFloat16s.BFloat16 and Int32. Where they are along a
# vector or one axis, taken as integers, is a fold on tiles; a
# `CartesianIndex` is no tile's element, and is made on the floor.
if CUDA_FUNCTIONAL
    @testset "extremes on cuTile" begin
        x = Float32.(sin.(reshape(1:(50000 * 3), 50000, 3)) .* 100)
        x[777, 1] = x[40000, 1] = 500
        x[123, 2] = x[9000, 2] = NaN
        x[:, 3] .= .-abs.(x[:, 3]) .- 1
        x[5, 3], x[6, 3], x[9, 3] = 0.0f0, -0.0f0, 0.0f0
        on_tiles = (x -> getindex.(argmax(x; dims = 1), 1), x -> argmax(x[:, 1]))
        found_in_floats(r) = map(v -> v isa Union{BFloat16s.BFloat16, AbstractArray{BFloat16s.BFloat16}} ? Float32.(v) : v, found(r))
        @testset "$(eltype(x))" for x in (x, BFloat16s.BFloat16.(x), Int32.(replace(round.(x), NaN => 0)))
            reference = x isa AbstractArray{BFloat16s.BFloat16} ? Float32.(x) : x
            for f in (on_tiles..., x -> findmax(x; dims = 1), x -> findmin(x[1:1000, :]; dims = 2), argmin, x -> extrema(x; dims = 1))
                ex = prepare(trace(f, x), CuArray(x); target = T.CUDATarget(cuTile))
                f in on_tiles && @test !any(k -> k isa C.FloorKernels, ex.kernels)
                @test isequal(found_in_floats(outputs(execute!(ex))), found(f(reference)))
            end
        end
    end
end

# Scans on cuTile, a block per tile of the kept axes sweeping the scanned ones:
# along the first axis, a middle one, the last, and two at once; builtin folds
# and folds of the user's, a recurrence among them, whose combine does not
# commute; with maps around them and a fold beside them; and reversed, in a
# cumulative sum's derivative. Base's `accumulate` by a function of the
# user's runs in sequence, and an exclusive scan, which derivatives make, has
# no tile realization yet: both stay on the floor.
if CUDA_FUNCTIONAL
    @testset "scans on cuTile" begin
        x = single(data(1000, 64))
        a, b = single(0.9 .+ 0.1 .* data(700, 32; scale = 0.3)), single(data(700, 32; scale = 0.7))
        scans = [
            (x -> cumsum(x; dims = 1), (x,)), (x -> cumsum(x; dims = 2), (x,)), (cumsum, (single(data(5000)),)),
            (x -> cumsum(x; dims = 2), (single(data(16, 300, 8)),)), (x -> accumulate(+, x), (single(data(30, 40)),)),
            (x -> cumprod(1 .+ 0.001f0 .* x; dims = 1), (x,)), (x -> accumulate(max, x; dims = 1), (x,)),
            ((a, b) -> accumulate(RECURRENCE, tuple.(a, b); dims = 1), (a, b)),
            (x -> accumulate(VARIANCE, x; dims = 1), (single(data(300, 20)),)),
            (x -> exp.(cumsum(sin.(x); dims = 1) .* 0.01f0), (x,)), (x -> (cumsum(x; dims = 1), sum(x; dims = 1)), (x,)),
            (x -> cumsum(x; dims = 1), (BFloat16s.BFloat16.(data(500, 16)),)),
            (x -> T.gradient(x -> sum(cumsum(x; dims = 2) .^ 2), x), (single(data(16, 300, 8)),)),
        ]
        sketches(ex) = [k.schedule.sketch for k in ex.kernels if hasproperty(k, :schedule)]
        for (f, args) in scans
            p = trace(f, args...)
            ex = prepare(p, map(CuArray, args)...; target = T.CUDATarget(cuTile))
            @test :scan in sketches(ex)
            @test agree(C.leaves(outputs(execute!(ex))), C.leaves(interpret(p, args...)); rtol = 1.0e-4)
        end
        for (f, args) in [((a, b) -> recurrence(a, b), (single(data(700)), single(data(700)))),
            (x -> T.gradient(x -> sum(cumprod(1 .+ 0.001f0 .* x; dims = 1)), x), (x,))]
            p = trace(f, args...)
            ex = prepare(p, map(CuArray, args)...; target = T.CUDATarget(cuTile))
            @test !(:scan in sketches(ex))
            @test agree(C.leaves(outputs(execute!(ex))), C.leaves(interpret(p, args...)); rtol = 1.0e-4)
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
    C.contract(shared, q, (1, 2, 3, 4), (1, 2, 5, 4), (3, 5, 2, 4))
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

# The dropout fixture's hash, wrapping Int32 multiplies and logical shifts on
# tiles, is claimed by the generator and then fails to build: the assembler
# rejects the bytecode cuTile emits for it. The floor runs it, correct, and
# the failed build warns once.
const INTEGER_HASH = "an integer hash (wrapping multiplies, logical shifts) on Int32 tiles (cuTile's bytecode)"
if CUDA_FUNCTIONAL
    @testset "gap: $INTEGER_HASH" begin
        x, s = single(data(64, 48)), Int32[7]
        p = trace((x, s) -> Ops.dropout(x, s; rate = 0.25), x, s)
        ex = prepare(p, CuArray(x), CuArray(s); target = T.CUDATarget(cuTile))
        @test_broken !(only(ex.kernels) isa C.FloorKernels)
        @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, x, s)])
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
# the batch alone, and its scatter sums by key in the same kernel.
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
            @test !any(k -> k isa C.FloorKernels, ex.kernels)
            @test agree(filter(!isnothing, C.leaves(outputs(execute!(ex)))), filter(!isnothing, C.leaves(interpret(p, args...))); rtol = 1.0e-4)
        end
    end
end

# A gather's adjoint scatters on the device, a sum by key in the kernel that
# computes its elements, on cuTile and on the floor alike: each element adds
# atomically, or on cuTile each box of a row's elements, so the sums at an
# index selected more than once come in no fixed order and differ from the
# interpreter's by rounding, which is the order every kernel is held to. With
# rows narrower than a tile and fewer tokens than one, repeated indices, two
# index arrays, and ranges and integers beside an index array.
const SCATTERS = [
    ("embedding", (E, t, g) -> sum(E[:, t] .* g), (data(64, 109), Int32.(mod1.(37 .* (1:1024) .^ 2, 109)), data(64, 1024; scale = 0.37))),
    ("embedding of short rows", (E, t, g) -> sum(E[:, t] .* g), (data(4, 5), Int32[2, 5, 1, 2, 3, 5, 5], data(4, 7))),
    ("repeated indices in two arrays", (x, i, j) -> sum(abs2, x[i, j]), (data(8, 6), [3, 1, 3, 8, 3], Int32[6, 6, 1])),
    ("a strided range and an integer beside index arrays", x -> sum(abs2, x[2:2:8, [1, 4, 4]]) + sum(abs2, x[3, [2, 2]]), (data(8, 6),)),
    ("a reversed range", x -> sum(abs2, x[8:-3:1, [2, 2, 5]]), (data(8, 6),)),
]

if CUDA_FUNCTIONAL
    @testset "scatters on the device" begin
        @testset "$name, $where" for (name, f, args) in SCATTERS, (where, target) in (("cuTile", T.CUDATarget(cuTile)), ("the floor", T.CUDATarget(:full, true, ())))
            args = map(single, args)
            p = differentiate(trace(f, args...))
            ex = prepare(p, map(CuArray, args)...; target)
            scatters = [k for (k, g) in enumerate(ex.plan.groups) if any(r -> ex.plan.graph.values[r].definition isa C.Keyed, g.reductions)]
            @test !isempty(scatters) && all(k -> (ex.kernels[k] isa C.FloorKernels) == (where == "the floor"), scatters)
            # An embedding's rows add on cuTile as boxes, a TMA reduction each.
            if where == "cuTile" && name == "embedding"
                ext = Base.get_extension(Tempest, :cuTileExt)
                @test all(k -> any(t -> occursin("atomic_store_add", string(ext.source(t))), ex.kernels[k].kernels), scatters)
            end
            reference = C.leaves(interpret(p, args...))[2:2]
            @test agree(C.leaves(outputs(execute!(ex)))[2:2], reference; rtol = 1.0e-5)
            # A scatter runs on the device, so the program can be captured.
            capture!(ex)
            @test agree(C.leaves(outputs(replay!(ex)))[2:2], reference; rtol = 1.0e-5)
        end
    end
end

# Writes into arrays the program holds, at positions it is given, on the
# device: a token's keys appended to a cache per sequence, a column at a
# position, and a write's gradient, at distinct positions as the interpreter
# writes them; where elements land at one position, each position holds one
# of them, which the device does not fix. A cache the step replaces is
# appended in place.
const DEVICE_WRITES = [
    ("a cache appended per sequence", appended, (data(4, 16, 2, 3), data(4, 2, 3), Int32[3, 16, 1])),
    ("a column at a position held", (K, k, p) -> (y = copy(K); y[:, p[1]] .= k; y), (data(8, 16), data(8), Int32[5])),
    ("a loss through a write", (x, v, i) -> (y = copy(x); y[:, i] .= v; sum(abs2, tanh.(y))), (data(4, 9), data(4, 3), Int32[7, 2, 5])),
]

if CUDA_FUNCTIONAL
    @testset "writes on the device" begin
        targets = (("cuTile", T.CUDATarget(cuTile)), ("the floor", T.CUDATarget(:full, true, ())))
        @testset "$name, $where" for (name, f, args) in DEVICE_WRITES, (where, target) in targets
            args = map(single, args)
            traced = trace(f, args...)
            for p in (f(args...) isa Number ? (traced, differentiate(traced)) : (traced,))
                ex = prepare(p, map(CuArray, args)...; target)
                @test all(k -> (k isa C.FloorKernels) == (where == "the floor"), ex.kernels)
                @test agree(filter(!isnothing, C.leaves(outputs(execute!(ex)))), filter(!isnothing, C.leaves(interpret(p, args...))); rtol = 1.0e-5)
            end
        end
        @testset "a cache appended in place, $where" for (where, target) in targets
            check_appended_in_place(CuArray, target)
        end
        x, v, i = single(data(64, 9)), single(data(64, 300)), Int32.(mod1.(1:300, 7))
        @testset "repeated positions, $where" for (where, target) in targets
            ex = prepare(trace((x, v, i) -> (y = copy(x); y[:, i] .= v; y), x, v, i), CuArray(x), CuArray(v), CuArray(i); target)
            y = Array(outputs(execute!(ex)))
            @test all(((f, j),) -> any(a -> y[f, j] == v[f, a], findall(==(j), i)), Iterators.product(1:64, 1:7))
            @test y[:, 8:9] == x[:, 8:9]
        end
    end
end

# Attention generated from its reference: one kernel, a fold over keys whose
# scores and logsumexp are recomputed in it, online or in two passes.
const GENERATED_ATTENTION = [
    ("causal", (16, 2, 2, 200, 2), true),
    ("causal, grouped heads", (16, 4, 2, 200, 2), true),
    ("grouped heads", (16, 4, 2, 200, 2), false),
    ("causal, odd sizes", (32, 3, 1, 70, 3), true),
    ("one head", (8, 1, 1, 133, 1), false),
]

# The folds a program's streaming plan recomputes where they are read, as
# numbered in the graph `prepare` plans for `target`.
streaming(p, target) = C.recomputable_folds(C.permits_rewrites(target) ? C.rewrite(C.lower(p)) : C.lower(p))

if CUDA_FUNCTIONAL
    @testset "attention generated on cuTile" begin
        @testset "$name" for (name, (D, H, KV, S, B), causal) in GENERATED_ATTENTION
            q, k, v = single(data(D, H, S, B; scale = 0.1)), single(data(D, KV, S, B; scale = 0.2)), single(data(D, KV, S, B; scale = 0.3))
            p = trace((q, k, v) -> Ops.attention(q, k, v; causal), q, k, v)
            for online in (true, false)
                target = T.CUDATarget(cuTile; online)
                ex = prepare(p, map(CuArray, (q, k, v))...; target, recompute = streaming(p, target))
                kernel = only(ex.kernels)
                @test !(kernel isa C.FloorKernels)
                @test occursin("online softmax", string(kernel)) == online
                # A causal mask is a formula: the steps it rules out are
                # skipped, where the queries have a tile axis of their own.
                # Grouped heads may share it with the group, which the
                # formula's bound over a tile cannot see through.
                (!causal || H == KV) && @test occursin("masked steps skipped", string(kernel)) == causal
                @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, q, k, v)]; rtol = 1.0e-4)
            end
        end
        # Softmax spelled with a maximum and a sum: rewritten into the same one
        # kernel, at full precision and in TF32.
        q, k, v = single(data(16, 2, 70, 2; scale = 0.1)), single(data(16, 2, 70, 2; scale = 0.2)), single(data(16, 2, 70, 2; scale = 0.3))
        p = trace(q, k, v) do q, k, v
            s = C.contract(k, q, (1, 2, 3, 4), (1, 2, 5, 4), (3, 2, 5, 4)) .* 0.25f0
            s = ifelse.(reshape(1:size(s, 1), :, 1, 1) .<= reshape(1:size(s, 3), 1, 1, :), s, -Inf32)
            e = exp.(s .- maximum(s; dims = 1))
            C.contract(v, e ./ sum(e; dims = 1), (1, 2, 3, 4), (3, 2, 5, 4), (1, 2, 5, 4))
        end
        for precision in (:full, :tf32)
            target = T.CUDATarget(cuTile; precision)
            ex = prepare(p, map(CuArray, (q, k, v))...; target, recompute = streaming(p, target))
            @test occursin("online softmax", string(only(ex.kernels)))
            @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, q, k, v)]; rtol = precision === :tf32 ? 1.0e-2 : 1.0e-4)
        end
        # In half precision the rewrite reads the exponential through the
        # product's conversion to Float32: one kernel, online, the weights
        # rounded to BFloat16 for the tensor cores.
        qh, kh, vh = map(x -> BFloat16.(single(x)), (data(16, 2, 70, 2; scale = 0.1), data(16, 2, 70, 2; scale = 0.2), data(16, 2, 70, 2; scale = 0.3)))
        p = trace((q, k, v) -> Ops.attention(q, k, v; causal = true), qh, kh, vh)
        ex = prepare(p, map(CuArray, (qh, kh, vh))...; target = T.CUDATarget(cuTile), recompute = streaming(p, T.CUDATarget(cuTile)))
        @test occursin("online softmax", string(only(ex.kernels)))
        @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, qh, kh, vh)]; rtol = 1.0e-2)
        # The reference's gradient: every product generated, also those that
        # sum over key heads shared by one query head each.
        q, k, v = single(data(16, 2, 70, 2; scale = 0.1)), single(data(16, 2, 70, 2; scale = 0.2)), single(data(16, 2, 70, 2; scale = 0.3))
        d = differentiate(trace((q, k, v) -> sum(abs2, Ops.attention(q, k, v; causal = true)), q, k, v))
        ex = prepare(d, map(CuArray, (q, k, v))...; target = T.CUDATarget(cuTile))
        @test all(d -> !(d isa C.FloorKernels), ex.kernels)
        @test agree(filter(!isnothing, C.leaves(outputs(execute!(ex)))), filter(!isnothing, C.leaves(interpret(d, q, k, v))); rtol = 1.0e-4)
        # Its streaming plan, as FlashAttention's backward: the scores and
        # their gradient recomputed in every kernel that reads them, the
        # log-sum-exp in a sweep over keys of its own, and nothing the size of
        # the scores stored. The key gradients sum over the queries after each
        # key and the query gradients over the keys before each query, in one
        # sweep, each fold stepping where its mask leaves it work.
        ex = prepare(d, map(CuArray, (q, k, v))...; target = T.CUDATarget(cuTile), recompute = streaming(d, T.CUDATarget(cuTile)))
        @test all(d -> !(d isa C.FloorKernels), ex.kernels)
        @test all(v -> !ex.plan.stored[v] || prod(ex.plan.graph.values[v].dims) < 70 * 70, eachindex(ex.plan.graph.values))
        @test any(d -> occursin("folds stepping by their masks", string(d)), ex.kernels)
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
        ex = prepare(p, map(CuArray, (q, k, v, x))...; target = T.CUDATarget(cuTile), recompute = streaming(p, T.CUDATarget(cuTile)))
        @test length(ex.kernels) == 1 && !(only(ex.kernels) isa C.FloorKernels)
        @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, q, k, v, x)]; rtol = 1.0e-4)
    end

    # A layer's gradient whose sums recompute the hidden product they read:
    # generated as folds that recompute it, stepping at a product's depths.
    @testset "sums that recompute a product, generated" begin
        x, w1, b1 = single(data(64, 256; scale = 0.5)), single(data(256, 64; scale = 0.05)), single(data(256; scale = 0.1))
        w2, b2 = single(data(64, 256; scale = 0.05)), single(data(64; scale = 0.1))
        d = differentiate(trace((x, w1, b1, w2, b2) -> sum(abs2, w2 * tanh.(w1 * x .+ b1) .+ b2), x, w1, b1, w2, b2))
        target = T.CUDATarget(cuTile)
        ex = prepare(d, map(CuArray, (x, w1, b1, w2, b2))...; target, recompute = streaming(d, target))
        @test all(k -> !(k isa C.FloorKernels), ex.kernels)
        @test agree(filter(!isnothing, C.leaves(outputs(execute!(ex)))), filter(!isnothing, C.leaves(interpret(d, x, w1, b1, w2, b2))); rtol = 1.0e-4)
    end

    # Decode attention: one query per head over a cache of keys, each
    # sequence's read up to a length the device holds, with query heads
    # grouped over key heads as Vallmo's cache holds them, (features,
    # positions, key heads, sequences). The length is a mask whose formula
    # reads it, and the sweep over keys stops where it ends, whether the
    # scores are recomputed in it or stored before.
    @testset "attention up to lengths the device holds" begin
        function decode(q, k, v, len)
            s = Einops.@einsum(k, q, "d n h b, d (g h) b -> n g h b", h = size(k, 3)) .* 0.25f0
            s = ifelse.((Int32(1):Int32(size(k, 2))) .<= reshape(len, 1, 1, 1, :), s, -Inf32)
            Einops.@einsum(v, exp.(s .- T.logsumexp(s; dims = 1)), "d n h b, n g h b -> d (g h) b")
        end
        q, k, v = single(data(16, 8, 3; scale = 0.1)), single(data(16, 200, 2, 3; scale = 0.2)), single(data(16, 200, 2, 3; scale = 0.3))
        n = Int32[37, 200, 1]
        p = trace(decode, q, k, v, n)
        target = T.CUDATarget(cuTile)
        for recompute in ((), streaming(p, target))
            ex = prepare(p, map(CuArray, (q, k, v, n))...; target, recompute)
            @test any(k -> occursin("masked steps skipped", string(k)), ex.kernels)
            @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, q, k, v, n)]; rtol = 1.0e-4)
            # Another length is the same program.
            bind!(ex, 4 => CuArray(Int32[200, 5, 64]))
            @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, q, k, v, Int32[200, 5, 64])]; rtol = 1.0e-4)
        end
        # Its claims are timed at the cache's length, so that the sweep runs
        # whole, as it is priced, whatever lengths it is prepared with.
        graph = C.lower(p)
        planned = C.plan(graph)
        d = C.Decider(target, graph; measure = false)
        foreach(g -> C.buffers!(d, planned, g), planned.groups)
        @test Array(d.buffers[only(i for i in graph.arguments if graph.values[i].eltype === Int32)]) == fill(Int32(200), 3)
        # Scores stored with their mask, as a step whose layers share one mask
        # stores them: a block whose positions all lie past the length sweeps
        # no features and stores minus infinity.
        scored(q, k, v, len) = (s = Einops.@einsum(k, q, "d n h b, d (g h) b -> n g h b", h = size(k, 3)) .* 0.25f0;
            s = ifelse.((Int32(1):Int32(size(k, 2))) .<= reshape(len, 1, 1, 1, :), s, -Inf32);
            (Einops.@einsum(v, exp.(s .- T.logsumexp(s; dims = 1)), "d n h b, n g h b -> d (g h) b"), s))
        q3, k3, v3 = single(data(64, 8, 3; scale = 0.1)), single(data(64, 200, 2, 3; scale = 0.2)), single(data(64, 200, 2, 3; scale = 0.3))
        p = trace(scored, q3, k3, v3, n)
        ex = withenv(() -> prepare(p, map(CuArray, (q3, k3, v3, n))...; target), "TEMPEST_PRODUCT_TILE" => "64,4,16")
        @test any(k -> occursin("masked tiles skipped", string(k)), ex.kernels)
        for m in (n, Int32[200, 5, 64])
            bind!(ex, 4 => CuArray(m))
            (o, s), (o′, s′) = map(Array, C.leaves(outputs(execute!(ex)))), interpret(p, q3, k3, v3, m)
            @test agree([o], [o′]; rtol = 1.0e-4)
            @test isinf.(s) == isinf.(s′) && agree([s[isfinite.(s)]], [s′[isfinite.(s′)]]; rtol = 1.0e-4)
        end
        # Over a BFloat16s.BFloat16 cache, the probabilities rounded to it
        # before their product, as Vallmo's decode rounds them: the product
        # reads the length only through their mask.
        half(q, k, v, len) = (s = Einops.@einsum(k, q, "d n h b, d (g h) b -> n g h b", h = size(k, 3)) .* 0.25f0;
            s = ifelse.((Int32(1):Int32(size(k, 2))) .<= reshape(len, 1, 1, 1, :), s, -Inf32);
            Einops.@einsum(v, BFloat16s.BFloat16.(exp.(s .- T.logsumexp(s; dims = 1))), "d n h b, n g h b -> d (g h) b"))
        qh, kh, vh = BFloat16s.BFloat16.(q), BFloat16s.BFloat16.(k), BFloat16s.BFloat16.(v)
        p = trace(half, qh, kh, vh, n)
        ex = prepare(p, map(CuArray, (qh, kh, vh, n))...; target)
        @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, qh, kh, vh, n)]; rtol = 1.0e-2)
        # Scores whose query a mask in memory chooses lane by lane, as a
        # rotation's stored mask does, stepping 128 deep along the features:
        # tileiras 13.4 gets the product wrong when the mask is loaded as Bool,
        # so it is loaded as bytes.
        chosen(q, k, m) = Einops.@einsum(k, BFloat16s.BFloat16.(ifelse.(m, q, -q)), "d n h b, d (g h) b -> n g h b", h = size(k, 3))
        q2, k2 = single(data(256, 8, 1; scale = 0.1)), BFloat16s.BFloat16.(single(data(256, 128, 2, 1; scale = 0.2)))
        m = isodd.(reshape(1:length(q2), size(q2)) .÷ 3)
        p = trace(chosen, q2, k2, m)
        ex = withenv(() -> prepare(p, map(CuArray, (q2, k2, m))...; target), "TEMPEST_PRODUCT_TILE" => "128,4,128")
        @test occursin("tile 128×4×128", string(only(ex.kernels)))
        @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, q2, k2, m)]; rtol = 1.0e-2)
        # With a key head per query head, the scores of one query are a product
        # of one row, no product of tiles: its sweep over keys is a fold that
        # recomputes them, generated as one.
        k1, v1 = single(data(16, 200, 8, 3; scale = 0.2)), single(data(16, 200, 8, 3; scale = 0.3))
        p = trace(decode, q, k1, v1, n)
        ex = prepare(p, map(CuArray, (q, k1, v1, n))...; target, recompute = streaming(p, target))
        @test !(only(ex.kernels) isa C.FloorKernels)
        @test agree(C.leaves(outputs(execute!(ex))), [interpret(p, q, k1, v1, n)]; rtol = 1.0e-4)
    end
end
