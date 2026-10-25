# The storage of prepared programs: values share slots where their lifetimes
# allow, a write takes the storage of the contents it starts from, and results
# replace the arguments they are prepared to replace, a training step's
# parameters and Adam's state among them; on the CPU, and on CUDA with cuTile
# and the floor.
include("../fixtures/programs.jl")

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

if CUDA_FUNCTIONAL
    @testset "a training step as one program, on CUDA" begin
        check_steps(CuArray; rtol = 1.0e-3, by_reference = (1, 2))
        check_steps(CuArray; rtol = 1.0e-3)
        check_steps(CuArray; rtol = 1.0e-3, by_reference = (1, 2), captured = true)
        ex = check_steps(CuArray; rtol = 1.0e-3, captured = true)
        @test ex.captured !== nothing
        check_replacing(CuArray)
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
