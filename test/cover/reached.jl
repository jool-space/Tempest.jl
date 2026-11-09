# Decisions kept and replayed (see `Tempest.Compiler.reach!`): a program
# prepared again replays what its cover reached, planning once and pricing
# nothing, and the key and the evidence they rest on miss on every change a
# decision reads. Each testset keeps its decisions, record and rates in a
# directory of its own.

# `f()` with the tuning directory a new one, holding the device's rates file
# where there is a device, and the preparations verifying or not.
function kept(f; verify = false)
    rates = CUDA_FUNCTIONAL ? C.rates_path(T.CUDATarget()) : nothing
    withenv("TEMPEST_CACHE" => mktempdir(), "TEMPEST_VERIFY" => string(verify)) do
        if rates !== nothing && isfile(rates)
            mkpath(C.tuning_directory())
            cp(rates, C.rates_path(T.CUDATarget()))
        end
        f()
    end
end

# What `f()` returns, with the deciders it replayed and those the cover decided.
function counted(f)
    before = C.reached_counts()
    x = f()
    after = C.reached_counts()
    (x, after.replayed - before.replayed, after.covered - before.covered)
end

decisions_files() = (dir = joinpath(C.tuning_directory(), "decisions"); isdir(dir) ? readdir(dir; join = true) : String[])

# Decisions as a named tuple of their fields.
fields(r) = NamedTuple{fieldnames(C.Reached)}(map(n -> getfield(r, n), fieldnames(C.Reached)))

# Decisions as kept, changed by `change(fields)`, and kept again.
rekeep(change, path) = C.store_reached(C.Reached(change(fields(C.load_reached(basename(path))))...))

mlp(x, w) = sum(abs2, tanh.(w * x) .+ 1)

@testset "a program prepared again replays what its cover reached" begin
    x, w = data(8, 4), data(8, 8)
    kept() do
        a, replayed, covered = counted(() -> prepare(trace(mlp, x, w), x, w))
        @test (replayed, covered) == (0, 1)
        @test length(decisions_files()) == 1
        b, replayed, covered = counted(() -> prepare(trace(mlp, x, w), x, w))
        @test (replayed, covered) == (1, 0)
        @test sprint(explain, a) == sprint(explain, b)
        @test agree(C.leaves(outputs(execute!(b))), [mlp(x, w)])
        r = C.load_reached(basename(only(decisions_files())))
        @test r.settled && r.how === :none && length(r.groups) == length(b.plan.groups)
    end
    # Verifying, a preparation replays and reaches them by the cover too,
    # and throws where the two differ.
    kept(verify = true) do
        prepare(trace(mlp, x, w), x, w)
        _, replayed, covered = counted(() -> prepare(trace(mlp, x, w), x, w))
        @test (replayed, covered) == (1, 1)
        path = only(decisions_files())
        rekeep(r -> merge(r, (; groups = [r.groups[1] * " "; r.groups[2:end]])), path)
        @test occursin("replayed differ", message(() -> prepare(trace(mlp, x, w), x, w)))
        rekeep(r -> merge(r, (; decisions = reverse(r.decisions))), path)
        @test occursin("replayed differ", message(() -> prepare(trace(mlp, x, w), x, w)))
    end
    # Unverified, decisions that no longer form the plan kept are the cover's to reach.
    kept() do
        prepare(trace(mlp, x, w), x, w)
        rekeep(r -> merge(r, (; groups = [r.groups[1] * " "; r.groups[2:end]])), only(decisions_files()))
        _, replayed, covered = counted(() -> prepare(trace(mlp, x, w), x, w))
        @test (replayed, covered) == (0, 1)
    end
end

@testset "a recurrence's steps are replayed with the program" begin
    function interior(u)
        i = reshape(Int32(1):Int32(size(u, 1)), :, 1)
        j = reshape(Int32(1):Int32(size(u, 2)), 1, :)
        (i .> 1) .& (i .< size(u, 1)) .& (j .> 1) .& (j .< size(u, 2))
    end
    function diffuse(u, inside)
        up, down = vcat(u[1:1, :], u[1:(end - 1), :]), vcat(u[2:end, :], u[end:end, :])
        left, right = hcat(u[:, 1:1], u[:, 1:(end - 1)]), hcat(u[:, 2:end], u[:, end:end])
        ifelse.(inside, u .+ eltype(u)(0.2) .* (up .+ down .+ left .+ right .- 4 .* u), u)
    end
    evolve(u, q) = (inside = interior(u); last(accumulate((v, s) -> diffuse(v, inside) .+ s, eachslice(q; dims = 3); init = u)))
    heat(u, q) = T.withgradient(v -> sum(abs2, evolve(v, q)), u)
    args = (data(8, 8), data(8, 8, 5; scale = 0.01))
    kept() do
        a, replayed, covered = counted(() -> prepare(trace(heat, args...), args...))
        @test (replayed, covered) == (0, 3)
        @test length(C.load_reached(basename(only(decisions_files()))).steps) == 2
        b, replayed, covered = counted(() -> prepare(trace(heat, args...), args...))
        @test (replayed, covered) == (3, 0)
        @test count(k -> k isa C.SteppedKernels, b.kernels) == 2
        @test sprint(explain, a) == sprint(explain, b)
        @test agree(C.leaves(outputs(execute!(b))), C.leaves(heat(args...)); rtol = 1.0e-10)
    end
    # A step the decisions kept does not hold is the cover's, and the program's with it.
    kept() do
        prepare(trace(heat, args...), args...)
        rekeep(r -> merge(r, (; steps = r.steps[1:1])), only(decisions_files()))
        _, replayed, covered = counted(() -> prepare(trace(heat, args...), args...))
        @test covered == 3
    end
end

key(f, args...; target = T.CPUTarget(), kw...) = C.decisions_key(C.Decider(target, C.lower(trace(f, args...)); kw...))

@testset "the key differs for every change a decision reads" begin
    x = data(8, 4)
    kept() do
        @test key(x -> sum(abs2, x), x) == key(x -> sum(abs2, x), x)
        @test key(x -> sum(abs2, x), x) != key(x -> sum(abs2, x), data(8, 5))
        @test key(x -> sum(abs2, x), x) != key(x -> sum(abs2, x), single(x))
        # A closure is its method and what it captures: a number, a
        # function, an array.
        scaled(a) = x -> sum(x .* a)
        @test key(scaled(2.0), x) == key(scaled(2.0), x)
        @test key(scaled(2.0), x) != key(scaled(3.0), x)
        applied(g) = x -> sum(g.(x))
        @test key(applied(sin), x) != key(applied(cos), x)
        gathered(i) = x -> sum(x[i, :])
        @test key(gathered([1, 3]), x) != key(gathered([2, 3]), x)
        # What the decider is given: folds pinned or kept, sums riding, the
        # steps of a step, how it measures.
        graph = C.lower(trace(x -> sum(abs2, x .* sum(x; dims = 1)), x))
        fold = findfirst(v -> v.definition isa C.Reduction, graph.values)
        given = [C.decisions_key(C.Decider(T.CPUTarget(), graph; kw...))
                 for kw in ((;), (; recompute = (fold,)), (; keep = (fold,)), (; steps = 2), (; measure = :dry))]
        @test allunique(given)
        # The target and its numerics, without a device when nothing is measured.
        cuda(precision) = C.decisions_key(C.Decider(T.CUDATarget(; precision), graph; measure = :dry))
        @test cuda(:full) != cuda(:tf32) && cuda(:full) != given[end]
        # The device by its name, and Tempest's own code by its identity.
        text = C.canonical_text(C.key_parts(C.Decider(T.CPUTarget(), graph))...)
        @test occursin(repr(C.device_name(T.CPUTarget())), text)
        @test occursin(C.package_identity(Tempest), text)
        # A setting an implementation reads.
        @test withenv(() -> key(x -> sum(abs2, x), x), "TEMPEST_PRODUCT_TILE" => "64,64,32") != key(x -> sum(abs2, x), x)
    end
    # A result replacing an argument a fold reads keeps that fold stored: another key.
    update(w, x) = (sum(w .* x), w .- 0.1 .* x)
    w = data(8, 4; scale = 0.5)
    kept() do
        prepare(trace(update, w, x), w, x)
        _, replayed, covered = counted(() -> prepare(trace(update, w, x), w, x; replaces = (2 => 1,)))
        @test (replayed, covered) == (0, 1)
        @test length(decisions_files()) == 2
    end
end

# Claims of an implementation outside any package are methods the key cannot name.
struct Elsewhere <: C.Implementation end

@testset "decisions an implementation outside any package reached are not kept" begin
    x, w = data(8, 4), data(8, 8)
    kept() do
        @test key(mlp, x, w; target = T.CPUTarget(Elsewhere())) === nothing
        prepare(trace(mlp, x, w), x, w; target = T.CPUTarget(Elsewhere()))
        @test isempty(decisions_files())
    end
end

@testset "decisions rest on the record and the rates they read" begin
    x, w = data(8, 4), data(8, 8)
    kept() do
        prepare(trace(mlp, x, w), x, w)
        r = C.load_reached(basename(only(decisions_files())))
        @test !isempty(r.record) && all(isnothing ∘ last, r.record)
        # A time recorded since for a claim they read.
        C.save_record(C.record_path(T.CPUTarget()), first(first(r.record)), 1.0e-6)
        _, replayed, covered = counted(() -> prepare(trace(mlp, x, w), x, w))
        @test (replayed, covered) == (0, 1)
        # Reached again on that record, they are replayed.
        _, replayed, covered = counted(() -> prepare(trace(mlp, x, w), x, w))
        @test (replayed, covered) == (1, 0)
    end
    # Reached while something was measured, fitted or failed to build, they
    # are not settled: the cover reaches them again.
    kept() do
        prepare(trace(mlp, x, w), x, w)
        rekeep(r -> merge(r, (; settled = false)), only(decisions_files()))
        _, replayed, covered = counted(() -> prepare(trace(mlp, x, w), x, w))
        @test (replayed, covered) == (0, 1)
    end
    if CUDA_FUNCTIONAL
        xs = map(x -> CuArray(single(x)), (x, w))
        kept() do
            prepare(trace(mlp, map(single, (x, w))...), xs...)
            r = C.load_reached(basename(only(decisions_files())))
            @test !isempty(r.calibration)
            # A rate fitted again: the rates file's last line for a key stands.
            # Fitted again to the value it had, as a rate of zero doubled is,
            # it changes no value the decisions read: they replay.
            rate, value = first(filter(!isnothing ∘ last, r.calibration))
            C.save_record(C.rates_path(T.CUDATarget()), rate, value)
            _, replayed, covered = counted(() -> prepare(trace(mlp, map(single, (x, w))...), xs...))
            @test (replayed, covered) == (1, 0)
            # Doubled, a rate other than zero is another: they do not.
            rate, value = first(filter(((_, v),) -> !isnothing(v) && !iszero(v), r.calibration))
            C.save_record(C.rates_path(T.CUDATarget()), rate, 2 * value)
            _, replayed, covered = counted(() -> prepare(trace(mlp, map(single, (x, w))...), xs...))
            @test (replayed, covered) == (0, 1)
        end
    end
end

@testset "a preparation measuring nothing replays the settled decisions of one measuring close calls" begin
    x, w = data(8, 4), data(8, 8)
    kept() do
        prepare(trace(mlp, x, w), x, w)
        path = only(decisions_files())
        d = C.Decider(T.CPUTarget(), C.rewrite(C.lower(trace(mlp, x, w)); rounded = C.rounded_backward(T.CPUTarget())))
        @test C.decisions_key(d) == basename(path)
        whole = fields(C.load_reached(basename(path)))
        # Its own unsettled, it replays only those kept as a decider
        # measuring close calls would keep them, not every option.
        rekeep(r -> merge(r, (; settled = false)), path)
        for (level, replays) in ((:close, true), (:all, false))
            key = C.decisions_key(d; measure = level)
            C.store_reached(C.Reached(merge(whole, (; key, how = level))...))
            _, replayed, covered = counted(() -> prepare(trace(mlp, x, w), x, w))
            @test (replayed, covered) == (replays ? (1, 0) : (0, 1))
            rm(joinpath(dirname(path), key))
        end
    end
end

@testset "a table in memory holds what its file reads back" begin
    # The record and the rates file have one writer, `keep!`: what it keeps,
    # whatever a target's timer returned, a decider in another process reads
    # back as it was kept, and the evidence noted on the one holds on the
    # other. A Float32 printed as itself would read back as another value.
    entries = ["a Float32" => 4.0255986f-7, "a third" => 1 / 3, "zero" => 0.0f0, "tiny" => 1.0e-300, "an integer" => 3]
    kept() do
        for path in (C.record_path(T.CPUTarget()), C.rates_path(T.CPUTarget()))
            table = C.Noted()
            for (key, value) in entries
                @test C.keep!(table, path, key, value) === Float64(value)
            end
            foreach(((key, _),) -> table[key], entries)
            again = C.Noted(C.load_record(path))
            @test again.table == table.table
            @test C.holds(again, C.evidence(table))
        end
    end
end

@testset "a cover that wrote as it decided is followed by another, until one writes nothing" begin
    x, w = data(8, 4), data(8, 8)
    graph = C.lower(trace(mlp, x, w))
    kept() do
        for (settle, covers) in ((false, 1), (true, 2))
            d = C.Decider(T.CPUTarget(), graph)
            # A time measured as the cover decided.
            d.record["a claim measured"] = 1.0e-6
            _, replayed, covered = counted(() -> C.reach!(d; settle))
            @test (replayed, covered) == (0, covers)
            @test C.written(d.record) == !settle
        end
    end
end

@testset "a decisions file cut short, or another's, is not read" begin
    x, w = data(8, 4), data(8, 8)
    kept() do
        prepare(trace(mlp, x, w), x, w)
        path = only(decisions_files())
        whole = read(path)
        for bytes in (whole[1:(end ÷ 2)], whole[1:10], rand(UInt8, 64), UInt8[])
            write(path, bytes)
            @test C.load_reached(basename(path)) === nothing
            _, replayed, covered = counted(() -> prepare(trace(mlp, x, w), x, w))
            @test (replayed, covered) == (0, 1)
            @test C.load_reached(basename(path)) !== nothing
        end
        # Another program's decisions under this one's key.
        prepare(trace((x, w) -> sum(w * x), x, w), x, w)
        other = only(filter(!=(path), decisions_files()))
        cp(other, path; force = true)
        @test C.load_reached(basename(path)) === nothing
        _, replayed, covered = counted(() -> prepare(trace(mlp, x, w), x, w))
        @test (replayed, covered) == (0, 1)
        @test !any(endswith(".part"), decisions_files())
    end
end

if CUDA_FUNCTIONAL
    @testset "a preparation replayed on the device compiles, measures and plans nothing" begin
        x, w = single(data(64, 32)), single(data(64, 64))
        xs = (CuArray(x), CuArray(w))
        kept() do
            a = prepare(trace(mlp, x, w), xs...)
            record = C.record_path(T.CUDATarget())
            lines = isfile(record) ? countlines(record) : 0
            compiled = C.cache_counts().compiled
            b, replayed, covered = counted(() -> prepare(trace(mlp, x, w), xs...))
            @test (replayed, covered) == (1, 0)
            @test C.cache_counts().compiled == compiled
            @test (isfile(record) ? countlines(record) : 0) == lines
            @test sprint(explain, a) == sprint(explain, b)
            @test isapprox(only(Array(only(C.leaves(outputs(execute!(b)))))), mlp(x, w); rtol = 1.0e-4)
        end
        # What a preparation measuring close calls settled, one measuring
        # nothing replays, and verifying, its cover reaches it too.
        kept(verify = true) do
            prepare(trace(mlp, x, w), xs...; measure = :close)
            _, replayed, covered = counted(() -> prepare(trace(mlp, x, w), xs...; measure = :none))
            @test (replayed, covered) == (1, 1)
        end
    end
end
