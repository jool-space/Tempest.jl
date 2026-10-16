# Prepared programs compute what programs compute, through generated kernels,
# on the CPU (and on CUDA when it is loaded and functional).
using Test
include("cases.jl")

scalar(x) = ndims(x) == 0 ? x[] : x
agree(a, b; rtol = 1.0e-10) = all(map((x, y) -> isapprox(scalar(Array(x)), y; rtol, atol = rtol * 1.0e-2), a, b))
host(x) = x isa AbstractArray ? x : x

single(x) = x isa AbstractArray{<:AbstractFloat} ? Float32.(x) : x isa AbstractFloat ? Float32(x) : x

function check_prepared(device, expected_target; rtol = 1.0e-10, precision = identity)
    @testset "$name" for (name, f, args) in PROGRAMS
        args = map(precision, args)
        for p in (trace(f, args...), differentiate(trace(f, args...)))
            reference = filter(!isnothing, T.leaves(interpret(p, args...)))
            ex = prepare(p, map(device, args)...)
            @test ex.target == expected_target
            got = filter(!isnothing, T.leaves(execute!(ex)))
            @test agree(got, reference; rtol)
        end
    end
end

@testset "prepared on the CPU" begin
    check_prepared(identity, T.CPUTarget())
end

using CUDACore
if CUDACore.functional()
    @testset "prepared on CUDA" begin
        check_prepared(x -> x isa AbstractArray ? CuArray(x) : x, T.CUDATarget())
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
            @test any(d -> !(d isa T.FloorKernels) && d.schedule.sketch === :product, ex.kernels)
            @test isapprox(Array(only(T.leaves(execute!(ex)))), interpret(p, w, x, b); rtol = target.precision === :tf32 ? 1.0e-2 : 1.0e-4)
        end
        # A conversion to BFloat16 is part of the epilogue: one kernel, and
        # only the BFloat16 result is stored.
        p = trace((w, x, b) -> Core.BFloat16.(gelu.(w * x .+ b)), w, x, b)
        ex = prepare(p, CuArray(w), CuArray(x), CuArray(b); target = T.CUDATarget(cuTile; precision = :tf32))
        @test length(ex.kernels) == 1 && only(ex.kernels).schedule.sketch === :product
        y = only(T.leaves(execute!(ex)))
        @test eltype(y) === Core.BFloat16
        @test isapprox(Float32.(Array(y)), Float32.(interpret(p, w, x, b)); rtol = 1.0e-2)
        # Half operands accumulate in Float32 and round once.
        h, y = Float16.(data(64, 48; scale = 0.1)), Float16.(data(48, 32))
        p = trace((h, y) -> h * y, h, y)
        ex = prepare(p, CuArray(h), CuArray(y); target = T.CUDATarget(cuTile))
        @test any(d -> !(d isa T.FloorKernels) && d.schedule.sketch === :product, ex.kernels)
        @test isapprox(Array(only(T.leaves(execute!(ex)))), interpret(p, h, y); rtol = eps(Float16))
        # A product spelled as a broadcast and a sum is a product too.
        a, c = Float32.(data(64, 48; scale = 0.1)), Float32.(data(48, 80))
        ex = prepare(trace(spelled_product, a, c), CuArray(a), CuArray(c); target = T.CUDATarget(cuTile))
        @test any(d -> !(d isa T.FloorKernels) && d.schedule.sketch === :product, ex.kernels)
        @test isapprox(Array(only(T.leaves(execute!(ex)))), a * c; rtol = 1.0e-4)
        # A long sum over a small output splits among blocks (split-K) and merges.
        u, z = Float32.(data(64, 3000; scale = 0.01)), Float32.(data(3000, 48))
        for target in (T.CUDATarget(cuTile), T.CUDATarget(cuTile; precision = :tf32))
            p = trace((u, z) -> u * z, u, z)
            ex = prepare(p, CuArray(u), CuArray(z); target)
            @test any(d -> !(d isa T.FloorKernels) && d.schedule.sketch === :product && d.schedule.splits > 1, ex.kernels)
            @test isapprox(Array(only(T.leaves(execute!(ex)))), interpret(p, u, z); rtol = target.precision === :tf32 ? 1.0e-2 : 1.0e-4)
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
            ((a, b) -> a * b, (Float16.(a), Float16.(b))), ((a, b) -> a * b, (Core.BFloat16.(a), Core.BFloat16.(b)))]
        for (f, args) in layouts, target in (T.CUDATarget(cuBLAS), T.CUDATarget(cuBLAS; precision = :tf32))
            p = trace(f, args...)
            ex = prepare(p, map(CuArray, args)...; target)
            @test length(built(ex)) == 1
            rtol = eltype(args[1]) === Float32 ? (target.precision === :tf32 ? 1.0e-2 : 1.0e-5) : 2 * eps(eltype(args[1]))
            @test isapprox(Float32.(Array(only(T.leaves(execute!(ex))))), Float32.(interpret(p, args...)); rtol)
        end
        # With work after the product, cuBLAS's candidate is the product
        # alone, and the work after it a kernel of its own.
        w = Float32.(data(96; scale = 0.3))
        p = trace((a, b, w) -> tanh.(a * b .+ w), a, b, w)
        ex = prepare(p, CuArray(a), CuArray(b), CuArray(w); target = T.CUDATarget(cuBLAS))
        @test length(built(ex)) == 1 && length(ex.accepted) == 1 && length(ex.kernels) == 2
        @test isapprox(Array(only(T.leaves(execute!(ex)))), interpret(p, a, b, w); rtol = 1.0e-5)
        ex = prepare(trace((a, b) -> a * b, a, b), CuArray(a), CuArray(b); target = T.CUDATarget(cuTile))
        @test isempty(built(ex))
    end
    @testset "choosing between claims" begin
        # Half-precision products cuBLAS and cuTile both claim, close by
        # estimate: measured at preparation and kept in the tuning record.
        a, b = Float16.(data(1024, 256; scale = 0.1)), Float16.(data(256, 1024))
        p = trace((a, b) -> a * b, a, b)
        target = T.CUDATarget()
        prepared() = prepare(p, CuArray(a), CuArray(b); target)
        ex = prepared()
        x = only(ex.decisions)
        @test x.how === :measured && length(x.options) == 2
        @test occursin("chosen by measurement", sprint(explain, ex))
        path = T.record_path(target)
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
        @test isapprox(Array(only(T.leaves(execute!(prepared())))), interpret(p, a, b); rtol = 2 * eps(Float16))
    end
    @testset "implementation sets" begin
        w, x, b = Float32.(data(96, 80; scale = 0.1)), Float32.(data(80, 112)), Float32.(data(96))
        p = trace((w, x, b) -> gelu.(w * x .+ b), w, x, b)
        # Loading a package makes its implementations available; a target
        # names the ones it uses, and the floor is always there.
        @test T.used_packages(T.CUDATarget()) == [cuBLAS, cuTile]
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

if CUDACore.functional()
    @testset "several blocks with cuTile, Float32" begin
        @testset "$name" for (name, f, args) in LARGE
            args = map(single, args)
            for p in (trace(f, args...), differentiate(trace(f, args...)))
                reference = T.leaves(interpret(p, args...))
                ex = prepare(p, map(x -> x isa AbstractArray ? CuArray(x) : x, args)...)
                @test all(d -> !(d isa T.FloorKernels), ex.kernels)
                @test agree(T.leaves(execute!(ex)), reference; rtol = 2.0e-3)
            end
        end
    end
end
