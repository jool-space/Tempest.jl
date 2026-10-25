# What a prepared program does when run: `execute!`, `bind!`, `capture!` and
# `replay!` on one executable; and a preparation without building, which
# decides as a built one does and runs nothing.

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
